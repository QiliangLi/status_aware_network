"""盘感知臂 TFS-D：v3 之上的每盘账本（对抗二轮线 A，设计 §2.2）。

信息边界零变更：块计划用 sim/cq.storage.disk_plan（公开确定性函数，
引擎与策略同函数同结果）；prefix 由 类名:h_tokens 推导（engine._members_h
文档：公开确定性信息）；在跑流剩余量来自 PublicFlow 公开字段。

与 v3 的唯一差分：第 3 步总账本 → 每盘账本；放行条件对每盘独立判定。
块分布均匀时逐动作退化为 v3（frac≈1/盘数，每盘条件之和≈总条件）。
"""
from __future__ import annotations

from sim.cq.policies import bw_demand
from sim.cq.storage import disk_plan
from sim.cq.types import Action, JointAction
from adv.tfs_versions import TFSAdv

V3_FLAGS = {"wl", "classfreeze", "coldabs", "active", "captol",
            "dnmed", "ovswitch"}


def _prefix_of_synthetic(class_id: str, h_tokens: int) -> str:
    """与 engine._members_h 的合成负载口径一致："{class_id}:{h_tokens}"。"""
    return f"{class_id}:{h_tokens}"


class TFSDisk(TFSAdv):
    """TFS-D：每盘一本账（v3 全部开关 + 盘维度替换总账本）。

    disk_tol：每盘容差。1.10 版在 P1 形态（分散流摊入+热盘临界组合）过度
    拒收——满需求份额静态记账把 15% 内的拉伸也拒了；1.25 为二轮防守修正。
    """

    DISK_TOL = 1.10
    CAP_MED_N = 9            # 与 dnmed 同窗口：报价中位数做当前带宽

    def __init__(self, hist: dict = None, disk_tol: float = None):
        super().__init__(flags=set(V3_FLAGS), hist=hist)
        if disk_tol:
            self.DISK_TOL = disk_tol
        self._bw_ring = []   # 本类自带（避免与父类状态纠缠）

    # -- 盘几何 --
    def _disks_of(self, scn) -> int:
        b_peak = max(float(v) for _t, v in scn.storage.b_schedule)
        disk_bps = float(scn.storage.disk_bps_gbps)
        return max(1, int(-(-b_peak // disk_bps)))   # ceil

    def _fracs(self, class_id: str, h: int, disks: int):
        """(frac_per_disk, d 之外不变)——返回 {disk: 字节占比}。"""
        if h <= 0 or disks <= 1:
            return None
        plan = disk_plan(_prefix_of_synthetic(class_id, h), h, disks)
        tot = sum(nb for _d, _c, nb in plan)
        if tot <= 0:
            return None
        return {d: nb / tot for d, _c, nb in plan}

    def _disk_caps(self, snap, disks: int):
        """每盘容量 = 当前带宽观测(中位) ÷ 盘数 × 容差。"""
        bw = float(snap.b_ref)
        q = snap.quote
        if q is not None:
            self._bw_ring.append(float(q.bw_hat_gbps))
            if len(self._bw_ring) > self.CAP_MED_N:
                self._bw_ring.pop(0)
            if self._bw_ring:
                bw = sorted(self._bw_ring)[len(self._bw_ring) // 2]
        per = bw / disks * self.DISK_TOL
        return per

    def decide(self, snap, scn) -> JointAction:
        if scn.storage.alloc != "max_min_disk":
            return super().decide(snap, scn)   # 非盘物理：回退 v3
        disks = self._disks_of(scn)
        d, d_heavy, C, _hr, wake = self._d_map(snap)
        cap_d = self._disk_caps(snap, disks)
        now = float(snap.now)
        L = snap._profile_cfg.L
        req = {r.rid: r for r in snap.requests}
        # 过载开关（同 v3）
        if True:
            queued0 = [r for r in snap.requests if r.rid in snap.queued]
            depth = len(queued0)
            m = max(1, len(snap.workers))
            if depth > self.OV_ON * m:
                self._ov_run += 1
            elif depth < 0.5 * m:
                self._ov_run = 0
                self._ov_on = False
            if self._ov_run >= self.OV_K:
                self._ov_on = True

        queued = [r for r in snap.requests if r.rid in snap.queued]
        slack = {r.rid: float(r.deadline_s) - now - float(r.T0_s)
                 for r in queued}
        hv = {r.rid: self._is_heavy(snap, r.rid, d, d_heavy) for r in queued}
        fracs = {}      # rid -> {disk: frac}（只给大读算）
        for r in queued:
            if hv[r.rid]:
                fracs[r.rid] = self._fracs(r.class_id, r.h_tokens, disks)

        def key(r):
            if self._ov_on:
                return (0, float(r.deadline_s), r.rid)
            return (0, slack[r.rid], float(r.deadline_s), r.rid)

        order = sorted(queued, key=key)
        anchors = self._aged_anchors(snap, scn)
        # 每盘账本：Σ 在跑大读(未进末层)的 d × frac
        led = [0.0] * disks
        for pw in snap.workers:
            if not pw.members:
                continue
            rid0 = pw.members[0]
            r0 = req.get(rid0)
            if r0 is None or not self._is_heavy(snap, rid0, d, d_heavy) \
                    or pw.layers_done >= L:
                continue
            f = self._fracs(r0.class_id, r0.h_tokens, disks)
            if f:
                for dd, fr in f.items():
                    led[dd] += d.get(rid0, 0.0) * fr
        acts, used = [], set()
        for wid in sorted(snap.idle_workers):
            pick = None
            for r in order:
                if r.rid in used:
                    continue
                if hv[r.rid]:
                    f = fracs[r.rid]
                    if f is None:
                        pick = r.rid          # 大读但零块（不可能，兜底）
                        break
                    ok = True
                    for dd, fr in f.items():
                        if led[dd] > 1e-9 and \
                                led[dd] + d[r.rid] * fr > cap_d:
                            ok = False        # 每盘独立判定；账本为 0 的盘保底放行
                            break
                    if not ok:
                        continue
                pick = r.rid
                break
            if pick is None:
                rest = [r for r in order if r.rid not in used]
                if not any(not hv[r.rid] for r in rest):
                    for a in anchors:
                        if a in snap.queued and a not in used:
                            pick = a
                            break
            if pick is None:
                acts.append(Action("WAIT", wid, (), wake))
                continue
            acts.append(Action("DISPATCH", wid, (pick,)))
            used.add(pick)
            if hv[pick]:
                f = fracs.get(pick)
                if f:
                    for dd, fr in f.items():
                        led[dd] += d[pick] * fr
        return JointAction(tuple(acts))


def make_tfs_disk(disk_tol: float = None):
    def factory():
        return TFSDisk(disk_tol=disk_tol)
    factory.version = "v3+disk" + (f"(tol{disk_tol})" if disk_tol else "")
    return factory
