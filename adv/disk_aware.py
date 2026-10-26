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
        if getattr(self, "feas_mode", False):
            return self._decide_feas(snap, scn)
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


class TFSDiskFeas(TFSDisk):
    """每盘期限可行性放行（容差两难的解法设计，20261026）。

    静态份额账本的根因：把「满需求份额」恒定记账，用一条全局容差代理
    「拉伸可否被期限吸收」——P1（在住者宽松，拉伸无害）需要松、P4（在住
    者紧、对齐簇）需要紧，无常量可兼顾。解法：容差不设常量，直接按每盘
    waterfill 比例缩放的代数预测做期限算术——
      放行 c ⇔ 对每个盘 d：新 Σ需求 Σ'=D_d+d_c·f_cd，缩放 s=max(1,Σ'/cap_d)；
      候选与所有在住者 i 的预测完成时刻 now + s·(剩余读量/d_i) ≤ 各自期限。
    仍是零推演（一步代数，非模拟）；全部公开量。"""

    feas_mode = True

    def _decide_feas(self, snap, scn) -> JointAction:
        disks = self._disks_of(scn)
        d, d_heavy, C, _hr, wake = self._d_map(snap)
        cap_d = self._disk_caps(snap, disks) / self.DISK_TOL  # 回除容差=纯容量
        now = float(snap.now)
        L = snap._profile_cfg.L
        req = {r.rid: r for r in snap.requests}
        queued = [r for r in snap.requests if r.rid in snap.queued]
        slack = {r.rid: float(r.deadline_s) - now - float(r.T0_s)
                 for r in queued}

        def key(r):
            return (0, slack[r.rid], float(r.deadline_s), r.rid)

        order = sorted(queued, key=key)
        # 在住者状态：每盘需求份额、剩余读量、期限
        members = []          # (d_i, frac, rem_read_time, deadline, layers_done)
        D = [0.0] * disks
        for pw in snap.workers:
            if not pw.members:
                continue
            rid0 = pw.members[0]
            r0 = req.get(rid0)
            if r0 is None:
                continue
            di = d.get(rid0, 0.0)
            if di <= 0 or pw.layers_done >= L:
                continue
            f = self._fracs(r0.class_id, r0.h_tokens, disks)
            if not f:
                continue
            rem_layers = L - pw.layers_done
            rem_read = rem_layers / L * (float(r0.T0_s) * di and 1)  # 占位，下行算
            # 自然读时间 = 剩余读量 / d；剩余读量 = V·rem_layers/L，V/d = T0 的读部分
            # 用 V/d = (T0 − 纯算) 近似：直接 rem_layers × (V_layer/d)
            V_layer = float(snap._profile_cfg.kappa_gb_per_token_layer) * r0.h_tokens
            rem_read = rem_layers * V_layer / di
            members.append((di, f, rem_read, float(r0.deadline_s)))
            for dd, fr in f.items():
                D[dd] += di * fr

        def admit(cand) -> bool:
            dc = d.get(cand.rid, 0.0)
            if dc <= 0:
                return True                      # 零读恒可行
            f = self._fracs(cand.class_id, cand.h_tokens, disks)
            if not f:
                return True
            prof = snap._profile_cfg
            V_layer_c = float(prof.kappa_gb_per_token_layer) * cand.h_tokens
            nat = L * V_layer_c / dc             # 候选自然读时间（=Σ层 V/d）
            # 短路 1：候选加入后所有涉及盘都不超容量 → 无拉伸，直接放行
            stretched = [dd for dd, fr in f.items()
                         if D[dd] + dc * fr > cap_d]
            if not stretched:
                return True
            # 候选自身：now + s·nat ≤ deadline
            for dd in stretched:
                s_str = (D[dd] + dc * f[dd]) / cap_d
                if now + s_str * nat > float(cand.deadline_s):
                    return False
            # 短路 2：只查被拉伸盘上有份额的在住者
            for di, fm, rem_read, dl in members:
                hit = [dd for dd in fm if dd in stretched and fm[dd] > 0]
                if not hit:
                    continue
                dd0 = max(hit, key=lambda x: (D[x] + dc * f.get(x, 0.0)) / cap_d)
                s_str = (D[dd0] + dc * f.get(dd0, 0.0)) / cap_d
                if now + s_str * rem_read > dl:
                    return False
            return True

        acts, used = [], set()
        for wid in sorted(snap.idle_workers):
            pick = None
            for r in order:
                if r.rid in used:
                    continue
                if admit(r):
                    pick = r.rid
                    break
            if pick is None:
                acts.append(Action("WAIT", wid, (), wake))
                continue
            acts.append(Action("DISPATCH", wid, (pick,)))
            used.add(pick)
            # 放行后即时更新在住者视图（同刻多 worker）
            r0 = req[pick]
            di = d.get(pick, 0.0)
            if di > 0:
                f = self._fracs(r0.class_id, r0.h_tokens, disks)
                prof = snap._profile_cfg
                V_layer = float(prof.kappa_gb_per_token_layer) * r0.h_tokens
                members.append((di, f, L * V_layer / di, float(r0.deadline_s)))
                if f:
                    for dd, fr in f.items():
                        D[dd] += di * fr
        return JointAction(tuple(acts))


def make_tfs_disk(disk_tol: float = None):
    def factory():
        return TFSDisk(disk_tol=disk_tol)
    factory.version = "v3+disk" + (f"(tol{disk_tol})" if disk_tol else "")
    return factory
