"""简单基线策略、统一填批函数与老化保护 wrapper（§3、§7.1）。

所有策略签名 decide(snapshot, scenario) -> JointAction，只见 ObservableSnapshot。
简单策略填批规则（§3.3）：排序后取队首逐项加入，遇到第一项使任一硬约束失败即
停止本批，不跳过该项继续塞后项；随后按 worker 编号升序给下一个空闲 worker
继续填批。
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from .profile import is_feasible, layer_read_gb
from .types import Action, JointAction, RequestSpec


def _specs_of(snap):
    """把快照请求转回轻量 RequestSpec（供 is_feasible/profile 使用）。"""
    out = {}
    for r in snap.requests:
        out[r.rid] = RequestSpec(
            rid=r.rid, arrival_s=r.arrival_s, h_tokens=r.h_tokens,
            u_tokens=r.u_tokens, class_id="", T0_s=r.T0_s,
            deadline_s=r.deadline_s)
    return out


def fill_batch(order: Sequence, specs: dict, limits, cfg, exclude: set,
               anchor: Optional[int] = None) -> Tuple[int, ...]:
    """排序 order（rid 列表）前缀贪心填批：首项必须可行（singleton 已验证），
    之后逐项尝试，第一项失败即停止。anchor 存在时作为首成员强制加入。"""
    members: List[RequestSpec] = []
    rids: List[int] = []
    if anchor is not None and anchor not in exclude:
        members.append(specs[anchor])
        rids.append(anchor)
    for rid in order:
        if rid in exclude or rid in rids:
            continue
        if is_feasible(members + [specs[rid]], limits, cfg):
            members.append(specs[rid])
            rids.append(rid)
        else:
            break   # 不跳过该项继续塞后项
    return tuple(rids)


def fill_batch_checked(order, specs, limits, cfg, exclude, anchor=None,
                       extra=None):
    """带能力约束钩子的填批（同 §3.3 语义）。"""
    members: List[RequestSpec] = []
    rids: List[int] = []
    if anchor is not None and anchor not in exclude:
        members.append(specs[anchor])
        rids.append(anchor)
    for rid in order:
        if rid in exclude or rid in rids:
            continue
        trial = members + [specs[rid]]
        ok = is_feasible(trial, limits, cfg)
        if ok and extra is not None:
            ok = extra(trial)
        if ok:
            members.append(specs[rid])
            rids.append(rid)
        else:
            break
    return tuple(rids)


def aged_anchor(snap, guard_w) -> Optional[int]:
    """等待 >= W_guard 的最早 (arrival,rid) 超限请求（§7.1）。"""
    if guard_w is None or guard_w <= 0:
        return None
    cands = []
    for r in snap.requests:
        if r.rid in snap.queued:
            wait = float(snap.now) - float(r.arrival_s)
            if wait >= float(guard_w):
                cands.append((r.arrival_s, r.rid))
    if not cands:
        return None
    return min(cands)[1]


class SimplePolicy:
    """排序驱动的填批策略基类；key_fn(rid, req, snap) 返回排序 tuple。
    feasibility_extra 可注入能力约束（如 E21-B 同象限限批）。"""

    def __init__(self, pid: str, key_fn, guard: bool = True, pair_mode: bool = False,
                 feasibility_extra=None):
        self.pid = pid
        self.key_fn = key_fn
        self.guard = guard
        self.pair_mode = pair_mode
        self.r_ref: Optional[float] = None   # 训练冻结的互补参考比
        self.feasibility_extra = feasibility_extra

    def _feasible(self, members, limits, cfg):
        if self.feasibility_extra is not None and not self.feasibility_extra(members):
            return False
        return is_feasible(members, limits, cfg)

    def _order(self, snap, scn) -> List[int]:
        specs = _specs_of(snap)
        queued = [r for r in snap.requests if r.rid in snap.queued]
        if self.pair_mode:
            return self._pair_order(queued, specs, snap, scn)
        keyed = sorted(queued, key=lambda r: self.key_fn(r, snap))
        return [r.rid for r in keyed]

    def _pair_order(self, queued, specs, snap, scn):
        """最老请求为锚，其余按互补距离 abs(log r_i + log r_a - 2 log r_ref)。"""
        import math
        anchored = sorted(queued, key=lambda r: (r.arrival_s, r.rid))
        if not anchored:
            return []
        a = anchored[0]

        def ratio(r):
            V = float(layer_read_gb(specs[r.rid], scn.profile))
            K = float(r.T0_s)
            if V <= 0 or K <= 0:
                return None
            return V / K

        ra = ratio(a)
        if not self.r_ref:
            import math as _m
            pos = [ratio(r.rid) for r in queued if ratio(r.rid)]
            self.r_ref = (_m.exp(sum(_m.log(x) for x in pos) / len(pos))
                          if pos else 1.0)
        r_ref = self.r_ref

        def dist(r):
            ri = ratio(r)
            if ri is None or ra is None or ra <= 0:
                return (1, 0.0, r.arrival_s, r.rid)   # 零读取排后，按到达序
            d = abs(math.log(ri) + math.log(ra) - 2 * math.log(r_ref))
            return (0, d, r.arrival_s, r.rid)

        rest = sorted(anchored[1:], key=dist)
        return [a.rid] + [r.rid for r in rest]

    def decide(self, snap, scn) -> JointAction:
        specs = _specs_of(snap)
        order = self._order(snap, scn)
        anchors = []
        if self.guard:
            anchors = self._aged_anchors(snap, scn)
        acts = []
        exclude: set = set()
        for w in sorted(snap.idle_workers):
            anchor = anchors[0] if anchors else None
            members = fill_batch_checked(
                order, specs, scn.limits, scn.profile, exclude, anchor,
                extra=self.feasibility_extra)
            if not members:
                continue
            acts.append(Action("DISPATCH", w, members))
            exclude |= set(members)
            if anchors and anchors[0] in members:
                anchors = anchors[1:]
        return JointAction(tuple(acts))

    def _aged_anchors(self, snap, scn) -> List[int]:
        """全部超限请求按 (arrival,rid) 升序；依次作为各 worker 的强制锚点。"""
        gw = getattr(scn, "guard_w", None)
        if not gw or gw <= 0:
            return []
        cands = sorted(
            [(r.arrival_s, r.rid) for r in snap.requests
             if r.rid in snap.queued
             and float(snap.now) - float(r.arrival_s) >= float(gw)])
        return [rid for _, rid in cands]


# ---------------------------------------------------------------------------
# 具体策略：排序键（§3.1）
# ---------------------------------------------------------------------------

def make_fcfs():
    return SimplePolicy("cq_fcfs", lambda r, s: (r.arrival_s, r.rid))


def make_lpm():
    return SimplePolicy("cq_lpm", lambda r, s: (-r.h_tokens, r.arrival_s, r.rid))


def make_spt():
    return SimplePolicy("cq_spt", lambda r, s: (r.T0_s, r.arrival_s, r.rid))


def make_edf():
    return SimplePolicy("cq_edf", lambda r, s: (r.deadline_s, r.arrival_s, r.rid))


def make_slack():
    def key(r, s):
        slack = float(r.deadline_s) - float(s.now) - s.pred_singleton_s(r.rid)
        return (slack, r.arrival_s, r.rid)
    return SimplePolicy("cq_slack", key)


def make_pair(r_ref: Optional[float] = None):
    p = SimplePolicy("cq_pair", None, pair_mode=True)
    p.r_ref = r_ref
    return p


class GuardedEDF(SimplePolicy):
    """MPC/local 的固定 continuation π（§7.1）：cq_edf + 老化保护。"""

    def __init__(self):
        super().__init__("cq_edf#pi", lambda r, s: (r.deadline_s, r.arrival_s, r.rid))


SIMPLE_POLICIES = {
    "cq_fcfs": make_fcfs,
    "cq_lpm": make_lpm,
    "cq_spt": make_spt,
    "cq_edf": make_edf,
    "cq_slack": make_slack,
    "cq_pair": make_pair,
}
