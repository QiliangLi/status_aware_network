"""TFS 规则式调度器不变量单测(设计 docs/TFS规则式调度器-20261014.md §4.4-2)。

U1 配额守恒:D1f 三类混合全 run,每次 decide 后重流并发 ≤ C;WAIT 唤醒
严格未来且有界。/ U2 填空与守门形态:t=0 突发 16A+16B+16C、32 空闲 →
轻流(C/B)填满、重流(A)恰 C=4 条。/ U3 slack 序:slack 与 deadline 序
相反的构造下 TFS 取 slack 小者(EDF 取另一者,对照)。/ U4 锚点让位与
豁免:锚点不越过未派轻流;配额满且无轻流时锚点豁免超额派发。/
U5 确定性(CRN)。/ U6 V=0 兼容:纯 C 排空无 stalled、makespan=8c。/
U7 退化等价:单类重流快照 TFS≡GuardedEDFBW(逐动作)。
"""
import sys
from dataclasses import replace
from fractions import Fraction as F

import pytest

sys.path.insert(0, ".")

from sim.cq.engine import CqEngine
from sim.cq.observable import Observable
from sim.cq.policies import (GuardedEDF, GuardedEDFBW, TFS, bw_demand,
                             make_edf, make_fcfs, make_tfs)
from sim.cq.profile import batch_compute_s, make_T0
from sim.cq.simrun import run_case
from sim.cq.types import RequestSpec
from sim.experiments import e26_ab as e26
from sim.experiments.e26f_third import C_A_CLASS, d1f_seed, d1f_specs


def _snap_of(scn, specs):
    """t=0 到达后的快照(物理步不派发,策略人工调用;沿 test_cq_mpcv2 模式)。"""
    obs = Observable(scn, numeric=float)
    eng = CqEngine(scn, specs, GuardedEDF(), numeric=float,
                   fallback_policy=GuardedEDF(), observable=obs)
    obs.attach(eng)
    eng._physical_step(0.0)
    return obs.snapshot(0.0)


def _cls_of(specs, rid):
    return {s.rid: s.class_id for s in specs}[rid]


# ---------------------------------------------------------------------------
# U1 配额守恒 + WAIT 合法(全 run,D1f 三类小规模)
# ---------------------------------------------------------------------------

class _Recorder(TFS):
    def __init__(self):
        super().__init__()
        self.heavy_after = []
        self.wakes = []

    def decide(self, snap, scn):
        ja = super().decide(snap, scn)
        _d, _dh, _C, heavy_run, _w = bw_demand(snap, self._hist)
        extra = sum(1 for a in ja.actions if a.kind == "DISPATCH"
                    and _d[a.members[0]] >= _dh)
        self.heavy_after.append((heavy_run + extra, _C))
        for a in ja.actions:
            if a.kind == "WAIT":
                self.wakes.append((float(a.wake_at), float(snap.now)))
        return ja


def test_u1_quota_invariant_full_run():
    specs, _c = d1f_specs(16, d1f_seed(2, 0), C_A_CLASS, n=40)
    scn = e26.e26_scenario(4)
    pol = _Recorder()
    eng = run_case(scn, specs, pol, numeric=float, seed=0)
    assert eng.status == "done"
    assert all(rr.F_s is not None for rr in eng.w.requests.values())
    # E26 guard_w=0 → 无老化豁免,重并发恒 ≤ C
    assert all(h <= c for h, c in pol.heavy_after), pol.heavy_after
    assert all(w > now and w - now <= 0.2 + 1e-9
               for w, now in pol.wakes)


# ---------------------------------------------------------------------------
# U2 填空与守门形态:t=0 突发(32 空闲,16A+16B+16C 排队)
# ---------------------------------------------------------------------------

def test_u2_burst_shape():
    events = ["A"] * 16 + ["B"] * 16 + ["C"] * 16
    cmap = {"A": e26.A_CLASS, "B": e26.B_CLASS, "C": C_A_CLASS}
    specs = [RequestSpec(i, 0.0, *cmap[cl], cl, F(1), F(1))
             for i, cl in enumerate(events)]
    T0 = make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    specs = [RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens, s.class_id,
                         T0[s.rid], float(T0[s.rid]) * 4) for s in specs]
    scn = e26.e26_scenario(4)
    snap = _snap_of(scn, specs)
    d, d_heavy, C, _hr, _w = bw_demand(snap, TFS()._hist)
    assert C == 4                          # 120/29.18(A=P75)→4
    ja = TFS().decide(snap, scn)
    disp = [a.members[0] for a in ja.actions if a.kind == "DISPATCH"]
    by_cls = {"A": 0, "B": 0, "C": 0}
    for rid in disp:
        by_cls[_cls_of(specs, rid)] += 1
    # 填空:16 条零读 C 全派;守门:重流 A 恰 4=C;余位由轻流 B 填
    assert by_cls == {"A": 4, "B": 12, "C": 16}, by_cls
    assert len(disp) == 32                 # 32 空闲全用满,无 WAIT


# ---------------------------------------------------------------------------
# U3 slack 序:与 deadline 序相反时取 slack 小者
# ---------------------------------------------------------------------------

def test_u3_slack_order():
    # r0: T0=1.9s/deadline=2.0s → slack=0.1;r1: T0=0.2s/deadline=1.5s → slack=1.3
    # deadline 序取 r1(1.5<2.0);slack 序取 r0(0.1<1.3)
    specs = [RequestSpec(0, 0.0, *e26.B_CLASS, "B", F("1.9"), F("2.0")),
             RequestSpec(1, 0.0, *e26.B_CLASS, "B", F("0.2"), F("1.5"))]
    scn = replace(e26.e26_scenario(4), m_workers=1)
    snap = _snap_of(scn, specs)
    ja_t = TFS().decide(snap, scn)
    ja_e = make_edf().decide(snap, scn)
    pick_t = next(a.members[0] for a in ja_t.actions if a.kind == "DISPATCH")
    pick_e = next(a.members[0] for a in ja_e.actions if a.kind == "DISPATCH")
    assert pick_t == 0 and pick_e == 1     # TFS 取 slack 小者,EDF 取 deadline 早者


# ---------------------------------------------------------------------------
# U4 锚点:不越过未派轻流;配额满且无轻流时豁免超额
# ---------------------------------------------------------------------------

class _FixedAnchors(TFS):
    def __init__(self, anchors):
        super().__init__()
        self._fixed = anchors

    def _aged_anchors(self, snap, scn):
        return self._fixed


def _specs_mix(n_a, n_b):
    cmap = {"A": e26.A_CLASS, "B": e26.B_CLASS}
    events = ["A"] * n_a + ["B"] * n_b
    specs = [RequestSpec(i, 0.0, *cmap[cl], cl, F(1), F(1))
             for i, cl in enumerate(events)]
    T0 = make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    return [RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens, s.class_id,
                        T0[s.rid], float(T0[s.rid]) * 4) for s in specs]


def test_u4_anchor_yield_and_exemption():
    scn = e26.e26_scenario(4)
    # 场景一:8A+2B、锚点=a5(重)。slack 序 A 先:配额 4 条;B 补位;B 派尽后
    # 锚点才豁免 → a5 恰派一次(第 7 个派发),不越过 B
    specs = _specs_mix(8, 2)
    snap = _snap_of(scn, specs)
    ja = _FixedAnchors([4]).decide(snap, scn)
    disp = [a.members[0] for a in ja.actions if a.kind == "DISPATCH"]
    a_disp = [r for r in disp if _cls_of(specs, r) == "A"]
    b_disp = [r for r in disp if _cls_of(specs, r) == "B"]
    assert len(a_disp) == 5 and 4 in a_disp       # 4 配额 + 1 锚点豁免
    assert len(b_disp) == 2                       # 轻流全派(锚点未越位)
    # 场景二:同构造、无锚点 → 恰 4A,余 worker WAIT
    snap2 = _snap_of(scn, _specs_mix(8, 2))
    ja2 = TFS().decide(snap2, scn)
    disp2 = [a.members[0] for a in ja2.actions if a.kind == "DISPATCH"]
    waits = [a for a in ja2.actions if a.kind == "WAIT"]
    assert sum(1 for r in disp2 if _cls_of(specs, r) == "A") == 4
    assert len(waits) == 32 - 6                   # 4A+2B 派尽,余 26 个 WAIT


# ---------------------------------------------------------------------------
# U5 确定性(CRN):同流两跑 F_s 逐位一致
# ---------------------------------------------------------------------------

def test_u5_determinism():
    specs, _c = d1f_specs(16, 424, C_A_CLASS, n=30)
    scn = e26.e26_scenario(4)
    e1 = run_case(scn, specs, make_tfs(), numeric=float, seed=0)
    e2 = run_case(scn, specs, make_tfs(), numeric=float, seed=0)
    same = sum(1 for rid in e1.w.requests
               if e1.w.requests[rid].F_s == e2.w.requests[rid].F_s)
    assert same == len(specs)


# ---------------------------------------------------------------------------
# U6 V=0 兼容:纯 C 排空无 stalled、makespan=8c 解析吻合
# ---------------------------------------------------------------------------

def test_u6_pure_c_no_stall():
    scn = e26.e26_scenario(4)
    specs = [RequestSpec(i, 0.0, 0, 2657, "C", F(1), F(1))
             for i in range(12)]
    T0 = make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    specs = [RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens, "C",
                         T0[s.rid], float(T0[s.rid]) * 4) for s in specs]
    eng = run_case(scn, specs, make_tfs(), numeric=float, seed=0)
    assert all(eng.w.requests[r].F_s is not None for r in range(12))
    stall_w = sum(float(e) - float(s)
                  for wk in eng.w.workers.values()
                  for (s, e, st, _b, _l) in wk.segments if st == "STALL")
    assert stall_w == 0.0
    c8 = 8 * float(batch_compute_s([specs[0]], e26.E26_PROFILE))
    assert abs(float(eng.w.t) - c8) < 1e-9


# ---------------------------------------------------------------------------
# U7 退化等价:单类重流快照 TFS ≡ GuardedEDFBW(逐动作)
# ---------------------------------------------------------------------------

def test_u7_equivalence_single_class_heavy():
    specs = _specs_mix(16, 0)                  # 全 A(重流单类,T0 全同)
    scn = e26.e26_scenario(4)
    snap = _snap_of(scn, specs)
    ja_t = TFS().decide(snap, scn)
    ja_b = GuardedEDFBW().decide(snap, scn)
    assert ja_t == ja_b                        # slack 序≡deadline 序(单类)


def test_u7_all_light_dispatch_all():
    specs = _specs_mix(0, 16)                  # 全 B(轻流)
    scn = e26.e26_scenario(4)
    snap = _snap_of(scn, specs)
    ja = TFS().decide(snap, scn)
    disp = [a.members[0] for a in ja.actions if a.kind == "DISPATCH"]
    assert sorted(disp) == list(range(16))     # 轻流恒可派:全部派出
    assert len([a for a in ja.actions if a.kind == "WAIT"]) == 32 - 16
