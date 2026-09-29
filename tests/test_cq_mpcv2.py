"""mpc-v2 修正（设计文档 20260929）单测：T1 BW-EDF 不变量 / T2 组合器不变量 /
T3 口径开关回归 / T4 健康计数。

旧口径默认 (base="edf", composer=False) 与现行逐位等价由既有金标
（test_cq_e25 的 29.0/31.5、test_cq_e26 的 U6/U7）覆盖，此处只断言默认值。
"""
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, ".")

from sim.cq.engine import CqEngine
from sim.cq.observable import Observable
from sim.cq.policies import GuardedEDF, GuardedEDFBW, bw_demand
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.types import Action, JointAction
from sim.experiments.e26_ab import A_CLASS, B_CLASS, E26_PROFILE, e26_scenario
from sim.cq.types import RequestSpec
from sim.cq.profile import make_T0


def _world(n_round=1, alpha=4):
    order = ["A", "B", "B"] * (32 * n_round)
    specs = [RequestSpec(i, 0.0, *(A_CLASS if c == "A" else B_CLASS), c, F(1), F(1))
             for i, c in enumerate(order)]
    T0 = make_T0(specs, E26_PROFILE, F(120), F(200))
    return [RequestSpec(s.rid, 0.0, s.h_tokens, s.u_tokens, s.class_id,
                        T0[s.rid], float(T0[s.rid]) * alpha) for s in specs]


# ---------------------------------------------------------------------------
# T3 口径开关：默认=旧口径；est 闭环 π 随 base 切换
# ---------------------------------------------------------------------------

def test_t3_defaults_and_pi_switch():
    pol = MPCPolicy(pid="cq_mpc", H=1, theta="S", c_ref=0.2)
    assert pol.base == "edf" and pol.composer is False
    assert isinstance(pol.pi, GuardedEDF)
    pol2 = MPCPolicy(pid="cq_mpc", H=1, theta="S", c_ref=0.2,
                     base="bw_edf", composer=True)
    assert isinstance(pol2.pi, GuardedEDFBW)


# ---------------------------------------------------------------------------
# T1 BW-EDF 不变量：确定性 / 全完成 / 派发时刻重并发 ≤ C / WAIT 唤醒合法
# ---------------------------------------------------------------------------

class _Recorder(GuardedEDFBW):
    """记录每次 decide 后的重并发与 WAIT 唤醒时刻。"""
    def __init__(self):
        super().__init__()
        self.heavy_after = []
        self.wakes = []

    def decide(self, snap, scn):
        ja = super().decide(snap, scn)
        _d, _dh, _C, heavy_run, wake = bw_demand(snap, self._hist)
        extra_heavy = sum(1 for a in ja.actions if a.kind == "DISPATCH"
                          and _d[a.members[0]] >= _dh)
        self.heavy_after.append((heavy_run + extra_heavy, _C))
        for a in ja.actions:
            if a.kind == "WAIT":
                self.wakes.append((float(a.wake_at), float(snap.now)))
        return ja


def test_t1_bwedf_invariants():
    specs = _world(n_round=1)
    scn = e26_scenario(4)
    pol = _Recorder()
    eng = run_case(scn, specs, pol)
    assert eng.status == "done"
    assert all(rr.F_s is not None for rr in eng.w.requests.values())
    # 重并发 ≤ C（无老化豁免：E26 guard_w=0）
    assert all(h <= c for h, c in pol.heavy_after), pol.heavy_after
    # WAIT 唤醒严格未来且有界
    assert all(w > now and w - now <= 0.2 + 1e-9
               for w, now in pol.wakes)
    # 确定性：同流两跑同 SLO
    pol2 = _Recorder()
    eng2 = run_case(scn, specs, pol2)
    same = sum(1 for rid in eng.w.requests
               if eng.w.requests[rid].F_s == eng2.w.requests[rid].F_s)
    assert same == len(specs)


def test_t1_bwedf_burst_shape():
    """t=0 突发（32 空闲、32A+64B 排队）：首动作派 A 数 ≤ C（E26=4）。"""
    specs = _world(n_round=1)
    scn = e26_scenario(4)
    obs = Observable(scn, numeric=float)
    eng = CqEngine(scn, specs, GuardedEDFBW(), numeric=float,
                   fallback_policy=GuardedEDF(), observable=obs)
    obs.attach(eng)
    eng._physical_step(0.0)
    snap = obs.snapshot(0.0)
    pol = GuardedEDFBW()
    ja = pol.decide(snap, scn)
    n_a = sum(1 for a in ja.actions if a.kind == "DISPATCH"
              and next(r for r in snap.requests if r.rid == a.members[0]).h_tokens
              == A_CLASS[0])
    _d, _dh, C, _hr, _w = bw_demand(snap, pol._hist)
    assert C == 4, (_dh, C)          # 120/29.18 → 4（E26 甜点）
    assert n_a <= C
    assert n_a >= 1                  # 轻流优先不阻塞，重流受限


# ---------------------------------------------------------------------------
# T2 组合器不变量：合法 / 重数 ≤ cap 上界 / 确定性 / 排评分队首
# ---------------------------------------------------------------------------

def test_t2_composer_invariants():
    specs = _world(n_round=1)
    scn = e26_scenario(4)
    obs = Observable(scn, numeric=float)
    eng = CqEngine(scn, specs, GuardedEDF(), numeric=float,
                   fallback_policy=GuardedEDF(), observable=obs)
    obs.attach(eng)
    eng._physical_step(0.0)
    snap = obs.snapshot(0.0)
    pol = MPCPolicy(pid="cq_mpc", H=1, theta="S", c_ref=0.2,
                    base="bw_edf", composer=True)
    from sim.cq.search import build_batches, build_forecast_engine
    est0 = build_forecast_engine(snap, scn, pol.pi)
    node_snap = est0.observable.snapshot(est0.w.t)
    batches = build_batches(node_snap, scn, pol.pi, pol.K_req, pol.K_batch)
    acts = pol._joint_actions(est0, node_snap, scn, batches,
                              sorted(snap.idle_workers))
    d, d_heavy, C, _hr, _w = bw_demand(node_snap, pol._d_hist)
    queued = set(node_snap.queued)
    idle = set(sorted(snap.idle_workers))
    composed = pol._compose_bw(node_snap, sorted(snap.idle_workers))
    assert 1 <= len(composed) <= 3
    for a in composed:
        for x in a.actions:
            if x.kind == "DISPATCH":
                assert set(x.members) <= queued
                assert x.worker_id in idle
            else:
                assert x.wake_at > float(node_snap.now)
        # 联合动作内重数 ≤ 网格上界 2C
        n_heavy = sum(1 for x in a.actions if x.kind == "DISPATCH"
                      and d[x.members[0]] >= d_heavy)
        assert n_heavy <= 2 * C
    # 组合候选排在评分队列最前
    keys = [a.key() for a in composed]
    assert acts[:len(keys)] == keys or all(
        any(a.key() == k for a in acts[:3]) for k in keys)
    # 确定性
    composed2 = pol._compose_bw(node_snap, sorted(snap.idle_workers))
    assert [a.key() for a in composed] == [a.key() for a in composed2]


# ---------------------------------------------------------------------------
# T4 健康计数：v2 全链路跑通，n_tie/n_deviate 合法
# ---------------------------------------------------------------------------

def test_t4_v2_health_counters():
    specs = _world(n_round=1)
    pol = MPCPolicy(pid="cq_mpc", H=1, theta="S", c_ref=0.2, budget_s=3600.0,
                    base="bw_edf", composer=True)
    eng = run_case(e26_scenario(4), specs, pol)
    assert eng.status == "done"
    h = pol.health()
    assert h["n_searched"] >= 1 and h["n_fallback"] == 0
    assert 0 <= h["n_deviate"] <= h["n_searched"]
    assert h["n_tie"] >= 0
