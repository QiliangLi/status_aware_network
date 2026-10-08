"""CPA（闭式管道接纳）单元测试（设计文档 docs/CPA闭式管道接纳算法设计-20261008.md §6）。

U1 未失速公式与引擎解析吻合；U2 饱和公式对拍（探针口径 ≤0.2ms）；
U3 存活阈值 k*=16 由引擎兑现；U4 CPA 决策=完整族最优点（16+WAIT）且
动作合法；U5 端到端确定性与不劣于 FCFS 底线；U6 参照工具（DPO/OPT）
口径自洽。
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from sim.cq.policies import make_fcfs
from sim.cq.poly import (CPAPolicy, wave_F, survival_cap, mixed_wave_F,
                         decision_point_optimum, offline_optimum,
                         _snapshot_of)
from sim.cq.simrun import run_case
from sim.cq.metrics import summarize
from sim.experiments import e26_ab as e26

V_A = float(F(4096, 10**9) * e26.A_CLASS[0])       # 0.17565 GB/层
V_B = float(F(4096, 10**9) * e26.B_CLASS[0])
S_A = V_A / 120.0                                   # 1.4638ms
C_A, C_B = 0.00602, 0.02859
T0_A = V_A / 120.0 + 8 * C_A
DL_A = 4 * T0_A


def mk_specs(classes, arrivals=None, alpha=4):
    arrivals = arrivals or [0.0] * len(classes)
    specs = []
    for i, c in enumerate(classes):
        h, u = e26.A_CLASS if c == "A" else e26.B_CLASS
        specs.append(e26.RequestSpec(i, float(arrivals[i]), h, u, c, F(1), F(1)))
    T0 = e26.make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    return [e26.RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens,
                            s.class_id, T0[s.rid],
                            float(s.arrival_s) + alpha * float(T0[s.rid]))
            for s in specs]


def run_fcfs_fs(specs):
    eng = run_case(e26.e26_scenario(4), specs, make_fcfs(), numeric=float,
                   seed=0)
    return [float(eng.w.requests[i].F_s) for i in range(len(specs))]


# -- U1/U2: 公式层对拍 -------------------------------------------------------

def test_u1_unsaturated_formula_exact():
    """k=1：F = s + L·c 与引擎逐位吻合（1e-9）。

    k∈[2,7] 为层0阻塞+到期提升的过渡区，闭式偏差 0.1~1.6ms（设计文档
    §5 披露；见 test_u2b_transition_zone），k≥8 全饱和段回到精确。"""
    Fs = run_fcfs_fs(mk_specs(["A"] * 1))
    assert Fs[0] == pytest.approx(S_A + 8 * C_A, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("k,tol", [(2, 3e-4), (4, 2e-3), (5, 2e-3),
                                   (7, 2e-3)])
def test_u2b_transition_zone(k, tol):
    """过渡区（k=2..7）统一式偏差有界（实测 ≤1.6ms @k=4）。"""
    Fs = run_fcfs_fs(mk_specs(["A"] * k))
    expect = wave_F(k, k, S_A, C_A, 8)
    assert abs(max(Fs) - expect) <= tol, (k, max(Fs), expect)


@pytest.mark.parametrize("k", [8, 12, 16, 17, 20, 32])
def test_u2_saturated_formula_within_probe_tolerance(k):
    """全饱和段（k≥8）统一式 F_last=L·k·s+c 对拍（≤0.2ms 绝对误差）。"""
    Fs = run_fcfs_fs(mk_specs(["A"] * k))
    expect = wave_F(k, k, S_A, C_A, 8)
    assert max(Fs) == pytest.approx(expect, abs=2e-4), (k, max(Fs), expect)


def test_u2b_mixed_saturated_exact():
    """混合饱和段 β=1 精确：16A+16B 同刻，A 末位 ≈ 229.3ms（引擎事实 4）。"""
    specs = mk_specs(["A"] * 16 + ["B"] * 16)
    eng = run_case(e26.e26_scenario(4), specs, make_fcfs(), numeric=float,
                   seed=0)
    fa = [float(eng.w.requests[i].F_s) for i in range(16)]
    expect = 16 * S_A + (8 - 1) * (16 * S_A + 16 * (V_B / 120.0)) + C_A
    assert max(fa) == pytest.approx(expect, rel=0.02), (max(fa), expect)


# -- U3: 存活阈值 -----------------------------------------------------------

def test_u3_survival_cap_formula():
    assert survival_cap(0.0, DL_A, S_A, C_A, 8) == 16


@pytest.mark.parametrize("k,alive", [(16, 16), (20, 0)])
def test_u3_survival_engine(k, alive):
    """引擎兑现：16 全活；20 全灭（α=4，空管道同刻）。"""
    Fs = run_fcfs_fs(mk_specs(["A"] * k))
    got = sum(1 for x in Fs if x <= DL_A)
    assert got == alive


def test_u3b_hold_b_saves_a():
    """事实 4：16A 先行（B 延后 0.3s）A 全活 vs 同刻 A 全灭。"""
    specs_same = mk_specs(["A"] * 16 + ["B"] * 16)
    eng = run_case(e26.e26_scenario(4), specs_same, make_fcfs(), seed=0)
    a_same = sum(1 for i in range(16)
                 if float(eng.w.requests[i].F_s) <= DL_A)
    specs_hold = mk_specs(["A"] * 16 + ["B"] * 16,
                          arrivals=[0.0] * 16 + [0.3] * 16)
    eng2 = run_case(e26.e26_scenario(4), specs_hold, make_fcfs(), seed=0)
    a_hold = sum(1 for i in range(16)
                 if float(eng2.w.requests[i].F_s) <= DL_A)
    assert (a_same, a_hold) == (0, 16)


# -- U4: CPA 决策 ------------------------------------------------------------

def test_u4_cpa_picks_survival_threshold():
    """32A 突发：form 模式首决策 = 派 16 + WAIT 16（完整族优于截断网格）。"""
    scn = e26.e26_scenario(4)
    _eng, snap = _snapshot_of(scn, mk_specs(["A"] * 32))
    act = CPAPolicy(mode="form").decide(snap, scn)
    d = [a for a in act.actions if a.kind == "DISPATCH"]
    w = [a for a in act.actions if a.kind == "WAIT"]
    assert len(d) == 16 and len(w) == 16


def test_u4b_cpa_holds_b_at_mixed_burst():
    """16A+16B 突发：rollout 模式选 16A+16WAIT（不让 B 切入 A 波）。"""
    scn = e26.e26_scenario(4)
    _eng, snap = _snapshot_of(scn, mk_specs(["A"] * 16 + ["B"] * 16))
    act = CPAPolicy(mode="rollout").decide(snap, scn)
    d = [a for a in act.actions if a.kind == "DISPATCH"]
    assert len(d) == 16
    assert all(next(r for r in snap.requests
                    if r.rid == a.members[0]).h_tokens == e26.A_CLASS[0]
               for a in d)


def test_u4c_action_valid_and_deterministic():
    """CPA 端到端：无 stale invalid、无回退；两次运行逐位一致。"""
    specs = mk_specs(["A", "B", "B"] * 8)   # 24 任务 D2 迷你
    scn = e26.e26_scenario(4)
    outs = []
    for _ in range(2):
        pol = CPAPolicy(mode="form")
        eng = run_case(scn, specs, pol, numeric=float, seed=0)
        assert eng.n_stale_invalid == 0
        s = summarize(eng, scn)
        outs.append((s["slo_success"], round(float(eng.w.t), 9),
                     pol.n_decides))
    assert outs[0] == outs[1]


def test_u4d_cpa_not_worse_than_fcfs_mini():
    """底线：D2 迷你格点 CPA ≥ FCFS（SLO 数）。"""
    specs = mk_specs(["A", "B", "B"] * 8)
    scn = e26.e26_scenario(4)
    s_poly = summarize(run_case(scn, specs, CPAPolicy(mode="form"),
                                seed=0), scn)
    s_fcfs = summarize(run_case(scn, specs, make_fcfs(), seed=0), scn)
    assert s_poly["slo_success"] >= s_fcfs["slo_success"]


# -- U6: 参照工具 ------------------------------------------------------------

def test_u6_offline_opt_small():
    r = offline_optimum(mk_specs(["A"] * 4 + ["B"] * 2),
                        e26.e26_scenario(4), time_budget_s=60)
    assert r["status"] == "optimal" and r["best_score"][0] == 0


def test_u4e_symmetry_guard():
    """普适性质疑回应（v3.3）：形状数≈请求数时守卫触发，回退 BW 底座。

    mooncake 逐条异形（212 请求 233 形状）下规则层退化为 V 降序贪心
    （实测 40.6% vs 底座 61.3%）——适用域必须显式声明并回退。
    """
    shapes = [(h, u) for h in range(1, 24) for u in (128, 256)]
    specs = []
    for i, (h, u) in enumerate(shapes):
        specs.append(e26.RequestSpec(i, 0.0, h, u, "mooncake", F(1), F(1)))
    T0 = e26.make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    specs = [e26.RequestSpec(s.rid, 0.0, s.h_tokens, s.u_tokens, s.class_id,
                             T0[s.rid], 4 * float(T0[s.rid]))
             for s in specs]
    scn = e26.e26_scenario(4)
    _eng, snap = _snapshot_of(scn, specs)
    pol = CPAPolicy(mode="form")
    act = pol.decide(snap, scn)
    base = pol.pi.decide(snap, scn)
    assert pol.n_guard >= 1
    assert act.key() == base.key()


def test_u6b_decision_point_optimum():
    r = decision_point_optimum(mk_specs(["A"] * 8 + ["B"] * 8),
                               e26.e26_scenario(4))
    assert r["n_actions"] == 81 and r["best"] is not None
    assert r["best"][0][0] == 0      # 该例无 miss，最优=TTFT 裁决
