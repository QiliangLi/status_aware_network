"""E26 A/B 双类带宽争抢实验测试（大纲 §5 U1–U5 + 推演修复回归 U6）。

U1 标定对拍 / U2 生成器确定性与配比 / U3 batch=1 不变量 / U4 字节守恒 /
U5 冒烟 run 无未完成；U6 为 20260928 build_forecast_engine 批/流 id 错配
修复的回归（E26 全读工况下推演世界必须能排空、MPC 必须有候选被评分）。
"""
import sys
from fractions import Fraction as F

import numpy as np
import pytest

sys.path.insert(0, ".")

from sim.cq.profile import batch_compute_s, layer_read_gb, singleton_K
from sim.cq.simrun import run_case
from sim.cq.policies import make_fcfs
from sim.cq.search import MPCPolicy
from sim.cq.timeseries import aggregate_run
from sim.experiments import e26_ab as e26
from sim.experiments.e26_ab import (A_CLASS, B_CLASS, E26_PROFILE,
                                    d1_specs, d2_specs, e26_scenario)


def _spec(cls, arrival, rid, alpha=8):
    h, u = A_CLASS if cls == "A" else B_CLASS
    s = type("S", (), {})()  # 轻量替代，直接构造 RequestSpec 于生成器内
    from sim.cq.types import RequestSpec
    return RequestSpec(rid=rid, arrival_s=arrival, h_tokens=h, u_tokens=u,
                       class_id=cls, T0_s=F(1), deadline_s=F(1))


# ---------------------------------------------------------------------------
# U1 标定对拍
# ---------------------------------------------------------------------------

def test_u1_calibration():
    ra = _spec("A", 0.0, 0)
    rb = _spec("B", 0.0, 1)
    cA = float(batch_compute_s([ra], E26_PROFILE))
    cB = float(batch_compute_s([rb], E26_PROFILE))
    assert abs(cA - 0.00602) < 1e-9
    assert abs(cB - 0.02859) < 1e-9
    vA = float(layer_read_gb(ra, E26_PROFILE))
    vB = float(layer_read_gb(rb, E26_PROFILE))
    assert abs(vA - float(E26_PROFILE.kappa_gb_per_token_layer * A_CLASS[0])) < 1e-12
    assert abs(vB - float(E26_PROFILE.kappa_gb_per_token_layer * B_CLASS[0])) < 1e-12
    assert abs(vA - 0.175648768) < 1e-9            # 175.65 MB（十进制）
    assert abs(vB - 0.038498304) < 1e-9            # 38.5 MB
    kA = float(singleton_K(ra, E26_PROFILE, F(120), F(200)))
    kB = float(singleton_K(rb, E26_PROFILE, F(120), F(200)))
    assert abs(kA - (vA / 120.0 + 8 * cA)) < 1e-9   # t_read < c → 无惩罚项
    assert abs(kB - (vB / 120.0 + 8 * cB)) < 1e-9
    assert abs(kA - 0.04962) < 5e-5 and abs(kB - 0.22904) < 5e-5
    assert abs(kA / kB - 0.2167) < 1e-3
    # 派生量（大纲 §1）：总读取 2587 GB → 排空下限 21.56s；λ_sat=178
    total_read = e26.N_A * 8 * vA + e26.N_B * 8 * vB
    assert abs(total_read / 2587.0 - 1.0) < 1e-3
    assert abs(total_read / 120.0 - 21.56) < 5e-3
    e_read = total_read / e26.N_TOTAL
    assert abs(120.0 / e_read - e26.LAMBDA_SAT) < 0.5


# ---------------------------------------------------------------------------
# U2 生成器确定性与配比
# ---------------------------------------------------------------------------

def test_u2_generator_determinism_and_mix():
    s1 = d1_specs(160.0, e26.d1_seed(2, 0), 8)
    s2 = d1_specs(160.0, e26.d1_seed(2, 0), 8)
    assert len(s1) == e26.N_TOTAL
    assert [x.arrival_s for x in s1] == [x.arrival_s for x in s2]
    assert [x.class_id for x in s1] == [x.class_id for x in s2]
    assert sum(1 for x in s1 if x.class_id == "A") == e26.N_A
    assert sum(1 for x in s1 if x.class_id == "B") == e26.N_B
    s3 = d1_specs(160.0, e26.d1_seed(2, 1), 8)
    assert [x.arrival_s for x in s3] != [x.arrival_s for x in s1]  # 不同 rep 不同流
    # deadline = arrival + α·T0
    for x in s1[:50]:
        assert abs(float(x.deadline_s) - (float(x.arrival_s) + 8.0 * float(x.T0_s))) < 1e-12
    # D2 结构：40 轮 × 96 条（32A+64B），轮内 (A,B,B)×32 交错，全轮同时到达
    d2 = d2_specs(1.0, 8)
    assert len(d2) == e26.N_TOTAL
    for r in range(40):
        blk = d2[r * 96:(r + 1) * 96]
        assert all(abs(float(x.arrival_s) - r * 1.0) < 1e-12 for x in blk)
        assert sum(1 for x in blk if x.class_id == "A") == 32
        assert [x.class_id for x in blk[:3]] == ["A", "B", "B"]


# ---------------------------------------------------------------------------
# U3 batch=1 不变量
# ---------------------------------------------------------------------------

def test_u3_batch_size_one_invariant():
    specs = d2_specs(1.0, 8)[:96]            # 1 轮 96 条
    eng = run_case(e26_scenario(8), specs, make_fcfs())
    assert eng.w.batches, "应有批产生"
    assert all(len(b.members) == 1 for b in eng.w.batches.values())


# ---------------------------------------------------------------------------
# U4 字节守恒
# ---------------------------------------------------------------------------

def test_u4_byte_conservation(tmp_path):
    specs = d2_specs(1.0, 8)[:96]
    eng = run_case(e26_scenario(8), specs, make_fcfs(), record_intervals=True)
    assert eng.status == "done"
    served = float(eng.w.storage.actual_integral_gb)
    total_v = sum(8 * float(layer_read_gb(s, E26_PROFILE)) for s in specs)
    assert abs(served - total_v) < 1e-6 * max(1.0, total_v)
    # aggregate_run 三恒等式（含 Σserved=actual_integral）不抛错
    ts = aggregate_run(eng, float(eng.w.t) * 1.1, K=50)
    rows = [r for r in ts["rows"] if r["width_s"] > 0]
    assert abs(sum(r["served_gb"] for r in rows) - served) < 1e-6 * served
    # 甘特四元组守恒：Σ(四态) = m×T_end
    buckets = e26.gantt_collect(eng, float(eng.w.t) * 1.1, K=50)
    tot = sum(v for bk in buckets for row in bk for v in row)
    assert abs(tot - 32 * float(eng.w.t)) < 1e-6 * 32 * float(eng.w.t)


# ---------------------------------------------------------------------------
# U5 冒烟 run 无未完成
# ---------------------------------------------------------------------------

def test_u5_smoke_no_unfinished():
    specs = d2_specs(1.0, 8)[:96]
    eng = run_case(e26_scenario(8), specs, make_fcfs())
    assert eng.status == "done"
    assert all(rr.F_s is not None for rr in eng.w.requests.values())


# ---------------------------------------------------------------------------
# U6 推演修复回归：全读工况下 MPC 推演可排空、有候选评分
# ---------------------------------------------------------------------------

def test_u6_forecast_id_fix_mpc_scores():
    """20260928 修复前：est 批 id=worker_id+1 与重注入流真实 batch_id 错配，

    含在飞读取的推演必然 stalled_no_event → n_fallback≈n_decides、n_scored=0
    （E26 全读工况）。修复后小世界（24 条全读、1 次突发）MPC 必须有候选
    被真实评分且全部请求完成。
    """
    order = (["A", "B", "B"] * 8)
    from sim.cq.types import RequestSpec
    specs = []
    for i, cls in enumerate(order):
        h, u = A_CLASS if cls == "A" else B_CLASS
        specs.append(RequestSpec(i, 0.0, h, u, cls, F(1), F(1)))
    from sim.cq.profile import make_T0
    T0 = make_T0(specs, E26_PROFILE, F(120), F(200))
    specs = [RequestSpec(s.rid, 0.0, s.h_tokens, s.u_tokens, s.class_id,
                         T0[s.rid], float(T0[s.rid]) * 8.0) for s in specs]
    pol = MPCPolicy(pid="cq_mpc", H=1, theta="S", c_ref=0.2,
                    budget_s=3600.0, max_events=20000)
    eng = run_case(e26_scenario(8), specs, pol)
    assert eng.status == "done"
    assert all(rr.F_s is not None for rr in eng.w.requests.values())
    assert pol.n_scored > 0, "修复前此断言失败（推演世界卡死、零评分）"


# ---------------------------------------------------------------------------
# U7 搜索健康监测（20260929）：计数器存在性/一致性/健康口径
# ---------------------------------------------------------------------------

def test_u7_mpc_health_counters():
    """mpc/local 必须暴露可审计计数：n_decides≥n_searched≥1、零回退、
    偏离≤搜索、health() 比率落在 [0,1]。这是防止 20260928 型静默退化
    再发生的常设监测（tools/cq_mpc_health.py 消费同一组字段）。"""
    order = (["A", "B", "B"] * 8)
    from sim.cq.types import RequestSpec
    from sim.cq.profile import make_T0
    specs = [RequestSpec(i, 0.0, *(A_CLASS if cls == "A" else B_CLASS), cls,
                         F(1), F(1)) for i, cls in enumerate(order)]
    T0 = make_T0(specs, E26_PROFILE, F(120), F(200))
    specs = [RequestSpec(s.rid, 0.0, s.h_tokens, s.u_tokens, s.class_id,
                         T0[s.rid], float(T0[s.rid]) * 8.0) for s in specs]
    for pid, local in (("cq_mpc", False), ("cq_local", True)):
        pol = MPCPolicy(pid=pid, local=local, H=1, theta="S", c_ref=0.2,
                        budget_s=3600.0, max_events=20000)
        eng = run_case(e26_scenario(8), specs, pol)
        assert eng.status == "done"
        h = pol.health()
        assert h["n_decides"] >= h["n_searched"] >= 1, pid
        assert h["n_fallback"] == 0, (pid, "修复后全读工况不允许推演失败")
        assert 0 <= h["n_deviate"] <= h["n_searched"], pid
        for k in ("search_ratio", "fallback_ratio", "deviate_ratio"):
            v = h[k]
            assert v is None or 0.0 <= v <= 1.0, (pid, k, v)
