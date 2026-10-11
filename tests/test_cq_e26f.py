"""E26f C 类标定 / V=0 引擎路径 / D1f 生成器不变量测试（设计 §6-2/§6-3）。

U1 C 标定（两型）：solve_c_u 二分解命中目标（容差 2e-6，整数 u 离散限制）
且为邻点最近者；V=0；T0=8c 精确（Fraction，无首层读）；C_A 期限早于 A。
U2 V=0 引擎路径：纯 C 小规模 run 排空、无正宽 STALL、makespan=8c 解析吻合；
混合小规模 cq_local run 推演占位 bid 路径走通（V=0 批无读流 → forecast
用 -(wid+1) 占位；若该路径坏，fallback 比例≈100%，参照 20260928 修复）。
U3 生成器不变量：1:1:1、A 簇恰 16+余簇（同刻 rid 连续）、B/C 每次单条、
确定性、CRN（同 seed 跨 C 变体边际流逐位同）、伪码逐位对拍、指数矩。
"""
import sys
from fractions import Fraction as F

import numpy as np
import pytest

sys.path.insert(0, ".")

from sim.experiments import e26_ab as e26
from sim.experiments.e26f_third import (ALPHA, C_A_CLASS, C_B_CLASS, C_LEVELS,
                                        C_CLASSES, LAM_EACH, N_EACH,
                                        d1f_seed, d1f_specs, solve_c_u,
                                        _c_pure)
from sim.cq.policies import make_fcfs
from sim.cq.profile import batch_compute_s, layer_read_gb, make_T0
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.types import RequestSpec


def _c_spec(u: int) -> RequestSpec:
    return RequestSpec(rid=0, arrival_s=F(0), h_tokens=0, u_tokens=int(u),
                       class_id="C", T0_s=F(1), deadline_s=F(1))


# ---------------------------------------------------------------------------
# U1 C 类标定（两型）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,u", [(0.00602, 2657), (0.02859, 8110)])
def test_u1_c_calibration(target, u):
    # 二分解命中且为邻点最近者（整数 u 离散限制，容差 2e-6）
    assert solve_c_u(target) == u
    c = _c_pure(u)
    assert abs(c - target) <= 2e-6
    for du in (-1, 1):
        assert abs(_c_pure(u + du) - target) > abs(c - target)
    # V=0：零读取（不建流）
    rs = _c_spec(u)
    assert layer_read_gb(rs, e26.E26_PROFILE) == 0
    # T0=8c 精确（无首层读：t_read=0 → K=L·c），Fraction 逐位相等
    T0 = make_T0([rs], e26.E26_PROFILE, F(120), F(200))[0]
    assert T0 == 8 * batch_compute_s([rs], e26.E26_PROFILE)


def test_u1_c_deadline_structure():
    # C_A 型：u=2657、期限宽度 192.7ms，早于 A 的 198.5ms（无首层读偏置）
    assert C_A_CLASS == (0, 2657) and C_B_CLASS == (0, 8110)
    t0_c = float(make_T0([_c_spec(2657)], e26.E26_PROFILE, F(120), F(200))[0])
    rsa = RequestSpec(1, F(0), *e26.A_CLASS, "A", F(1), F(1))
    t0_a = float(make_T0([rsa], e26.E26_PROFILE, F(120), F(200))[1])
    assert abs(4 * t0_c - 0.1927) < 5e-4      # C_A 期限 192.7ms
    assert abs(4 * t0_a - 0.1985) < 5e-4      # A 期限 198.5ms
    assert t0_c < t0_a                         # C_A 期限早于 A
    # C_B 型：T0=228.7ms、期限 914.9ms（与 B 同档宽）
    t0_cb = float(make_T0([_c_spec(8110)], e26.E26_PROFILE, F(120), F(200))[0])
    assert abs(4 * t0_cb - 0.9149) < 5e-4
    # η 语义：两型 u 均 ≥N_sat=512 → aN 项无放大（c 随 u 严格增已由二分单调性覆盖）
    assert _c_pure(2657) < _c_pure(8110)


# ---------------------------------------------------------------------------
# U2 V=0 引擎路径
# ---------------------------------------------------------------------------

def test_u2_pure_c_run_no_stall():
    """纯 C 小规模 run：12 条 C_A 同刻到达 → 32 worker 全空闲即并行，
    makespan=8c=48.17ms 解析吻合；全程无正宽 STALL（V=0 不建流）。"""
    scn = e26.e26_scenario(ALPHA)
    specs = [RequestSpec(rid=i, arrival_s=0.0, h_tokens=0, u_tokens=2657,
                         class_id="C", T0_s=F(1), deadline_s=F(1))
             for i in range(12)]
    T0 = make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    specs = [RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens, "C",
                         T0[s.rid], float(s.arrival_s) + ALPHA * float(T0[s.rid]))
             for s in specs]
    eng = run_case(scn, specs, make_fcfs(), numeric=float, seed=0)
    assert all(eng.w.requests[r].F_s is not None for r in range(12))  # 排空
    stall_w = sum(float(e) - float(s)
                  for wk in eng.w.workers.values()
                  for (s, e, st, _b, _l) in wk.segments if st == "STALL")
    assert stall_w == 0.0                     # 无正宽 IO 等待
    c8 = 8 * float(batch_compute_s([_c_spec(2657)], e26.E26_PROFILE))
    assert abs(float(eng.w.t) - c8) < 1e-9    # makespan=8c 解析吻合


def test_u2_forecast_placeholder_bid_path():
    """混合小规模 cq_local run：V=0 活动批无读流 → forecast 用 -(wid+1)
    占位重建。若占位路径坏（参照 20260928 批/流 id 错配症状），
    n_fallback/n_searched≈100%；此处要求 ≤5% 且有搜索有评分、全部排空。
    场景保证决策点同时有队列积压（生成候选）与活动 C 批（触发占位路径）。"""
    scn = e26.e26_scenario(ALPHA)
    events = ([("A", 0.0)] * 6 + [("B", 0.0)] * 6 + [("C", 0.0)] * 6
              + [("A", 0.05)] * 3 + [("B", 0.05)] * 3 + [("C", 0.05)] * 3)
    cmap = {"A": e26.A_CLASS, "B": e26.B_CLASS, "C": C_A_CLASS}
    specs = [RequestSpec(i, t, *cmap[cl], cl, F(1), F(1))
             for i, (cl, t) in enumerate(events)]
    T0 = make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    specs = [RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens, s.class_id,
                         T0[s.rid],
                         float(s.arrival_s) + ALPHA * float(T0[s.rid]))
             for s in specs]
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    pol = MPCPolicy(pid="cq_local", local=True, H=1, theta="S",
                    c_ref=c_ref, budget_s=3600.0, max_events=20000,
                    base="bw_edf", composer=True)
    eng = run_case(scn, specs, pol, numeric=float, seed=0)
    assert all(eng.w.requests[r].F_s is not None for r in range(len(specs)))
    h = pol.health()
    assert h["n_searched"] > 0 and h["n_scored"] > 0
    assert h["n_fallback"] / h["n_searched"] <= 0.05   # 占位 bid 路径走通


# ---------------------------------------------------------------------------
# U3 D1f 生成器不变量
# ---------------------------------------------------------------------------

def _times(specs, cls):
    return [float(s.arrival_s) for s in specs if s.class_id == cls]


@pytest.mark.parametrize("cell,variant,base", [C_LEVELS[0], C_LEVELS[1]])
def test_u3_mix_1_1_1(cell, variant, base):
    specs, centers = d1f_specs(16, d1f_seed(base, 0), C_CLASSES[variant])
    cs = [s.class_id for s in specs]
    assert len(specs) == 3 * N_EACH == 360
    for cl in "ABC":
        assert cs.count(cl) == N_EACH            # 1:1:1
    assert sorted(s.rid for s in specs) == list(range(360))
    # 到达期 ≈1.22s（每类 120 条 × 1/98；宽容差）
    assert 0.9 < max(float(s.arrival_s) for s in specs) < 1.8


def test_u3_cluster_structure_and_rid_contiguous():
    specs, centers = d1f_specs(16, d1f_seed(2, 0), C_A_CLASS)
    assert len(centers) == 8                     # ceil(120/16)=7×16+8
    a = [(s.rid, float(s.arrival_s)) for s in specs if s.class_id == "A"]
    # 同刻切块：每块恰 16 条（末块 8），块内 rid 连续
    sizes, i = [], 0
    while i < len(a):
        t0 = a[i][1]
        j = i
        while j < len(a) and a[j][1] == t0:
            j += 1
        sizes.append(j - i)
        assert [r for r, _ in a[i:j]] == list(range(a[i][0], a[j - 1][0] + 1))
        i = j
    assert sizes == [16] * 7 + [8]
    assert centers == sorted(set(t for _, t in a))
    assert all(b > a_ for a_, b in zip(centers, centers[1:]))


@pytest.mark.parametrize("c_class", [C_A_CLASS, C_B_CLASS])
def test_u3_b_c_stream_single(c_class):
    specs, _ = d1f_specs(16, 99, c_class)
    for cl in "BC":
        bt = _times(specs, cl)
        assert len(bt) == N_EACH and len(set(bt)) == N_EACH   # 每次恰 1 条
        assert all(b > a_ for a_, b in zip(bt, bt[1:]))


def test_u3_determinism_and_crn():
    s1, c1 = d1f_specs(16, d1f_seed(2, 0), C_A_CLASS)
    s2, c2 = d1f_specs(16, d1f_seed(2, 0), C_A_CLASS)
    assert c1 == c2                              # 确定性
    assert [(s.rid, s.class_id, float(s.arrival_s), s.u_tokens) for s in s1] == \
           [(s.rid, s.class_id, float(s.arrival_s), s.u_tokens) for s in s2]
    s3, _ = d1f_specs(16, d1f_seed(2, 1), C_A_CLASS)          # 补测 seed=不同流
    assert [(s.rid, s.class_id) for s in s3] != [(s.rid, s.class_id) for s in s1]
    # CRN：同 seed 跨 C 变体 → 三流边际逐位同（RNG 抽取序列与 C 类无关）
    s4, _ = d1f_specs(16, d1f_seed(2, 0), C_B_CLASS)
    assert [(s.rid, s.class_id, float(s.arrival_s)) for s in s4] == \
           [(s.rid, s.class_id, float(s.arrival_s)) for s in s1]
    # 但 C 的 u/T0/期限随变体不同（两格点工况差异的根）
    cA = {s.rid: s for s in s1 if s.class_id == "C"}
    cB = {s.rid: s for s in s4 if s.class_id == "C"}
    assert all(cA[r].u_tokens != cB[r].u_tokens for r in cA)
    assert all(float(cA[r].T0_s) != float(cB[r].T0_s) for r in cA)


def test_u3_pseudocode_bitwise():
    """伪码逐位对拍：手工重实现设计 §8.2 三条 RNG 流，防实现走样。"""
    seed = d1f_seed(3, 0)
    rng = np.random.Generator(np.random.PCG64(seed))
    t, a_times, centers = 0.0, [], []
    while len(a_times) < N_EACH:
        t += rng.exponential(16 / LAM_EACH)
        m = min(16, N_EACH - len(a_times))
        a_times.extend([t] * m)
        centers.append(t)
    b_times, t = [], 0.0
    while len(b_times) < N_EACH:
        t += rng.exponential(1 / LAM_EACH)
        b_times.append(t)
    c_times, t = [], 0.0
    while len(c_times) < N_EACH:
        t += rng.exponential(1 / LAM_EACH)
        c_times.append(t)
    specs, cs = d1f_specs(16, seed, C_B_CLASS)
    assert cs == centers
    assert _times(specs, "A") == [float(x) for x in a_times]
    assert _times(specs, "B") == [float(x) for x in b_times]
    assert _times(specs, "C") == [float(x) for x in c_times]


def test_u3_exponential_moments():
    """B/C 泊松矩检验：大样本间隔均值≈1/λ、CV≈1（E26e E3/E5 口径）。"""
    specs, _ = d1f_specs(16, 23, C_A_CLASS, n=4000)
    for cl in "BC":
        at = np.array(_times(specs, cl))
        gaps = np.diff(np.concatenate([[0.0], at]))
        assert abs(gaps.mean() - 1 / LAM_EACH) / (1 / LAM_EACH) < 0.05
        assert abs(np.std(gaps) / gaps.mean() - 1.0) < 0.05


def test_u3_deadline_formula():
    specs, _ = d1f_specs(16, 3, C_A_CLASS)
    for s in specs:
        assert abs(float(s.deadline_s)
                   - (float(s.arrival_s) + ALPHA * float(s.T0_s))) < 1e-12
    # C_A 期限宽度（192.7ms）全队列最短：期限序 C 排 A 前（设计 §1-4）
    widths = {cl: [float(s.deadline_s) - float(s.arrival_s) for s in specs
                   if s.class_id == cl] for cl in "ABC"}
    assert np.median(widths["C"]) < np.median(widths["A"]) < \
        np.median(widths["B"])
