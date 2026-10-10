"""E26e D1c 生成器不变量测试（设计 §7-2 / §6-3）。

E1 比例恒定 / E2 簇恰 k 条+余数簇（同刻+rid 连续）/ E3 k=1 退化（每簇恰 1
条+间隔指数矩；与 e26.d1_specs 的逐位对拍不可实现——机制不同（全局泊松+
1:2 置换 vs 双独立 1:1 流），以"与设计伪码逐位一致 + 边际过程泊松"等价
替代，结果文档 §局限披露）/ E4 伪码逐位对拍（手工重实现 RNG 循环）/ E5
指数矩检验 / E6 确定性 / E7 同 seed 同流（CRN）/ E8 B 流每次 1 条 / E9
deadline=arrival+α·T0 与 A 期限 198ms 预测吻合。
"""
import sys
from fractions import Fraction as F

import numpy as np
import pytest

sys.path.insert(0, ".")

from sim.experiments.e26e_cluster import (ALPHA, LAM, N_A, N_B, d1c_seed,
                                          d1c_specs)


def _classes(specs):
    return [s.class_id for s in specs]


def _a_times(specs):
    return [float(s.arrival_s) for s in specs if s.class_id == "A"]


def _b_times(specs):
    return [float(s.arrival_s) for s in specs if s.class_id == "B"]


# ---------------------------------------------------------------------------
# E1 比例恒定：1:1、总量 240、rid 重排后 A/B 恰各 120
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [4, 16])
def test_e1_mix_constant(k):
    specs, centers = d1c_specs(k, d1c_seed(0 if k == 4 else 1, 0))
    cs = _classes(specs)
    assert len(specs) == N_A + N_B == 240
    assert cs.count("A") == N_A and cs.count("B") == N_B
    assert sorted(s.rid for s in specs) == list(range(240))
    # 到达期 ≈1.22s（设计 §2.1）：120 条 B 期望跨度 120/98≈1.22s，宽容差
    t_end = max(float(s.arrival_s) for s in specs)
    assert 0.9 < t_end < 1.7


# ---------------------------------------------------------------------------
# E2 簇恰 k 条 + 余数簇：每簇同刻、rid 连续、簇数=ceil(120/k)
# ---------------------------------------------------------------------------

def test_e2_cluster_structure():
    for k, base in ((4, 0), (16, 1)):
        specs, centers = d1c_specs(k, d1c_seed(base, 0))
        n_cl = int(np.ceil(N_A / k))
        assert len(centers) == n_cl
        assert n_cl == {4: 30, 16: 8}[k]
        # 按 rid 顺序切块：A 的 rid 序列应恰按簇分组（每簇同刻、rid 连续）
        a = [s for s in specs if s.class_id == "A"]
        sizes, i = [], 0
        while i < len(a):
            t0 = float(a[i].arrival_s)
            j = i
            while j < len(a) and float(a[j].arrival_s) == t0:
                j += 1
            sizes.append(j - i)
            i = j
        assert sizes == [k] * (N_A // k) + ([N_A % k] if N_A % k else [])
        # 簇心时刻与块到达时刻一致、严格递增
        seen = []
        for s in a:
            t = float(s.arrival_s)
            if not seen or seen[-1] != t:
                seen.append(t)
        assert seen == centers
        assert all(b > a_ for a_, b in zip(centers, centers[1:]))


# ---------------------------------------------------------------------------
# E3 k=1 退化：每簇恰 1 条（=每次 1 条的泊松流）、间隔指数
# ---------------------------------------------------------------------------

def test_e3_k1_degenerate_poisson():
    specs, centers = d1c_specs(1, 7, n_a=4000, n_b=4000)
    assert len(centers) == 4000
    at = _a_times(specs)
    assert len(set(at)) == 4000          # 每簇恰 1 条：无同刻 A
    gaps = np.diff([0.0] + at)           # 首间隔也 ~Exp（t=0 起累加）
    assert (gaps > 0).all()
    # 间隔均值 ≈ 1/λ_A=1/98 s（n=4000，SE≈1.6%，容差 5%）
    assert abs(gaps.mean() - 1 / (LAM / 2)) / (1 / (LAM / 2)) < 0.05


# ---------------------------------------------------------------------------
# E4 伪码逐位对拍：手工重实现设计 §8.2 RNG 循环，防实现走样
# ---------------------------------------------------------------------------

def test_e4_pseudocode_bitwise():
    k, seed = 16, d1c_seed(1, 0)
    rng = np.random.Generator(np.random.PCG64(seed))
    t, a_times, centers = 0.0, [], []
    while len(a_times) < N_A:
        t += rng.exponential(k / (LAM / 2))
        m = min(k, N_A - len(a_times))
        a_times.extend([t] * m)
        centers.append(t)
    b_times, t = [], 0.0
    while len(b_times) < N_B:
        t += rng.exponential(1 / (LAM / 2))
        b_times.append(t)
    specs, cs = d1c_specs(k, seed)
    assert cs == centers
    assert _a_times(specs) == [float(x) for x in a_times]
    assert _b_times(specs) == [float(x) for x in b_times]


# ---------------------------------------------------------------------------
# E5 簇心间隔指数矩：均值 ≈ k/λ_A（k=16 大样本）
# ---------------------------------------------------------------------------

def test_e5_cluster_gap_exponential_moment():
    n_big = 32000                         # /16 → 2000 簇心间隔，SE≈2.3%
    specs, centers = d1c_specs(16, 23, n_a=n_big, n_b=10)
    gaps = np.diff(centers)
    k_exp = 16 / (LAM / 2)                # ≈163ms（设计 §2.1）
    assert len(gaps) == n_big // 16 - 1
    assert abs(gaps.mean() - k_exp) / k_exp < 0.07
    assert abs(np.std(gaps) / gaps.mean() - 1.0) < 0.05   # 指数：CV=1


# ---------------------------------------------------------------------------
# E6 确定性：同 seed 两次生成逐位一致
# ---------------------------------------------------------------------------

def test_e6_determinism():
    for k, base in ((4, 0), (16, 1)):
        s1, c1 = d1c_specs(k, d1c_seed(base, 0))
        s2, c2 = d1c_specs(k, d1c_seed(base, 0))
        assert c1 == c2
        assert [(s.rid, s.class_id, float(s.arrival_s)) for s in s1] == \
               [(s.rid, s.class_id, float(s.arrival_s)) for s in s2]


# ---------------------------------------------------------------------------
# E7 CRN：同格点（k, seed）同流；不同 k 同基号族流不同
# ---------------------------------------------------------------------------

def test_e7_crn_same_cell_same_stream():
    s1, _ = d1c_specs(16, d1c_seed(1, 0))
    s2, _ = d1c_specs(16, d1c_seed(1, 0))
    assert [(s.rid, s.class_id) for s in s1] == [(s.rid, s.class_id) for s in s2]
    s3, _ = d1c_specs(16, d1c_seed(1, 1))   # 补测 seed=不同流
    assert [(s.rid, s.class_id) for s in s3] != [(s.rid, s.class_id) for s in s1]


# ---------------------------------------------------------------------------
# E8 B 流每次 1 条：B 到达时刻两两不同
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [4, 16])
def test_e8_b_stream_single(k):
    specs, _ = d1c_specs(k, 99)
    bt = _b_times(specs)
    assert len(bt) == N_B and len(set(bt)) == N_B
    assert all(b > a_ for a_, b in zip(bt, bt[1:]))


# ---------------------------------------------------------------------------
# E9 deadline=arrival+α·T0；A 类 T0≈49.5ms → 期限 198ms（设计 §2.2）
# ---------------------------------------------------------------------------

def test_e9_deadline_and_T0():
    specs, _ = d1c_specs(4, 3)
    for s in specs:
        assert abs(float(s.deadline_s)
                   - (float(s.arrival_s) + ALPHA * float(s.T0_s))) < 1e-12
    t0_a = [float(s.T0_s) for s in specs if s.class_id == "A"]
    assert abs(np.median(t0_a) - 0.0495) < 0.003     # A 期限 198ms/4
    t0_b = [float(s.T0_s) for s in specs if s.class_id == "B"]
    assert np.median(t0_b) > np.median(t0_a) * 3    # B 慢类 T0 更长
