"""max-min fair 存储层主线单测（docs/存储带宽maxmin替换设计-20261016.md §8）。

覆盖：waterfill 向量（源设计 §3.4 两例 + 退化性质）、用户确认的 40/30/20
动态重分配规范算例、新到达重配全部在途 IO 时刻、盘数派生、HRW 放置
（确定性/计数/均匀性/一致性）、逐块公平签名（组速率 ∝ 块数）、多盘对称
端到端解析对拍、hypothetical 与实际一致、ps_share 回归锚、配置校验。
"""
from __future__ import annotations

import math

import pytest
import simpy

from sim.config import RunSpec, StorageConfig
from sim.request import build_request
from sim.simrun import cls_demand_gbps
from sim.storage import (SharedKVStorage, Transfer, disk_of, disk_plan,
                         waterfill)


# ---------------------------------------------------------------------------
# waterfill 向量
# ---------------------------------------------------------------------------

def test_waterfill_source_vectors():
    # 源设计 §3.4 供不应求（迭代到不动点）
    assert waterfill(100, [10, 28, 100, 100]) == pytest.approx([10, 28, 31, 31])
    # 供过于求：按需分 + 剩余均分
    assert waterfill(40, [2, 4, 8]) == pytest.approx(
        [10 + 2 / 3, 12 + 2 / 3, 16 + 2 / 3])
    # 均匀需求退化 C/N（两种容量情形）
    assert waterfill(30, [5, 5, 5]) == pytest.approx([10, 10, 10])
    assert waterfill(12, [5, 5, 5]) == pytest.approx([4, 4, 4])
    # 单请求 = 全容量（超需求）
    assert waterfill(40, [30]) == pytest.approx([40])
    assert waterfill(20, [30]) == pytest.approx([20])
    # Σd 恰等于 C 边界：alloc = d
    assert waterfill(40, [10, 30]) == pytest.approx([10, 30])
    # d > C：不预截断，均分
    assert waterfill(10, [30, 30]) == pytest.approx([5, 5])
    # 空集
    assert waterfill(40, []) == []
    # 守恒：非空需求下分配和恒等于容量（供过于求时剩余也全部分掉）
    for cap, ds in [(100, [10, 28, 100, 100]), (40, [2, 4, 8]), (10, [30, 30])]:
        assert sum(waterfill(cap, ds)) == pytest.approx(cap)


# ---------------------------------------------------------------------------
# 用户确认的动态重分配规范算例（40/30/20 -> [20,20]）
# ---------------------------------------------------------------------------

def _run_with_driver(cfg, driver, until=5.0):
    env = simpy.Environment()
    s = SharedKVStorage(env, "p0", cfg)
    times = {}

    def watch(rid, ev):
        yield ev
        times[rid] = env.now

    def wrap():
        yield from driver(env, s, watch, times)

    env.process(wrap())
    env.run(until=until)
    return s, times


def test_dynamic_reallocation_40_30_20():
    """独占超需求分满（30->40）；第二请求到达后两者同降为 20；完成后回升。"""
    cfg = StorageConfig(b_total=40.0, disk_bps=40.0, bw_policy="max_min_fair")

    def driver(env, s, watch, times):
        e1 = s.submit(1, 40.0, 30.0)
        env.process(watch(1, e1))
        yield env.timeout(0.005 + 1e-9)          # 过 t_base 门控
        ready = [g for g in s.active if g.gate_until <= env.now]
        assert s._mm_rates(ready, env.now) == pytest.approx([40.0])
        yield env.timeout(0.5 - env.now)
        e2 = s.submit(2, 10.0, 20.0)             # 在途提交
        env.process(watch(2, e2))
        yield env.timeout(0.005 + 1e-9)
        ready = [g for g in s.active if g.gate_until <= env.now]
        assert sorted(s._mm_rates(ready, env.now)) == pytest.approx([20.0, 20.0])

    s, times = _run_with_driver(cfg, driver)
    # 解析：0.005~0.505 r1@40；0.505~1.005 双双@20（r2 完成）；
    # 1.005~ r1@40（回升），r1 剩 10 -> 0.25s
    assert times[2] == pytest.approx(0.505 + 10.0 / 20.0, abs=1e-9)
    assert times[1] == pytest.approx(1.005 + (20.0 - 10.0) / 40.0, abs=1e-9)
    assert s.bytes_served == pytest.approx(50.0)


def test_new_arrival_retimes_all_inflight():
    """用户强调的语义：新请求提交后，所有在途请求的 IO 完成时刻都会变。

    3 条同质在途（C=30）各自 1.005 完成；t=0.4 第 4 条到达后全部重配为
    7.5 GB/s，在途三条完成时刻推迟到 1.205，第 4 条 1.3383。
    """
    cfg = StorageConfig(b_total=30.0, disk_bps=30.0, bw_policy="max_min_fair")

    def driver(env, s, watch, times):
        for rid in (1, 2, 3):
            e = s.submit(rid, 10.0, 15.0)
            env.process(watch(rid, e))
        yield env.timeout(0.4)
        e4 = s.submit(4, 10.0, 15.0)
        env.process(watch(4, e4))

    s, times = _run_with_driver(cfg, driver)
    for rid in (1, 2, 3):
        # 0.405 前各 10（30/3），已传 4；0.405 起 7.5：剩 6 -> 0.8s
        assert times[rid] == pytest.approx(0.405 + 6.0 / 7.5, abs=1e-9)
    # r4：0.405~1.205 @7.5 传 6；之后独占 30：剩 4 -> 2/15
    assert times[4] == pytest.approx(1.205 + 4.0 / 30.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 盘数派生与放置
# ---------------------------------------------------------------------------

def test_disks_derivation():
    assert StorageConfig(b_total=120.0).disks == 3
    assert StorageConfig(b_total=100.0).disks == 3     # 33.3/盘
    assert StorageConfig(b_total=80.0).disks == 2
    assert StorageConfig(b_total=60.0).disks == 2
    assert StorageConfig(b_total=40.0).disks == 1
    assert StorageConfig(b_total=25.0).disks == 1
    assert StorageConfig(b_total=120.0, disk_bps=120.0).disks == 1   # fabric 口径
    for bt in (120.0, 100.0, 80.0, 60.0, 40.0, 25.0):
        c = StorageConfig(b_total=bt)
        assert c.b_total / c.disks <= 40.0 + 1e-12    # 单盘不超物理上限


def test_placement_determinism_and_counts():
    assert disk_of("p0", "A", 7, 3) == disk_of("p0", "A", 7, 3)
    g = 1e-3
    plan = disk_plan("p0", "A", 128 * 512, 3, 512, g)
    assert sum(k for (_d, k, _b) in plan) == 128
    assert sum(b for (_d, _k, b) in plan) == pytest.approx(128 * 512 * g)
    # 末块部分字节
    plan2 = disk_plan("p0", "A", 10 * 512 + 100, 3, 512, g)
    assert sum(k for (_d, k, _b) in plan2) == 11
    assert sum(b for (_d, _k, b) in plan2) == pytest.approx((10 * 512 + 100) * g)
    # 均匀性宽松界（128 块 / 3 盘，期望 42.7，HRW 不会过度倾斜）
    fracs = [k / 128 for (_d, k, _b) in plan]
    assert all(0.20 <= f <= 0.47 for f in fracs)
    # 一致性：3 -> 4 盘只重映射少数块（HRW 性质）
    moved = sum(1 for i in range(128)
                if disk_of("p0", "A", i, 4) != disk_of("p0", "A", i, 3))
    assert 0 < moved < 64


def test_per_block_fairness_signature():
    """同盘等需求下组速率 ∝ 块数（vs 聚合 PS 的等速率）——逐块公平签名。"""
    cfg = StorageConfig(b_total=80.0, disk_bps=40.0, bw_policy="max_min_fair")
    env = simpy.Environment()
    s = SharedKVStorage(env, "p0", cfg)
    g1 = Transfer(1, 8.0, 0.0, demand=15.0, disk=0, n_io=8)
    g2 = Transfer(2, 2.0, 0.0, demand=15.0, disk=0, n_io=2)
    rates = s._mm_rates([g1, g2], 0.0)
    assert rates[0] / rates[1] == pytest.approx(4.0)     # 8 块 : 2 块
    # 完成时间同时（等需求同盘逐块公平）：8/(8a) == 2/(2a)
    assert g1.remaining / rates[0] == pytest.approx(g2.remaining / rates[1])


def test_multidisk_symmetric_analytic():
    """两同类请求同池（2 盘）：同计划 -> 每盘对半 -> 各总速率 40，解析完成。"""
    cfg = StorageConfig(b_total=80.0, disk_bps=40.0, bw_policy="max_min_fair",
                        block_tokens=512)
    g = 1e-3
    n_tokens = 16 * 512
    nbytes = n_tokens * g

    def driver(env, s, watch, times):
        for rid in (1, 2):
            e = s.submit(rid, nbytes, 30.0, s.plan_for("A", n_tokens, g))
            env.process(watch(rid, e))
        yield env.timeout(0)

    s, times = _run_with_driver(cfg, driver)
    # 每盘：两请求同块数同需求 -> 每盘各 20 GB/s；完成由最慢盘（块数最多的盘）决定
    plan = disk_plan("p0", "A", n_tokens, 2, 512, g)
    slow_disk_gb = max(b for (_d, _k, b) in plan)
    for rid in (1, 2):
        assert times[rid] == pytest.approx(0.005 + slow_disk_gb / 20.0, abs=1e-9)
    assert s.bytes_served == pytest.approx(2 * nbytes)


def test_heterogeneous_edge_analytic_and_hypothetical():
    """Σd=C 边界（聚合单盘）：alloc=d；r1 完成后 r2 独占获剩余加成；Oracle 一致。"""
    cfg = StorageConfig(b_total=40.0, disk_bps=40.0, bw_policy="max_min_fair")
    env = simpy.Environment()
    s = SharedKVStorage(env, "p0", cfg)
    times = {}

    def watch(rid, ev):
        yield ev
        times[rid] = env.now

    e1 = s.submit(1, 12.0, 30.0)
    env.process(watch(1, e1))
    # Oracle 预测：r1 活跃（门控中），插入 r2（6GB, demand=10）
    hyp = s.hypothetical_fetch_time(6.0, 10.0)
    e2 = s.submit(2, 6.0, 10.0)
    env.process(watch(2, e2))
    env.run(until=5.0)
    assert times[1] == pytest.approx(0.005 + 12.0 / 30.0, abs=1e-9)
    # r2：0.005~0.405 @10 传 4；r1 完成后独占 40（10+剩余30），剩 2 -> 0.05
    assert times[2] == pytest.approx(0.405 + 2.0 / 40.0, abs=1e-9)
    assert hyp == pytest.approx(times[2], abs=1e-9)


def test_ps_share_regression_anchor():
    """ps_share 臂 = 旧模型：等分、忽略 demand/plan，解析时间线不变。"""
    cfg = StorageConfig(b_total=40.0, bw_policy="ps_share")

    def driver(env, s, watch, times):
        for rid in (1, 2):
            e = s.submit(rid, 20.0, None, None)   # demand/plan 被忽略
            env.process(watch(rid, e))
        yield env.timeout(0)

    s, times = _run_with_driver(cfg, driver)
    for rid in (1, 2):
        assert times[rid] == pytest.approx(0.005 + 20.0 / 20.0, abs=1e-9)


def test_config_validation():
    with pytest.raises(ValueError):
        SharedKVStorage(simpy.Environment(), "p", StorageConfig(bw_policy="bogus"))
    with pytest.raises(ValueError):
        SharedKVStorage(simpy.Environment(), "p", StorageConfig(disk_bps=0.0))
    with pytest.raises(ValueError):
        SharedKVStorage(simpy.Environment(), "p", StorageConfig(block_tokens=0))
    # 多盘 max_min 缺 plan 报错（防静默聚合退化）
    env = simpy.Environment()
    s = SharedKVStorage(env, "p", StorageConfig(b_total=80.0, bw_policy="max_min_fair"))
    assert s.disks == 2
    with pytest.raises(ValueError):
        s.submit(1, 10.0, 30.0)
    # fabric 口径（disk_bps=b_total -> 单盘）无需 plan
    s2 = SharedKVStorage(env, "fab", StorageConfig(b_total=120.0, disk_bps=120.0,
                                                   bw_policy="max_min_fair"))
    assert s2.disks == 1
    s2.submit(1, 10.0, 30.0)   # 不抛


def test_demand_derivation_and_request_injection():
    spec = RunSpec(exp="t", policy="x", seed=0)
    dem = cls_demand_gbps(spec)
    d_ref = spec.model.kv_gb_per_token * 65536 / 0.410
    assert abs(d_ref - 52.37) < 0.02
    for c in ("A", "B", "C", "D"):
        assert dem[c] == pytest.approx(d_ref)
    # 按类覆盖 + 全局缩放
    from dataclasses import replace
    spec2 = replace(replace(spec, wl=replace(spec.wl, cls_demand_scale=(("A", 0.25),))),
                    wl=replace(replace(spec.wl, cls_demand_scale=(("A", 0.25),)),
                               demand_scale=2.0))
    dem2 = cls_demand_gbps(spec2)
    assert dem2["A"] == pytest.approx(d_ref * 0.5)
    assert dem2["B"] == pytest.approx(d_ref * 2.0)
    # build_request 注入
    req = build_request(1, 0.0, "A", True, spec.wl, spec.model, demand_gbps=dem["A"])
    assert req.demand_gbps == pytest.approx(d_ref)


def test_bg_breakpoint_switches_allocation():
    """背景负载断点改变容量 -> 分配切换（解析吻合）。"""
    cfg = StorageConfig(b_total=40.0, disk_bps=40.0, bw_policy="max_min_fair",
                        bg_schedule=((0.0, 0.0), (0.2, 24.0)))   # cap 40 -> 16

    def driver(env, s, watch, times):
        e = s.submit(1, 40.0, 20.0)
        env.process(watch(1, e))
        yield env.timeout(0)

    s, times = _run_with_driver(cfg, driver)
    # 0.005~0.2 @40（传 7.8）；0.2~ @16：剩 32.2 -> 2.0125
    assert times[1] == pytest.approx(0.2 + 32.2 / 16.0, abs=1e-9)
