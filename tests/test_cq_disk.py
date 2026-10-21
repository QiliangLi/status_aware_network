"""cq 线盘维度存储模型单测（docs/E26f-盘维度存储实验设计-20261021.md §7）。

覆盖：HRW 放置（计数守恒/末块部分字节/确定性/增盘少重映射/与主线同键同
落点）、waterfill_weighted 与逐块展开逐位等价、每盘分配解析性质（逐块需求
守恒/独占全盘/均匀=B/N/倾斜完成由最慢盘决定/boost 同比例抬升）、组级完成
重配、clone 与 forecast 重注入的盘组穿透、fcfs/max_min 回归锚（disk_bps
被忽略）、端到端守恒。
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from sim.cq.config import CqScenario, ProfileConfig, StorageConfig
from sim.cq.engine import CqEngine, WorldState
from sim.cq.policies import make_fcfs
from sim.cq.search import MPCPolicy, build_forecast_engine
from sim.cq.simrun import run_case
from sim.cq.storage import (BLOCK_TOKENS, StorageSim, disk_of, disk_plan,
                            plan_bytes, plan_n_blocks, waterfill,
                            waterfill_weighted)
from sim.cq.types import HardLimits, RequestSpec

KAPPA = F(4096, 10**9)          # 现行画像参数（GB/token/层）


# ---------------------------------------------------------------------------
# §7-1 放置
# ---------------------------------------------------------------------------

def test_disk_plan_conservation_and_last_block_partial():
    for h in (42883, 9399, 512, 513, 1, 1024):
        p = disk_plan(f"A:{h}", h, 3, BLOCK_TOKENS, KAPPA)
        assert plan_n_blocks(p) == -(-h // BLOCK_TOKENS)      # 计数和 = n_b
        assert plan_bytes(p) == KAPPA * h                     # 字节和 = κ·h
        assert sum(c for _d, c, _b in p) == plan_n_blocks(p)
        for d, c, b in p:
            assert 0 <= d < 3 and c >= 1
            # 末块部分字节：非末盘组字节=整块；末盘组 ≤ 整块字节
            full = KAPPA * (c * BLOCK_TOKENS)
            assert b <= full
    # 末块部分字节精确：42883 = 83×512 + 387（设计 §2.1 的"407"系笔误，
    # 守恒断言 κ·h 已在上面锁定；结果文档将记录此修正）
    h = 42883
    p = dict((d, (c, b)) for d, c, b in disk_plan("A:42883", h, 3,
                                                  BLOCK_TOKENS, KAPPA))
    n_full = sum(c for d, (c, _b) in p.items())
    assert n_full == 84
    total = plan_bytes(tuple((d, c, b) for d, (c, b) in p.items()))
    assert total == KAPPA * (83 * 512 + 387)


def test_disk_plan_deterministic():
    for h in (42883, 9399, 1000):
        a = disk_plan(f"B:{h}", h, 3, BLOCK_TOKENS, KAPPA)
        b = disk_plan(f"B:{h}", h, 3, BLOCK_TOKENS, KAPPA)
        assert a == b
        assert disk_of(f"B:{h}", 5, 3) == disk_of(f"B:{h}", 5, 3)


def test_shrink_disks_remaps_minority():
    """增删盘只重映射少数块（HRW 一致性）：3 盘→2 盘（disk_bps 40→60）。"""
    h = 42883
    n_b = -(-h // BLOCK_TOKENS)
    m3 = [disk_of("A:42883", i, 3) for i in range(n_b)]
    m2 = [disk_of("A:42883", i, 2) for i in range(n_b)]
    on_d2 = sum(1 for d in m3 if d == 2)
    changed = sum(1 for a, b in zip(m3, m2) if a != b)
    # 只有原落在被删盘(2)上的块可能移动；且是少数
    assert changed <= on_d2
    assert changed < n_b / 2


def test_cross_line_disk_of_matches_mainline():
    """与主线 sim/storage.py::disk_of 同键同落点（跨线一致性对拍）。"""
    from sim.storage import disk_of as main_disk_of
    from sim.storage import disk_plan as main_disk_plan
    prefixes = ["A:42883", "B:9399", "C:0", "conv:123", "x:77"]
    for pid in prefixes:
        for disks in (2, 3, 4):
            for idx in range(40):
                assert disk_of(pid, idx, disks) == main_disk_of(
                    "cq", pid, idx, disks), (pid, disks, idx)
    # 计划（计数）同构：主线 float 版 vs cq Fraction 版
    for pid, h in (("A:42883", 42883), ("B:9399", 9399)):
        for disks in (2, 3):
            ml = main_disk_plan("cq", pid, h, disks, BLOCK_TOKENS,
                                float(KAPPA))
            cq = disk_plan(pid, h, disks, BLOCK_TOKENS, KAPPA)
            assert {d: c for d, c, _b in ml} == {d: c for d, c, _b in cq}, \
                (pid, disks)
            ml_bytes = {d: nb for d, _c, nb in ml}
            for d, _c, b in cq:
                assert abs(float(b) - ml_bytes[d]) <= 1e-9 * max(1.0, float(b))


# ---------------------------------------------------------------------------
# §7-2 每盘分配
# ---------------------------------------------------------------------------

def test_waterfill_weighted_equals_expanded():
    import random
    rng = random.Random(20261021)
    for _ in range(300):
        m = rng.randint(1, 6)
        weighted = [(F(rng.randint(1, 60)), rng.randint(1, 7))
                    for _ in range(m)]
        cap = F(rng.randint(0, 200))
        expanded = []
        for d, c in weighted:
            expanded.extend([d] * c)
        a1 = waterfill(cap, expanded, F(0))
        got = []
        for (d, c), (al, _c) in zip(weighted, waterfill_weighted(
                cap, weighted, F(0))):
            got.extend([al] * c)
        assert got == a1, (cap, weighted)


def _st_disk(numeric=F, b=F(120), disk_bps=F(40), kappa=F(1),
             block_tokens=10):
    return StorageSim(((F(0), b),), F(200), numeric=numeric,
                      alloc="max_min_disk", disk_bps=disk_bps,
                      block_tokens=block_tokens, kappa=kappa)


def test_per_block_demand_conservation():
    """逐块需求聚合守恒：Σ 组(逐块需求×块数) = q。"""
    st = _st_disk(kappa=F(1), block_tokens=10)
    f = st.submit(1, 1, 0, F(0), F(40), F(15), None, prefix_id="p", h_tokens=40)
    assert f.n_blocks == 4
    total = st.zero
    for g in st.groups[f.submit_seq]:
        per = f.q_gbps / f.n_blocks
        total += per * g.n_blocks
    assert total == f.q_gbps
    assert plan_bytes(tuple()) == 0


def test_single_flow_exclusive_full_disk_and_slowest_disk():
    """单流独占：每盘组速率=全盘容量；完成=最慢盘组（解析对拍）。"""
    st = _st_disk(kappa=F(1), block_tokens=10)      # B=120, 3 盘, cap=40
    f = st.submit(1, 1, 0, F(0), F(40), F(15), None, prefix_id="p", h_tokens=40)
    st.allocate(F(0))
    plan = dict((d, (c, b)) for d, c, b in
                disk_plan("p", 40, 3, 10, F(1)))
    gs = {g.disk: g for g in st.groups[f.submit_seq]}
    assert set(gs) == set(plan)
    for d, g in gs.items():
        assert g.rate_gbps == F(40)                  # 独占全盘
    # 完成由最慢盘（字节最大者）决定
    t_done = max(gs[d].remaining_gb / gs[d].rate_gbps for d in gs)
    comp = dict(st.completions(F(0)))
    assert set(comp) == {(f.submit_seq, d) for d in gs}   # 组级完成事件
    assert max(comp.values()) == t_done
    assert min(comp.values()) == min(
        gs[d].remaining_gb / gs[d].rate_gbps for d in gs)


def test_uniform_flows_aggregate_equals_b_over_n():
    """同类同质 N 流：每盘公平份额聚合 = B/N（与单池 max_min 期望重合）。"""
    st = _st_disk(kappa=F(1), block_tokens=10)
    N = 4
    for i in range(N):
        st.submit(i + 1, i + 1, 0, F(0), F(40), F(15), None,
                  prefix_id="same", h_tokens=40)
    st.allocate(F(0))
    for f in st.flows.values():
        # 每盘 N 组均分（等需求等块数）→ 组速率 cap/N → 流聚合 Σ_d cap/N
        assert f.rate_gbps == F(120) / N
    # 单池 max_min 同场景对照：等需求均分 B/N
    st1 = StorageSim(((F(0), F(120)),), F(200), numeric=F, alloc="max_min")
    for i in range(N):
        st1.submit(i + 1, i + 1, 0, F(0), F(40), F(15), None)
    st1.allocate(F(0))
    assert all(f.rate_gbps == F(120) / N for f in st1.flows.values())


def test_tilted_completion_analytic():
    """3:1 倾斜：两异前缀流，解析对拍每盘组速率与各自完成时刻。"""
    st = _st_disk(kappa=F(1), block_tokens=10)
    h = 40
    fa = st.submit(1, 1, 0, F(0), F(40), F(10), None, prefix_id="pa", h_tokens=h)
    fb = st.submit(2, 2, 0, F(0), F(40), F(10), None, prefix_id="pb", h_tokens=h)
    st.allocate(F(0))
    na, nb = fa.n_blocks, fb.n_blocks
    assert na == nb == 4
    by_disk = {}
    for f in (fa, fb):
        for g in st.groups[f.submit_seq]:
            by_disk.setdefault(g.disk, []).append(g)
    # 解析：每盘上两组等需求等块结构 → waterfill 精确值
    for d, gs in by_disk.items():
        k_total = sum(g.n_blocks for g in gs)
        per = fa.q_gbps / na
        # 展开为 k_total 个等需求 claimant：超订与否皆均分 cap
        exp_block = min(per, F(40) / k_total) if per * k_total > F(40) else \
            per + (F(40) - per * k_total) / k_total
        for g in gs:
            assert g.rate_gbps == exp_block * g.n_blocks
    for f in (fa, fb):
        gs = st.groups[f.submit_seq]
        t_flow = max(g.remaining_gb / g.rate_gbps for g in gs)
        comp = dict(st.completions(F(0)))
        assert max(v for (s, _d), v in comp.items() if s == f.submit_seq) \
            == t_flow


def test_group_completion_reallocates_that_disk():
    """组完成触发重配：先完成的组摘除后，同盘剩余组立即吃满。"""
    st = _st_disk(kappa=F(1), block_tokens=10)
    # p1: 4 块；p2: 4 块——先找各自计划，构造同盘对峙且字节不同
    f1 = st.submit(1, 1, 0, F(0), F(40), F(15), None, prefix_id="pa",
                   h_tokens=40)
    f2 = st.submit(2, 2, 0, F(0), F(20), F(15), None, prefix_id="pb",
                   h_tokens=20)
    st.allocate(F(0))
    # 推进到 f2（字节小）在某盘的组完成
    done = st.completed_groups()
    assert done == []
    comp = dict(st.completions(F(0)))
    t_first = min(comp.values())
    st.advance(F(0), t_first)
    done = st.completed_groups()
    assert done
    for seq, disk in done:
        g = st.remove_group(seq, disk)
        assert g is not None
    st.allocate(t_first)
    # 摘除后每盘Σ速率仍 ≤ 盘容量，且无组速率越过 40
    by_disk = {}
    for seq, gs in st.groups.items():
        for g in gs:
            by_disk.setdefault(g.disk, []).append(g.rate_gbps)
    for d, rates in by_disk.items():
        assert sum(rates) <= F(40) + F(1, 10**9)
    # f1 独占的盘吃满 40
    for seq, gs in st.groups.items():
        for g in gs:
            if sum(1 for _s, gl in st.groups.items()
                   for _g in gl if _g.disk == g.disk) == 1:
                assert g.rate_gbps == F(40)


def test_boost_scales_per_block_demand():
    """boost 后逐块需求同比例抬升：q→q_max 后盘上份额按新需求 waterfill。"""
    st = _st_disk(kappa=F(1), block_tokens=10)
    f1 = st.submit(1, 1, 0, F(0), F(400), F(10), None, prefix_id="pa",
                   h_tokens=400)
    f2 = st.submit(2, 2, 0, F(0), F(400), F(10), None, prefix_id="pb",
                   h_tokens=400)
    f1.due_s = F(1)                       # 引擎外手工设到期（到期提升路径）
    st.allocate(F(0))
    r1_before = {g.disk: g.rate_gbps for g in st.groups[f1.submit_seq]}
    st.advance(F(0), F(1))
    assert not st.completed_groups()      # t=1 仍在途（V 大速率有限）
    assert st.boost_due(F(1))
    assert f1.q_gbps == F(200) and f1.boosted
    st.allocate(F(1))
    # 逐块需求 10/40 → 50/40：共享盘上 f1 单块需求超 f2，份额抬升
    r1_after = {g.disk: g.rate_gbps for g in st.groups[f1.submit_seq]}
    assert any(r1_after[d] > r1_before[d] for d in r1_before)
    # 聚合需求守恒仍成立（q 抬升后逐块需求=q/n_b）
    per_total = sum((f1.q_gbps / f1.n_blocks) * g.n_blocks
                    for g in st.groups[f1.submit_seq])
    assert per_total == F(200)


# ---------------------------------------------------------------------------
# §7-4 回归锚：fcfs / max_min 忽略 disk_bps，行为逐位不变
# ---------------------------------------------------------------------------

def test_fcfs_maxmin_ignore_disk_bps_bitwise():
    specs = [RequestSpec(0, F(0), 4096, 512, "A", F(1), F(10)),
             RequestSpec(1, F(0), 4096, 512, "A", F(1), F(10))]
    base = CqScenario(scenario_name="t", profile=ProfileConfig(),
                      storage=StorageConfig(b_schedule=((F(0), F(40)),)),
                      limits=HardLimits(n_max=1), m_workers=2)
    for alloc in ("fcfs", "max_min"):
        outs = []
        for dbps in (F(40), F(100)):
            scn = replace(base, storage=replace(base.storage, alloc=alloc,
                                                disk_bps_gbps=dbps))
            eng = run_case(scn, specs, make_fcfs(), numeric=float, seed=0)
            outs.append([float(eng.w.requests[i].F_s) for i in (0, 1)])
        assert outs[0] == outs[1], alloc            # 逐位一致


def test_config_disk_validation_and_derivation():
    # disks 派生只看 B 峰值与 disk_bps（与 alloc 无关；仅 max_min_disk 使用它）
    assert StorageConfig(b_schedule=((F(0), F(120)),)).disks == 3
    sc = StorageConfig(b_schedule=((F(0), F(120)),), alloc="max_min_disk")
    assert sc.disks == 3                              # ceil(120/40)
    sc2 = StorageConfig(b_schedule=((F(0), F(120)),), alloc="max_min_disk",
                        disk_bps_gbps=F(60))
    assert sc2.disks == 2
    sc3 = StorageConfig(b_schedule=((F(0), F(80)), (F(100), F(200)),
                                    (F(300), F(80))),
                        alloc="max_min_disk")
    assert sc3.disks == 5                             # 峰值 200 口径
    with pytest.raises(ValueError):
        StorageConfig(alloc="bogus")
    with pytest.raises(ValueError):
        StorageConfig(alloc="max_min_disk", disk_bps_gbps=F(0))
    js = sc.to_json()
    assert js["alloc"] == "max_min_disk" and js["disks"] == 3
    assert js["disk_bps_gbps"] == "40"


def test_submit_requires_prefix_in_disk_mode():
    st = _st_disk()
    with pytest.raises(ValueError):
        st.submit(1, 1, 0, F(0), F(40), F(15), None)              # 缺 prefix
    with pytest.raises(ValueError):
        st.submit(1, 1, 0, F(0), F(40), F(15), None,
                  prefix_id="p", h_tokens=0)                      # 空计划
    st2 = StorageSim(((F(0), F(40)),), F(200), numeric=F)         # 聚合模式不受限
    st2.submit(1, 1, 0, F(0), F(40), F(15), None)


# ---------------------------------------------------------------------------
# §7-3/§7-5 完成语义、clone 与 forecast 穿透
# ---------------------------------------------------------------------------

def _disk_scenario(n_max=1, alloc="max_min_disk"):
    return CqScenario(
        scenario_name="t",
        profile=ProfileConfig(L=4, kappa_gb_per_token_layer=KAPPA),
        storage=StorageConfig(b_schedule=((F(0), F(120)),), alloc=alloc),
        limits=HardLimits(n_max=n_max, token_max=F(8192), workspace_gb=F(8)),
        m_workers=2)


def test_flow_completes_at_slowest_disk_group():
    """引擎级（Fraction 精确）：单请求独占 → 每层 read_ready − 提交时刻 =
    max_d(bytes_d)/40（最慢盘决定）；单池 max_min 同场景 = V/120（无倾斜）。"""
    scn = _disk_scenario()
    spec = RequestSpec(0, F(0), 42883, 256, "A", F(1), F(10))
    eng = CqEngine(scn, [spec], make_fcfs(), numeric=F)
    from sim.cq.observable import Observable
    obs = Observable(scn, numeric=F, seed=0)
    eng.observable = obs
    obs.attach(eng)
    eng.run()
    assert eng.w.all_done()
    b = eng.w.batches[0]
    plan = disk_plan("A:42883", 42883, 3, BLOCK_TOKENS, KAPPA)
    t_read = max(nb for _d, _c, nb in plan) / F(40)
    assert b.read_ready[0] == t_read
    for l in range(1, scn.profile.L):
        assert b.read_ready[l] - b.C[l - 1] == t_read      # 提交于 C_{l-1}
    # 单池 max_min 对照：V/120 < 最慢盘 max_bytes/40（倾斜损耗 = 差值）
    scn1 = replace(scn, storage=replace(scn.storage, alloc="max_min"))
    eng1 = CqEngine(scn1, [spec], make_fcfs(), numeric=F)
    obs1 = Observable(scn1, numeric=F, seed=0)
    eng1.observable = obs1
    obs1.attach(eng1)
    eng1.run()
    b1 = eng1.w.batches[0]
    assert b1.read_ready[0] == KAPPA * 42883 / F(120)
    assert b1.read_ready[0] < b.read_ready[0]


def test_clone_preserves_disk_groups():
    scn = _disk_scenario()
    specs = [RequestSpec(0, F(0), 42883, 256, "A", F(1), F(10)),
             RequestSpec(1, F(0), 9399, 4096, "B", F(1), F(10))]
    from sim.cq.simrun import run_case as _rc
    eng = _rc(scn, specs, make_fcfs(), numeric=float, seed=0,
              drain_budget_s=F("0.02"))            # 早停，保留在途流
    w = eng.w
    if not w.storage.flows:
        return                                     # 全部读完了（无在途可验）
    st = w.storage
    assert st.uses_groups and st.disks == 3
    wc = w.clone()
    stc = wc.storage
    assert stc.uses_groups and stc.disks == st.disks
    assert stc.kappa == st.kappa and stc.disk_bps == st.disk_bps
    assert set(stc.groups) == set(st.groups)
    for seq in st.groups:
        for g, gc in zip(st.groups[seq], stc.groups[seq]):
            assert (gc.disk, gc.n_blocks, gc.total_gb, gc.remaining_gb,
                    gc.rate_gbps) == (g.disk, g.n_blocks, g.total_gb,
                                      g.remaining_gb, g.rate_gbps)
    # 深拷贝：改原世界不影响 clone
    seq0 = next(iter(st.groups))
    st.groups[seq0][0].remaining_gb = F(12345)
    assert stc.groups[seq0][0].remaining_gb != F(12345)


def test_forecast_reinjection_rebuilds_same_groups():
    """推演重注入：估计世界流的盘组与真实流的完整计划同构（disk/块数/字节）。

    逐盘进度非公开（served 只有流级报告）→ 估计世界按完整计划 + 比例分摊
    remaining 重建（信息边界内的最优重建）；组结构必须与 disk_plan 逐位一致。
    快照在运行中抓取（层 1 流首提时刻）——run() 尾部 FINAL 结算会清
    worker.batch_id，事后快照不是策略可见的真实状态。
    """
    from sim.cq.engine import _prefix_of
    from sim.cq.observable import Observable
    scn = _disk_scenario()
    specs = [RequestSpec(0, F(0), 42883, 256, "A", F(1), F(10)),
             RequestSpec(1, F(0), 9399, 4096, "B", F(1), F(10))]

    class _SnapObs(Observable):
        snap = None

        def on_flow_submit(self, f, t):
            super().on_flow_submit(f, t)
            if _SnapObs.snap is None and f.layer >= 1:
                _SnapObs.snap = self.snapshot(t)

    obs = _SnapObs(scn, numeric=float, seed=0)
    eng = CqEngine(scn, specs, make_fcfs(), numeric=float, observable=obs)
    obs.attach(eng)
    eng.run()
    assert eng.w.all_done()
    snap = _SnapObs.snap
    assert snap is not None and any(not f.completed for f in snap.flows)
    est = build_forecast_engine(snap, scn, make_fcfs())
    spec_of = {s.rid: s for s in specs}
    est_by_fid = {fl.flow_id: fl for fl in est.w.storage.flows.values()}
    checked = 0
    for f in snap.flows:
        if f.completed:
            continue
        ef = est_by_fid.get(f.flow_id)
        assert ef is not None, f.flow_id
        # 真实批成员的完整计划（公开确定性重建）
        est_b = est.w.batches.get(f.batch_id)
        assert est_b is not None and est_b.members
        plan = {}
        for rid in est_b.members:
            s = spec_of[rid]
            for d, c, nb in disk_plan(_prefix_of(s), s.h_tokens, 3,
                                      BLOCK_TOKENS, KAPPA):
                cd, cb = plan.get(d, (0, 0.0))
                plan[d] = (cd + c, cb + float(nb))
        eg = {g.disk: (g.n_blocks, float(g.total_gb)) for g in
              est.w.storage.groups[ef.submit_seq]}
        assert eg == plan, (f.flow_id, eg, plan)
        checked += 1
    assert checked >= 1


def test_mpc_disk_mode_zero_fallback():
    """MPC 推演与真实同物理：小场景 fb=0（健康审计口径的 min 版）。"""
    scn = _disk_scenario()
    specs = [RequestSpec(0, F(0), 42883, 256, "A", F(1), F(10)),
             RequestSpec(1, F("0.3"), 9399, 4096, "B", F(1), F(10)),
             RequestSpec(2, F("0.6"), 0, 2657, "C", F(1), F(10))]
    pol = MPCPolicy(pid="cq_mpc", H=1, theta="S", c_ref=0.05,
                    budget_s=30.0, max_events=20000, base="edf",
                    composer=False)
    eng = run_case(scn, specs, pol, numeric=float, seed=0)
    assert eng.status == "done"
    assert pol.n_decides > 0
    assert pol.n_fallback == 0, pol.health()       # fb=0


# ---------------------------------------------------------------------------
# §7-6 端到端守恒（缩放冒烟与区分度在实验阶段做）
# ---------------------------------------------------------------------------

def test_end_to_end_conservation_disk_mode():
    scn = _disk_scenario()
    specs = [RequestSpec(0, F(0), 42883, 256, "A", F(1), F(10)),
             RequestSpec(1, F(0), 9399, 4096, "B", F(1), F(10)),
             RequestSpec(2, F(0), 0, 2657, "C", F(1), F(10))]
    eng = run_case(scn, specs, make_fcfs(), numeric=float, seed=0,
                   record_intervals=True)
    st = eng.w.storage
    assert eng.status == "done" and eng.w.all_done()
    # 服务量守恒：actual_integral == ΣV（全完成）
    served = float(st.actual_integral_gb)
    v_sum = 4 * (KAPPA * 42883 + KAPPA * 9399)     # L=4 层 × A/B 各 1 条
    assert abs(served - float(v_sum)) <= 1e-9 * float(v_sum)
    # 每盘速率不超容量（所有区段）
    for (_t0, _t1, rates) in st.disk_interval_log:
        assert len(rates) == 3
        for r in rates:
            assert float(r) <= 40.0 + 1e-9
    # 区段账本平行：Σ盘速率 == interval_log 总速率
    for (a, _b, r, _q), (_a2, _b2, dr) in zip(st.interval_log,
                                              st.disk_interval_log):
        assert abs(float(r) - sum(float(x) for x in dr)) <= 1e-9 * 40
    # fcfs 策略臂在盘模式：整跑可复现
    eng2 = run_case(scn, specs, make_fcfs(), numeric=float, seed=0)
    assert [float(r.F_s) for r in eng2.w.requests.values()] == \
        [float(r.F_s) for r in eng.w.requests.values()]
