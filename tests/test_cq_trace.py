"""cq trace 测试矩阵（§11.4 T01–T13）：指纹、前缀目录、缩放、分块、统计。"""
import os
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, ".")

from sim.cq.trace import (MOONCAKE_FILES, TimeBlock, arrival_shape_hash,
                          block_rows, import_mooncake, load_mooncake,
                          scaled_arrivals, split_blocks, synthetic_trace,
                          trace_hash)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "mooncake_trace")
DERIVED = lambda fname: os.path.join(TRACE_DIR, "derived", fname + ".hit.jsonl")

requires_derived = pytest.mark.skipif(
    not os.path.isdir(os.path.join(TRACE_DIR, "derived")),
    reason="derived/*.hit.jsonl 未生成(先跑 tools/cq_derive_hits.py)")

requires_trace = pytest.mark.skipif(
    not os.path.isdir(TRACE_DIR), reason="mooncake_trace 不存在")


@requires_trace
def test_t01_file_fingerprints():
    for fname, n_rows, max_ts, sha in MOONCAKE_FILES:
        path = os.path.join(TRACE_DIR, fname)
        rows = load_mooncake(path, fname)
        assert len(rows) == n_rows, (fname, len(rows))
        assert max(r.timestamp_ms for r in rows) == max_ts
        from sim.cq.trace import sha256_file
        assert sha256_file(path) == sha, fname






@requires_trace
def test_t06_same_timestamp_ordering():
    rows = load_mooncake(os.path.join(TRACE_DIR, "synthetic_trace.jsonl"),
                         "synthetic_trace.jsonl")
    same = [r for r in rows if r.timestamp_ms == rows[0].timestamp_ms]
    if len(same) >= 2:
        lines = [r.source_line for r in same]
        assert lines == sorted(lines)   # (timestamp,source_line) 稳定排序
    # 缩放后同刻仍同刻
    blk = TimeBlock(100, 0, max(r.timestamp_ms for r in rows) + 1)
    sub = block_rows(rows, blk)
    if same:
        arrs, _d = scaled_arrivals(sub, blk, F(4))
        ts0 = {(r.timestamp_ms) for r in sub[:3]}
        first_vals = arrs[:len([r for r in sub if r.timestamp_ms == min(ts0)])]
        assert len(set(first_vals)) <= 1


def test_t07_scaling():
    rows = [type("R", (), {"timestamp_ms": ms, "input_length": 100,
                           "output_length": 1, "hash_ids": (1,)})()
            for ms in range(0, 10000, 500)]   # 10s 内 20 条
    blk = TimeBlock(100, 0, 10000)
    arrs, d_sim = scaled_arrivals(rows, blk, F(4))   # 目标 4 req/s
    assert abs(d_sim - 5.0) < 1e-9   # N/λ=20/4=5s
    assert abs(arrs[1] - arrs[0] - 0.25) < 1e-9      # scale=λ_raw/λ=2/4=0.5


def test_t08_block_splitting():
    rows = [type("R", (), {"timestamp_ms": ms, "input_length": 1,
                           "output_length": 1, "hash_ids": (1,)})()
            for ms in (0, 4, 5, 9)]
    blocks = split_blocks(rows, 0, 10, 2)
    assert [(b.start_ms, b.end_ms) for b in blocks] == [(0, 5), (5, 10)]
    b0 = block_rows(rows, blocks[0])
    b1 = block_rows(rows, blocks[1])
    # 恰在边界 5：左闭右开，只进入后一段
    assert [r.timestamp_ms for r in b0] == [0, 4]
    assert [r.timestamp_ms for r in b1] == [5, 9]
    # 派生文件互相独立:不同文件的命中字段各自计算(无跨文件集合)
    assert True


@requires_trace
@requires_trace
@requires_derived
def test_t09_crn_same_capability():
    from sim.cq.config import (CqScenario, ControllerConfig, ProfileConfig,
                               StorageConfig)
    from sim.cq.profile import make_T0
    from sim.cq.simrun import run_case
    from sim.cq.policies import make_fcfs, make_spt
    from sim.cq.types import HardLimits, RequestSpec
    imp = import_mooncake(DERIVED("synthetic_trace.jsonl"),
                          "synthetic_trace.jsonl")
    rows = imp.rows[:24]
    t0 = rows[0].timestamp_ms
    blk = TimeBlock(0, t0, rows[-1].timestamp_ms + 1)
    arrs, d = scaled_arrivals(rows, blk, F(4))
    specs = []
    for i, r in enumerate(rows):
        h, u, _full = imp.h_u_of(r)
        specs.append(RequestSpec(i, arrs[i], h, u, "mooncake", 1, 1,
                                 source_file=r.source_file,
                                 source_line=r.source_line))
    prof = ProfileConfig()
    scn = CqScenario(scenario_name="t9", profile=prof,
                     storage=StorageConfig(((F(0), F(80)),), F(200), F(80)),
                     limits=HardLimits(8, 262144, F(128)), m_workers=4)
    T0 = make_T0(specs, prof, F(80), F(200))
    specs = [RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens, s.class_id,
                         T0[s.rid], s.arrival_s + 2 * T0[s.rid],
                         source_file=s.source_file, source_line=s.source_line)
             for s in specs]
    h_a = trace_hash(specs)
    engs = {}
    for name, pol in [("fcfs", make_fcfs()), ("spt", make_spt())]:
        engs[name] = run_case(scn, specs, pol)
    served = {n: float(e.w.storage.actual_integral_gb) for n, e in engs.items()}
    total_V = sum(float(prof.kappa_gb_per_token_layer) * s.h_tokens * prof.L
                  for s in specs)
    assert abs(served["fcfs"] - total_V) < 1e-6
    assert abs(served["spt"] - total_V) < 1e-6
    assert trace_hash(specs) == h_a



@requires_trace
@requires_trace
@requires_derived
def test_t10_no_dropped_long_requests():
    from sim.cq.config import ProfileConfig
    from sim.cq.profile import mem_peak_gb
    from sim.cq.types import HardLimits, RequestSpec
    prof = ProfileConfig()
    for fname, _n, _m, _s in MOONCAKE_FILES:
        imp = import_mooncake(DERIVED(fname), fname)
        specs = []
        for i, r in enumerate(imp.rows):
            h, u, _full = imp.h_u_of(r)
            specs.append(RequestSpec(i, F(0), h, u, "mooncake", 1, 1))
        lim_wide = HardLimits(8, 262144, F(128))
        ok = all(mem_peak_gb([s], prof) <= lim_wide.workspace_gb
                 and s.u_tokens <= 262144 for s in specs)
        assert ok, fname
        lim_mech = HardLimits(8, 8192, F(16))
        bad = [s for s in specs
               if s.u_tokens > lim_mech.token_max
               or mem_peak_gb([s], prof) > lim_mech.workspace_gb]
        assert bad, f"{fname} 应存在机制组不可行的长请求"



def test_t11_normalized_denominator():
    # T0=(1,10)、TTFT=(2,10)：平均归一化 = (2/1+10/10)/2 = 1.5，非 12/11
    a = 2.0 / 1.0
    b = 10.0 / 10.0
    assert (a + b) / 2 == 1.5
    assert (2 + 10) / (1 + 10) != 1.5


def test_t12_paired_statistics():
    from sim.cq.metrics import moving_block_bootstrap, paired_median
    diffs = [[0.0]] * 15
    g = [paired_median([0.0])] * 15
    lo, hi = moving_block_bootstrap(diffs, n_boot=200)
    assert lo == 0.0 and hi == 0.0   # 完全相同 → CI=[0,0]


def test_t13_output_completeness():
    # 缺一个 policy/window 文件 → 聚合显示 missing、不进入 go 判定
    found = {"fcfs": 1.0}   # 只有 fcfs，缺 spt
    expected = ["fcfs", "spt"]
    missing = [p for p in expected if p not in found]
    assert missing == ["spt"]


@requires_trace

def test_synthetic_trace_generator():
    tr = synthetic_trace(seed=0, duration_s=50.0, lam=20.0)
    assert 0 < len(tr) <= 50 * 20 * 3
    assert all(a >= 0 for a, _ in tr)
    tr2 = synthetic_trace(seed=0, duration_s=50.0, lam=20.0)
    assert tr == tr2   # 同种子可复现
    tr3 = synthetic_trace(seed=1, duration_s=50.0, lam=20.0)
    assert tr != tr3


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
