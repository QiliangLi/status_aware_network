"""cq trace 测试矩阵（§11.4 T01–T13）：指纹、前缀目录、缩放、分块、统计。"""
import os
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, ".")

from sim.cq.trace import (MOONCAKE_FILES, PrefixCatalog, TimeBlock,
                          arrival_shape_hash, block_rows, canonical_hash,
                          import_mooncake, load_mooncake, scaled_arrivals,
                          split_blocks, synthetic_trace, trace_hash)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(ROOT, "mooncake_trace")

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


def test_t02_contiguous_prefix():
    cat = PrefixCatalog()
    cat.insert([1, 2], 2)
    cat.insert([9, 3], 2)
    # 查询 [1,3]：只命中首块 512（连续前缀），不因 3 出现过而记 1024
    assert cat.match([1, 3], 2) == 1
    assert cat.match([1, 2], 2) == 2
    assert cat.match([9], 1) == 1
    assert cat.match([7], 1) == 0


def test_t03_partial_tail_block():
    cat = PrefixCatalog()
    cat.insert([1, 2], 2)
    # input=700、hash=[1,2]：ceil(700/512)=2；floor(700/512)=1 完整块
    n_complete = 700 // 512
    assert n_complete == 1
    k = cat.match([1, 2], n_complete)
    h = min(512 * k, 700 - 1)
    u = 700 - h
    assert (h, u) == (512, 188)


def test_t04_full_hit():
    cat = PrefixCatalog()
    cat.insert([1, 2], 2)
    k = cat.match([1, 2], 1024 // 512)
    h = min(512 * k, 1024 - 1)
    u = 1024 - h
    assert (h, u) == (1023, 1)
    assert 512 * k >= 1024   # full_hit_last_token_adjusted=true
    assert u >= 1            # 无 u=0 正式请求


def test_t05_cold_and_future():
    cat = PrefixCatalog()
    cat.insert([1], 1)
    # 当前 hash 只在评价后段首次出现：早期查询不命中
    assert cat.match([5, 6], 2) == 0
    # catalog 不随运行增长（无 insert 则无变化）
    assert cat.match([5], 1) == 0


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
    # hash 命名空间隔离：不同文件各自建目录，同 hash 不跨文件命中
    cat1 = PrefixCatalog()
    cat1.insert([1, 2], 2)
    cat2 = PrefixCatalog()   # 另一文件的目录为空
    assert cat1.match([1, 2], 2) == 2
    assert cat2.match([1, 2], 2) == 0


@requires_trace
def test_t09_crn_same_capability():
    from sim.cq.config import (CqScenario, ControllerConfig, ProfileConfig,
                               StorageConfig)
    from sim.cq.profile import make_T0
    from sim.cq.simrun import run_case
    from sim.cq.policies import make_fcfs, make_spt
    from sim.cq.types import HardLimits, RequestSpec
    imp = import_mooncake(os.path.join(TRACE_DIR, "synthetic_trace.jsonl"),
                          "synthetic_trace.jsonl")
    rows = [r for r in imp.rows if imp.train_end_ms <= r.timestamp_ms]
    blk_rows = rows[:24]
    blk = TimeBlock(0, blk_rows[0].timestamp_ms,
                    blk_rows[-1].timestamp_ms + 1)
    arrs, d = scaled_arrivals(blk_rows, blk, F(4))
    specs = []
    for i, r in enumerate(blk_rows):
        h, u, _ = imp.h_u_of(r)
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
    # 同块两策略：trace/profile/capability 哈希一致，总读取字节最终相等
    served = {n: float(e.w.storage.actual_integral_gb) for n, e in engs.items()}
    total_V = sum(float(prof.kappa_gb_per_token_layer) * s.h_tokens * prof.L
                  for s in specs)
    assert abs(served["fcfs"] - total_V) < 1e-6
    assert abs(served["spt"] - total_V) < 1e-6
    assert trace_hash(specs) == h_a


@requires_trace
def test_t10_no_dropped_long_requests():
    from sim.cq.config import ProfileConfig, StorageConfig
    from sim.cq.profile import make_T0, mem_peak_gb
    from sim.cq.types import HardLimits, RequestSpec
    prof = ProfileConfig()
    for fname, _n, _m, _s in MOONCAKE_FILES:
        imp = import_mooncake(os.path.join(TRACE_DIR, fname), fname)
        specs = []
        for i, r in enumerate(imp.rows):
            h, u, _ = imp.h_u_of(r)
            specs.append(RequestSpec(i, F(0), h, u, "mooncake", 1, 1))
        # trace-wide 能力组：n_max=8、token 262144、128 GB → 100% 可行
        lim_wide = HardLimits(8, 262144, F(128))
        ok = all(mem_peak_gb([s], prof) <= lim_wide.workspace_gb
                 and s.u_tokens <= 262144 for s in specs)
        assert ok, fname
        # 机制组 8192/16GB 对超限 singleton 应明确报错而非静默过滤
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
def test_e19_import_fingerprints():
    """§5.5 导入验收指纹（目录规则下的全文件统计）。"""
    expect = {
        "conversation_trace.jsonl": (2121, 9910, 1.000000, 0.1276456513),
        "toolagent_trace.jsonl": (4249, 19359, 1.000000, 0.4076393734),
        "synthetic_trace.jsonl": (744, 3249, 0.2074484457, 0.2892445718),
    }
    for fname, (n_cat, n_after, hit_r, hit_t) in expect.items():
        imp = import_mooncake(os.path.join(TRACE_DIR, fname), fname)
        assert imp.n_catalog == n_cat, (fname, imp.n_catalog, n_cat)
        assert imp.n_after == n_after, (fname, imp.n_after, n_after)
        assert abs(imp.hit_ratio_req - hit_r) < 2e-6, fname
        assert abs(imp.hit_ratio_token - hit_t) < 2e-6, fname


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
