"""trace 导入与生成：本地 Mooncake 三文件、冻结目录 trie、分块/缩放、合成 trace。

- Mooncake 原始字段 timestamp(相对毫秒)/input_length/output_length/hash_ids(512 token 块)；
- 主缓存假设固定 frozen_history_catalog：前 20% 请求的完整块构成前缀 trie，
  运行期不增删（§5.5）；
- 命中取 h=min(512*k, input_length-1)，保证至少一次末 token forward。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from .types import RequestSpec, frac

BLOCK = 512

MOONCAKE_FILES: Tuple[Tuple[str, int, int, str], ...] = (
    ("conversation_trace.jsonl", 12031, 3536999,
     "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df"),
    ("toolagent_trace.jsonl", 23608, 3536999,
     "48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71"),
    ("synthetic_trace.jsonl", 3993, 1022025,
     "bd070915a98fc0ed264d7cfef2ce746002eb3076a695ec31ba2674c0111ec131"),
)

TRACE_LABEL = {
    "conversation_trace.jsonl": "Conversation",
    "toolagent_trace.jsonl": "ToolAgent",
    "synthetic_trace.jsonl": "Mooncake-Synthetic",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    """规范 JSON（key 排序、紧凑分隔符、禁 NaN/Inf）的 SHA256。"""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class RawTraceRow:
    source_file: str
    source_line: int
    timestamp_ms: int
    input_length: int
    output_length: int
    hash_ids: Tuple[int, ...]
    # 首现命中派生字段(原始文件为默认值;派生文件读入后填充)
    hit_tokens: int = 0
    u_tokens: int = 0
    full_hit_adjusted: bool = False


def load_mooncake(path: str, fname: str) -> List[RawTraceRow]:
    """逐行读入原始 trace 并校验:合法 JSON、非负长度、hash 数=ceil(input/512)、
    按 (timestamp,source_line) 稳定排序(不读命中字段,供指纹核对与预处理用)。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ts, il = int(obj["timestamp"]), int(obj["input_length"])
            ol = int(obj["output_length"])
            hs = tuple(int(x) for x in obj["hash_ids"])
            if ts < 0 or il < 0 or ol < 0:
                raise ValueError(f"{fname}:{i} 负长度/时间戳")
            need = (il + BLOCK - 1) // BLOCK
            if len(hs) != need:
                raise ValueError(f"{fname}:{i} hash 数 {len(hs)} != ceil({il}/512)={need}")
            rows.append(RawTraceRow(fname, i, ts, il, ol, hs))
    if not rows:
        raise ValueError(f"{fname} 空文件")
    rows.sort(key=lambda r: (r.timestamp_ms, r.source_line))
    return rows


# ---------------------------------------------------------------------------
# 首现命中派生文件的导入(first_seen 单模式,变更设计 v1.4)
# ---------------------------------------------------------------------------


@dataclass
class TraceImport:
    """一份派生 trace(hit 已预计算)的导入结果与全量统计。"""

    fname: str
    rows: List[RawTraceRow]
    n_rows: int
    token_hit_ratio: float
    request_hit_ratio: float
    max_u: int
    full_hit_adjusted: int
    d_raw_s: Fraction
    derived_sha256: str

    def h_u_of(self, row: RawTraceRow) -> Tuple[int, int, bool]:
        """命中信息直接来自派生字段(h+u=input、u>=1 已在导入时校验)。"""
        return row.hit_tokens, row.u_tokens, row.full_hit_adjusted


def import_mooncake(derived_path: str, fname: str) -> TraceImport:
    """读 derived/<名>.hit.jsonl 并校验,返回全量统计。"""
    rows = []
    tot_in = tot_hit = n_hit = n_full = 0
    max_u = 0
    with open(derived_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ts, il = int(obj["timestamp"]), int(obj["input_length"])
            ol = int(obj["output_length"])
            hs = tuple(int(x) for x in obj["hash_ids"])
            hit = int(obj["hit_tokens"])
            u = int(obj["u_tokens"])
            full = bool(obj.get("full_hit_adjusted", False))
            if ts < 0 or il < 0 or ol < 0:
                raise ValueError(f"{fname}:{i} 负长度/时间戳")
            need = (il + BLOCK - 1) // BLOCK
            if len(hs) != need:
                raise ValueError(f"{fname}:{i} hash 数 {len(hs)} != ceil({il}/512)")
            if hit + u != il or u < 1 or hit > il - 1:
                raise ValueError(f"{fname}:{i} hit/u 字段非法: {hit}+{u}!={il}")
            rows.append(RawTraceRow(fname, i, ts, il, ol, hs, hit, u, full))
            tot_in += il
            tot_hit += hit
            n_hit += 1 if hit > 0 else 0
            n_full += 1 if full else 0
            max_u = max(max_u, u)
    if not rows:
        raise ValueError(f"{fname} 派生文件为空")
    rows.sort(key=lambda r: (r.timestamp_ms, r.source_line))
    d_raw_ms = max(r.timestamp_ms for r in rows) + 1
    return TraceImport(
        fname=fname, rows=rows, n_rows=len(rows),
        token_hit_ratio=tot_hit / tot_in,
        request_hit_ratio=n_hit / len(rows),
        max_u=max_u, full_hit_adjusted=n_full,
        d_raw_s=frac(d_raw_ms) / 1000,
        derived_sha256=sha256_file(derived_path))


# ---------------------------------------------------------------------------
# 时间块与缩放（§5.6）
# ---------------------------------------------------------------------------


@dataclass
class TimeBlock:
    block_id: int
    start_ms: int
    end_ms: int


def split_blocks(rows: Sequence[RawTraceRow], start_ms: int, end_ms: int,
                 n_blocks: int) -> List[TimeBlock]:
    """等时切块，边界左闭右开；整数毫秒 + 有理比例计算。"""
    span = end_ms - start_ms
    blocks = []
    for i in range(n_blocks):
        s = start_ms + (Fraction(span) * Fraction(i, n_blocks)).__floor__()
        e = start_ms + (Fraction(span) * Fraction(i + 1, n_blocks)).__floor__()
        blocks.append(TimeBlock(i, s, e))   # 统一窗口编号 0..N-1(全量打分)
    return blocks


def block_rows(rows: Sequence[RawTraceRow], blk: TimeBlock) -> List[RawTraceRow]:
    return [r for r in rows if blk.start_ms <= r.timestamp_ms < blk.end_ms]


def scaled_arrivals(rows_in_block: Sequence[RawTraceRow], blk: TimeBlock,
                    lam: Fraction, numeric=float) -> Tuple[List[object], object]:
    """目标负载 λ 下的缩放：scale=λ_raw/λ，arrival_sim=(ts/1000-s)*scale，
    D_sim=(e-s)*scale=N/λ。只缩放到达时间，保留同刻 burst。"""
    n = len(rows_in_block)
    if n == 0:
        return [], numeric(0)
    span_s = frac(blk.end_ms - blk.start_ms) / 1000
    lam_raw = frac(n) / span_s
    scale = lam_raw / lam
    arr = [numeric(frac(r.timestamp_ms) / 1000 - frac(blk.start_ms) / 1000) * numeric(scale)
           for r in rows_in_block]
    d_sim = numeric(span_s * scale)
    return arr, d_sim


def trace_hash(specs: Sequence[RequestSpec]) -> str:
    """规范 trace 哈希：含到达/类/h/u/T0/deadline。"""
    obj = [[int(s.rid), str(s.arrival_s), s.class_id, int(s.h_tokens),
            int(s.u_tokens), str(s.T0_s), str(s.deadline_s)] for s in specs]
    return canonical_hash(obj)


def arrival_shape_hash(specs: Sequence[RequestSpec]) -> str:
    """不含 SLO 的到达形状哈希（配对 SLO 扫描）。"""
    obj = [[int(s.rid), str(s.arrival_s), s.class_id, int(s.h_tokens),
            int(s.u_tokens)] for s in specs]
    return canonical_hash(obj)


# ---------------------------------------------------------------------------
# 合成 trace（§5.3）：Poisson / 突发，NumPy PCG64(SeedSequence([seed,stream]))
# ---------------------------------------------------------------------------


def synthetic_trace(seed: int, duration_s: float, lam: float,
                    class_probs: Optional[Sequence[float]] = None,
                    burst: bool = False) -> List[Tuple[float, str]]:
    """生成合成到达 (arrival_s, class_id)。stream 0 到达、1 类。"""
    import numpy as np
    rng0 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 0])))
    rng1 = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1])))
    if class_probs is None:
        class_probs = [0.25, 0.25, 0.25, 0.25]
    cids = [c[0] for c in
            (("HS_US", 512, 128), ("HS_UL", 512, 2048),
             ("HL_US", 8192, 128), ("HL_UL", 8192, 2048))]
    exps = rng0.exponential(1.0, size=max(1, int(duration_s * lam * 3)))
    out = []
    t = 0.0
    for w in exps:
        if burst:
            # 每 20s 周期：[0,10) 强度 1.6λ，[10,20) 强度 0.4λ；由累计强度逆映射
            w_scaled = w / lam
            # 逆映射：给定 Exp(1) 增量，找 Δt 使 ∫λ(t)dt = w
            t = _burst_advance(t, w_scaled, lam)
        else:
            t = t + w / lam
        if t >= duration_s:
            break
        out.append(t)
    cum = np.cumsum(class_probs)
    u = rng1.random(size=len(out))
    cls = np.searchsorted(cum, u, side="right").clip(0, 3)
    return [(float(a), cids[int(c)]) for a, c in zip(out, cls)]


def _burst_advance(t: float, w: float, lam: float) -> float:
    """突发强度 λ(t)：周期 20s，[0,10) 为 1.6λ、[10,20) 为 0.4λ。通过累计
    强度的逆函数逐个映射相同 Exp(1) 增量，跨阶段不丢弃等待时间。"""
    import math
    period = 20.0
    hi, lo = 1.6 * lam, 0.4 * lam
    remaining = w * lam  # 以“标准 Exp(1)×λ”为单位的服务量
    t = float(t)
    while remaining > 1e-15:
        ph = t % period
        if ph < 10.0:
            rate, seg = hi, 10.0 - ph
        else:
            rate, seg = lo, period - ph
        need = seg * rate
        if remaining <= need:
            return t + remaining / rate
        remaining -= need
        t += seg
    return t


def bandwidth_perturbation(b: Fraction, numeric=float):
    """共享带宽扰动 ((0,B),(60,0.5B),(100,B))，所有策略共享（§5.3）。"""
    half = b / 2
    return ((numeric(0), numeric(b)), (numeric(60), numeric(half)),
            (numeric(100), numeric(b)))
