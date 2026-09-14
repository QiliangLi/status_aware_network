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


# ---------------------------------------------------------------------------
# 前缀 trie 目录
# ---------------------------------------------------------------------------


class PrefixCatalog:
    """完整块前缀 trie：插入建目录请求的完整 hash 前缀路径；查询沿同一路径走，
    遇首个缺块停止。只匹配连续前缀，不做集合命中。"""

    def __init__(self):
        self._children: Dict[int, "PrefixCatalog"] = {}
        self.size_blocks = 0

    def insert(self, hash_ids: Sequence[int], n_complete: int):
        node = self
        for hid in hash_ids[:n_complete]:
            nxt = node._children.get(hid)
            if nxt is None:
                nxt = PrefixCatalog()
                node._children[hid] = nxt
            node = nxt
        node.size_blocks = max(node.size_blocks, n_complete)

    def match(self, hash_ids: Sequence[int], n_complete: int) -> int:
        """返回连续命中的完整块数 k。"""
        node = self
        k = 0
        for hid in hash_ids[:n_complete]:
            nxt = node._children.get(hid)
            if nxt is None:
                break
            node = nxt
            k += 1
        return k


@dataclass
class RawTraceRow:
    source_file: str
    source_line: int
    timestamp_ms: int
    input_length: int
    output_length: int
    hash_ids: Tuple[int, ...]


def load_mooncake(path: str, fname: str) -> List[RawTraceRow]:
    """逐行读入并校验：合法 JSON、非负长度、hash 数=ceil(input/512)、按
    (timestamp,source_line) 稳定排序。"""
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


@dataclass
class TraceImport:
    """一个 Mooncake 文件的完整导入结果（含冻结目录与区间划分）。"""

    fname: str
    rows: List[RawTraceRow]
    d_raw_s: Fraction
    catalog: PrefixCatalog
    build_end_ms: int           # 0.2*D_raw 对应毫秒边界
    train_end_ms: int           # 0.4*D_raw
    n_catalog: int
    n_after: int
    hit_ratio_req: float
    hit_ratio_token: float
    max_u: int
    full_hit_adjusted: int
    file_sha256: str

    def h_u_of(self, row: RawTraceRow) -> Tuple[int, int, bool]:
        """查询冻结目录得 (h, u, full_hit_last_token_adjusted)。"""
        n_complete = row.input_length // BLOCK
        k = self.catalog.match(row.hash_ids, n_complete)
        h = min(BLOCK * k, row.input_length - 1)
        u = row.input_length - h
        return h, u, (BLOCK * k >= row.input_length)

    def split_bounds_ms(self) -> Tuple[int, int]:
        """[0,0.2D) 建目录、[0.2D,0.4D) 训练、[0.4D,D) 评价；整数毫秒边界。"""
        return self.build_end_ms, self.train_end_ms


def import_mooncake(path: str, fname: str) -> TraceImport:
    """导入单个 Mooncake 文件并构建冻结目录（§5.5 合同）。"""
    rows = load_mooncake(path, fname)
    max_ts = max(r.timestamp_ms for r in rows)
    d_raw_ms = max_ts + 1
    build_end = (Fraction(d_raw_ms) * Fraction(1, 5)).__floor__()
    train_end = (Fraction(d_raw_ms) * Fraction(2, 5)).__floor__()
    cat = PrefixCatalog()
    n_cat = 0
    for r in rows:
        if r.timestamp_ms < build_end:
            cat.insert(r.hash_ids, r.input_length // BLOCK)
            n_cat += 1
        else:
            break
    after = [r for r in rows if r.timestamp_ms >= build_end]
    n_hit = 0
    sum_h = 0
    sum_in = 0
    max_u = 0
    adj = 0
    for r in after:
        h, u, full = None, None, None
        h, u, full = _hu(cat, r)
        if h > 0:
            n_hit += 1
        sum_h += h
        sum_in += r.input_length
        max_u = max(max_u, u)
        adj += 1 if full else 0
    return TraceImport(
        fname=fname, rows=rows, d_raw_s=frac(d_raw_ms) / 1000, catalog=cat,
        build_end_ms=build_end, train_end_ms=train_end, n_catalog=n_cat,
        n_after=len(after), hit_ratio_req=n_hit / len(after),
        hit_ratio_token=sum_h / sum_in, max_u=max_u, full_hit_adjusted=adj,
        file_sha256=sha256_file(path))


def _hu(cat: PrefixCatalog, r: RawTraceRow) -> Tuple[int, int, bool]:
    n_complete = r.input_length // BLOCK
    k = cat.match(r.hash_ids, n_complete)
    h = min(BLOCK * k, r.input_length - 1)
    return h, r.input_length - h, (BLOCK * k >= r.input_length)


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
        blocks.append(TimeBlock(i if n_blocks == 5 else 100 + i, s, e))
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
