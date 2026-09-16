"""E25 时间序列聚合：锚定桶网格、区间求交、解析容量与守恒校验（规格 §5）。

只读观察者实现：从已完成 run 的 engine 事后聚合三路数据——
  1) worker 三态区段（WorkerState.segments，run() 的 FINAL 结算保证覆盖 [0,T_end]）；
  2) 存储速率区段（StorageSim.advance 在 record_intervals=True 时记录的
     (t0,t1,Σrate,Σq) 账本，默认关闭，e19–e24 轨迹零影响）；
  3) 积压区间（每请求 [arrival, F或T_end) 的到达-完成事件扫描，时间加权）。
桶容量 ∫B 由 schedule 闭式解析积分，独立于引擎累计器（守恒恒等式 3 的复算侧）。
"""
from __future__ import annotations

import bisect
import gzip
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

K_BUCKETS = 200          # 常规桶数；溢出桶下标为 K（[T_anchor, ∞)）
TOL = 1e-9               # 相对容差基准（时间秒 / 字节 GB 同口径）


class TsConservationError(AssertionError):
    """守恒恒等式失败（规格 §5.6）：该 run 不得进入图表。"""


def anchor_T(T_end_fcfs) -> float:
    """T_anchor = 0.1s 网格上取整的 1.1×T_end(FCFS)（规格 §5.1）。"""
    return math.ceil(1.1 * float(T_end_fcfs) * 10.0) / 10.0


def bucket_intersections(s, e, delta, K: int = K_BUCKETS):
    """区间 [s,e) 与桶网格求交 → [(i, ov)]；i==K 为溢出桶 [KΔ,∞)。

    左闭右开：端点重合时边界值归属右侧桶的左端点；零长/零交不输出。
    """
    s, e, delta = float(s), float(e), float(delta)
    if delta <= 0 or e <= s:
        return []
    if s >= K * delta:
        return [(K, e - s)]
    out = []
    # ceil−1e-9 抵御 e/delta 因浮点略低于真实整数而漏掉最后一个部分桶
    i1 = min(int(math.ceil(e / delta - 1e-9)), K)
    for i in range(int(s // delta), i1 + 1):
        lo = i * delta
        if i == K:
            ov = e - max(s, lo)
        else:
            ov = min(e, lo + delta) - max(s, lo)
        if ov > 0:
            out.append((i, ov))
    return out


def sweep_backlog(events, T_end, delta, K: int = K_BUCKETS):
    """到达(+1)/完成(-1)事件扫描常值积压区间 → (qsum, qmax)。

    qsum 为时间加权和（桶内 Σ len×时长），qmax 为桶内峰值；事件平均与时间
    加权平均不同，禁止混用（规格 §5.5 例 2）。
    """
    n = K + 1
    qsum = [0.0] * n
    qmax = [0] * n
    T_end = float(T_end)
    cnt = 0
    prev = 0.0
    for t, d in sorted((float(t), d) for t, d in events):
        if t < 0.0:
            t = 0.0
        if t > T_end:
            t = T_end
        if t > prev:
            for i, ov in bucket_intersections(prev, t, delta, K):
                qsum[i] += cnt * ov
                if cnt > qmax[i]:
                    qmax[i] = cnt
        cnt += d
        if t > prev:
            prev = t
    if prev < T_end:
        for i, ov in bucket_intersections(prev, T_end, delta, K):
            qsum[i] += cnt * ov
            if cnt > qmax[i]:
                qmax[i] = cnt
    return qsum, qmax


# ---------------------------------------------------------------------------
# B(t) schedule 的解析积分（独立于引擎累计器）
# ---------------------------------------------------------------------------

def _sched_floats(b_schedule) -> List[Tuple[float, float]]:
    return [(float(t), float(v)) for t, v in b_schedule]


def schedule_integral(b_schedule, lo: float, hi: float) -> float:
    """∫B dt over [lo,hi)，阶梯 schedule 闭式分段积分。"""
    if hi <= lo:
        return 0.0
    sched = _sched_floats(b_schedule)
    total = 0.0
    for j, (seg_start, v) in enumerate(sched):
        seg_end = sched[j + 1][0] if j + 1 < len(sched) else math.inf
        s2, e2 = max(lo, seg_start), min(hi, seg_end)
        if e2 > s2:
            total += v * (e2 - s2)
    return total


def capacity_per_bucket(b_schedule, delta: float, K: int, T_end: float
                        ) -> List[float]:
    """逐桶容量 ∫B over [iΔ, min((i+1)Δ,T_end))；溢出桶 [T_anchor,T_end)。"""
    T_end = float(T_end)
    out = []
    for i in range(K):
        lo = i * delta
        hi = min((i + 1) * delta, T_end)
        out.append(schedule_integral(b_schedule, lo, hi) if hi > lo else 0.0)
    out.append(schedule_integral(b_schedule, K * delta, T_end))
    return out


def b_max_in(b_schedule, lo: float, hi: float) -> float:
    """[lo,hi) 内可用带宽的最大值（FG03：逐桶速率不超过桶内 B 峰值）。"""
    sched = _sched_floats(b_schedule)
    best = 0.0
    for j, (seg_start, v) in enumerate(sched):
        seg_end = sched[j + 1][0] if j + 1 < len(sched) else math.inf
        if min(hi, seg_end) > max(lo, seg_start):
            best = max(best, v)
    return best


# ---------------------------------------------------------------------------
# 守恒校验（§5.6）：对建成/载入的 rows 复算
# ---------------------------------------------------------------------------

def check_rows_conservation(rows: List[dict], m_workers: int, T_end: float,
                            storage_served=None, storage_requested=None,
                            storage_capacity=None, tol: float = TOL) -> None:
    """三条恒等式；storage_* 给出引擎累计参照（None 跳过该项）。失败抛错。"""
    T_end = float(T_end)

    def _close(a, b, what):
        if abs(a - b) > tol * max(1.0, abs(float(b))):
            raise TsConservationError(
                f"守恒失败[{what}]: {a!r} != {b!r}")

    dev = sum(r["compute_s"] + r["stall_s"] + r["idle_s"] for r in rows)
    _close(dev, m_workers * T_end, "Σ(compute+stall+idle)=m×T_end")
    served = sum(r["served_gb"] for r in rows)
    if storage_served is not None:
        _close(served, float(storage_served), "Σserved=actual_integral")
    if storage_requested is not None:
        _close(sum(r["requested_gb"] for r in rows),
               float(storage_requested), "Σrequested=requested_integral")
    cap = sum(r["capacity_gb"] for r in rows)
    if storage_capacity is not None:
        _close(cap, float(storage_capacity), "Σcapacity=∫B[0,T_end)")


# ---------------------------------------------------------------------------
# 主聚合入口
# ---------------------------------------------------------------------------

def aggregate_run(engine, T_anchor, K: int = K_BUCKETS, *, cell: str = "",
                  policy: str = "", theta=None, mode: str = "",
                  tol: float = TOL) -> dict:
    """从已完成 run 的 engine 事后聚合（规格 §5.3 算法 TS 的只读实现）。"""
    w = engine.w
    T_end = float(w.t)
    T_anchor = float(T_anchor)
    meta = {
        "cell": cell, "policy": policy, "theta": theta, "mode": mode,
        "T_anchor_s": T_anchor, "K": K, "T_end_s": T_end,
        "delta_s": (T_anchor / K) if (T_anchor > 0 and K > 0) else 0.0,
        "b_schedule": _sched_floats(w.storage.b_schedule),
        "m_workers": len(w.workers),
        "overflow_used": bool(T_anchor > 0 and T_end > T_anchor + 1e-12),
    }
    delta = meta["delta_s"]
    if T_anchor <= 0 or delta <= 0:
        meta["note"] = "empty_run"
        return {"meta": meta, "rows": []}

    comp = [0.0] * (K + 1)
    stall = [0.0] * (K + 1)
    idle = [0.0] * (K + 1)
    tgt = {"COMPUTE": comp, "STALL": stall, "IDLE": idle}
    # 每 NPU 分桶三态（可选扩展字段，向后兼容；供每实例小倍图）
    m = len(w.workers)
    wid_index = {wid: idx for idx, wid in enumerate(sorted(w.workers))}
    state_idx = {"COMPUTE": 0, "STALL": 1, "IDLE": 2}
    per_w = [[[0.0, 0.0, 0.0] for _ in range(m)] for _ in range(K + 1)]
    for wk in w.workers.values():
        wi = wid_index[wk.worker_id]
        for (s, e, state, _b, _l) in wk.segments:
            acc = tgt.get(state)
            if acc is None:
                continue
            si = state_idx[state]
            for i, ov in bucket_intersections(float(s), float(e), delta, K):
                acc[i] += ov
                per_w[i][wi][si] += ov

    served = [0.0] * (K + 1)
    requested = [0.0] * (K + 1)
    for (t0, t1, rate, q) in getattr(w.storage, "interval_log", None) or []:
        for i, ov in bucket_intersections(float(t0), float(t1), delta, K):
            served[i] += float(rate) * ov
            requested[i] += float(q) * ov

    events = []
    for rr in w.requests.values():
        a = max(0.0, float(rr.spec.arrival_s))
        end = float(rr.F_s) if rr.F_s is not None else T_end
        events.append((a, 1))
        events.append((min(end, T_end), -1))
    qsum, qmax = sweep_backlog(events, T_end, delta, K)

    capacity = capacity_per_bucket(w.storage.b_schedule, delta, K, T_end)

    arrs = sorted(max(0.0, float(rr.spec.arrival_s))
                  for rr in w.requests.values())
    dones = sorted(float(rr.F_s) for rr in w.requests.values()
                   if rr.F_s is not None)

    rows: List[dict] = []
    n_req = len(w.requests)
    for i in range(K + 1):
        lo = i * delta
        hi_nom = (i + 1) * delta if i < K else T_end
        hi = min(hi_nom, T_end)
        width = max(0.0, hi - lo) if lo < T_end else 0.0
        if i == K and width <= 0:
            continue   # 溢出桶未用到（策略快于锚）
        hi_ref = hi if width > 0 else T_end
        rows.append({
            "i": i, "t_start_s": lo, "width_s": width,
            "compute_s": comp[i], "stall_s": stall[i], "idle_s": idle[i],
            "served_gb": served[i], "capacity_gb": capacity[i],
            "requested_gb": requested[i],
            "queue_mean": (qsum[i] / width) if width > 0 else None,
            "queue_max": qmax[i],
            "cum_arrived": bisect.bisect_right(arrs, hi_ref + 1e-9),
            "cum_done": bisect.bisect_right(dones, hi_ref + 1e-9),
            "workers": per_w[i],
        })
        if rows[-1]["cum_arrived"] > n_req or rows[-1]["cum_done"] > n_req:
            raise TsConservationError("累计口径越界")

    check_rows_conservation(
        rows, len(w.workers), T_end,
        storage_served=w.storage.actual_integral_gb,
        storage_requested=w.storage.requested_integral_gb,
        storage_capacity=w.storage.integrated_capacity_gb, tol=tol)
    return {"meta": meta, "rows": rows}


def row_fractions(row: dict, m_workers: int):
    """三态占比 (compute, stall, idle)/(m×桶宽)；空尾桶（width<=0）返回 None，
    图上层据此留白而非画 0（FG02）。"""
    width = row["width_s"]
    if width <= 0:
        return None
    denom = m_workers * width
    return (row["compute_s"] / denom, row["stall_s"] / denom,
            row["idle_s"] / denom)


def save_ts(path: str, obj: dict) -> None:
    """gzip JSON 落盘；禁 NaN/Infinity（规格 §5.4）。"""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"),
                  allow_nan=False)


def load_ts(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
