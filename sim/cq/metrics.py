"""cq 指标：cohort 口径、右删失、summary、配对收益与块 bootstrap（§10）。

不复用旧 metrics.py 的分母。有限工作集 warmup=0、全体计入；在线 cohort 为
[warmup, arrival_stop) 到达请求。未完成请求 TTFT 右删失：F=null、
ttft_lower=censor-arrival，整 cohort 均值报下界。
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from .profile import batch_compute_s, singleton_K


def request_rows(engine) -> List[dict]:
    """逐请求行：rid、a、h/u、T0、deadline、batch、dispatch、F、分解。"""
    w = engine.w
    rows = []
    for rid in sorted(w.requests):
        rr = w.requests[rid]
        s = rr.spec
        b = w.batches.get(rr.batch_id) if rr.batch_id is not None else None
        queue = None if rr.dispatch_s is None else float(rr.dispatch_s) - float(s.arrival_s)
        comp = stall = None
        if b is not None and rr.F_s is not None:
            # 请求视角：batch_compute 对每个成员重复出现（仅解释 TTFT）
            comp = sum(b.c_layers)
            stall = (float(rr.F_s) - float(b.dispatch_s)) - comp
        rows.append({
            "rid": rid, "arrival_s": float(s.arrival_s),
            "h_tokens": s.h_tokens, "u_tokens": s.u_tokens,
            "T0_s": float(s.T0_s), "deadline_s": float(s.deadline_s),
            "batch_id": rr.batch_id, "dispatch_s": None if rr.dispatch_s is None else float(rr.dispatch_s),
            "F_s": None if rr.F_s is None else float(rr.F_s),
            "queue_s": queue, "batch_compute_s": comp, "batch_stall_s": stall,
            "class_id": s.class_id, "source_file": s.source_file,
            "source_line": s.source_line, "state": rr.state})
    return rows


def summarize(engine, scenario, warmup=None, arrival_stop=None) -> dict:
    """生成 summary.json 的核心指标（§10.2）。"""
    w = engine.w
    specs = w.specs
    arr = {rid: float(rr.spec.arrival_s) for rid, rr in w.requests.items()}
    stop = (arrival_stop if arrival_stop is not None
            else max(arr.values(), default=0.0) + 1e-9)
    wp = float(warmup) if warmup is not None else 0.0
    cohort = [rid for rid in sorted(w.requests) if wp <= arr[rid] < stop]
    censor = float(w.t)

    done = [rid for rid in cohort if w.requests[rid].F_s is not None]
    und = [rid for rid in cohort if w.requests[rid].F_s is None]
    ttft = {rid: (float(w.requests[rid].F_s) - arr[rid]) for rid in done}
    ttft_lb = {}
    for rid in cohort:
        if rid in ttft:
            ttft_lb[rid] = ttft[rid]
        else:
            ttft_lb[rid] = max(0.0, censor - arr[rid])

    def q(vals, p):
        if not vals:
            return None
        import numpy as np
        return float(np.quantile(np.array(vals), p, method="linear"))

    slo_ok = 0
    slo_unknown = 0
    slo_fail = 0
    for rid in cohort:
        s = specs[rid]
        if rid in done:
            if float(w.requests[rid].F_s) <= float(s.deadline_s):
                slo_ok += 1
            else:
                slo_fail += 1
        else:
            if float(s.deadline_s) <= censor:
                slo_fail += 1
            else:
                slo_unknown += 1
    n = len(cohort)
    window = max(stop - wp, 1e-12)

    # 设备分解（固定窗 = 全程积分，有限工作集下二者一致；在线场景标注口径）
    compute = sum(float(wk.compute_s) for wk in w.workers.values())
    stall = sum(float(wk.stall_s) for wk in w.workers.values())
    idle = sum(float(wk.idle_s) for wk in w.workers.values())
    mH = float(scenario.m_workers) * float(w.t)

    # 批效率（G 用纯计算：Σ singleton L*c / batch L*c）
    g_list, e_tok = [], []
    sum_single_k = 0.0
    sum_batch_k = 0.0
    for b in w.batches.values():
        mem = [specs[r] for r in b.members]
        sk = sum(float(batch_compute_s([specs[r]], scenario.profile)) * scenario.profile.L
                 for r in b.members)
        bk = sum(float(c) for c in b.c_layers)
        if bk > 0:
            g_list.append(sk / bk)
        N = sum(specs[r].u_tokens for r in b.members)
        if bk > 0:
            e_tok.append(N / bk)
        sum_single_k += sk
        sum_batch_k += bk
    served = float(w.storage.actual_integral_gb)
    cap = float(w.storage.integrated_capacity_gb)
    req_int = float(w.storage.requested_integral_gb)

    mean_lb = sum(ttft_lb.values()) / n if n else None
    norm_lb = (sum(ttft_lb[rid] / float(specs[rid].T0_s) for rid in cohort) / n
               if n else None)
    mean_full = (sum(ttft.values()) / len(done)) if done else None

    M = max((float(w.requests[rid].F_s) for rid in done), default=None)
    M = M if not und else None
    M_lb = max((float(w.requests[rid].F_s) if rid in done else censor)
               for rid in cohort) if cohort else None

    return {
        "n_cohort": n, "n_done": len(done), "n_unfinished": len(und),
        "censor_s": censor,
        "ttft_mean_lower": mean_lb,
        "ttft_mean_completed_only": mean_full,
        "ttft_norm_mean_lower": norm_lb,
        "ttft_p50_lower": q(sorted(ttft_lb.values()), 0.50),
        "ttft_p95_lower": q(sorted(ttft_lb.values()), 0.95),
        "ttft_p99_lower": q(sorted(ttft_lb.values()), 0.99),
        "is_lower_bound": bool(und),
        "slo_success": slo_ok, "slo_fail": slo_fail, "slo_unknown": slo_unknown,
        "slo_rate_interval": [slo_ok / n, (slo_ok + slo_unknown) / n] if n else None,
        "goodput_req_s": slo_ok / window,
        "makespan_M": M, "makespan_M_lower": M_lb,
        "device_compute_s": compute, "device_stall_s": stall,
        "device_idle_s": idle, "device_total_mh_s": mH,
        "storage_served_gb": served,
        "storage_utilization": (served / cap) if cap > 0 else None,
        "grant_ratio": (served / req_int) if req_int > 0 else None,
        "batch_G_mean": (sum(g_list) / len(g_list)) if g_list else None,
        "batch_G_total": (sum_single_k / sum_batch_k) if sum_batch_k > 0 else None,
        "E_token_median": q(e_tok, 0.5),
        "n_batches": len(w.batches),
        "status": engine.status,
        "n_events": engine.events_processed,
        "ctrl_time_s": engine.ctrl_time_s,
        "n_fallback": engine.n_fallback,
        "n_stale_invalid": engine.n_stale_invalid,
    }


# ---------------------------------------------------------------------------
# 配对收益与 bootstrap（§10.3）
# ---------------------------------------------------------------------------


def latency_gain(J_b: float, J_p: float) -> Optional[float]:
    """100*(J_b-J_p)/J_b；J_b<=0 时 None。"""
    if J_b is None or J_p is None or J_b <= 0:
        return None
    return 100.0 * (J_b - J_p) / J_b


def slo_gain_pp(A_b: float, A_p: float) -> float:
    return 100.0 * (A_p - A_b)


def goodput_gain_pct(G_b, G_p):
    if not G_b:
        return None
    return 100.0 * (G_p - G_b) / G_b


def paired_median(gains: Sequence[float]):
    import numpy as np
    if not gains:
        return None
    return float(np.median(np.array(gains)))


def moving_block_bootstrap(diffs_per_block: Sequence[Sequence[float]],
                           n_boot: int = 10000, block_len: int = 3,
                           seed: int = 20260915) -> Tuple[float, float]:
    """Mooncake 时间块的循环 moving-block bootstrap：块长 3，抽 5 个起点各取
    连续 3 块共 15 项，计算配对中位数，重复 n_boot 次；返回 2.5/97.5 分位。"""
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(seed))
    k = len(diffs_per_block)
    if k == 0:
        return (0.0, 0.0)
    meds = []
    for _ in range(n_boot):
        picked = []
        for _s in range(max(1, (k + block_len - 1) // block_len)):
            start = rng.integers(0, k)
            for j in range(block_len):
                picked.extend(diffs_per_block[(start + j) % k])
        meds.append(float(np.median(np.array(picked))))
    return (float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975)))


def ci_label(ci):
    return "-" if ci is None else f"[{ci[0]:.2f},{ci[1]:.2f}]"
