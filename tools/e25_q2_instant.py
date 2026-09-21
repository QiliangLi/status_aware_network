"""三问深答（20260921）：瞬时口径的带宽需求/供给分析（interval_log 重跑）。

用 StorageSim.interval_log（每次速率变化一个区段，毫秒级分辨率）重跑 FCFS，
回答：B=80 时瞬时需求是否超容量、超多久；B=20 时过载时段的真实占比与时长。
不落任何正式结果，仅 stdout + results/cq/eval/e25_q2/instant.json 供文档引用。
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.getcwd())

from sim.cq.metrics import summarize
from sim.cq.simrun import run_case
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                       MooncakeSource, build_policy,
                                       default_scenario)
from sim.experiments.e25_factor import fw_specs
from sim.cq.profile import ProfileConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "cq", "eval", "e25_q2", "instant.json")
FILES = {"conversation": "conversation_trace.jsonl",
         "toolagent": "toolagent_trace.jsonl",
         "synthetic": "synthetic_trace.jsonl"}


def episodes(log, b):
    """Σq > B 的连续区段 → 时长列表（秒）。"""
    out = []
    cur = 0.0
    for t0, t1, rate, q in log:
        w = float(t1) - float(t0)
        if w <= 0:
            continue
        if float(q) > b:
            cur += w
        else:
            if cur > 0:
                out.append(cur)
            cur = 0.0
    if cur > 0:
        out.append(cur)
    return out


def analyze(log, b):
    seg = [(float(t1) - float(t0), float(q), float(rate))
           for t0, t1, rate, q in log if float(t1) > float(t0)]
    tot = sum(w for w, _, _ in seg)
    qs = sorted(q for _, q, _ in seg)

    def tq(p):
        return qs[min(len(qs) - 1, int(p * len(qs)))]

    over_w = sum(w for w, q, _ in seg if q > b)
    touch_w = sum(w for w, q, r in seg if r >= 0.999 * b)
    excess_gb = sum((q - r) * w for w, q, r in seg if q > r)
    eps = episodes(log, b)
    eps.sort()
    return {
        "B": b,
        "sim_time_s": tot,
        "n_segments": len(seg),
        "seg_med_ms": 1000 * sorted(w for w, _, _ in seg)[len(seg) // 2],
        "q_p50": tq(0.5), "q_p90": tq(0.9), "q_p99": tq(0.99), "q_max": qs[-1],
        "rate_max": max(r for _, _, r in seg),
        "instant_over_frac": over_w / tot,          # 瞬时 Σq>B 时间占比
        "instant_touch_frac": touch_w / tot,        # 瞬时 Σrate≥B 时间占比
        "excess_gb": excess_gb,                     # 被削掉的需求积分
        "n_episodes": len(eps),
        "ep_med_ms": 1000 * eps[len(eps) // 2] if eps else 0.0,
        "ep_p90_ms": 1000 * eps[int(0.9 * len(eps))] if eps else 0.0,
        "ep_max_ms": 1000 * eps[-1] if eps else 0.0,
    }


def main():
    srcs = {}
    out = {}
    cfgs = []
    for k, f in FILES.items():
        cfgs.append((k, "OL", 80.0, 0.6))
        cfgs.append((k, "OL", 20.0, 0.6))
    cfgs.append(("synthetic", "FW", 80.0, None))     # local 掉 6.3pp 的格点
    cfgs.append(("toolagent", "OL", 80.0, 0.9))      # Q3 高负载对照

    for key, mode, B, rho in cfgs:
        if key not in srcs:
            srcs[key] = MooncakeSource(FILES[key], TRACE_DIR_DEFAULT)
        src = srcs[key]
        w = 9
        scn = default_scenario(B_gbps=B, m=4, alpha=F(4),
                               limits=TRACE_WIDE_LIMITS)
        if mode == "OL":
            lam = src.lam0 * F(str(rho))
            specs, d_sim = src.window_specs(w, lam, F(4), duration_cap=None,
                                            limits=TRACE_WIDE_LIMITS)
        else:
            specs = fw_specs(src, w, F(4), ProfileConfig(), 128)
        pol = build_policy("cq_fcfs", "S", 0.01, H=1)
        eng = run_case(scn, specs, pol, numeric=float, seed=w,
                       record_intervals=True)
        s = analyze(eng.w.storage.interval_log, float(B))
        s.update(trace=key, mode=mode, rho=rho, n_req=len(specs))
        cid = f"{key}|{mode}|B{B:g}" + (f"|rho{rho:g}" if rho else "")
        out[cid] = s
        print(f"{cid:32s} 瞬时Σq>B {s['instant_over_frac']:6.1%} | "
              f"瞬时贴满 {s['instant_touch_frac']:6.1%} | "
              f"Σq p50/p90/max {s['q_p50']:6.1f}{s['q_p90']:7.1f}{s['q_max']:8.1f} | "
              f"过载段 n={s['n_episodes']:5d} 中位 {s['ep_med_ms']:7.1f}ms "
              f"p90 {s['ep_p90_ms']:8.1f}ms 最长 {s['ep_max_ms']:9.1f}ms | "
              f"削掉 {s['excess_gb']:8.2f} GB")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
