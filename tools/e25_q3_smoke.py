"""三问深答（20260921）Q3：候选"存储间歇过载"格点的冒烟验证。

从 B=80 需求分布反推出候选 B（Tool B15ρ0.6 / Tool B25ρ0.9 / Conv B10ρ0.9 /
Syn B20ρ0.9 对照），每格跑 fcfs/edf/local-T/mpc-T（w9 单窗，理想上限档），
核对：sat（桶均利用率≥90% 占比）、STALL/口径2、comp、队列、mpc vs edf/local
的 TTFT 配对差。单窗冒烟，不做统计声明。
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.getcwd())

from sim.cq.config import ProfileConfig
from sim.cq.metrics import summarize
from sim.cq.profile import singleton_K
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.types import RequestSpec
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                       MooncakeSource, build_policy,
                                       c_ref_of, default_scenario)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "cq", "eval", "e25_q2", "q3_smoke.json")
MPC_BUDGET_S = 3600.0
FILES = {"conversation": "conversation_trace.jsonl",
         "toolagent": "toolagent_trace.jsonl",
         "synthetic": "synthetic_trace.jsonl"}

# (trace, B, rho, m, mode, 说明)；rho=None 表示 FW（128 条 t=0）
CELLS = [
    ("toolagent", 15.0, 0.6, 4, "OL", "推荐：B 降到需求 p90~p99 区间"),
    ("toolagent", 25.0, 0.9, 4, "OL", "推荐：中带宽+高负载"),
    ("toolagent", 20.0, 0.6, 4, "OL", "对照：现有 B20 格点（sat 5.5%）"),
    ("conversation", 10.0, 0.9, 4, "OL", "推荐：Conv 需求小，需更低 B"),
    ("synthetic", 20.0, 0.9, 4, "OL", "对照：预测存储过载但调度仍无对象"),
    ("toolagent", 50.0, 0.3, 8, "OL", "推荐：m=8 计算欠载 + 8 路并发读放大需求"),
    ("synthetic", 20.0, None, 4, "FW", "预测：深队列+大读取+低 B 放大 mpc-local 分化"),
    ("synthetic", 80.0, None, 4, "FW", "对照：现有 FW B80 格点（local -6.3pp SLO）"),
]
POL_OL = [("cq_fcfs", ""), ("cq_edf", ""), ("cq_local", "T"), ("cq_mpc", "T")]
POL_FW = [("cq_fcfs", ""), ("cq_edf", ""), ("cq_local", "S"), ("cq_mpc", "S")]


def ts_metrics(eng, b, k=200):
    rows = []
    for t0, t1, rate, q in eng.w.storage.interval_log:
        w = float(t1) - float(t0)
        if w > 0:
            rows.append((w, float(rate), float(q)))
    tot = sum(w for w, _, _ in rows)
    # 桶均口径（E25.2 地图同款）：[0,T_end] 等切 200 桶，桶均速率/B≥0.9
    t_end = max(float(t1) for t0, t1, r, q in eng.w.storage.interval_log)
    buckets = [0.0] * k
    for t0, t1, r, q in eng.w.storage.interval_log:
        w0, w1 = float(t0), float(t1)
        i0, i1 = int(w0 / t_end * k), min(k, int(w1 / t_end * k + 1e-9))
        for i in range(i0, i1):
            a = max(w0, i * t_end / k)
            b_ = min(w1, (i + 1) * t_end / k)
            if b_ > a:
                buckets[i] += r * (b_ - a)
    sat_bucket = sum(1 for x in buckets if x >= 0.9 * b * (t_end / k)) / k
    return {
        "sat_instant": sum(w for w, r, _ in rows if r >= 0.9 * b) / tot,
        "sat_bucket": sat_bucket,
        "rate_mean": sum(r * w for w, r, _ in rows) / tot,
    }


def main():
    srcs = {}
    out = {}
    for key, B, rho, m, mode, note in CELLS:
        if key not in srcs:
            srcs[key] = MooncakeSource(FILES[key], TRACE_DIR_DEFAULT)
        src = srcs[key]
        scn = default_scenario(B_gbps=B, m=m, alpha=F(4),
                               limits=TRACE_WIDE_LIMITS)
        prof = ProfileConfig()
        if mode == "FW":
            from sim.experiments.e25_factor import fw_specs
            specs = fw_specs(src, 9, F(4), prof, 128)
            d_sim = None
            pols = POL_FW
        else:
            # λ0 锚随 m 变化（与 e25_regime.lam0_m 同式）
            hus = [(r.hit_tokens, r.u_tokens) for r in src.imp.rows]
            e_v = sum(F(prof.kappa_gb_per_token_layer) * h
                      for h, _u in hus) / len(hus)
            ks = [singleton_K(RequestSpec(0, 0, h, u, "x", 1, 1), prof,
                              F(80), F(200)) for h, u in hus]
            lam0_m = min(F(80) / e_v, F(m) / (sum(ks) / len(ks)))
            lam = lam0_m * F(str(rho))
            specs, d_sim = src.window_specs(9, lam, F(4), duration_cap=None,
                                            limits=TRACE_WIDE_LIMITS)
            pols = POL_OL
        c_ref = c_ref_of(specs, scn)
        cid = (f"{key}|{mode}|B{B:g}|" + (f"rho{rho:g}|" if rho is not None else "")
               + f"m{m}|w9")
        res = {"note": note, "n_req": len(specs), "B": B, "rho": rho,
               "m": m, "mode": mode}
        for pid, th in pols:
            if pid in ("cq_mpc", "cq_local"):
                pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1,
                                theta=th, c_ref=c_ref, budget_s=MPC_BUDGET_S)
            else:
                pol = build_policy(pid, th, c_ref, H=1)
            eng = run_case(scn, specs, pol, numeric=float, seed=9,
                           record_intervals=True)
            s = summarize(eng, scn, arrival_stop=d_sim)
            mt = ts_metrics(eng, B)
            comp = sum(wk.compute_s for wk in eng.w.workers.values())
            stall = sum(wk.stall_s for wk in eng.w.workers.values())
            span = float(eng.w.t)
            denom = m * span
            ttft = s.get("ttft_mean_lower") or s.get("ttft_mean_completed_only")
            slo_iv = s.get("slo_rate_interval") or [0.0, 0.0]
            res[pid + ("" if not th else "_" + th)] = {
                "ttft": ttft, "slo": slo_iv[0],
                "comp": comp / denom, "stall": stall / denom,
                "ku2": comp / (comp + stall) if comp + stall > 0 else 1.0,
                "sat_instant": mt["sat_instant"],
                "sat_bucket": mt["sat_bucket"],
                "q_peak": s.get("queue_peak"),
            }
            print(f"  {cid} {pid:9s}{th:2s} ttft={ttft:.3f} "
                  f"slo={slo_iv[0]:.3f} comp={comp/denom:.1%} "
                  f"stall={stall/denom:.1%} sat瞬={mt['sat_instant']:.1%} "
                  f"sat桶={mt['sat_bucket']:.1%}")
        out[cid] = res
        tag = "T" if mode == "OL" else "S"
        base = res["cq_edf"]["ttft"]
        for p in (f"cq_local_{tag}", f"cq_mpc_{tag}"):
            if base:
                res[p]["gain_vs_edf_pct"] = 100 * (base - res[p]["ttft"]) / base
        if mode == "OL":
            print(f"{cid}: mpc vs edf TTFT {res['cq_mpc_T']['gain_vs_edf_pct']:+.1f}% | "
                  f"local vs edf {res['cq_local_T']['gain_vs_edf_pct']:+.1f}% | "
                  f"mpc-local 差 {100*(res['cq_local_T']['ttft']-res['cq_mpc_T']['ttft'])/res['cq_local_T']['ttft']:+.1f}%")
        else:
            print(f"{cid}: SLO edf={res['cq_edf']['slo']:.3f} "
                  f"local={res['cq_local_S']['slo']:.3f} "
                  f"mpc={res['cq_mpc_S']['slo']:.3f} | "
                  f"local-mpc 差 {100*(res['cq_local_S']['slo']-res['cq_mpc_S']['slo']):+.1f}pp")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
