"""三问深答 v2（20260921）补充冒烟：B=12（更强过载）、m8+B30（更强存储压力），
并给 B15/B50 补齐桶均 sat（aggregate_run 正确口径，与 E25.2 地图一致）。"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.getcwd())

import numpy as np

from sim.cq.config import ProfileConfig
from sim.cq.metrics import summarize
from sim.cq.profile import singleton_K
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.timeseries import aggregate_run, anchor_T
from sim.cq.types import RequestSpec
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                       MooncakeSource, build_policy,
                                       c_ref_of, default_scenario)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "cq", "eval", "e25_q2", "q3_smoke2.json")
MPC_BUDGET_S = 3600.0
FILES = {"toolagent": "toolagent_trace.jsonl"}
CELLS = [
    ("toolagent", 12.0, 0.6, 4, "更强过载"),
    ("toolagent", 30.0, 0.3, 8, "m8 更强存储压力"),
    ("toolagent", 15.0, 0.6, 4, "补桶均 sat"),
    ("toolagent", 50.0, 0.3, 8, "补桶均 sat"),
]
POL = [("cq_fcfs", ""), ("cq_edf", ""), ("cq_local", "T"), ("cq_mpc", "T")]


def sat_two_ways(eng, b):
    seg = [(float(t1) - float(t0), float(r)) for t0, t1, r, q
           in eng.w.storage.interval_log if float(t1) > float(t0)]
    tot = sum(w for w, _ in seg)
    inst = sum(w for w, r in seg if r >= 0.9 * b) / tot
    ts = aggregate_run(eng, anchor_T(float(eng.w.t)), cell="x", policy="x",
                       mode="OL")
    utils = [r["served_gb"] / r["width_s"] / b for r in ts["rows"]
             if r["width_s"] > 0 and r["capacity_gb"] > 0]
    buck = float(np.mean([u >= 0.9 for u in utils])) if utils else 0.0
    return inst, buck


def main():
    srcs = {}
    out = {}
    for key, B, rho, m, note in CELLS:
        if key not in srcs:
            srcs[key] = MooncakeSource(FILES[key], TRACE_DIR_DEFAULT)
        src = srcs[key]
        prof = ProfileConfig()
        hus = [(r.hit_tokens, r.u_tokens) for r in src.imp.rows]
        e_v = sum(F(prof.kappa_gb_per_token_layer) * h for h, _u in hus) / len(hus)
        ks = [singleton_K(RequestSpec(0, 0, h, u, "x", 1, 1), prof, F(80), F(200))
              for h, u in hus]
        lam = min(F(80) / e_v, F(m) / (sum(ks) / len(ks))) * F(str(rho))
        scn = default_scenario(B_gbps=B, m=m, alpha=F(4), limits=TRACE_WIDE_LIMITS)
        specs, d_sim = src.window_specs(9, lam, F(4), duration_cap=None,
                                        limits=TRACE_WIDE_LIMITS)
        c_ref = c_ref_of(specs, scn)
        cid = f"{key}|OL|B{B:g}|rho{rho:g}|m{m}|w9"
        res = {"note": note, "B": B, "rho": rho, "m": m}
        for pid, th in POL:
            if pid in ("cq_mpc", "cq_local"):
                pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1,
                                theta=th, c_ref=c_ref, budget_s=MPC_BUDGET_S)
            else:
                pol = build_policy(pid, th, c_ref, H=1)
            eng = run_case(scn, specs, pol, numeric=float, seed=9,
                           record_intervals=True)
            s = summarize(eng, scn, arrival_stop=d_sim)
            inst, buck = sat_two_ways(eng, B)
            comp = sum(wk.compute_s for wk in eng.w.workers.values())
            stall = sum(wk.stall_s for wk in eng.w.workers.values())
            span = float(eng.w.t)
            ttft = s.get("ttft_mean_lower") or s.get("ttft_mean_completed_only")
            slo_iv = s.get("slo_rate_interval") or [0.0, 0.0]
            res[pid + ("" if not th else "_" + th)] = {
                "ttft": ttft, "slo": slo_iv[0], "comp": comp / (m * span),
                "stall": stall / (m * span), "sat_instant": inst,
                "sat_bucket": buck}
            print(f"  {cid} {pid:9s}{th:2s} ttft={ttft:.3f} "
                  f"comp={comp/(m*span):.1%} stall={stall/(m*span):.1%} "
                  f"sat瞬={inst:.1%} sat桶={buck:.1%}")
        base = res["cq_edf"]["ttft"]
        for p in ("cq_local_T", "cq_mpc_T"):
            res[p]["gain_vs_edf_pct"] = 100 * (base - res[p]["ttft"]) / base
        print(f"{cid}: mpc vs edf {res['cq_mpc_T']['gain_vs_edf_pct']:+.1f}% | "
              f"local {res['cq_local_T']['gain_vs_edf_pct']:+.1f}%")
        out[cid] = res
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
