"""三问深答补充（20260921）：桶均 sat 口径对账 + FW 5 窗 local/mpc 分化验证。

① 用与 e25_regime.analyze 完全相同的方法（aggregate_run 200 桶）重算
   Tool B20ρ0.6 的桶均 sat，对账 map.json 的 5.5%（我的手写分桶曾算出 0）；
② FW Syn/Conv × B∈{20,80} × 5 窗 {0,4,9,14,19} × {edf, local-S, mpc-S}，
   以 5 窗中位口径（与报告 §4.3 表同口径）回答"降 B 是否放大 local-mpc 分化"。
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.getcwd())

import numpy as np

from sim.cq.config import ProfileConfig
from sim.cq.metrics import summarize
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.timeseries import aggregate_run, anchor_T
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                       MooncakeSource, build_policy,
                                       c_ref_of, default_scenario)
from sim.experiments.e25_factor import fw_specs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "cq", "eval", "e25_q2", "fw5.json")
MPC_BUDGET_S = 3600.0
FILES = {"conversation": "conversation_trace.jsonl",
         "synthetic": "synthetic_trace.jsonl"}
W5 = [0, 4, 9, 14, 19]


def bucket_sat(eng, b):
    """与 e25_regime.analyze 同款：aggregate_run 200 桶，桶均速率/B≥0.9 占比。"""
    ts = aggregate_run(eng, anchor_T(float(eng.w.t)), cell="x", policy="x",
                       mode="OL")
    utils = [r["served_gb"] / r["width_s"] / b for r in ts["rows"]
             if r["width_s"] > 0 and r["capacity_gb"] > 0]
    return float(np.mean([u >= 0.9 for u in utils])) if utils else 0.0


def main():
    srcs = {}
    out = {}

    # ① 桶均 sat 对账（OL Tool B20ρ0.6，与 map.json 5.5% 比）
    src = MooncakeSource("toolagent_trace.jsonl", TRACE_DIR_DEFAULT)
    scn = default_scenario(B_gbps=20.0, m=4, alpha=F(4), limits=TRACE_WIDE_LIMITS)
    lam = src.lam0 * F("0.6")
    specs, d_sim = src.window_specs(9, lam, F(4), duration_cap=None,
                                    limits=TRACE_WIDE_LIMITS)
    pol = build_policy("cq_fcfs", "S", 0.01, H=1)
    eng = run_case(scn, specs, pol, numeric=float, seed=9, record_intervals=True)
    s = bucket_sat(eng, 20.0)
    out["bucket_sat_check"] = {"cell": "toolagent|OL|B20|rho0.6|w9|fcfs",
                               "sat": s, "map_ref": 0.055}
    print(f"① 桶均 sat 对账：Tool B20ρ0.6 w9 FCFS = {s:.1%}（map.json 参考 5.5%）")

    # ② FW 5 窗分化
    for key, fname in FILES.items():
        if key not in srcs:
            srcs[key] = MooncakeSource(fname, TRACE_DIR_DEFAULT)
        src = srcs[key]
        for B in (20.0, 80.0):
            scn = default_scenario(B_gbps=B, m=4, alpha=F(4),
                                   limits=TRACE_WIDE_LIMITS)
            cell = {}
            for w in W5:
                specs = fw_specs(src, w, F(4), ProfileConfig(), 128)
                if len(specs) < 16:
                    continue
                c_ref = c_ref_of(specs, scn)
                for pid in ("cq_edf", "cq_local", "cq_mpc"):
                    if pid == "cq_edf":
                        pol = build_policy(pid, "S", c_ref, H=1)
                    else:
                        pol = MPCPolicy(pid=pid, local=(pid == "cq_local"),
                                        H=1, theta="S", c_ref=c_ref,
                                        budget_s=MPC_BUDGET_S)
                    eng = run_case(scn, specs, pol, numeric=float, seed=w,
                                   record_intervals=False)
                    sm = summarize(eng, scn, arrival_stop=None)
                    slo_iv = sm.get("slo_rate_interval") or [0.0, 0.0]
                    cell.setdefault(pid, []).append(slo_iv[0])
            med = {p: float(np.median(v)) for p, v in cell.items()}
            cid = f"{key}|FW|B{B:g}"
            out[cid] = {"per_window": cell, "median": med,
                        "local_minus_mpc_pp": 100 * (med["cq_local"] - med["cq_mpc"])}
            print(f"② {cid}: SLO 5窗中位 edf={med['cq_edf']:.3f} "
                  f"local={med['cq_local']:.3f} mpc={med['cq_mpc']:.3f} | "
                  f"local−mpc = {100*(med['cq_local']-med['cq_mpc']):+.1f}pp | "
                  f"逐窗 local {[f'{x:.3f}' for x in cell['cq_local']]} vs "
                  f"mpc {[f'{x:.3f}' for x in cell['cq_mpc']]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
