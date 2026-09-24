"""找 mpc vs local 差距最大的场景：FW Synthetic 参数扫描（20260921）。

机制前提（三问深答 §3.5 + 独立核查 §2.3）：两推演的分岔条件是
Σmin(q_i,B)>B（多条并发读流各自的申请都大于零头）——B 太低时单流
也被压平（两想象相同）、B 太高时谁都不挤（两想象相同），甜点在中段。
本扫描实证回答"B/m/α 设多少时 local−mpc 的配对差距最大"。

Phase A 筛 B：FW Syn θ=S m=4 α=4，B∈{40,60,80,120,160}，窗 {0,9}；
Phase B 加深：取 Phase A 差距最大的 B，跑 5 窗 + m=8 变体 + α=2 变体。
指标：SLO 与 TTFT 的逐窗配对差（local−mpc，正值=local 更差）。
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
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                       MooncakeSource, build_policy,
                                       c_ref_of, default_scenario)
from sim.experiments.e25_factor import fw_specs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "cq", "eval", "e25_q2", "div_scan.json")
MPC_BUDGET_S = 3600.0
SRC = MooncakeSource("synthetic_trace.jsonl", TRACE_DIR_DEFAULT)
PROF = ProfileConfig()


def run_cell(B, m, alpha, windows, tag):
    """一个 (B, m, α) 格点 × 窗口 × {edf, local-S, mpc-S}，返回逐窗配对差。"""
    out = {"B": B, "m": m, "alpha": alpha, "windows": windows, "per_window": {}}
    diffs_slo, diffs_ttft = [], []
    for w in windows:
        specs = fw_specs(SRC, w, F(alpha), PROF, 128)
        if len(specs) < 16:
            continue
        scn = default_scenario(B_gbps=B, m=m, alpha=F(alpha),
                               limits=TRACE_WIDE_LIMITS)
        c_ref = c_ref_of(specs, scn)
        row = {}
        for pid in ("cq_edf", "cq_local", "cq_mpc"):
            if pid == "cq_edf":
                pol = build_policy(pid, "S", c_ref, H=1)
            else:
                pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1,
                                theta="S", c_ref=c_ref, budget_s=MPC_BUDGET_S)
            eng = run_case(scn, specs, pol, numeric=float, seed=w,
                           record_intervals=False)
            s = summarize(eng, scn, arrival_stop=None)
            ttft = s.get("ttft_mean_lower") or s.get("ttft_mean_completed_only")
            row[pid] = {"slo": (s.get("slo_rate_interval") or [0.0])[0],
                        "ttft": ttft}
        d_slo = row["cq_local"]["slo"] - row["cq_mpc"]["slo"]
        d_ttft = (row["cq_local"]["ttft"] - row["cq_mpc"]["ttft"]
                  ) / max(row["cq_local"]["ttft"], 1e-9)
        out["per_window"][w] = {"cells": row, "d_slo_pp": 100 * d_slo,
                                "d_ttft_pct": 100 * d_ttft}
        diffs_slo.append(100 * d_slo)
        diffs_ttft.append(100 * d_ttft)
        print(f"  [{tag}] B{B:g} m{m} α{alpha:g} w{w}: "
              f"edf {row['cq_edf']['slo']:.3f} local {row['cq_local']['slo']:.3f} "
              f"mpc {row['cq_mpc']['slo']:.3f} → local−mpc {100*d_slo:+.2f}pp "
              f"(TTFT {100*d_ttft:+.2f}%)")
    out["median_d_slo_pp"] = (float(np.median(diffs_slo)) if diffs_slo else 0.0)
    out["max_d_slo_pp"] = max(diffs_slo) if diffs_slo else 0.0
    out["median_d_ttft_pct"] = (float(np.median(diffs_ttft))
                                if diffs_ttft else 0.0)
    print(f"[{tag}] B{B:g} m{m} α{alpha:g}: 配对中位 {out['median_d_slo_pp']:+.2f}pp "
          f"最大 {out['max_d_slo_pp']:+.2f}pp（SLO，local−mpc，正=local 差）")
    return out


def main():
    results = {}

    # Phase A：筛 B（窗 {0,9}——0 是已知分化窗、9 是常规窗）
    print("== Phase A：筛 B（FW Syn θ=S m=4 α=4，窗 {0,9}）==")
    phaseA = {}
    for B in (40.0, 60.0, 80.0, 120.0, 160.0):
        phaseA[B] = run_cell(B, 4, 4, [0, 9], "A")
    results["phaseA"] = phaseA

    # Phase B：差距最大的 B → 5 窗 + m8 + α2 变体
    best_B = max(phaseA, key=lambda b: abs(phaseA[b]["median_d_slo_pp"]))
    print(f"== Phase B：最优 B={best_B:g} 的 5 窗与变体 ==")
    pb = {}
    pb["base5"] = run_cell(best_B, 4, 4, [0, 4, 9, 14, 19], "B")
    pb["m8"] = run_cell(best_B, 8, 4, [0, 4, 9, 14, 19], "B")
    pb["alpha2"] = run_cell(best_B, 4, 2, [0, 4, 9, 14, 19], "B")
    results["phaseB_bestB"] = best_B
    results["phaseB"] = pb

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
