"""E25 完备性补跑：基线格点（FW+OL）扩展到全部 20 个窗口，并对比
5 窗中位数 vs 20 窗中位数（检验窗口选择偏差）。

复用 run_cell 与既有 progress 续跑语义；只补基线（factor_line=baseline），
因子线不扩展。产出：results/cq/eval/e25/ 记录追加 + 控制台对比表。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.getcwd())

from sim.experiments import e25_factor as e5
from sim.cq.trace import MOONCAKE_FILES
from sim.experiments.cq_common import MooncakeSource, print_progress
from sim.cq.trace import TRACE_LABEL

D = "results/cq/eval/e25"


def main():
    records, progress = e5.load_existing(D)
    srcs = {f: MooncakeSource(f, e5.TRACE_DIR_DEFAULT) for f, _n, _m, _s in MOONCAKE_FILES}
    done0 = len(progress)
    for fname, _n, _m, _s in MOONCAKE_FILES:
        src = srcs[fname]
        for w in range(20):
            for mode, rho in (("FW", None), ("OL", 0.6)):
                cell = {"factor_line": "baseline", "mode": mode, "file": fname,
                        "rho": rho, "B": 80.0, "alpha": 4, "cap": "C",
                        "profile": "default", "window": w, "_stage": "eval"}
                if e5.cell_id(cell) in progress:
                    continue
                out = e5.run_cell(src, cell, D, records, progress,
                                  n_req=128, duration_cap=None)
                print_progress(f"补跑 {e5.cell_id(cell)} -> {len(out)} 条")
    print_progress(f"补跑完成：{done0} -> {len(progress)} 格点")

    # ---- 对比：5 窗 vs 20 窗（OL 基线，关键口径） ----
    def med(vs):
        vs = [v for v in vs if v is not None]
        return float(np.median(vs)) if vs else None

    base = [r for r in records if r["factor_line"] == "baseline" and r["mode"] == "OL"]
    W5 = [0, 4, 9, 14, 19]
    print("\n=== OL 基线：5 窗 vs 20 窗 中位数对比 ===")
    print(f"{'trace':16s} {'指标':14s} {'5窗':>10s} {'20窗':>10s} {'漂移':>8s}")
    for fname, _n, _m, _s in MOONCAKE_FILES:
        for label, pid, th, key in [
                ("fcfs SLO", "cq_fcfs", "S", lambda r: r["slo_success"]/r["n_cohort"]),
                ("edf SLO", "cq_edf", "S", lambda r: r["slo_success"]/r["n_cohort"]),
                ("mpc SLO", "cq_mpc", "S", lambda r: r["slo_success"]/r["n_cohort"]),
                ("mpc-vs-edf TTFT%", "pair", "T", None)]:
            if key is not None:
                v5 = med([key(r) for r in base if r["file"] == fname and r["policy"] == pid
                          and r["theta"] == th and r["window"] in W5])
                v20 = med([key(r) for r in base if r["file"] == fname and r["policy"] == pid
                           and r["theta"] == th])
                print(f"{TRACE_LABEL[fname]:16s} {label:14s} {v5:10.3f} {v20:10.3f} "
                      f"{100*(v20-v5)/max(abs(v5),1e-9):+7.1f}%")
            else:
                idx = defaultdict(dict)
                for r in base:
                    if r["file"] == fname and r["theta"] == "T":
                        idx[r["window"]][r["policy"]] = r
                def gains(ws):
                    gs = []
                    for kk, pols in idx.items():
                        if kk in ws and "cq_edf" in pols and "cq_mpc" in pols:
                            b = pols["cq_edf"]["ttft_mean_lower"]
                            k2 = pols["cq_mpc"]["ttft_mean_lower"]
                            gs.append(100*(b-k2)/b)
                    return med(gs)
                g5, g20 = gains(W5), gains(set(idx.keys()))
                print(f"{TRACE_LABEL[fname]:16s} {label:14s} {g5:9.1f}% {g20:9.1f}% "
                      f"{g20-g5:+7.1f}pp")


if __name__ == "__main__":
    main()
