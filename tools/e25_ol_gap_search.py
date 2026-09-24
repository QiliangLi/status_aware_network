"""OL 模式下搜索 mpc vs local 差距最大的格点（用户目标，20260921）。

只跑 OL（trace 真实节奏）；策略 {cq_local, cq_mpc}（+可选 edf）；逐窗配对
差距 d_ttft=(ttft_l−ttft_m)/ttft_l、d_slo=slo_m−slo_l（正=mpc 更好）。
结果渐进追加到 results/cq/eval/e25_q2/ol_gap_search.json，可多轮续跑。

用法：
  python tools/e25_ol_gap_search.py            # 跑 ROUNDS 里全部未完成格点
  python tools/e25_ol_gap_search.py --round 2  # 只跑某一轮
"""
from __future__ import annotations

import json
import os
import sys
import time
from fractions import Fraction as F

sys.path.insert(0, os.getcwd())

import numpy as np

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
OUT = os.path.join(REPO, "results", "cq", "eval", "e25_q2", "ol_gap_search.json")
MPC_BUDGET_S = 3600.0
W5 = [0, 4, 9, 14, 19]
FILES = {"con": "conversation_trace.jsonl",
         "too": "toolagent_trace.jsonl",
         "syn": "synthetic_trace.jsonl"}

# 每轮一组格点：dict(trace, B, rho, m, alpha, theta)
ROUNDS = {
    1: [  # 归档挖掘指向的区域：m=2 为核心
        dict(trace="too", B=80.0, rho=0.6, m=2, alpha=4, theta="S"),   # 已知 7.6%TTFT 的格点（复测+补 edf）
        dict(trace="too", B=80.0, rho=0.6, m=2, alpha=8, theta="S"),   # m2 × α8 叠加
        dict(trace="too", B=80.0, rho=0.9, m=2, alpha=4, theta="S"),   # m2 × 高负载
        dict(trace="too", B=60.0, rho=0.6, m=2, alpha=4, theta="S"),   # m2 × 甜带 B
        dict(trace="too", B=80.0, rho=0.6, m=2, alpha=4, theta="T"),   # TTFT 方向
        dict(trace="con", B=20.0, rho=0.3, m=2, alpha=4, theta="S"),   # Conv 最强格 × m2
        dict(trace="con", B=20.0, rho=0.3, m=4, alpha=8, theta="S"),   # Conv 最强格 × α8
        dict(trace="syn", B=80.0, rho=0.9, m=2, alpha=4, theta="S"),   # Syn × m2
    ],
}


def lam0_of(src: MooncakeSource, m: int) -> F:
    prof = ProfileConfig()
    hus = [(r.hit_tokens, r.u_tokens) for r in src.imp.rows]
    e_v = sum(F(prof.kappa_gb_per_token_layer) * h for h, _u in hus) / len(hus)
    ks = [singleton_K(RequestSpec(0, 0, h, u, "x", 1, 1), prof, F(80), F(200))
          for h, u in hus]
    return min(F(80) / e_v, F(m) / (sum(ks) / len(ks)))


def run_cell(cell, srcs):
    key = f"{cell['trace']}|OL|B{cell['B']:g}|rho{cell['rho']:g}|m{cell['m']}|a{cell['alpha']:g}|th{cell['theta']}"
    tr = cell["trace"]
    if tr not in srcs:
        srcs[tr] = MooncakeSource(FILES[tr], TRACE_DIR_DEFAULT)
    src = srcs[tr]
    scn = default_scenario(B_gbps=cell["B"], m=cell["m"], alpha=F(cell["alpha"]),
                           limits=TRACE_WIDE_LIMITS)
    lam = lam0_of(src, cell["m"]) * F(str(cell["rho"]))
    res = {"cell": cell, "per_window": {}, "t_wall_s": 0.0}
    t0 = time.time()
    d_ttft, d_slo = [], []
    for w in cell.get("windows", W5):
        specs, d_sim = src.window_specs(w, lam, F(cell["alpha"]),
                                        duration_cap=None,
                                        limits=TRACE_WIDE_LIMITS)
        if len(specs) < 16:
            continue
        c_ref = c_ref_of(specs, scn)
        row = {}
        for pid in ("cq_edf", "cq_local", "cq_mpc"):
            if pid == "cq_edf" and w != W5[2]:      # edf 只跑 w9 做上下文
                continue
            if pid == "cq_edf":
                pol = build_policy(pid, cell["theta"], c_ref, H=1)
            else:
                pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1,
                                theta=cell["theta"], c_ref=c_ref,
                                budget_s=MPC_BUDGET_S)
            eng = run_case(scn, specs, pol, numeric=float, seed=w,
                           record_intervals=False)
            s = summarize(eng, scn, arrival_stop=d_sim)
            ttft = s.get("ttft_mean_lower") or s.get("ttft_mean_completed_only")
            row[pid] = {"ttft": ttft,
                        "slo": (s.get("slo_rate_interval") or [0.0, 0.0])[0]}
        dt = 100 * (row["cq_local"]["ttft"] - row["cq_mpc"]["ttft"]
                    ) / max(row["cq_local"]["ttft"], 1e-9)
        ds = 100 * (row["cq_mpc"]["slo"] - row["cq_local"]["slo"])
        res["per_window"][w] = {"row": row, "d_ttft_pct": dt, "d_slo_pp": ds}
        d_ttft.append(dt)
        d_slo.append(ds)
        print(f"  {key} w{w}: local ttft={row['cq_local']['ttft']:.3f} "
              f"mpc={row['cq_mpc']['ttft']:.3f} → {dt:+.2f}% | "
              f"local slo={row['cq_local']['slo']:.3f} "
              f"mpc={row['cq_mpc']['slo']:.3f} → {ds:+.2f}pp", flush=True)
    res["median_d_ttft_pct"] = float(np.median(d_ttft)) if d_ttft else 0.0
    res["median_d_slo_pp"] = float(np.median(d_slo)) if d_slo else 0.0
    res["max_abs_d_ttft_pct"] = max((abs(x) for x in d_ttft), default=0.0)
    res["max_abs_d_slo_pp"] = max((abs(x) for x in d_slo), default=0.0)
    res["t_wall_s"] = round(time.time() - t0, 1)
    print(f"[cell] {key}: TTFT中位{res['median_d_ttft_pct']:+.2f}% "
          f"SLO中位{res['median_d_slo_pp']:+.2f}pp | "
          f"单窗最大 |TTFT|{res['max_abs_d_ttft_pct']:.2f}% "
          f"|SLO|{res['max_abs_d_slo_pp']:.2f}pp ({res['t_wall_s']}s)", flush=True)
    return key, res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None)
    ap.add_argument("--out", default=None, help="输出 JSON（默认 ol_gap_search.json）")
    ap.add_argument("--cells", default=None,
                    help="JSON 列表覆盖本轮格点（用于并行 worker）")
    args = ap.parse_args()
    out_path = args.out or OUT
    rounds = ROUNDS
    if args.cells:
        rounds = {args.round or 99: json.loads(args.cells)}
    db = json.load(open(out_path)) if os.path.exists(out_path) else {}
    srcs = {}
    for rnd, cells in sorted(rounds.items()):
        if args.round is not None and rnd != args.round:
            continue
        print(f"== Round {rnd}: {len(cells)} 格点 ==", flush=True)
        for cell in cells:
            key = (f"{cell['trace']}|OL|B{cell['B']:g}|rho{cell['rho']:g}"
                   f"|m{cell['m']}|a{cell['alpha']:g}|th{cell['theta']}")
            if key in db:
                print(f"  跳过已完成: {key}")
                continue
            k, res = run_cell(cell, srcs)
            db[k] = res
            json.dump(db, open(out_path, "w"), ensure_ascii=False, indent=1)
    # 汇总
    print("\n== 全部格点汇总（按 |TTFT中位|+|SLO中位| 排序）==")
    rank = sorted(db.items(),
                  key=lambda kv: -(abs(kv[1]["median_d_ttft_pct"])
                                   + abs(kv[1]["median_d_slo_pp"])))
    for k, v in rank:
        print(f"{k:44s} TTFT中位{v['median_d_ttft_pct']:+7.2f}% "
              f"SLO中位{v['median_d_slo_pp']:+6.2f}pp | "
              f"最大 {v['max_abs_d_ttft_pct']:6.2f}%/{v['max_abs_d_slo_pp']:5.2f}pp")


if __name__ == "__main__":
    main()
