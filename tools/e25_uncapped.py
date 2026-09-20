"""E25.3 事件上限消融（20260920，应用户提问）：

"如果把 2 万事件上限取消，对 FW 和 OL 各有什么影响？"

设计：把 MPC 的 max_forecast_events 从 2 万放开到 1000 万（等效取消；
CPU 预算仍 3600s 不构成约束），只重跑 MPC（θ 见下），与既有 2 万预算
记录（E25.1 主矩阵，同种子同配置）逐窗配对对比：
- FW 基线（B=80, α=4, m=4, 128 条封闭工作集）× 3 trace × 5 窗
  × θ∈{M,S,T}——深队列场景，预算已知在 n≈128 起部分绑定；
- OL ρ∈{0.9, 1.1}（B=80, α=4）× 3 trace × 5 窗 × θ∈{S,T}——
  在线场景里队列最深的两档（q_med 20–74），预算是否绑定未实测过；
  ρ=0.6 已有"2 万→1000 万逐位不变"的验证（§5.7），不重跑。

输出：results/cq/eval/e25_uncapped/records.json + 对比表
（ΔTTFT/ΔSLO/n_scored/ctrl_p50 的 2万 vs 1000万）。

用法：python tools/e25_uncapped.py run|compare|all
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.getcwd())

from sim.cq.metrics import summarize
from sim.cq.profile import make_T0
from sim.cq.simrun import run_case
from sim.cq.trace import MOONCAKE_FILES
from sim.cq.types import RequestSpec
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                       MooncakeSource, c_ref_of,
                                       default_scenario)
from sim.experiments.e25_factor import PROFILES, fw_specs

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "results", "cq", "eval", "e25_uncapped")
E25 = os.path.join(REPO, "results", "cq", "eval", "e25")
W5 = [0, 4, 9, 14, 19]
MAX_EVENTS = 10_000_000          # 等效取消 2 万上限
BUDGET_S = 3600.0

FW_JOBS = [dict(mode="FW", rho=None, thetas=("M", "S", "T"))]
OL_JOBS = [dict(mode="OL", rho=0.9, thetas=("S", "T")),
           dict(mode="OL", rho=1.1, thetas=("S", "T"))]


def run():
    from sim.cq.search import MPCPolicy
    os.makedirs(ROOT, exist_ok=True)
    recs_path = os.path.join(ROOT, "records.json")
    prog_path = os.path.join(ROOT, "progress.json")
    recs = (json.load(open(recs_path, encoding="utf-8"))
            if os.path.exists(recs_path) else [])
    prog = (json.load(open(prog_path, encoding="utf-8"))
            if os.path.exists(prog_path) else {})
    srcs = {}
    for job in FW_JOBS + OL_JOBS:
        for fname, _n, _m, _s in MOONCAKE_FILES:
            src = srcs.get(fname) or MooncakeSource(fname, TRACE_DIR_DEFAULT)
            srcs[fname] = src
            for w in W5:
                for th in job["thetas"]:
                    key = f"{fname}|{job['mode']}|rho{job['rho']}|w{w}|{th}"
                    if key in prog:
                        continue
                    if job["mode"] == "FW":
                        specs = fw_specs(src, w, F(4), PROFILES["default"],
                                         128)
                        d_sim = None
                    else:
                        lam = src.lam0 * F(str(job["rho"]))
                        specs, d_sim = src.window_specs(
                            w, lam, F(4), duration_cap=None,
                            limits=TRACE_WIDE_LIMITS)
                    if len(specs) < 16:
                        prog[key] = "skip"
                        continue
                    scn = default_scenario(B_gbps=80.0, alpha=F(4),
                                           limits=TRACE_WIDE_LIMITS)
                    c_ref = c_ref_of(specs, scn)
                    pol = MPCPolicy(pid="cq_mpc", H=1, theta=th,
                                    c_ref=c_ref, budget_s=BUDGET_S,
                                    max_events=MAX_EVENTS)
                    eng = run_case(scn, specs, pol, numeric=float, seed=w)
                    s = summarize(eng, scn,
                                  arrival_stop=(d_sim if d_sim else None))
                    ctrl = sorted(d["ctrl_s"] for d in eng.decisions) or [0.0]
                    s.update({"policy": "cq_mpc", "theta": th, "file": fname,
                              "mode": job["mode"], "rho": job["rho"],
                              "window": w, "n_req": len(specs),
                              "ctrl_p50_ms": 1000.0 * ctrl[len(ctrl) // 2],
                              "n_scored": getattr(pol, "n_scored", None),
                              "n_overrun": getattr(pol, "n_overrun", None),
                              "makespan": s.get("makespan_M_lower"),
                              "code_rev": "e253-uncapped-20260920"})
                    recs.append(s)
                    prog[key] = "done"
                    json.dump(recs, open(recs_path, "w", encoding="utf-8"),
                              ensure_ascii=False)
                    json.dump(prog, open(prog_path, "w", encoding="utf-8"),
                              ensure_ascii=False)
                    print(f"done {key} scored={s['n_scored']} "
                          f"P50={s['ctrl_p50_ms']:.0f}ms")
    print(f"E25.3 run 完成：{len(recs)} 条记录 -> {recs_path}")


def compare():
    new = json.load(open(os.path.join(ROOT, "records.json"), encoding="utf-8"))
    old = json.load(open(os.path.join(E25, "e25_records.json"),
                         encoding="utf-8"))
    out = []
    print(f"{'trace':12s} {'mode':3s} {'ρ':>4s} {'θ':2s} | "
          f"{'2万:TTFT':>8s} {'1000万':>8s} {'Δ':>7s} | "
          f"{'2万:SLO':>7s} {'1000万':>7s} | {'scored 2万/1000万':>16s} | "
          f"{'P50 2万→1000万':>14s}")
    for job in FW_JOBS + OL_JOBS:
        for fname, _n, _m, _s in MOONCAKE_FILES:
            for th in job["thetas"]:
                rows = []
                for w in W5:
                    o = next((r for r in old if r["file"] == fname
                              and r["mode"] == job["mode"]
                              and r.get("rho") == job["rho"]
                              and r["factor_line"] in ("baseline", "line_rho")
                              and r["window"] == w and r["policy"] == "cq_mpc"
                              and r.get("theta") == th), None)
                    n = next((r for r in new if r["file"] == fname
                              and r["mode"] == job["mode"]
                              and r.get("rho") == job["rho"]
                              and r["window"] == w and r.get("theta") == th),
                             None)
                    if o and n:
                        rows.append((o, n))
                if not rows:
                    continue
                med = lambda src, k: float(np.median([r[k] for r in src]))
                o_ttft = med([o for o, _ in rows], "ttft_mean_lower")
                n_ttft = med([n for _, n in rows], "ttft_mean_lower")
                o_slo = med([o for o, _ in rows], "slo_success") / \
                    med([o for o, _ in rows], "n_cohort")
                n_slo = med([n for _, n in rows], "slo_success") / \
                    med([n for _, n in rows], "n_cohort")
                o_sc = med([o for o, _ in rows], "n_scored")
                n_sc = med([n for _, n in rows], "n_scored")
                o_p50 = med([o for o, _ in rows], "ctrl_p50_ms")
                n_p50 = med([n for _, n in rows], "ctrl_p50_ms")
                ident = all(o["ttft_mean_lower"] == n["ttft_mean_lower"]
                            and o["slo_success"] == n["slo_success"]
                            for o, n in rows)
                d = (o_ttft - n_ttft) / o_ttft * 100
                print(f"{fname.split('_')[0]:12s} {job['mode']:3s} "
                      f"{job['rho'] if job['rho'] else '-':>4} {th:2s} | "
                      f"{o_ttft:8.3f} {n_ttft:8.3f} {d:+6.1f}% | "
                      f"{o_slo:7.3f} {n_slo:7.3f} | "
                      f"{o_sc:7.0f}/{n_sc:7.0f} | "
                      f"{o_p50:6.0f}→{n_p50:6.0f}ms "
                      f"{'逐位相同' if ident else ''}")
                out.append(dict(file=fname, mode=job["mode"], rho=job["rho"],
                                theta=th, ttft_20k=o_ttft, ttft_10m=n_ttft,
                                slo_20k=o_slo, slo_10m=n_slo,
                                scored_20k=o_sc, scored_10m=n_sc,
                                p50_20k=o_p50, p50_10m=n_p50,
                                identical=ident))
    json.dump(out, open(os.path.join(ROOT, "compare.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已保存 -> results/cq/eval/e25_uncapped/compare.json")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("run", "all"):
        run()
    if mode in ("compare", "all"):
        compare()


if __name__ == "__main__":
    main()
