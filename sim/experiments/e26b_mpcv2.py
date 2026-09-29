"""E26b：MPC 修正方案（设计文档 20260929）实施验证——240 规模重测。

目的：验证 M1（组合式候选）+M2（GuardedEDF-BW 延续/兜底）后 mpc 能否
实现带宽错峰（用户 20260930 指令：仅 240 任务规模；甘特图须呈错位形态，
不达预期则反思迭代）。

臂设计（口径消融，θ=S）：
  fcfs / edf                基线
  mpcS_old                  base=edf, composer=False（旧口径锚点）
  mpcS_v2                   base=bw_edf, composer=True（完整修正）
  mpcS_v2_baseOnly          仅 M2（量化兜底抬升）
  mpcS_v2_compOnly          仅 M1（量化组合器）
  localS_v2                 local 评分消融 + 完整修正（mpc-local 差距新口径）

格点：D2_b1.0_a4（全臂）、D2_b0.75_a4 / D2_b1.0_a8 / D1_r1.1_a4（核心臂）。
输出：results/cq/eval/e26b/（records 分片、逐 run 甘特/区间账本）。
并行：E26B_ONLY="D2_b1.0_a4,..." 过滤格点；E26B_SHARD 分片名。
"""
from __future__ import annotations

import gzip
import json
import os
import time
from typing import List

import numpy as np

from sim.cq.policies import make_edf, make_fcfs
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.metrics import summarize
from sim.experiments import e26_ab as e26
from sim.experiments.cq_common import (out_dir, print_progress,
                                       save_json)

# 用户指令：只在 240 任务规模测（20260930）
e26.SCALE_TOTAL = 240
e26.D2_ROUNDS = max(1, 240 // 96)

MPC_BUDGET_S = 3600.0
MPC_EVENTS = 20000


def arms(cell: str):
    full = [
        ("fcfs", "cq_fcfs", None, None, None),
        ("edf", "cq_edf", None, None, None),
        ("mpcS_old", "cq_mpc", "S", "edf", False),
        ("mpcS_v2", "cq_mpc", "S", "bw_edf", True),
        ("mpcS_v2_baseOnly", "cq_mpc", "S", "bw_edf", False),
        ("mpcS_v2_compOnly", "cq_mpc", "S", "edf", True),
        ("localS_v2", "cq_local", "S", "bw_edf", True),
    ]
    core = ["fcfs", "edf", "mpcS_old", "mpcS_v2"]
    return [a for a in full if a[0] in core] if cell != "D2_b1.0_a4" else full


def build_specs(cell: str):
    if cell.startswith("D2_b"):
        tb = float(cell.split("_b")[1].split("_")[0])
        return e26.d2_specs(tb, int(cell.rsplit("_a", 1)[1]))
    tag, alpha = cell.rsplit("_a", 1)
    lvl = {"D1_r0.5": 89.0, "D1_r0.7": 125.0, "D1_r0.9": 160.0,
           "D1_r1.0": 178.0, "D1_r1.1": 196.0}[tag]
    idx = {"D1_r0.5": 0, "D1_r0.7": 1, "D1_r0.9": 2,
           "D1_r1.0": 3, "D1_r1.1": 4}[tag]
    return e26.d1_specs(lvl, e26.d1_seed(idx, 0), int(alpha))


def run_cell(cell: str, d: str):
    specs = build_specs(cell)
    scn = e26.e26_scenario(int(cell.rsplit("_a", 1)[1]))
    alpha = int(cell.rsplit("_a", 1)[1])
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    anchor = None
    records = []
    for label, pid, theta, base, composer in arms(cell):
        if pid == "cq_fcfs":
            pol = make_fcfs()
        elif pid == "cq_edf":
            pol = make_edf()
        else:
            pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1,
                            theta=theta, c_ref=c_ref, budget_s=MPC_BUDGET_S,
                            max_events=MPC_EVENTS, base=base,
                            composer=composer)
        t0 = time.time()
        eng = run_case(scn, specs, pol, numeric=float, seed=0,
                       record_intervals=True)
        wall = time.time() - t0
        s = summarize(eng, scn)
        s["slo_rate"] = s["slo_success"] / s["n_cohort"]
        s.update(e26.class_metrics(eng, specs))
        s.update(e26.concurrent_a_quantiles(eng))
        s.update(e26.bw_interval_stats(eng.w.storage.interval_log))
        comp, stall = s["device_compute_s"], s["device_stall_s"]
        s["io_wait_share2"] = comp / (comp + stall) if comp + stall > 0 else None
        h = pol.health() if hasattr(pol, "health") else {}
        s.update({"policy": label, "pid": pid, "theta": theta,
                  "mpc_base": base, "composer": composer,
                  "cell": cell, "alpha": alpha, "n_total": 240,
                  "makespan_s": float(eng.w.t), "wall_s": wall,
                  "n_decides": h.get("n_decides"),
                  "n_searched": h.get("n_searched"),
                  "n_scored": h.get("n_scored"),
                  "n_fallback": h.get("n_fallback"),
                  "n_deviate": h.get("n_deviate"),
                  "n_tie": h.get("n_tie")})
        # 甘特 + 区间账本（工具读文件名约定）
        if anchor is None:
            anchor = float(eng.w.t) * 1.1
        g = e26.gantt_collect(eng, anchor)
        with gzip.open(os.path.join(d, f"{cell}/{label}.gantt.json.gz"),
                       "wt") as f:
            json.dump({"meta": {"cell": cell, "policy": label, "theta": None,
                                "T_anchor_s": anchor, "K": 200,
                                "m_workers": 32}, "buckets": g},
                      f, separators=(",", ":"))
        arr = np.asarray([(float(a), float(b), float(r), float(q))
                          for (a, b, r, q) in eng.w.storage.interval_log])
        np.savez_compressed(
            os.path.join(d, f"{cell}/{label}.intervals.npz"), intervals=arr)
        records.append(s)
        print_progress(
            f"E26b {cell} {label:18s} wall={wall:.0f}s "
            f"SLO={s['slo_success']}/{s['n_cohort']} A={100*s['A']['slo_rate']:.0f}% "
            f"B={100*s['B']['slo_rate']:.0f}% concA {s['a_conc_p50']:.0f}/"
            f"{s['a_conc_p90']:.0f}/{s['a_conc_max']:.0f} "
            f"dev={s['n_deviate']} tie={s['n_tie']} fb={s['n_fallback']}")
    return records


def main(seeds, procs=None, duration=None, stage="smoke", **kw):
    d = out_dir("eval", "e26b")
    only = [x for x in os.environ.get("E26B_ONLY", "").split(",") if x]
    shard = os.environ.get("E26B_SHARD", "main")
    cells = (["D2_b1.0_a4", "D2_b0.75_a4", "D2_b1.0_a8", "D1_r1.1_a4"]
             if not only else only)
    part = os.path.join(d, f"records_part_{shard}.json")
    records = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else []
    done = {(r["cell"], r["policy"]) for r in records}
    for cell in cells:
        if all((cell, a[0]) in done for a in arms(cell)):
            continue
        os.makedirs(os.path.join(d, cell), exist_ok=True)
        recs = run_cell(cell, d)
        records.extend(recs)
        save_json(part, records)
        done.update((r["cell"], r["policy"]) for r in recs)
    print_progress(f"E26b done -> {part} ({len(records)} 记录)")
    return {"n_records": len(records)}


if __name__ == "__main__":
    main([0])
