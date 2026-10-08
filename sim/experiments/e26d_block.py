"""E26d：块状到达（用户 20261008 指定）——每轮 32 条 A 同刻到达、64 条 B
同刻到达（A 在前），重复两轮（192 条）。对比交错序 D2：FCFS 的"天然错峰"
来自 (A,B,B) 交错；块状序下 FCFS 将先派整块 32A 直面洪水。

臂（5）：fcfs / edf / localS_v2.1 / mpcS_v2.1 / mpcT_v2.1（θ=T 首次上 v2.1）。
格点：D3_b1.0 / D3_b0.75（轮间隔，与 D2 对应可比）。
输出 results/cq/eval/e26d/。CRN：同格点全臂同一份流。
"""
from __future__ import annotations

import gzip
import json
import os
import time
from fractions import Fraction as F

import numpy as np

from sim.cq.policies import make_edf, make_fcfs
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.metrics import summarize
from sim.experiments import e26_ab as e26
from sim.experiments.cq_common import out_dir, print_progress, save_json

MPC_BUDGET_S = 3600.0
MPC_EVENTS = 20000
ARMS = [("fcfs", "cq_fcfs", None, None, None),
        ("edf", "cq_edf", None, None, None),
        ("localS_v2.1", "cq_local", "S", "bw_edf", True),
        ("mpcS_v2.1", "cq_mpc", "S", "bw_edf", True),
        ("mpcT_v2.1", "cq_mpc", "T", "bw_edf", True)]


def d3_specs(tb: float, alpha: int, rounds: int = 2):
    """块状到达：每轮 32 条 A 同刻 + 64 条 B 同刻（A 的 rid 在前 → FCFS 先派 A 块）。"""
    classes, arrivals = [], []
    for r in range(rounds):
        arrivals.extend([r * tb] * 32)
        classes.extend(["A"] * 32)
        arrivals.extend([r * tb] * 64)
        classes.extend(["B"] * 64)
    specs = [e26.RequestSpec(i, float(arrivals[i]),
                             *(e26.A_CLASS if c == "A" else e26.B_CLASS), c,
                             F(1), F(1)) for i, c in enumerate(classes)]
    T0 = e26.make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    return [e26.RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens,
                            s.class_id, T0[s.rid],
                            float(s.arrival_s) + alpha * float(T0[s.rid]))
            for s in specs]


def run_cell(cell: str, d: str):
    tb = float(cell.split("_b")[1])
    specs = d3_specs(tb, 4)
    scn = e26.e26_scenario(4)
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    anchor = None
    records = []
    for label, pid, theta, base, composer in ARMS:
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
        s["idle_frac"] = s["device_idle_s"] / s["device_total_mh_s"]
        h = pol.health() if hasattr(pol, "health") else {}
        s.update({"policy": label, "pid": pid, "theta": theta,
                  "mpc_base": base, "composer": composer, "cell": cell,
                  "alpha": 4, "n_total": 192, "makespan_s": float(eng.w.t),
                  "wall_s": wall, "n_decides": h.get("n_decides"),
                  "n_searched": h.get("n_searched"),
                  "n_scored": h.get("n_scored"),
                  "n_fallback": h.get("n_fallback"),
                  "n_deviate": h.get("n_deviate"),
                  "n_tie": h.get("n_tie")})
        if anchor is None:
            anchor = float(eng.w.t) * 1.1
        g = e26.gantt_collect(eng, anchor)
        with gzip.open(os.path.join(d, f"{cell}/{label}.gantt.json.gz"),
                       "wt") as f:
            json.dump({"meta": {"cell": cell, "policy": label, "theta": None,
                                "T_anchor_s": anchor, "K": 200,
                                "m_workers": 32}, "buckets": g},
                      f, separators=(",", ":"))
        arr = np.asarray([(float(a), float(b), float(r_), float(q))
                          for (a, b, r_, q) in eng.w.storage.interval_log])
        np.savez_compressed(
            os.path.join(d, f"{cell}/{label}.intervals.npz"), intervals=arr)
        records.append(s)
        print_progress(
            f"E26d {cell:10s} {label:12s} wall={wall:.0f}s "
            f"SLO={s['slo_success']}/{s['n_cohort']} "
            f"A={100*s['A']['slo_rate']:.0f}% B={100*s['B']['slo_rate']:.0f}% "
            f"ttft={s['ttft_mean_lower']:.3f} concA {s['a_conc_p50']:.0f}/"
            f"{s['a_conc_p90']:.0f}/{s['a_conc_max']:.0f} "
            f"dev={s['n_deviate']} tie={s['n_tie']}")
    return records


def main(seeds, procs=None, duration=None, stage="eval", **kw):
    d = out_dir("eval", "e26d")
    only = [x for x in os.environ.get("E26D_ONLY", "").split(",") if x]
    shard = os.environ.get("E26D_SHARD", "main")
    cells = only or ["D3_b1.0", "D3_b0.75"]
    part = os.path.join(d, f"records_part_{shard}.json")
    records = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else []
    done = {(r["cell"], r["policy"]) for r in records}
    for cell in cells:
        if all((cell, a[0]) in done for a in ARMS):
            continue
        os.makedirs(os.path.join(d, cell), exist_ok=True)
        recs = run_cell(cell, d)
        records.extend(recs)
        save_json(part, records)
    print_progress(f"E26d done -> {part} ({len(records)} 记录)")
    return {"n_records": len(records)}


if __name__ == "__main__":
    main([0])
