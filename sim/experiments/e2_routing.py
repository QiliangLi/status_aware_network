"""E2：存储热点下 KV-aware routing 是否放大热点；storage-aware routing 能否打破。

拓扑：4 workers / 2 backends；A(热点)只在 S_A，B 双方都有，C 在 S_A，D 在 S_B；
S_A 背景 80 GB/s (HOT)，S_B 背景 10 GB/s (IDLE)。
"""
from __future__ import annotations

from ..config import GpuConfig, RunSpec, StorageConfig, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
import os

STORAGES = (
    StorageConfig(bg_schedule=stable(80.0)),
    StorageConfig(bg_schedule=stable(10.0)),
)
WORKER_BACKEND = (0, 0, 1, 1)
CLASS_BACKENDS = (("A", (0,)), ("B", (0, 1)), ("C", (0,)), ("D", (1,)))
POLICIES = ["rr", "load", "kv", "p3", "p4"]
LABELS = {"rr": "RoundRobin", "load": "Load-aware", "kv": "KV-aware", "p3": "KV+Storage(P3)", "p4": "Oracle"}
GPU_BG = 0.2


def build_e2(seeds, duration=400.0):
    jobs = []
    for pol in POLICIES:
        for seed in seeds:
            spec = RunSpec(
                exp="e2", policy=pol, seed=seed, duration=duration, warmup=20.0, margin=60.0,
                storages=STORAGES, worker_backend=WORKER_BACKEND, class_backends=CLASS_BACKENDS,
                gpu=GpuConfig(bg_schedule=stable(GPU_BG)),
                collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]),
                out_dir=os.path.join(RESULTS_DIR, "e2") if seed == seeds[0] else None,
            )
            jobs.append(({"policy": pol, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    plt = setup_matplotlib()
    agg = df.groupby("policy").agg(
        goodput_med=("goodput", "median"),
        goodput_lo=("goodput", lambda s: np.percentile(s, 25)),
        goodput_hi=("goodput", lambda s: np.percentile(s, 75)),
        p95_med=("ttft_p95", "median"),
        p95_lo=("ttft_p95", lambda s: np.percentile(s, 25)),
        p95_hi=("ttft_p95", lambda s: np.percentile(s, 75)),
        misroute=("misroute_rate", "median"),
        sa_qmax=("s0_q_max", "median"),
    ).reindex(POLICIES)
    print("\n[E2] policy summary (median over seeds):")
    print(agg.round(3).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), constrained_layout=True)
    ax0, ax1, ax2, axt = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    x = np.arange(len(POLICIES))
    names = [LABELS[p] for p in POLICIES]
    ax0.bar(x, agg.goodput_med, color="#4c72b0")
    ax0.errorbar(x, agg.goodput_med,
                 yerr=[agg.goodput_med - agg.goodput_lo, agg.goodput_hi - agg.goodput_med],
                 fmt="none", ecolor="0.3", capsize=3)
    ax0.set_xticks(x, names, rotation=20, ha="right")
    ax0.set_ylabel("SLO goodput (req/s)")
    ax0.set_title("Goodput")
    ax1.bar(x, agg.p95_med, color="#c44e52")
    ax1.errorbar(x, agg.p95_med,
                 yerr=[agg.p95_med - agg.p95_lo, agg.p95_hi - agg.p95_med],
                 fmt="none", ecolor="0.3", capsize=3)
    ax1.set_xticks(x, names, rotation=20, ha="right")
    ax1.set_ylabel("TTFT P95 (s)")
    ax1.set_title("Tail latency")
    ax2.bar(x, agg.sa_qmax, color="#55a868")
    ax2.set_xticks(x, names, rotation=20, ha="right")
    ax2.set_ylabel("S_A queue depth max")
    ax2.set_title("Hotspot peak (Storage A)")

    import pandas as pd
    s0 = seeds[0]
    axt.set_title("Queue depth time series (seed 0)")
    for pol, color in [("kv", "#c44e52"), ("p3", "#4c72b0")]:
        path = os.path.join(RESULTS_DIR, "e2", f"ts_{pol}_s{s0}.csv")
        if os.path.exists(path):
            ts = pd.read_csv(path)
            axt.plot(ts.t, ts.s0_q.rolling(50).mean(), color=color, lw=1.4,
                     label=f"{LABELS[pol]}: S_A (hot)")
            axt.plot(ts.t, ts.s1_q.rolling(50).mean(), color=color, ls=":", lw=1.2,
                     label=f"{LABELS[pol]}: S_B (idle)")
    axt.set_xlabel("time (s)")
    axt.set_ylabel("queue depth (rolling mean)")
    axt.legend(fontsize=7)
    savefig(plt, "fig_c_e2_routing.png")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e2(seeds, duration=duration)
    print(f"[e2] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e2")
    analyze(pd.read_csv(out), seeds)
