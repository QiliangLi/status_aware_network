"""E4：热点 prefix 触发 KV restoration incast——动态策略能否保护尾延迟。"""
from __future__ import annotations

import os

from ..config import GpuConfig, RunSpec, StorageConfig, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib

BURST = (120.0, 150, 1.0, "A")   # t0, n, dur, class
WINDOW = (120.0, 180.0)
DURATION = 240.0
POLICIES = ["p0", "p1", "p2", "p3", "p4"]
LABELS = {"p0": "P0 AlwaysFetch", "p1": "P1 StaticCost", "p2": "P2 Dynamic",
          "p3": "P3 Dynamic+Route", "p4": "P4 Oracle"}
COLORS = {"p0": "#8c8c8c", "p1": "#c44e52", "p2": "#4c72b0", "p3": "#55a868", "p4": "#cca64c"}


def main(seeds, procs=None):
    import pandas as pd
    jobs = build_e4(seeds)
    print(f"[e4] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e4")
    analyze(pd.read_csv(out), seeds)


def build_e4(seeds):
    jobs = []
    for pol in POLICIES:
        for seed in seeds:
            spec = RunSpec(
                exp="e4", policy=pol, seed=seed, duration=DURATION, warmup=20.0, margin=90.0,
                storages=(StorageConfig(bg_schedule=stable(20.0)),),
                gpu=GpuConfig(bg_schedule=stable(0.4)),
                burst=BURST, window=WINDOW,
                collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]), save_requests=(seed == seeds[0]),
                out_dir=os.path.join(RESULTS_DIR, "e4") if seed == seeds[0] else None,
            )
            jobs.append(({"policy": pol, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    import pandas as pd
    plt = setup_matplotlib()
    agg = df.groupby("policy").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"), p99=("ttft_p99", "median"),
        win_goodput=("win_goodput", "median"), win_p95=("win_ttft_p95", "median"),
        win_p99=("win_ttft_p99", "median"), win_slo=("win_slo_rate", "median"),
        qmax=("s0_q_max", "median"), recompute=("recompute_count", "median"),
    ).reindex(POLICIES)
    print("\n[E4] burst-window [120,180s) metrics (median over seeds):")
    print(agg.round(3).to_string())

    s0 = seeds[0]
    ts = {p: pd.read_csv(os.path.join(RESULTS_DIR, "e4", f"ts_{p}_s{s0}.csv")) for p in POLICIES}

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 8.2), sharex=True, constrained_layout=True)
    t0, t1 = 90, 210
    for p in POLICIES:
        d = ts[p]
        m = (d.t >= t0) & (d.t <= t1)
        axes[0].plot(d.t[m], d.s0_q[m], color=COLORS[p], lw=1.1, label=LABELS[p])
        axes[1].plot(d.t[m], d.recompute_frac[m], color=COLORS[p], lw=1.1)
        axes[3].plot(d.t[m], d.roll_p95_ttft[m], color=COLORS[p], lw=1.1)
    d2 = ts["p2"]
    m2 = (d2.t >= t0) & (d2.t <= t1)
    axes[2].plot(d2.t[m2], d2.s0_fg[m2], color="#4c72b0", lw=1.1, label="storage fg GB/s (P2)")
    axes[2].plot(d2.t[m2], d2.w0_util[m2], color="#55a868", lw=1.1, label="GPU util w0 (P2)")
    axes[2].plot(d2.t[m2], d2.w1_util[m2], color="#55a868", lw=0.8, ls="--")

    axes[0].set_ylabel("Storage queue depth")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel("Recompute fraction")
    axes[2].set_ylabel("GB/s / util")
    axes[2].legend(fontsize=7)
    axes[3].set_ylabel("Rolling P95 TTFT (s)")
    axes[3].axhline(1.0, color="0.3", lw=0.8, ls="-.")
    axes[3].text(t0 + 2, 1.03, "SLO 1s", fontsize=7)
    for ax in axes:
        ax.axvspan(120, 121, color="0.8", alpha=0.5)
    axes[3].set_xlabel("time (s) — burst of 150 × prefix-A(32K) at t=120")
    fig.suptitle("E4: KV restoration incast (storage bg=20GB/s, GPU bg=40%)", fontsize=10)
    savefig(plt, "fig_b_e4_burst.png")

    # 恢复时间：rolling p95 在峰值之后首次回落到 burst 前基线 1.2× 以内
    print("\n[E4] recovery analysis (seed 0):")
    for p in POLICIES:
        d = ts[p].dropna(subset=["roll_p95_ttft"])
        base = d[(d.t > 60) & (d.t < 118)].roll_p95_ttft.median()
        thr = 1.2 * base
        after = d[d.t >= 121]
        peak_t = float(after.loc[after.roll_p95_ttft.idxmax()].t)
        peak_v = float(after.roll_p95_ttft.max())
        rec = after[(after.t >= peak_t) & (after.roll_p95_ttft <= thr)]
        t_rec = float(rec.t.iloc[0]) if len(rec) else float("nan")
        print(f"   {LABELS[p]:18s} baseline_p95={base:.2f}s  peak_p95={peak_v:.1f}s@t={peak_t:.0f}  recovery={t_rec:.0f}s")
