"""E15（问题⑧）：会话预测预取——用类级轮间隔 EMA 过滤"大概率浪费"的预取。

混合速度会话（每会话 50% 概率快/慢轮间隔），均不带回写以隔离变量。
假设 H8：predictive 的浪费率 ≤ 0.5 × gated，goodput 不降（≥ gated −1%）。
"""
from __future__ import annotations

import os

from ..config import NodeConfig, PrefetchConfig, PrefixClass, StorageConfig, stable, square
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

VARIANTS = ["none", "gated", "predictive"]
VLABELS = {"none": "No-prefetch", "gated": "Gated", "predictive": "Gated+Predictive"}
VCOLORS = {"none": "#8c8c8c", "gated": "#4c72b0", "predictive": "#55a868"}

NODES = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=square(35.0, 20.0, 40.0, 600.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0), ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)),
)
REPLICAS = (("A", ((0, "mem"),)), ("B", ((0, "mem"),)), ("C", ((0, "mem"),)), ("D", ((2, "ssd"),)))
CLASSES = (PrefixClass("A", 32768, 0.40), PrefixClass("B", 16384, 0.20),
           PrefixClass("C", 8192, 0.20), PrefixClass("D", 65536, 0.20))
SESSIONS = (1.6, 4.0, (0.6, 8.0))   # 混合速度：快会话 0.6s / 慢会话 8s


def build_e15(seeds, duration=400.0):
    jobs = []
    for var in VARIANTS:
        pf = None if var == "none" else PrefetchConfig(mode=var, writeback=False)
        topo = mk_topo(replicas=REPLICAS, nodes=NODES, local_cache_gb=12.0,
                       gpu_bgs=(stable(0.5),) * 4,
                       prefetch=pf)
        for seed in seeds:
            spec = mk_spec("e15", "joint2", seed, topo, duration=duration, classes=CLASSES,
                           slo=0.35, lam=6.0, sessions=SESSIONS,
                           out_dir=os.path.join(RESULTS_DIR, "e15") if seed == seeds[0] else None)
            jobs.append(({"variant": var, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    plt = setup_matplotlib()
    agg = df.groupby("variant").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"),
        floc=("frac_local", "median"), pgb=("prefetch_gb", "median"),
        waste=("prefetch_waste_frac", "median"),
    ).reindex(VARIANTS)
    print("\n[E15] variant summary (median over seeds):")
    print(agg.round(3).to_string())
    if agg.waste["gated"] and agg.waste["predictive"] is not None:
        ratio = agg.waste["predictive"] / agg.waste["gated"]
        print(f"\n[E15] waste ratio predictive/gated = {ratio:.2f} (H8 验收 <= 0.5)")
        print(f"[E15] goodput diff predictive-gated = {agg.goodput['predictive'] - agg.goodput['gated']:+.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    for ax, col, color, title, ylab in [
            (axes[0], "goodput", "#4c72b0", "Goodput", "SLO goodput (req/s)"),
            (axes[1], "pgb", "#ccb974", "Prefetch bytes", "GB"),
            (axes[2], "waste", "#c44e52", "Prefetch waste fraction", "wasted / prefetched")]:
        ax.bar(x, agg[col], color=color)
        for i, v in enumerate(VARIANTS):
            ax.text(i, agg[col][v], f"{agg[col][v]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [VLABELS[v] for v in VARIANTS], rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
    savefig(plt, "fig_e15_prefetch_pred.png")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e15(seeds, duration=duration)
    print(f"[e15] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    analyze(pd.read_csv(save_rows(rows, "e15")), seeds)
