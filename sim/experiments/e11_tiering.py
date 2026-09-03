"""E11（问题④）：跨层降级 + 冷类迁移 + 容量淘汰联动。

场景：3 节点，mem/ssd 双层，节点容量受限（mem 放不下所有类的双副本）；
6 个类热度每 100s 漂移一轮（份额轮转）——放置必须跟随热度。
引擎统一 joint2，只比存储侧控制器：
  none    —— 无控制器（静态初始放置）
  tiered  —— 跨层复制（目标含 ssd，按 util+容量压力选）
  full    —— tiered + 冷类迁移（回收 mem）+ 容量淘汰（防孤儿）
  nocap   —— 上界：容量无限 + 初始双副本静态放置
"""
from __future__ import annotations

import os

from ..config import CtrlConfig, NodeConfig, PrefixClass, StorageConfig, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

VARIANTS = ["none", "tiered", "full", "nocap"]
VLABELS = {"none": "Static(no-ctrl)", "tiered": "Cross-tier-repl",
           "full": "Tiered+migrate+evict", "nocap": "Nocap-static(upper)"}
VCOLORS = {"none": "#8c8c8c", "tiered": "#4c72b0", "full": "#55a868", "nocap": "#cca64c"}

NODES = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(10.0)),
               ssd=StorageConfig(b_total=25.0, bg_schedule=stable(2.0)), cap_gb=40.0),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0, bg_schedule=stable(10.0)),
               ssd=StorageConfig(b_total=25.0, bg_schedule=stable(2.0)), cap_gb=40.0),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0, bg_schedule=stable(5.0)),
               ssd=StorageConfig(b_total=15.0, bg_schedule=stable(1.0)), cap_gb=30.0),
)
CLASSES = (PrefixClass("A", 32768, 0.30), PrefixClass("B", 16384, 0.20),
           PrefixClass("C", 8192, 0.10), PrefixClass("D", 65536, 0.15),
           PrefixClass("E", 8192, 0.10), PrefixClass("F", 32768, 0.15))
DRIFT = (100.0,)
INIT_SINGLE = (("A", ((0, "mem"),)), ("B", ((1, "mem"),)), ("C", ((2, "mem"),)),
               ("D", ((0, "mem"),)), ("E", ((1, "mem"),)), ("F", ((2, "mem"),)))
INIT_DOUBLE = (("A", ((0, "mem"), (1, "mem"))), ("B", ((1, "mem"), (2, "mem"))),
               ("C", ((2, "mem"), (0, "mem"))), ("D", ((0, "ssd"), (1, "ssd"))),
               ("E", ((1, "mem"), (2, "mem"))), ("F", ((2, "mem"), (0, "mem"))))


def _ctrl(var):
    if var == "none" or var == "nocap":
        return None
    return CtrlConfig(interval=0.5, hot_util=0.78, exit_util=0.65, hold_s=2.0,
                      min_demand=0.6, max_replicas=3, cross_tier=True,
                      cold_demand=0.8, cap_evict=0.90, evict_target=0.75,
                      predictive=(var == "full"))


def build_e11(seeds, duration=400.0):
    jobs = []
    for var in VARIANTS:
        nodes = NODES
        replicas = INIT_SINGLE
        if var == "nocap":
            from dataclasses import replace
            nodes = tuple(replace(n, cap_gb=1e9) for n in NODES)
            replicas = INIT_DOUBLE
        topo = mk_topo(replicas=replicas, nodes=nodes, ctrl=_ctrl(var),
                       local_cache_gb=0.0, gpu_bgs=(stable(0.65),) * 4)
        for seed in seeds:
            spec = mk_spec("e11", "joint2", seed, topo, duration=duration, classes=CLASSES,
                           slo=0.5, lam=10.0, hit_ratio=0.85, drift=DRIFT,
                           collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]),
                           out_dir=os.path.join(RESULTS_DIR, "e11") if seed == seeds[0] else None)
            jobs.append(({"variant": var, "policy": "joint2", "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    import pandas as pd
    plt = setup_matplotlib()
    agg = df.groupby("variant").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"),
        nrepl=("n_replications", "median"), rgb=("replication_gb", "median"),
        evict=("n_evictions", "median"), orphan=("orphan_events", "median"),
        xtier=("cross_tier_replicas", "median"), q0max=("s0_q_max", "median"),
    ).reindex(VARIANTS)
    print("\n[E11] variant summary (median over seeds):")
    print(agg.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    names = [VLABELS[v] for v in VARIANTS]
    for ax, col, color, title, ylab in [
            (axes[0], "goodput", "#4c72b0", "Goodput", "SLO goodput (req/s)"),
            (axes[1], "p95", "#c44e52", "TTFT P95", "TTFT P95 (s)"),
            (axes[2], "xtier", "#55a868", "Cross-tier (ssd) replicas at end", "count")]:
        ax.bar(x, agg[col], color=color)
        for i, v in enumerate(VARIANTS):
            ax.text(i, agg[col][v], f"{agg[col][v]:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
    savefig(plt, "fig_e11_tiering.png")

    print("\n[E11] reading:")
    for v in VARIANTS:
        print(f"   {VLABELS[v]:22s} goodput={agg.goodput[v]:.2f} p95={agg.p95[v]:.2f} "
              f"repl={agg.nrepl[v]:.0f}({agg.rgb[v]:.0f}GB) evict={agg.evict[v]:.0f} "
              f"orphan={agg.orphan[v]:.0f} ssd_repl={agg.xtier[v]:.0f}")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e11(seeds, duration=duration)
    print(f"[e11] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e11")
    analyze(pd.read_csv(out), seeds)
