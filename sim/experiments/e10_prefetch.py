"""E10（问题②）：压力门控预取 + 重算回写 + 缓存协同放置。

会话型负载（多轮复用同一前缀，轮间隔 ~1.5s）；A/B/C 副本集中在 (n0,mem)，
背景方波 30±15（周期 40s）——gated 预取应在高压相位自动关闭，always 预取
则把额外流量砸进拥塞（预期复刻 E7 静态陷阱）。全部运行引擎侧 joint2。
变体：
  none        —— 无预取（基线）
  always      —— 无条件预取（反例）
  gated       —— 压力门控预取
  gated+wb    —— + 重算后回写共享存储
  gated+wb+coord —— + 本地缓存协同放置（类->偏好worker准入）
"""
from __future__ import annotations

import os

from ..config import NodeConfig, PrefetchConfig, PrefixClass, StorageConfig, stable, square
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

VARIANTS = ["none", "always", "gated", "gated_wb", "gated_wb_coord", "always_wb_coord"]
VLABELS = {"none": "No-prefetch", "always": "Always-prefetch", "gated": "Gated-prefetch",
           "gated_wb": "Gated+Writeback", "gated_wb_coord": "Gated+WB+Coord",
           "always_wb_coord": "Always+WB+Coord"}
VCOLORS = {"none": "#8c8c8c", "always": "#c44e52", "gated": "#4c72b0",
           "gated_wb": "#55a868", "gated_wb_coord": "#ccb974", "always_wb_coord": "#937860"}

NODES = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=square(35.0, 20.0, 40.0, 600.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0), ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)),
)
REPLICAS = (("A", ((0, "mem"),)), ("B", ((0, "mem"),)), ("C", ((0, "mem"),)), ("D", ((2, "ssd"),)))
CLASSES = (PrefixClass("A", 32768, 0.40), PrefixClass("B", 16384, 0.20),
           PrefixClass("C", 8192, 0.20), PrefixClass("D", 65536, 0.20))
SESSIONS = (1.6, 4.0, 1.2)   # 会话率 1/s，平均 4 轮，轮间隔 1.5s → ~4 req/s


def _prefetch_cfg(var):
    if var == "none":
        return None
    wb = var.endswith("_wb") or var.endswith("_coord")
    mode = "always" if var.startswith("always") else "gated"
    return PrefetchConfig(mode=mode, writeback=wb)


def build_e10(seeds, duration=400.0):
    jobs = []
    for var in VARIANTS:
        topo = mk_topo(replicas=REPLICAS, nodes=NODES, local_cache_gb=12.0, gpu_bgs=(stable(0.5),) * 4,
                       prefetch=_prefetch_cfg(var),
                       cache_mode="coord" if var.endswith("_coord") else "lru")
        for seed in seeds:
            spec = mk_spec("e10", "joint2", seed, topo, duration=duration, classes=CLASSES,
                           slo=0.35, lam=6.0, sessions=SESSIONS,
                           collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]),
                           out_dir=os.path.join(RESULTS_DIR, "e10") if seed == seeds[0] else None)
            jobs.append(({"variant": var, "policy": "joint2", "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    plt = setup_matplotlib()
    agg = df.groupby("variant").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"),
        floc=("frac_local", "median"), pgb=("prefetch_gb", "median"),
        waste=("prefetch_waste_frac", "median"), nwb=("n_writeback", "median"),
        redun=("cache_redundancy", "median"), cov=("cache_coverage", "median"),
        q0max=("s0_q_max", "median"),
    ).reindex(VARIANTS)
    print("\n[E10] variant summary (median over seeds):")
    print(agg.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    names = [VLABELS[v] for v in VARIANTS]
    for ax, col, color, title, ylab in [
            (axes[0], "goodput", "#4c72b0", "Goodput", "SLO goodput (req/s)"),
            (axes[1], "q0max", "#c44e52", "Hot-node queue peak (n0.mem)", "queue depth max"),
            (axes[2], "floc", "#55a868", "Local-cache hit fraction", "fraction")]:
        ax.bar(x, agg[col], color=color)
        for i, v in enumerate(VARIANTS):
            ax.text(i, agg[col][v], f"{agg[col][v]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
    savefig(plt, "fig_e10_prefetch.png")

    print("\n[E10] reading:")
    for v in VARIANTS:
        print(f"   {VLABELS[v]:18s} goodput={agg.goodput[v]:.2f} local={agg.floc[v]:.2f} "
              f"prefetch={agg.pgb[v]:.0f}GB waste={agg.waste[v]:.2f} "
              f"redundancy={agg['redun'][v]:.2f} coverage={agg['cov'][v]:.2f}")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e10(seeds, duration=duration)
    print(f"[e10] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e10")
    analyze(pd.read_csv(out), seeds)
