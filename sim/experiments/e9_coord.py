"""E9（Q5：如何避免热点、争抢和局部最优？）：局部贪心 vs 全局协同。

场景：A 双副本 (n0,mem)/(n1,mem)，两节点背景对称稳定（无静态偏好可言）；
t=120 注入 150 个 A 命中请求的同步 burst —— 所有请求同时读同一份陈旧 quote（200ms、σ=0.1），
逐请求贪心的"局部最优"会把整批流量砸向同一副本（羊群），协同策略需在无新鲜反馈时
靠调度器自身记账（在途影子项 + 滞回 + 抖动）完成分流。
  joint2  —— 局部贪心（逐请求取最小 quote）
  coord2  —— 本方向+协同（在途影子项 + 滞回 + 抖动）
  oracle2 —— 上界（真值预测自然分流）
观测：burst 窗口 P95/P99、副本队列峰值（羊群振荡）、goodput regret。
"""
from __future__ import annotations

import os

from ..config import NodeConfig, ObsConfig, StorageConfig, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

POLICIES = ["joint2", "coord2", "oracle2"]
PLABELS = {"joint2": "Greedy(Ours)", "coord2": "Coordinated(Ours+)", "oracle2": "Oracle"}
PCOLORS = {"joint2": "#4c72b0", "coord2": "#55a868", "oracle2": "#cca64c"}

NODES = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(35.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0, bg_schedule=stable(35.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)),
)
REPLICAS = (
    ("A", ((0, "mem"), (1, "mem"))),
    ("B", ((0, "mem"),)),
    ("C", ((1, "ssd"),)),
    ("D", ((2, "ssd"),)),
)
OBS = ObsConfig(interval=0.2, noise_sigma=0.1, signal="quote")
BURST = (120.0, 150, 1.0, "A")
WINDOW = (120.0, 220.0)


def build_e9(seeds, duration=400.0):
    topo = mk_topo(replicas=REPLICAS, nodes=NODES, local_cache_gb=0.0)
    jobs = []
    for pol in POLICIES:
        for seed in seeds:
            spec = mk_spec("e9", pol, seed, topo, duration=duration, lam=5.0, slo=0.8,
                           obs=OBS, burst=BURST, window=WINDOW,
                           collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]),
                           out_dir=os.path.join(RESULTS_DIR, "e9") if seed == seeds[0] else None)
            jobs.append(({"policy": pol, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    import pandas as pd
    plt = setup_matplotlib()
    agg = df.groupby("policy").agg(
        goodput=("goodput", "median"), win_p95=("win_ttft_p95", "median"),
        win_p99=("win_ttft_p99", "median"), win_slo=("win_slo_rate", "median"),
        q0max=("s0_q_max", "median"), q2max=("s2_q_max", "median"),
        flip=("replica_flip_rate", "median"),
    ).reindex(POLICIES)
    agg["herd_peak"] = agg[["q0max", "q2max"]].max(axis=1)
    print("\n[E9] policy summary (median over seeds):")
    print(agg.round(3).to_string())
    g0 = agg.win_slo["oracle2"]
    print(f"\n   burst-window SLO rate regret vs oracle: "
          + ", ".join(f"{PLABELS[p]}={100 * (g0 - agg.win_slo[p]):.1f}pp" for p in POLICIES))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    x = np.arange(len(POLICIES))
    names = [PLABELS[p] for p in POLICIES]
    for ax, col, color, title, ylab in [
            (axes[0], "win_slo", "#4c72b0", "Burst-window SLO rate", "SLO rate"),
            (axes[1], "win_p99", "#c44e52", "Burst-window TTFT P99", "TTFT P99 (s)"),
            (axes[2], "herd_peak", "#55a868", "Herd queue peak (single replica)", "queue depth max")]:
        ax.bar(x, agg[col], color=color)
        ax.set_xticks(x, names, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
    savefig(plt, "fig_e9_coord.png")

    s0 = seeds[0]
    fig2, ax = plt.subplots(figsize=(8.5, 3.2), constrained_layout=True)
    for pol, ls in [("joint2", "-"), ("coord2", "--")]:
        path = os.path.join(RESULTS_DIR, "e9", f"ts_{pol}_s{s0}.csv")
        if os.path.exists(path):
            ts = pd.read_csv(path)
            m = (ts.t > 100) & (ts.t < 220)
            ax.plot(ts.t[m], ts.s0_q[m], color=PCOLORS[pol], ls=ls, lw=1.2,
                    label=f"{PLABELS[pol]}: n0.mem")
            ax.plot(ts.t[m], ts.s2_q[m], color=PCOLORS[pol], ls=ls, lw=0.9, alpha=0.6,
                    label=f"{PLABELS[pol]}: n1.mem")
    ax.axvspan(120, 121, color="0.8", alpha=0.5)
    ax.set_xlabel("time (s) — burst of 150 x prefix-A at t=120")
    ax.set_ylabel("queue depth")
    ax.set_title("E9: synchronized burst — greedy herding vs coordinated splitting (seed 0)")
    ax.legend(fontsize=7, ncol=2)
    savefig(plt, "fig_e9_queues.png")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e9(seeds, duration=duration)
    print(f"[e9] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e9")
    analyze(pd.read_csv(out), seeds)
