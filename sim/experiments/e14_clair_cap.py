"""E14（问题⑦）：容量/副本约束下的先知对照。

E9b 证明资源平行、无约束的 burst 场景下先知 ≈ 逐请求贪心；本实验在
E11 的容量受限 + 热度漂移场景（无控制器）复检假设 H7：clairvoyant > oracle。
"""
from __future__ import annotations

import os

from ..config import NodeConfig, PrefixClass, StorageConfig, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo
from .e11_tiering import CLASSES, DRIFT, INIT_SINGLE, NODES

POLICIES = ["joint2", "oracle2", "clairvoyant2"]
PLABELS = {"joint2": "Greedy(Ours)", "oracle2": "Oracle(per-req)", "clairvoyant2": "Clairvoyant"}
PCOLORS = {"joint2": "#4c72b0", "oracle2": "#cca64c", "clairvoyant2": "#55a868"}


def build_e14(seeds, duration=400.0):
    topo = mk_topo(replicas=INIT_SINGLE, nodes=NODES, local_cache_gb=0.0,
                   gpu_bgs=(stable(0.65),) * 4)
    jobs = []
    for pol in POLICIES:
        for seed in seeds:
            spec = mk_spec("e14", pol, seed, topo, duration=duration, classes=CLASSES,
                           slo=0.5, lam=10.0, hit_ratio=0.85, drift=DRIFT,
                           out_dir=os.path.join(RESULTS_DIR, "e14") if seed == seeds[0] else None)
            jobs.append(({"policy": pol, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    import pandas as pd
    plt = setup_matplotlib()
    agg = df.groupby("policy").agg(
        goodput=("goodput", "median"), q1=("goodput", lambda s: np.percentile(s, 25)),
        q3=("goodput", lambda s: np.percentile(s, 75)), p95=("ttft_p95", "median"),
    ).reindex(POLICIES)
    print("\n[E14] policy summary (median [IQR] over seeds):")
    print(agg.round(3).to_string())
    c, o = agg.goodput["clairvoyant2"], agg.goodput["oracle2"]
    sep = (agg.q1["clairvoyant2"] > agg.q3["oracle2"]) or (agg.q3["clairvoyant2"] < agg.q1["oracle2"])
    print(f"\n[E14] clairvoyant - oracle = {c - o:+.3f} req/s ({(c / o - 1) * 100:+.1f}%), "
          f"IQR 分离: {sep}")

    fig, ax = plt.subplots(figsize=(6.4, 3.3), constrained_layout=True)
    x = np.arange(len(POLICIES))
    ax.bar(x, agg.goodput, color=[PCOLORS[p] for p in POLICIES],
           yerr=[agg.goodput - agg.q1, agg.q3 - agg.goodput], capsize=4, error_kw={"ecolor": "0.3"})
    for i, p in enumerate(POLICIES):
        ax.text(i, agg.q3[p], f"{agg.goodput[p]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [PLABELS[p] for p in POLICIES], rotation=12)
    ax.set_ylabel("SLO goodput (req/s)")
    ax.set_title("E14: capacity-constrained drift — greedy vs per-req oracle vs clairvoyant")
    savefig(plt, "fig_e14_clair_cap.png")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e14(seeds, duration=duration)
    print(f"[e14] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    analyze(pd.read_csv(save_rows(rows, "e14")), seeds)
