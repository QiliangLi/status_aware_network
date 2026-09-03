"""E13（问题⑥）：引擎能力 × 存储侧控制 的 2×2 因子实验。

假设 H6：控制器（predictive 复制）的收益在弱引擎（static2，无逃逸能力）下显著大于
强引擎（joint2，有重算/部分取回逃逸）下——即"引擎自稳挤占控制器收益"。
场景沿用 E8：热点类 A 单副本 (n0,mem) + GPU 稠忙（bg 0.7）+ 高命中。
"""
from __future__ import annotations

import os

from ..config import CtrlConfig, NodeConfig, PrefixClass, StorageConfig, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

ENGINES = ["static2", "joint2"]
CTRLS = ["none", "predictive"]
PRED = CtrlConfig(interval=0.5, hot_util=0.75, exit_util=0.65, hold_s=1.0,
                  min_demand=1.0, predictive=True, cross_tier=False)

NODES = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(30.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0, bg_schedule=stable(5.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0, bg_schedule=stable(5.0)),
               ssd=StorageConfig(b_total=15.0)),
)
CLASSES = (PrefixClass("A", 32768, 0.65), PrefixClass("B", 16384, 0.15),
           PrefixClass("C", 8192, 0.10), PrefixClass("D", 65536, 0.10))


def build_e13(seeds, duration=400.0):
    jobs = []
    for eng in ENGINES:
        for ctl in CTRLS:
            replicas = (("A", ((0, "mem"),)), ("B", ((1, "mem"),)),
                        ("C", ((1, "ssd"),)), ("D", ((2, "ssd"),)))
            topo = mk_topo(replicas=replicas, nodes=NODES, ctrl=PRED if ctl == "predictive" else None,
                           gpu_bgs=(stable(0.7),) * 4, local_cache_gb=0.0)
            for seed in seeds:
                spec = mk_spec("e13", eng, seed, topo, duration=duration, classes=CLASSES,
                               slo=0.75, lam=8.0, hit_ratio=0.85,
                               out_dir=os.path.join(RESULTS_DIR, "e13") if seed == seeds[0] else None)
                jobs.append(({"engine": eng, "ctrl": ctl, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    import pandas as pd
    plt = setup_matplotlib()
    agg = df.groupby(["engine", "ctrl"]).agg(
        goodput=("goodput", "median"), goodput_q1=("goodput", lambda s: np.percentile(s, 25)),
        goodput_q3=("goodput", lambda s: np.percentile(s, 75)),
        sloA=("slo_rate_A", "median"), p95=("ttft_p95", "median"),
        nrepl=("n_replications", "median"),
    ).reset_index()
    print("\n[E13] engine x ctrl summary (median over seeds):")
    print(agg.round(3).to_string(index=False))

    piv_g = agg.pivot(index="engine", columns="ctrl", values="goodput")
    piv_a = agg.pivot(index="engine", columns="ctrl", values="sloA")
    eff = {e: piv_g.loc[e, "predictive"] - piv_g.loc[e, "none"] for e in ENGINES}
    interaction = eff["static2"] - eff["joint2"]
    print(f"\n[E13] controller effect (goodput req/s): static2 {eff['static2']:+.3f}, "
          f"joint2 {eff['joint2']:+.3f}; interaction (static-joint) = {interaction:+.3f}")
    print(f"[E13] controller effect (sloA): "
          + ", ".join(f"{e}: {(piv_a.loc[e, 'predictive'] - piv_a.loc[e, 'none']):+.3f}" for e in ENGINES))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), constrained_layout=True)
    x = np.arange(len(ENGINES))
    w = 0.35
    for i, ctl in enumerate(CTRLS):
        axes[0].bar(x + (i - 0.5) * w, [piv_g.loc[e, ctl] for e in ENGINES], w, label=f"ctrl={ctl}")
        axes[1].bar(x + (i - 0.5) * w, [piv_a.loc[e, ctl] for e in ENGINES], w, label=f"ctrl={ctl}")
    for ax, ylab, title in [(axes[0], "SLO goodput (req/s)", "Goodput"),
                            (axes[1], "class-A SLO rate", "Hot class A SLO rate")]:
        ax.set_xticks(x, [f"engine={e}" for e in ENGINES])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=8)
    savefig(plt, "fig_e13_engine_ctrl.png")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e13(seeds, duration=duration)
    print(f"[e13] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    analyze(pd.read_csv(save_rows(rows, "e13")), seeds)
