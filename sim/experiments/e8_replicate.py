"""E8（Q4：何时预取、迁移或复制？）：存储侧主动复制对比。

场景：A 为重热点（share 0.6），副本仅在 (n0,mem) 且背景较高；n1/n2 空闲。
所有运行引擎侧均用 joint2（动态联合决策），差异只在存储侧控制器：
  none      —— 无控制器（只能靠重算逃逸）
  reactive  —— 源持续 HOT ≥2s 且类需求高才复制（滞回阈值）
  predictive—— 更低阈值 + 更短保持（提前触发）
  full      —— 上界：初始即全节点副本（无运行时复制成本）
"""
from __future__ import annotations

import os

from ..config import CtrlConfig, NodeConfig, PrefixClass, StorageConfig, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import LABELS, mk_spec, mk_topo

VARIANTS = ["none", "reactive", "predictive", "full"]
VLABELS = {"none": "No-ctrl", "reactive": "Reactive-repl",
           "predictive": "Predictive-repl", "full": "Full-static(oracle)"}
VCOLORS = {"none": "#8c8c8c", "reactive": "#4c72b0", "predictive": "#55a868", "full": "#cca64c"}

NODES = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(30.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0, bg_schedule=stable(5.0)),
               ssd=StorageConfig(b_total=25.0, bg_schedule=stable(2.0))),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0, bg_schedule=stable(5.0)),
               ssd=StorageConfig(b_total=15.0, bg_schedule=stable(2.0))),
)
CLASSES = (PrefixClass("A", 32768, 0.65), PrefixClass("B", 16384, 0.15),
           PrefixClass("C", 8192, 0.10), PrefixClass("D", 65536, 0.10))
HIT_RATIO = 0.85   # 高命中：把瓶颈压到存储侧，隔离"复制是否有价值"


def build_e8(seeds, duration=400.0):
    jobs = []
    for var in VARIANTS:
        for seed in seeds:
            if var == "full":
                replicas = (("A", ((0, "mem"), (1, "mem"), (2, "mem"))),
                            ("B", ((1, "mem"),)), ("C", ((1, "ssd"),)), ("D", ((2, "ssd"),)))
                ctrl = None
            else:
                replicas = (("A", ((0, "mem"),)),
                            ("B", ((1, "mem"),)), ("C", ((1, "ssd"),)), ("D", ((2, "ssd"),)))
                ctrl = {
                    "reactive": CtrlConfig(interval=0.5, hot_util=0.85, exit_util=0.65,
                                           hold_s=2.0, min_demand=1.0),
                    "predictive": CtrlConfig(interval=0.5, hot_util=0.85, exit_util=0.65,
                                             hold_s=2.0, min_demand=1.0, predictive=True),
                }.get(var)
            topo = mk_topo(replicas=replicas, nodes=NODES, ctrl=ctrl, gpu_bgs=(stable(0.7),) * 4,
                           local_cache_gb=0.0)
            spec = mk_spec("e8", "joint2", seed, topo, duration=duration, classes=CLASSES, slo=0.75,
                           hit_ratio=HIT_RATIO,
                           collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]),
                           save_requests=(seed == seeds[0]),
                           out_dir=os.path.join(RESULTS_DIR, "e8", var) if seed == seeds[0] else None)
            jobs.append(({"variant": var, "policy": "joint2", "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    import pandas as pd
    plt = setup_matplotlib()
    agg = df.groupby("variant").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"), p99=("ttft_p99", "median"),
        sloA=("slo_rate_A", "median"), nrepl=("n_replications", "median"),
        rgb=("replication_gb", "median"), fetch_gb=("fetch_gb", "median"),
        q0max=("s0_q_max", "median"),
    ).reindex(VARIANTS)
    print("\n[E8] variant summary (median over seeds):")
    print(agg.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    x = np.arange(len(VARIANTS))
    names = [VLABELS[v] for v in VARIANTS]
    axes[0].bar(x, agg.goodput, color=[VCOLORS[v] for v in VARIANTS])
    axes[0].set_xticks(x, names, rotation=20, ha="right")
    axes[0].set_ylabel("SLO goodput (req/s)")
    axes[0].set_title("Goodput")
    axes[1].bar(x, agg.sloA, color=[VCOLORS[v] for v in VARIANTS])
    axes[1].set_xticks(x, names, rotation=20, ha="right")
    axes[1].set_ylabel("SLO rate (class A)")
    axes[1].set_title("Hot class A SLO rate")
    axes[2].bar(x, agg.q0max, color=[VCOLORS[v] for v in VARIANTS])
    axes[2].set_xticks(x, names, rotation=20, ha="right")
    axes[2].set_ylabel("n0.mem queue max")
    axes[2].set_title("Hotspot peak")
    savefig(plt, "fig_e8_replicate.png")

    s0 = seeds[0]
    fig2, ax = plt.subplots(figsize=(8.5, 3.2), constrained_layout=True)
    for var in ["none", "reactive", "full"]:
        path = os.path.join(RESULTS_DIR, "e8", f"ts_joint2_s{s0}.csv")
        # 各 variant 覆盖写同一文件名会互相冲掉，改用 requests/summary 判别；此处用目录分 variant 的落盘
        path = os.path.join(RESULTS_DIR, "e8", var, f"ts_joint2_s{s0}.csv")
        if os.path.exists(path):
            ts = pd.read_csv(path)
            m = (ts.t > 20) & (ts.t < 240)
            ax.plot(ts.t[m], ts.s0_q[m].rolling(20).mean(), lw=1.1,
                    color=VCOLORS[var], label=f"{VLABELS[var]}: n0.mem q (rolling)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("queue depth")
    ax.set_title("E8: hot-node queue under replication variants (seed 0)")
    ax.legend(fontsize=7)
    savefig(plt, "fig_e8_queues.png")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e8(seeds, duration=duration)
    print(f"[e8] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e8")
    analyze(pd.read_csv(out), seeds)
