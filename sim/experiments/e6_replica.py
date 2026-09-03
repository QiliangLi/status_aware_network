"""E6（Q2：从哪个 KV 副本、哪条路径读取？）：副本/层/路径选择对比。

场景：热点类 A 三个副本 —— (n0,mem) 方波高压、(n1,ssd) 稳定低压但层慢、(n2,mem) 反相方波；
fabric 恒定背景。基线映射：
  nearest2   —— 主流默认 credit（mem 优先 + 最近路径，无压力感知）
  rrrep2     —— 副本间轮询（盲负载均衡）
  static2    —— AAFLOW+ 静态成本选副本
  joint2     —— 本方向（压力感知选副本）
  oracle2    —— 上界
"""
from __future__ import annotations

import os

from ..config import NodeConfig, StorageConfig, square, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import LABELS, mk_spec, mk_topo, offset_square

POLICIES = ["nearest2", "rrrep2", "static2", "joint2", "oracle2"]
PERIOD = 40.0

NODES = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=square(55.0, 20.0, PERIOD, 600.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0),
               ssd=StorageConfig(b_total=25.0, bg_schedule=stable(15.0))),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0, bg_schedule=offset_square(30.0, 15.0, PERIOD, 600.0, 10.0)),
               ssd=StorageConfig(b_total=15.0)),
)
REPLICAS = (
    ("A", ((0, "mem"), (1, "ssd"), (2, "mem"))),
    ("B", ((0, "mem"),)),
    ("C", ((1, "ssd"),)),
    ("D", ((2, "ssd"),)),
)


def build_e6(seeds, duration=400.0):
    topo = mk_topo(replicas=REPLICAS, nodes=NODES,
                   fabric=StorageConfig(b_total=120.0, bg_schedule=stable(20.0)))
    jobs = []
    for pol in POLICIES:
        for seed in seeds:
            spec = mk_spec("e6", pol, seed, topo, duration=duration, lam=9.0, slo=0.6,
                           collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]),
                           out_dir=os.path.join(RESULTS_DIR, "e6") if seed == seeds[0] else None)
            jobs.append(({"policy": pol, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    import pandas as pd
    plt = setup_matplotlib()
    agg = df.groupby("policy").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"),
        q0max=("s0_q_max", "median"), q0std=("s0_q_std", "median"),
        q4max=("s4_q_max", "median"),
        quote=("quote_mape", "median"), flip=("replica_flip_rate", "median"),
    ).reindex(POLICIES)
    # 副本选择分布（replica_n{node}_{tier}）
    for p in POLICIES:
        sub = df[df.policy == p]
        for col in [c for c in df.columns if c.startswith("replica_n")]:
            agg.loc[p, col] = sub[col].median()
    print("\n[E6] policy summary (median over seeds):")
    cols = ["goodput", "p95", "q0max", "quote", "flip"] + [c for c in agg.columns if c.startswith("replica_")]
    print(agg[cols].round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    x = np.arange(len(POLICIES))
    names = [LABELS[p] for p in POLICIES]
    axes[0].bar(x, agg.goodput, color="#4c72b0")
    axes[0].set_xticks(x, names, rotation=25, ha="right")
    axes[0].set_ylabel("SLO goodput (req/s)")
    axes[0].set_title("Goodput")
    axes[1].bar(x, agg.p95, color="#c44e52")
    axes[1].set_xticks(x, names, rotation=25, ha="right")
    axes[1].set_ylabel("TTFT P95 (s)")
    axes[1].set_title("Tail latency")
    rep_cols = ["replica_n0_mem", "replica_n1_ssd", "replica_n2_mem"]
    bottom = np.zeros(len(POLICIES))
    for col, color, lab in zip(rep_cols, ["#c44e52", "#ccb974", "#55a868"],
                               ["n0.mem (sq-wave hot)", "n1.ssd (slow, idle)", "n2.mem (anti-phase)"]):
        vals = agg[col].fillna(0.0).to_numpy()
        axes[2].bar(x, vals, bottom=bottom, color=color, label=lab)
        bottom += vals
    axes[2].set_xticks(x, names, rotation=25, ha="right")
    axes[2].set_ylabel("fetch decisions")
    axes[2].set_title("Replica choice distribution")
    axes[2].legend(fontsize=7)
    savefig(plt, "fig_e6_replica.png")

    # 时间序列：n0.mem 队列 + joint2 的副本选择轨迹
    s0 = seeds[0]
    ts_path = os.path.join(RESULTS_DIR, "e6", f"ts_joint2_s{s0}.csv")
    if os.path.exists(ts_path):
        ts = pd.read_csv(ts_path)
        m = (ts.t > 20) & (ts.t < 200)
        fig2, ax = plt.subplots(figsize=(8.5, 3.2), constrained_layout=True)
        ax.plot(ts.t[m], ts.s0_q[m], color="#c44e52", lw=1.1, label="n0.mem queue (hot sq-wave)")
        ax.plot(ts.t[m], ts.s4_q[m], color="#55a868", lw=1.0, label="n2.mem queue (anti-phase)")
        ax.plot(ts.t[m], ts.s2_q[m], color="#ccb974", lw=1.0, label="n1.ssd queue")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("queue depth")
        ax.set_title("E6: per-replica queue under joint2 (seed 0)")
        ax.legend(fontsize=7)
        savefig(plt, "fig_e6_queues.png")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e6(seeds, duration=duration)
    print(f"[e6] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e6")
    analyze(pd.read_csv(out), seeds)
