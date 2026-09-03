"""E7（Q3：拉取、部分重算还是完整重算？）：恢复动作空间对比。

场景：所有类副本集中在 (n0,mem) 且该层背景负载逼近饱和；GPU 有余量。
控制变量：所有策略共享同一动作空间（引擎能力相同），差异只在决策信息源。
  alwaysfetch2    —— 主流默认（命中即取，P0）
  static2         —— AAFLOW+（nominal 带宽二选一）
  partial_static2 —— CacheFlow（静态成本 F 网格 + I/O/计算重叠）
  joint2          —— 本方向（访问成本查询驱动的 F 选择）
  joint2_seq      —— 消融：joint2 但不重叠
  oracle2         —— 上界
两个负载格点：mem bg=40（中度）/ bg=55（重度，仅靠动态感知才能发现）。
"""
from __future__ import annotations

from ..config import NodeConfig, StorageConfig, stable
from .common import run_pool, save_rows, savefig, setup_matplotlib
from .v2common import LABELS, mk_spec, mk_topo

POLICIES = ["alwaysfetch2", "static2", "partial_static2", "joint2", "joint2_seq", "oracle2"]


def _nodes(bg: float):
    return (NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(bg)),
                       ssd=StorageConfig(b_total=25.0)),
            NodeConfig("n1", mem=StorageConfig(b_total=60.0), ssd=StorageConfig(b_total=25.0)),
            NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)))


REPLICAS = (
    ("A", ((0, "mem"),)),
    ("B", ((0, "mem"),)),
    ("C", ((0, "mem"),)),
    ("D", ((0, "mem"),)),
)


def build_e7(seeds, bg: float, duration=400.0):
    topo = mk_topo(replicas=REPLICAS, nodes=_nodes(bg))
    jobs = []
    for pol in POLICIES:
        for seed in seeds:
            spec = mk_spec("e7", pol, seed, topo, duration=duration, lam=8.0, slo=0.6,
                           gpu_bg=0.5)
            jobs.append(({"policy": pol, "seed": seed, "bg": bg}, spec))
    return jobs


def _panel(ax, df, title):
    import numpy as np
    agg = df.groupby("policy").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"),
        fetch_gb=("fetch_gb", "median"), rct=("recompute_tokens", "median"),
        f_partial=("frac_partial", "median"), f_rc=("frac_recompute", "median"),
    ).reindex(POLICIES)
    x = np.arange(len(POLICIES))
    ax.bar(x, agg.goodput, color=["#c9c9c9", "#c44e52", "#ccb974", "#4c72b0", "#7f9fc4", "#cca64c"])
    for i, p in enumerate(POLICIES):
        ax.text(i, agg.goodput[p] + 0.02, f"{agg.goodput[p]:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, [LABELS[p] for p in POLICIES], rotation=30, ha="right")
    ax.set_title(title, fontsize=9)
    return agg


def analyze(dfs, titles, bgs):
    import numpy as np
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), constrained_layout=True, sharey=False)
    aggs = [_panel(ax, df, t) for ax, df, t in zip(axes, dfs, titles)]
    axes[0].set_ylabel("SLO goodput (req/s)")
    savefig(plt, "fig_e7_partial.png")

    for bg, agg in zip(bgs, aggs):
        print(f"\n[E7 bg={bg}] policy summary (median over seeds):")
        print(agg.round(3).to_string())
        base = agg.fetch_gb["alwaysfetch2"]
        print(f"   fetch bytes saved vs AlwaysFetch: "
              + ", ".join(f"{LABELS[p]}={100 * (1 - agg.fetch_gb[p] / base):.0f}%" for p in POLICIES))


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    dfs, titles, bgs = [], [], []
    for bg, tag in [(40.0, "n0.mem bg=40 GB/s (offered ~115%)"),
                    (55.0, "n0.mem bg=55 GB/s (offered ~140%, deep)")]:
        jobs = build_e7(seeds, bg, duration=duration)
        print(f"[e7 bg={bg:.0f}] {len(jobs)} runs ...")
        rows = run_pool(jobs, procs)
        out = save_rows(rows, f"e7_bg{int(bg)}")
        dfs.append(pd.read_csv(out))
        titles.append(tag)
        bgs.append(bg)
    analyze(dfs, titles, bgs)
