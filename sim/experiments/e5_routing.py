"""E5（Q1：请求在哪个计算节点执行？）：共享存储下路由策略对比。

场景：w0/w1 GPU 忙、w2/w3 空闲；热点类 A 的本地缓存在忙的 w0 上（locality vs load 冲突）；
B 双副本、C 在 ssd、D 在远节点。基线映射：
  rr2 / load2            —— 无 KV 感知路由
  default2               —— 主流默认（Dynamo KV-credit 路由 + vLLM 命中即取）
  tensorcast2            —— TensorCast（locality+load 联合）
  joint2                 —— 本方向（访问成本查询联合路由）
  oracle2                —— 上界
"""
from __future__ import annotations

import os

from ..config import stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import LABELS, mk_spec, mk_topo

POLICIES = ["rr2", "load2", "default2", "tensorcast2", "joint2", "oracle2"]

REPLICAS = (
    ("A", ((0, "mem"),)),
    ("B", ((0, "mem"), (2, "mem"))),
    ("C", ((1, "ssd"),)),
    ("D", ((2, "ssd"),)),
)
GPU_BGS = (stable(0.6), stable(0.6), stable(0.2), stable(0.2))
SEED_LOCAL = ((0, "A"), (1, "B"))


def build_e5(seeds, duration=400.0):
    topo = mk_topo(replicas=REPLICAS, gpu_bgs=GPU_BGS, seed_local=SEED_LOCAL)
    jobs = []
    for pol in POLICIES:
        for seed in seeds:
            spec = mk_spec("e5", pol, seed, topo, duration=duration, lam=7.0, slo=0.6,
                           collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]),
                           out_dir=os.path.join(RESULTS_DIR, "e5") if seed == seeds[0] else None)
            jobs.append(({"policy": pol, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    plt = setup_matplotlib()
    agg = df.groupby("policy").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"), p99=("ttft_p99", "median"),
        local_used=("local_used_rate", "median"),
        w0u=("w0_util_mean", "median"), w2u=("w2_util_mean", "median"),
        fetch_gb=("fetch_gb", "median"), rct=("recompute_tokens", "median"),
    ).reindex(POLICIES)
    print("\n[E5] policy summary (median over seeds):")
    print(agg.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    x = np.arange(len(POLICIES))
    names = [LABELS[p] for p in POLICIES]
    for ax, col, color, title, ylab in [
            (axes[0], "goodput", "#4c72b0", "Goodput", "SLO goodput (req/s)"),
            (axes[1], "p95", "#c44e52", "TTFT P95", "TTFT P95 (s)"),
            (axes[2], "local_used", "#55a868", "Local-cache hit utilization", "used / available")]:
        ax.bar(x, agg[col], color=color)
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)
    savefig(plt, "fig_e5_routing.png")

    print("\n[E5] reading: locality-vs-load tension (w0/w1 busy hold A/B locally; w2/w3 idle):")
    for p in POLICIES:
        print(f"   {LABELS[p]:24s} goodput={agg.goodput[p]:.2f}  local_used="
              f"{(agg.local_used[p] if np.isfinite(agg.local_used[p]) else float('nan')):.2f}"
              f"  w0util={agg.w0u[p]:.2f} w2util={agg.w2u[p]:.2f}")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e5(seeds, duration=duration)
    print(f"[e5] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e5")
    analyze(pd.read_csv(out), seeds)
