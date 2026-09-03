"""E9b（问题③）：E9 oracle "P99 更高但 SLO 率更高" 矛盾的诊断 + 先知上界。

假设：oracle 是"完美信息下的逐请求贪心"，看不到同步 burst 的后续到达，
早期请求各自选当时最快的 GPU 重算 → 车道效应堆高尾部。
先知上界（clairvoyant2）用未来视线做 list-scheduling 平衡分配：
若其显著优于 oracle2，则证明"逐请求局部最优 ≠ 全局最优"，协同有独立价值。
诊断面板：oracle2 在 burst 窗口内的动作分布与按动作分解的 TTFT。
"""
from __future__ import annotations

import os

from ..config import NodeConfig, ObsConfig, StorageConfig, stable
from .common import RESULTS_DIR, run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

POLICIES = ["joint2", "oracle2", "clairvoyant2"]
PLABELS = {"joint2": "Greedy(Ours)", "oracle2": "Oracle(per-req)", "clairvoyant2": "Clairvoyant"}
PCOLORS = {"joint2": "#4c72b0", "oracle2": "#cca64c", "clairvoyant2": "#55a868"}

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


def build_e9b(seeds, duration=400.0):
    topo = mk_topo(replicas=REPLICAS, nodes=NODES, local_cache_gb=0.0)
    jobs = []
    for pol in POLICIES:
        for seed in seeds:
            spec = mk_spec("e9b", pol, seed, topo, duration=duration, lam=5.0, slo=0.8,
                           obs=OBS, burst=BURST, window=WINDOW,
                           collect_ts=(seed == seeds[0]), save_ts=(seed == seeds[0]),
                           save_requests=True,
                           out_dir=os.path.join(RESULTS_DIR, "e9b") if seed == seeds[0] else None)
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
        frac_rc=("frac_recompute", "median"),
    ).reindex(POLICIES)
    agg["herd_peak"] = agg[["q0max", "q2max"]].max(axis=1)
    print("\n[E9b] policy summary (median over seeds):")
    print(agg.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    x = np.arange(len(POLICIES))
    names = [PLABELS[p] for p in POLICIES]
    for ax, col, color, title, ylab in [
            (axes[0], "win_slo", "#4c72b0", "Burst-window SLO rate", "SLO rate"),
            (axes[1], "win_p99", "#c44e52", "Burst-window TTFT P99", "TTFT P99 (s)"),
            (axes[2], "herd_peak", "#55a868", "Herd queue peak", "queue depth max")]:
        ax.bar(x, agg[col], color=color)
        for i, p in enumerate(POLICIES):
            ax.text(i, agg[col][p], f"{agg[col][p]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, names, rotation=15, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylab)

    # 诊断：oracle2 burst 窗口动作分布 + 按动作 TTFT（seed 0 请求级 CSV）
    s0 = seeds[0]
    path = os.path.join(RESULTS_DIR, "e9b", f"requests_oracle2_s{s0}.csv")
    diag_title = "oracle2: no per-request CSV"
    if os.path.exists(path):
        rq = pd.read_csv(path)
        m = (rq.t_arr >= WINDOW[0]) & (rq.t_arr < WINDOW[1]) & (rq.cls == "A")
        sub = rq[m]
        acts = sub.action.value_counts()
        diag_title = f"oracle2 A-class actions in burst (n={len(sub)})"
        print("\n[E9b] diagnosis:", diag_title)
        print(acts.to_string())
        print("[E9b] TTFT by action (median / p99):")
        for a, g in sub.groupby("action"):
            print(f"   {a:10s} n={len(g):4d}  med={g.ttft.median():.2f}s  p99={g.ttft.quantile(0.99):.2f}s")
        inset = fig.add_axes([0.68, 0.62, 0.20, 0.30])
        acts.reindex(["fetch", "recompute", "prefill", "partial", "local"]).dropna().plot(
            kind="bar", ax=inset, color="#8172b2", rot=0, fontsize=6)
        inset.set_title(diag_title, fontsize=6.5, pad=2)
        inset.set_ylabel("count", fontsize=6)
        inset.tick_params(labelsize=6)
        inset.set_xlabel("")
    savefig(plt, "fig_e9b_clairvoyant.png")

    g0 = agg.win_slo["clairvoyant2"]
    print("\n[E9b] burst-window SLO rate gap vs clairvoyant:")
    for p in POLICIES:
        print(f"   {PLABELS[p]:18s} {100 * (g0 - agg.win_slo[p]):.1f}pp   "
              f"win_p99={agg.win_p99[p]:.2f}s")


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    jobs = build_e9b(seeds, duration=duration)
    print(f"[e9b] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    out = save_rows(rows, "e9b")
    analyze(pd.read_csv(out), seeds)
