"""E18（问题⑫⑬）：预取条目缓存保护/准入 + 流体最优批分配。

面板 A（⑫）：gated / session（E17a 复检）/ protect / session+protect，混合速度会话。
  H12：protect 使浪费率 <= 0.5 x gated 且 goodput 不降。
面板 B（⑬）：E9b 场景下 joint2 / oracle2 / clairvoyant2(list) / clairfluid2。
  H13：fluid 的 win_slo >= oracle2 - 1pp 且 win_p99 < oracle2。
"""
from __future__ import annotations

from ..config import (NodeConfig, ObsConfig, PrefetchConfig, PrefixClass, StorageConfig,
                      stable, square)
from .common import run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

# ---------- 面板 A 场景（同 E15/E17a） ----------
NODES_A = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=square(35.0, 20.0, 40.0, 600.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0), ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)),
)
REPLICAS_A = (("A", ((0, "mem"),)), ("B", ((0, "mem"),)), ("C", ((0, "mem"),)), ("D", ((2, "ssd"),)))
CLASSES_A = (PrefixClass("A", 32768, 0.40), PrefixClass("B", 16384, 0.20),
             PrefixClass("C", 8192, 0.20), PrefixClass("D", 65536, 0.20))
SESSIONS_A = (1.6, 4.0, (0.6, 8.0))
VARS_A = ["gated", "session", "protect", "session_protect"]
LABELS_A = {"gated": "Gated", "session": "+Pred(session)", "protect": "+Protect",
            "session_protect": "+Pred+Protect"}

# ---------- 面板 B 场景（同 E9b） ----------
NODES_B = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(35.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0, bg_schedule=stable(35.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)),
)
REPLICAS_B = (("A", ((0, "mem"), (1, "mem"))), ("B", ((0, "mem"),)),
              ("C", ((1, "ssd"),)), ("D", ((2, "ssd"),)))
OBS_B = ObsConfig(interval=0.2, noise_sigma=0.1, signal="quote")
BURST_B = (120.0, 150, 1.0, "A")
WINDOW_B = (120.0, 220.0)
POL_B = ["joint2", "oracle2", "clairvoyant2", "clairfluid2"]
LABEL_B = {"joint2": "Greedy", "oracle2": "Oracle(per-req)", "clairvoyant2": "Clair(list)",
           "clairfluid2": "Clair(fluid-opt)"}
COLOR_B = {"joint2": "#4c72b0", "oracle2": "#cca64c", "clairvoyant2": "#8172b2",
           "clairfluid2": "#55a868"}


def build_e18a(seeds, duration=400.0):
    jobs = []
    for var in VARS_A:
        mode = "session" if var.startswith("session") else "gated"
        topo = mk_topo(replicas=REPLICAS_A, nodes=NODES_A, local_cache_gb=12.0,
                       gpu_bgs=(stable(0.5),) * 4,
                       prefetch=PrefetchConfig(mode=mode, writeback=False,
                                               protect="protect" in var))
        for seed in seeds:
            spec = mk_spec("e18a", "joint2", seed, topo, duration=duration, classes=CLASSES_A,
                           slo=0.35, lam=6.0, sessions=SESSIONS_A)
            jobs.append(({"panel": "a", "variant": var, "seed": seed}, spec))
    return jobs


def build_e18b(seeds, duration=400.0):
    topo = mk_topo(replicas=REPLICAS_B, nodes=NODES_B, local_cache_gb=0.0)
    jobs = []
    for pol in POL_B:
        for seed in seeds:
            spec = mk_spec("e18b", pol, seed, topo, duration=duration, lam=5.0, slo=0.8,
                           obs=OBS_B, burst=BURST_B, window=WINDOW_B)
            jobs.append(({"panel": "b", "policy": pol, "seed": seed}, spec))
    return jobs


def _panel_a(ax, df):
    import numpy as np
    agg = df.groupby("variant").agg(
        goodput=("goodput", "median"), p95=("ttft_p95", "median"),
        pgb=("prefetch_gb", "median"), wgb=("prefetch_wasted_gb", "median"),
        waste=("prefetch_waste_frac", "median")).reindex(VARS_A)
    print("\n[E18a] variant summary:")
    print(agg.round(3).to_string())
    r = agg.waste["protect"] / agg.waste["gated"] if agg.waste["gated"] else float("nan")
    print(f"   waste ratio protect/gated = {r:.2f} (H12 验收 <= 0.5)")
    print(f"   goodput diff protect-gated = {agg.goodput['protect'] - agg.goodput['gated']:+.3f}")
    x = np.arange(len(VARS_A))
    ax.bar(x - 0.22, agg.goodput, 0.44, color="#4c72b0", label="goodput")
    axb = ax.twinx()
    axb.bar(x + 0.22, agg.waste, 0.44, color="#c44e52", label="waste frac")
    ax.set_xticks(x, [LABELS_A[v] for v in VARS_A], rotation=20, ha="right")
    ax.set_ylabel("SLO goodput (req/s)")
    axb.set_ylabel("prefetch waste fraction", color="#c44e52")
    ax.set_title("A: cache protection & admission", fontsize=9)


def _panel_b(ax, df):
    import numpy as np
    agg = df.groupby("policy").agg(
        win_slo=("win_slo_rate", "median"), win_p99=("win_ttft_p99", "median"),
        q0max=("s0_q_max", "median"), q2max=("s2_q_max", "median")).reindex(POL_B)
    agg["herd"] = agg[["q0max", "q2max"]].max(axis=1)
    print("\n[E18b] policy summary:")
    print(agg.round(3).to_string())
    print(f"   fluid vs oracle: dSLO={100 * (agg.win_slo['clairfluid2'] - agg.win_slo['oracle2']):+.1f}pp, "
          f"dp99={agg.win_p99['clairfluid2'] - agg.win_p99['oracle2']:+.2f}s")
    x = np.arange(len(POL_B))
    for i, p in enumerate(POL_B):
        ax.bar(i, agg.win_slo[p], color=COLOR_B[p])
        ax.text(i, agg.win_slo[p], f"{agg.win_slo[p]:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [LABEL_B[p] for p in POL_B], rotation=15, ha="right")
    ax.set_ylabel("burst-window SLO rate")
    ax.set_title("B: fluid-optimal batch assignment", fontsize=9)


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    rows = run_pool(build_e18a(seeds, duration=duration), procs)
    dfa = pd.read_csv(save_rows(rows, "e18a"))
    rows = run_pool(build_e18b(seeds, duration=duration), procs)
    dfb = pd.read_csv(save_rows(rows, "e18b"))
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5), constrained_layout=True)
    _panel_a(axes[0], dfa)
    _panel_b(axes[1], dfb)
    savefig(plt, "fig_e18_protect_fluid.png")
