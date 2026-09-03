"""E17（问题⑩⑪）：会话身份级预测预取 + 互补性机制验证。

面板 A（问题⑩）：gated / predictive-class / predictive-session，混合速度会话。
  H10：predictive-session 浪费率 <= 0.5 x gated，goodput 不降。
面板 B（问题⑪）：引擎 {static2, static2dyn, joint2} x ctrl {none, predictive}，E13 场景。
  H11：static2dyn 的控制器效应 >= 0.7 x joint2 的控制器效应（互补性=副本可见性）。
"""
from __future__ import annotations

from ..config import (CtrlConfig, NodeConfig, PrefetchConfig, PrefixClass, StorageConfig,
                      stable, square)
from .common import run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

# ---------- 面板 A 场景（同 E15） ----------
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
VARS_A = ["gated", "predictive", "session"]
LABELS_A = {"gated": "Gated", "predictive": "Gated+Pred(class)", "session": "Gated+Pred(session)"}

# ---------- 面板 B 场景（同 E13） ----------
PRED = CtrlConfig(interval=0.5, hot_util=0.75, exit_util=0.65, hold_s=1.0,
                  min_demand=1.0, predictive=True, cross_tier=False)
NODES_B = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(30.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0, bg_schedule=stable(5.0)),
               ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0, bg_schedule=stable(5.0)),
               ssd=StorageConfig(b_total=15.0)),
)
CLASSES_B = (PrefixClass("A", 32768, 0.65), PrefixClass("B", 16384, 0.15),
             PrefixClass("C", 8192, 0.10), PrefixClass("D", 65536, 0.10))
ENGINES_B = ["static2", "static2dyn", "joint2"]


def build_e17a(seeds, duration=400.0):
    jobs = []
    for var in VARS_A:
        topo = mk_topo(replicas=REPLICAS_A, nodes=NODES_A, local_cache_gb=12.0,
                       gpu_bgs=(stable(0.5),) * 4,
                       prefetch=PrefetchConfig(mode=var, writeback=False))
        for seed in seeds:
            spec = mk_spec("e17a", "joint2", seed, topo, duration=duration, classes=CLASSES_A,
                           slo=0.35, lam=6.0, sessions=SESSIONS_A)
            jobs.append(({"panel": "a", "variant": var, "seed": seed}, spec))
    return jobs


def build_e17b(seeds, duration=400.0):
    jobs = []
    for eng in ENGINES_B:
        for ctl in ("none", "predictive"):
            replicas = (("A", ((0, "mem"),)), ("B", ((1, "mem"),)),
                        ("C", ((1, "ssd"),)), ("D", ((2, "ssd"),)))
            topo = mk_topo(replicas=replicas, nodes=NODES_B,
                           ctrl=PRED if ctl == "predictive" else None,
                           gpu_bgs=(stable(0.7),) * 4, local_cache_gb=0.0)
            for seed in seeds:
                spec = mk_spec("e17b", eng, seed, topo, duration=duration, classes=CLASSES_B,
                               slo=0.75, lam=8.0, hit_ratio=0.85)
                jobs.append(({"panel": "b", "engine": eng, "ctrl": ctl, "seed": seed}, spec))
    return jobs


def _panel_a(ax, df):
    import numpy as np
    agg = df.groupby("variant").agg(
        goodput=("goodput", "median"), pgb=("prefetch_gb", "median"),
        waste=("prefetch_waste_frac", "median")).reindex(VARS_A)
    print("\n[E17a] variant summary:")
    print(agg.round(3).to_string())
    if np.isfinite(agg.waste["gated"]) and np.isfinite(agg.waste["session"]):
        r = agg.waste["session"] / agg.waste["gated"]
        print(f"   waste ratio session/gated = {r:.2f} (H10 验收 <= 0.5)")
        print(f"   goodput diff session-gated = {agg.goodput['session'] - agg.goodput['gated']:+.3f}")
    x = np.arange(len(VARS_A))
    ax.bar(x - 0.2, agg.goodput, 0.4, color="#4c72b0", label="goodput")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, agg.waste, 0.4, color="#c44e52", label="waste frac")
    ax.set_xticks(x, [LABELS_A[v] for v in VARS_A], rotation=15, ha="right")
    ax.set_ylabel("SLO goodput (req/s)")
    ax2.set_ylabel("prefetch waste fraction", color="#c44e52")
    ax.set_title("A: session-level predictive prefetch", fontsize=9)


def _panel_b(ax, df):
    import numpy as np
    agg = df.groupby(["engine", "ctrl"]).goodput.median().reset_index()
    piv = agg.pivot(index="engine", columns="ctrl", values="goodput").reindex(ENGINES_B)
    eff = {e: piv.loc[e, "predictive"] - piv.loc[e, "none"] for e in ENGINES_B}
    print("\n[E17b] engine x ctrl goodput:")
    print(piv.round(3).to_string())
    for e in ENGINES_B:
        print(f"   controller effect {e:10s}: {eff[e]:+.3f}")
    if eff["joint2"]:
        print(f"   static2dyn/joint2 effect ratio = {eff['static2dyn'] / eff['joint2']:.2f} "
              f"(H11 验收 >= 0.7)")
    x = np.arange(len(ENGINES_B))
    w = 0.35
    for i, ctl in enumerate(("none", "predictive")):
        ax.bar(x + (i - 0.5) * w, [piv.loc[e, ctl] for e in ENGINES_B], w, label=f"ctrl={ctl}")
    ax.set_xticks(x, ENGINES_B)
    ax.set_ylabel("SLO goodput (req/s)")
    ax.set_title("B: replica-visibility mechanism check", fontsize=9)
    ax.legend(fontsize=8)


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    rows = run_pool(build_e17a(seeds, duration=duration), procs)
    dfa = pd.read_csv(save_rows(rows, "e17a"))
    rows = run_pool(build_e17b(seeds, duration=duration), procs)
    dfb = pd.read_csv(save_rows(rows, "e17b"))
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5), constrained_layout=True)
    _panel_a(axes[0], dfa)
    _panel_b(axes[1], dfb)
    savefig(plt, "fig_e17_session_mech.png")
