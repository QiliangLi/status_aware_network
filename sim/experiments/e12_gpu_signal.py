"""E12（问题⑤）：GPU 侧预测误差层 + cascade2 基线对等比较。

面板 A（稳健性）：joint2 相对 static2 的 goodput 增益，随 GPU 估计噪声 σ × 陈旧度变化。
面板 B（对等比较）：cascade2 vs joint2（双方读同样的带误差 GPU + 存储 quote），
  差异只在 request 级预算/恢复决策 vs 全局联合调度；存储压力三档扫。
场景沿用 E7：全部副本集中在 (n0,mem)，动态背景；本地缓存关闭以隔离。
"""
from __future__ import annotations

from ..config import NodeConfig, ObsConfig, StorageConfig, stable
from .common import run_pool, save_rows, savefig, setup_matplotlib
from .v2common import LABELS, mk_spec, mk_topo

SIGMAS = [0.0, 0.1, 0.3]
INTERVALS_MS = [0, 50, 200]
BGS = [10.0, 30.0, 50.0]
POL_B = ["default2", "cascade2", "joint2", "oracle2"]
LABEL_B = {"default2": "Default(Dynamo+vLLM)", "cascade2": "Cascade-like",
           "joint2": "Joint(Ours)", "oracle2": "Oracle"}
COLOR_B = {"default2": "#8c8c8c", "cascade2": "#ccb974", "joint2": "#4c72b0", "oracle2": "#cca64c"}


def _nodes(bg: float):
    return (NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(bg)),
                       ssd=StorageConfig(b_total=25.0)),
            NodeConfig("n1", mem=StorageConfig(b_total=60.0), ssd=StorageConfig(b_total=25.0)),
            NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)))


REPLICAS = (("A", ((0, "mem"),)), ("B", ((0, "mem"),)), ("C", ((0, "mem"),)), ("D", ((0, "mem"),)))


def build_e12a(seeds, bg=45.0, duration=300.0):
    topo = mk_topo(replicas=REPLICAS, nodes=_nodes(bg), local_cache_gb=0.0)
    jobs = []
    for sigma in SIGMAS:
        for iv in INTERVALS_MS:
            for pol in ("joint2", "static2"):
                for seed in seeds:
                    obs = ObsConfig(gpu_interval=iv / 1000.0, gpu_noise=sigma)
                    spec = mk_spec("e12a", pol, seed, topo, duration=duration, lam=8.0, slo=0.6,
                                   gpu_bg=0.4, obs=obs)
                    jobs.append(({"policy": pol, "seed": seed, "sigma": sigma, "interval_ms": iv}, spec))
    return jobs


def build_e12b(seeds, duration=300.0):
    jobs = []
    for bg in BGS:
        topo = mk_topo(replicas=REPLICAS, nodes=_nodes(bg), local_cache_gb=0.0)
        for pol in POL_B:
            for seed in seeds:
                obs = ObsConfig(gpu_interval=0.05, gpu_noise=0.1)   # 对等：双方都带 GPU 误差
                spec = mk_spec("e12b", pol, seed, topo, duration=duration, lam=8.0, slo=0.7,
                               gpu_bg=0.5, obs=obs, guardband=1.1)
                jobs.append(({"policy": pol, "seed": seed, "bg": bg}, spec))
    return jobs


def _panel_a(ax, df):
    import numpy as np
    sub = df.groupby(["sigma", "interval_ms", "policy"]).goodput.median().reset_index()
    gains = {}
    for iv in INTERVALS_MS:
        for sigma in SIGMAS:
            a = sub[(sub.interval_ms == iv) & (sub.sigma == sigma) & (sub.policy == "joint2")].goodput
            b = sub[(sub.interval_ms == iv) & (sub.sigma == sigma) & (sub.policy == "static2")].goodput
            if len(a) and len(b):
                gains[(sigma, iv)] = (float(a.iloc[0]) / float(b.iloc[0]) - 1.0) * 100.0
    colors = {0: "#55a868", 50: "#4c72b0", 200: "#c44e52"}
    for iv in INTERVALS_MS:
        xs = [s for s in SIGMAS if (s, iv) in gains]
        ys = [gains[(s, iv)] for s in xs]
        ax.plot(xs, ys, marker="o", color=colors[iv], label=f"GPU staleness {iv} ms")
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.0f}%", fontsize=7, ha="left")
    base = gains.get((0.0, 0))
    if base is not None:
        ax.axhline(0.7 * base, color="0.4", lw=0.8, ls="--")
        ax.text(0.30, 0.7 * base, "70% retention", fontsize=6.5, va="bottom")
    ax.set_xticks(SIGMAS, [str(s) for s in SIGMAS])
    ax.set_xlabel("GPU estimate noise σ")
    ax.set_ylabel("goodput gain vs Static (%)")
    ax.set_title("A: joint2 robustness to GPU estimate error (bg=45)", fontsize=9)
    ax.legend(fontsize=7)
    print("\n[E12a] joint2 vs static2 gain (%) by (σ, staleness):")
    for (s, iv), g in sorted(gains.items()):
        print(f"   σ={s:.1f} stale={iv:3d}ms  gain={g:+.1f}%")
    return gains


def _panel_b(ax, df):
    import numpy as np
    agg = df.groupby(["bg", "policy"]).goodput.median().reset_index()
    width = 0.2
    xs = np.arange(len(BGS))
    for i, pol in enumerate(POL_B):
        ys = [agg[(agg.bg == bg) & (agg.policy == pol)].goodput.iloc[0] for bg in BGS]
        ax.bar(xs + (i - 1.5) * width, ys, width, color=COLOR_B[pol], label=LABEL_B[pol])
    ax.set_xticks(xs, [f"bg={int(b)}" for b in BGS])
    ax.set_ylabel("SLO goodput (req/s)")
    ax.set_title("B: request-level (Cascade-like) vs joint (Ours) — equal GPU/storage error", fontsize=9)
    ax.legend(fontsize=7)
    print("\n[E12b] head-to-head goodput:")
    for bg in BGS:
        row = {p: agg[(agg.bg == bg) & (agg.policy == p)].goodput.iloc[0] for p in POL_B}
        gap = (row["joint2"] / row["cascade2"] - 1.0) * 100 if row["cascade2"] else float("nan")
        print(f"   bg={int(bg):2d}  " + "  ".join(f"{LABEL_B[p]}={row[p]:.2f}" for p in POL_B)
              + f"  joint/cascade={gap:+.1f}%")


def main(seeds, procs=None, duration=300.0):
    import pandas as pd
    print(f"[e12a] {len(build_e12a(seeds[:2]))} pattern runs ...")
    rows = run_pool(build_e12a(seeds, duration=duration), procs)
    dfa = pd.read_csv(save_rows(rows, "e12a"))
    rows = run_pool(build_e12b(seeds, duration=duration), procs)
    dfb = pd.read_csv(save_rows(rows, "e12b"))

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), constrained_layout=True)
    _panel_a(axes[0], dfa)
    _panel_b(axes[1], dfb)
    savefig(plt, "fig_e12_gpu_signal.png")
