"""E3：存储侧信号设计——信号类型 × 新鲜度 × 噪声，哪个组合足够换来收益。

两个格点：
  hi  = bg 80 GB/s（offered≈119%，深度过载——压力巨大且持续）
  mid = bg 60 GB/s（offered≈99%，临界拥塞——动态变化更依赖及时感知）
"""
from __future__ import annotations

from ..config import GpuConfig, ObsConfig, RunSpec, StorageConfig, stable
from .common import run_pool, save_rows, savefig, setup_matplotlib

GPU = 0.4
SIGNALS = ["util", "queue", "bw", "quote"]
INTERVALS_MS = [0, 10, 50, 100, 200, 500, 1000]
SIGMAS = [0.0, 0.1, 0.3]
REF = ["p1", "p4"]


def build_e3(seeds, bg: float, duration=400.0):
    jobs = []
    for pol in REF:
        for seed in seeds:
            spec = RunSpec(exp="e3", policy=pol, seed=seed, duration=duration,
                           warmup=20.0, margin=60.0,
                           storages=(StorageConfig(bg_schedule=stable(bg)),),
                           gpu=GpuConfig(bg_schedule=stable(GPU)))
            jobs.append(({"policy": pol, "seed": seed, "signal": "", "interval_ms": -1, "sigma": -1}, spec))
    for sig in SIGNALS:
        for iv in INTERVALS_MS:
            for sigma in SIGMAS:
                for seed in seeds:
                    spec = RunSpec(
                        exp="e3", policy="p2", seed=seed, duration=duration, warmup=20.0, margin=60.0,
                        storages=(StorageConfig(bg_schedule=stable(bg)),),
                        gpu=GpuConfig(bg_schedule=stable(GPU)),
                        obs=ObsConfig(interval=iv / 1000.0, noise_sigma=sigma, signal=sig),
                    )
                    jobs.append(({"policy": "p2", "seed": seed, "signal": sig,
                                  "interval_ms": iv, "sigma": sigma}, spec))
    return jobs


def _panel(ax, df, title):
    import numpy as np
    g1 = float(np.median(df[df.policy == "p1"].goodput))
    g4 = float(np.median(df[df.policy == "p4"].goodput))
    sub = df[df.policy == "p2"].copy()
    sub["gain"] = (sub.goodput / g1 - 1.0) * 100.0
    pivot = sub.groupby(["signal", "sigma", "interval_ms"]).gain.median().reset_index()

    colors = {"util": "#8172b2", "queue": "#937860", "bw": "#55a868", "quote": "#4c72b0"}
    lss = {0.0: "-", 0.1: "--", 0.3: ":"}
    for sig in SIGNALS:
        for sigma in SIGMAS:
            d = pivot[(pivot.signal == sig) & (pivot.sigma == sigma)].sort_values("interval_ms")
            if len(d) == 0:
                continue
            show_label = (sig in ("quote", "queue") and sigma != 0.3) or (sig in ("util", "bw") and sigma == 0.1)
            ax.plot(d.interval_ms, d.gain, color=colors[sig], ls=lss[sigma], marker="o", ms=3,
                    label=f"{sig} σ={sigma}" if show_label else None)
    ax.axhline(10, color="0.4", lw=0.8, ls="-.")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(INTERVALS_MS, [str(x) for x in INTERVALS_MS])
    ax.set_xlabel("State staleness (ms) — 0 = live")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    print(f"\n[E3 {title}] P1={g1:.2f} P4={g4:.2f} (oracle headroom {(g4 / g1 - 1) * 100:.0f}%)")
    print(f"[E3 {title}] max staleness retaining >=90% of peak gain (sigma=0.1):")
    for sig in SIGNALS:
        d = pivot[(pivot.signal == sig) & (pivot.sigma == 0.1)].sort_values("interval_ms")
        if not len(d):
            continue
        peak = d.gain.max()
        ok = d[d.gain >= 0.9 * peak]
        worst = d[d.interval_ms == d.interval_ms.max()].gain.iloc[0]
        print(f"   {sig:6s} peak={peak:6.1f}%  up_to={int(ok.interval_ms.max()):4d} ms  @1000ms={worst:6.1f}%")
    return pivot


def analyze(dfs, titles):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, len(dfs), figsize=(11, 3.8), constrained_layout=True, sharey=True)
    if len(dfs) == 1:
        axes = [axes]
    pivots = []
    for ax, df, title in zip(axes, dfs, titles):
        pivots.append(_panel(ax, df, title))
    axes[0].set_ylabel("Goodput gain vs P1 (%)")
    fig.suptitle("E3 signal design: gain vs state staleness (P2 dynamic, GPU bg=40%)", fontsize=10)
    savefig(plt, "fig_d_e3_signal.png")
    return pivots


def main(seeds, procs=None, duration=400.0):
    import pandas as pd
    dfs, titles = [], []
    for bg, tag in [(80.0, "hi: offered~119% (deep oversubscription)"),
                    (60.0, "mid: offered~99% (borderline)")]:
        jobs = build_e3(seeds, bg, duration=duration)
        print(f"[e3 bg={bg:.0f}] {len(jobs)} runs ...")
        rows = run_pool(jobs, procs)
        out = save_rows(rows, f"e3_bg{int(bg)}")
        dfs.append(pd.read_csv(out))
        titles.append(f"storage bg={bg:.0f} GB/s — {tag}")
    analyze(dfs, titles)
