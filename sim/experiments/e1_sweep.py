"""E1：动态存储压力 × GPU 余量（go/no-go）。

E1a stable 背景网格；E1b 方波背景（mean±amp, period=60s）。
主对比：P2(dynamic) vs P1(static) 的 goodput 增益热图。
"""
from __future__ import annotations

from ..config import GpuConfig, RunSpec, StorageConfig, square, stable
from .common import run_pool, save_rows, savefig, setup_matplotlib

BG_LEVELS = [0, 20, 40, 60, 80, 100]
GPU_LEVELS = [0.2, 0.4, 0.6, 0.8, 0.9]
POLICIES = ["p0", "p1", "p2", "p4"]
FG_GBPS = None  # 由 build 时填充


def _fg(spec: RunSpec) -> float:
    return spec.fg_demand_gbps()


def build_e1a(seeds, duration=400.0):
    jobs = []
    for bg in BG_LEVELS:
        for gpu in GPU_LEVELS:
            for pol in POLICIES:
                for seed in seeds:
                    spec = RunSpec(
                        exp="e1a", policy=pol, seed=seed, duration=duration,
                        warmup=20.0, margin=60.0,
                        storages=(StorageConfig(bg_schedule=stable(float(bg))),),
                        gpu=GpuConfig(bg_schedule=stable(gpu)),
                    )
                    jobs.append(({"bg": bg, "gpu": gpu, "policy": pol, "seed": seed}, spec))
    return jobs


def build_e1b(seeds, duration=400.0, amp=40.0, period=60.0):
    jobs = []
    for bgm in [40, 60, 80]:
        for gpu in [0.2, 0.4, 0.6]:
            for pol in POLICIES:
                for seed in seeds:
                    spec = RunSpec(
                        exp="e1b", policy=pol, seed=seed, duration=duration,
                        warmup=20.0, margin=60.0,
                        storages=(StorageConfig(bg_schedule=square(float(bgm), amp, period, duration + 120)),),
                        gpu=GpuConfig(bg_schedule=stable(gpu)),
                    )
                    jobs.append(({"bg": bgm, "gpu": gpu, "policy": pol, "seed": seed,
                                  "amp": amp, "period": period}, spec))
    return jobs


def _gain_grid(df, numerator: str, base: str = "p1", value: str = "goodput"):
    """返回 (gpu_levels, offered_levels, gain_matrix%)，每格取种子中位数。"""
    import numpy as np
    fg = 39.0  # 与 WorkloadConfig 默认值一致的前台需求（GB/s），仅用于横轴标注
    bgs = sorted(df["bg"].unique())
    gpus = sorted(df["gpu"].unique(), reverse=True)
    g = np.full((len(gpus), len(bgs)), np.nan)
    for i, gp in enumerate(gpus):
        for j, bg in enumerate(bgs):
            sub = df[(df.gpu == gp) & (df.bg == bg)]
            a = sub[sub.policy == numerator][value]
            b = sub[sub.policy == base][value]
            if len(a) and len(b):
                g[i, j] = (float(np.median(a)) / float(np.median(b)) - 1.0) * 100.0
    offered = [round(bg + fg) for bg in bgs]
    return gpus, offered, g


def analyze(df, exp: str, fig_name: str, title_prefix: str):
    import numpy as np
    plt = setup_matplotlib()
    gpus, offered, g_p2 = _gain_grid(df, "p2")
    _, _, g_p4 = _gain_grid(df, "p4")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    for ax, g, title in zip(axes, [g_p2, g_p4],
                            [f"{title_prefix}: Dynamic(P2) vs Static(P1)",
                             f"{title_prefix}: Oracle(P4) vs Static(P1)"]):
        vmax = min(120.0, np.nanmax(g) if np.isfinite(np.nanmax(g)) else 100)
        im = ax.imshow(g, cmap="RdYlGn", vmin=-30, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(offered)), [f"{o}%" for o in offered])
        ax.set_yticks(range(len(gpus)), [f"{int(gp*100)}%" for gp in gpus])
        ax.set_xlabel("Storage offered load")
        ax.set_ylabel("GPU background load")
        ax.set_title(title)
        ax.grid(False)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                if np.isfinite(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Goodput gain %", shrink=0.85)
    savefig(plt, fig_name)

    # go/no-go 判读：P2/P1 增益 ≥10% 的格子
    ok = [(gpus[i], offered[j], g_p2[i, j]) for i in range(len(gpus)) for j in range(len(offered))
          if np.isfinite(g_p2[i, j]) and g_p2[i, j] >= 10]
    print(f"[{exp}] cells with P2/P1 gain >= 10%: {len(ok)}/{g_p2.size}")
    for gp, off, gv in sorted(ok, key=lambda x: -x[2])[:12]:
        print(f"   gpu_bg={gp:.0%} offered={off}% gain={gv:.0f}%")
    return g_p2


def main(seeds, procs=None, duration=400.0, exps=("e1a", "e1b")):
    import pandas as pd
    for exp in exps:
        builder = build_e1a if exp == "e1a" else build_e1b
        jobs = builder(seeds, duration=duration)
        print(f"[{exp}] {len(jobs)} runs ...")
        rows = run_pool(jobs, procs)
        out = save_rows(rows, exp)
        df = pd.read_csv(out)
        analyze(df, exp, f"fig_{'a' if exp == 'e1a' else 'e1b'}_{exp}_heatmap.png",
                "E1a stable bg" if exp == "e1a" else "E1b square-wave bg")
