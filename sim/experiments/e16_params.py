"""E16（问题⑨）：参数档位敏感性——检索到的现实慢/快两档下策略排序是否稳健。

参数来源（2026-09-04 检索，见改进方案 II §参数来源）：
  slow: ssd 13 GB/s（LMCache 实测读峰下界）、fabric 87 GB/s（4×200Gb RoCE / Mooncake）、path_lat 减半
  fast: fabric 190 GB/s（8×400Gb RoCE / Mooncake）、ssd 25 GB/s
场景：E7/E12b 式单副本拥塞（bg=30），GPU bg 0.5，γ=1.1。
"""
from __future__ import annotations

from ..config import NodeConfig, ObsConfig, StorageConfig, stable
from .common import run_pool, save_rows, savefig, setup_matplotlib
from .v2common import mk_spec, mk_topo

PROFILES = {
    "slow": dict(ssd=13.0, fabric=87.0, lat_scale=0.5),
    "fast": dict(ssd=25.0, fabric=190.0, lat_scale=1.0),
}
POLICIES = ["alwaysfetch2", "static2", "cascade2", "joint2", "oracle2"]
LABELS = {"alwaysfetch2": "AlwaysFetch", "static2": "Static(AAFLOW+)",
          "cascade2": "Cascade-like", "joint2": "Joint(Ours)", "oracle2": "Oracle"}
COLORS = {"alwaysfetch2": "#c9c9c9", "static2": "#c44e52", "cascade2": "#ccb974",
          "joint2": "#4c72b0", "oracle2": "#cca64c"}


def _nodes(profile, bg=30.0):
    p = PROFILES[profile]
    return (NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(bg)),
                       ssd=StorageConfig(b_total=p["ssd"])),
            NodeConfig("n1", mem=StorageConfig(b_total=60.0),
                       ssd=StorageConfig(b_total=p["ssd"])),
            NodeConfig("n2", mem=StorageConfig(b_total=40.0),
                       ssd=StorageConfig(b_total=max(10.0, p["ssd"] * 0.6))))


def build_e16(seeds, duration=300.0):
    jobs = []
    for prof, p in PROFILES.items():
        lat = tuple(tuple(x * p["lat_scale"] for x in row) for row in
                    ((0.002, 0.004, 0.008), (0.003, 0.002, 0.006),
                     (0.006, 0.003, 0.002), (0.008, 0.004, 0.002)))
        from ..config import TopoConfig
        topo = TopoConfig(n_workers=4, nodes=_nodes(prof),
                          fabric=StorageConfig(b_total=p["fabric"]), path_lat=lat,
                          local_cache_gb=0.0,
                          replicas=(("A", ((0, "mem"),)), ("B", ((0, "mem"),)),
                                    ("C", ((0, "mem"),)), ("D", ((0, "mem"),))))
        for pol in POLICIES:
            for seed in seeds:
                obs = ObsConfig(gpu_interval=0.05, gpu_noise=0.1)
                spec = mk_spec("e16", pol, seed, topo, duration=duration, lam=8.0, slo=0.7,
                               gpu_bg=0.5, obs=obs, guardband=1.1)
                jobs.append(({"profile": prof, "policy": pol, "seed": seed}, spec))
    return jobs


def analyze(df, seeds):
    import numpy as np
    plt = setup_matplotlib()
    agg = df.groupby(["profile", "policy"]).goodput.median().reset_index()
    print("\n[E16] goodput by (profile, policy):")
    print(agg.pivot(index="policy", columns="profile", values="goodput")
          .reindex(POLICIES).round(3).to_string())
    for prof in PROFILES:
        sub = agg[agg.profile == prof].set_index("policy").goodput
        rank = sub.rank(ascending=False).astype(int)
        print(f"   {prof}: ranking " + " > ".join(sub.sort_values(ascending=False).index))
    # 排序一致性
    ranks = {prof: list(agg[agg.profile == prof].set_index("policy").goodput
                        .sort_values(ascending=False).index) for prof in PROFILES}
    same = ranks["slow"] == ranks["fast"]
    print(f"\n[E16] ranking invariance across profiles: {same}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), constrained_layout=True, sharey=True)
    x = np.arange(len(POLICIES))
    for ax, prof in zip(axes, PROFILES):
        sub = agg[agg.profile == prof].set_index("policy").goodput.reindex(POLICIES)
        ax.bar(x, sub.values, color=[COLORS[p] for p in POLICIES])
        for i, v in enumerate(sub.values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x, [LABELS[p] for p in POLICIES], rotation=30, ha="right")
        fb = PROFILES[prof]["fabric"]
        ax.set_title(f"{prof}: fabric {fb:.0f} GB/s, ssd {PROFILES[prof]['ssd']:.0f} GB/s", fontsize=9)
        ax.set_ylabel("SLO goodput (req/s)")
    savefig(plt, "fig_e16_params.png")


def main(seeds, procs=None, duration=300.0):
    import pandas as pd
    jobs = build_e16(seeds, duration=duration)
    print(f"[e16] {len(jobs)} runs ...")
    rows = run_pool(jobs, procs)
    analyze(pd.read_csv(save_rows(rows, "e16")), seeds)
