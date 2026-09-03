"""v2 实验（E5-E9）公共：拓扑构建、标签与配色。"""
from __future__ import annotations

from ..config import (CtrlConfig, GpuConfig, ModelConfig, NodeConfig, ObsConfig,
                      PolicyConfig, PrefixClass, RunSpec, StorageConfig, TopoConfig,
                      WorkloadConfig, square, stable)

# 通用节点：mem 带宽各异、ssd 更慢；容量用于容量压力状态
NODES = (
    NodeConfig("n0", mem=StorageConfig(b_total=60.0), ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n1", mem=StorageConfig(b_total=60.0), ssd=StorageConfig(b_total=25.0)),
    NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)),
)
FABRIC = StorageConfig(b_total=120.0)
PATH_LAT = ((0.002, 0.004, 0.008), (0.003, 0.002, 0.006), (0.006, 0.003, 0.002), (0.008, 0.004, 0.002))

LABELS = {
    "rr2": "RoundRobin", "load2": "Load-aware", "default2": "Default(Dynamo+vLLM)",
    "tensorcast2": "TensorCast-like", "static2": "Static(AAFLOW+)",
    "partial_static2": "PartialStatic(CacheFlow)", "joint2": "Joint(Ours)",
    "joint2_seq": "Joint-noOverlap", "coord2": "Coordinated(Ours+)",
    "nearest2": "Nearest-credit", "rrrep2": "RR-replica",
    "alwaysfetch2": "AlwaysFetch", "oracle2": "Oracle",
}
COLORS = {
    "rr2": "#937860", "load2": "#8172b2", "default2": "#8c8c8c", "tensorcast2": "#64b5cd",
    "static2": "#c44e52", "partial_static2": "#ccb974", "joint2": "#4c72b0",
    "joint2_seq": "#7f9fc4", "coord2": "#55a868", "nearest2": "#b5b5b5",
    "rrrep2": "#a1c9a1", "alwaysfetch2": "#c9c9c9", "oracle2": "#cca64c",
}


def offset_square(mean: float, amp: float, period: float, until: float, phase: float):
    """带初相的方波（phase 秒内取高电平，随后按 period/2 交替）。"""
    segs, t = [], 0.0
    hi = True
    while t < until + period:
        segs.append((t, mean + amp if hi else mean - amp))
        step = phase if hi and t == 0.0 else period / 2
        t += step
        hi = not hi
    return tuple(segs)


def mk_topo(replicas=(), gpu_bgs=(), ctrl=None, seed_local=(), nodes=NODES,
            fabric=FABRIC, local_cache_gb=12.0, prefetch=None, cache_mode="lru"):
    return TopoConfig(n_workers=4, nodes=nodes, fabric=fabric, path_lat=PATH_LAT,
                      local_cache_gb=local_cache_gb, replicas=replicas, ctrl=ctrl,
                      gpu_bgs=gpu_bgs, seed_local=seed_local,
                      prefetch=prefetch, cache_mode=cache_mode)


def mk_spec(exp, policy, seed, topo, duration=400.0, lam=5.0, classes=None,
            obs=None, margin=0.10, slo=1.0, gpu_bg=0.0, hit_ratio=None, burst=None, guardband=1.2,
            sessions=None,
            window=None, collect_ts=False, save_ts=False, save_requests=False, out_dir=None):
    kw = dict(lam=lam, ttft_slo=slo)
    if hit_ratio is not None:
        kw["hit_ratio"] = hit_ratio
    if classes:
        kw["classes"] = classes
    wl = WorkloadConfig(**kw)
    return RunSpec(
        exp=exp, policy=policy, seed=seed, duration=duration, warmup=20.0, margin=60.0,
        topo=topo, gpu=GpuConfig(bg_schedule=stable(gpu_bg)), wl=wl,
        obs=obs if obs is not None else ObsConfig(),
        pol=PolicyConfig(margin=margin, guardband=guardband),
        burst=burst, window=window, sessions=sessions,
        collect_ts=collect_ts, save_ts=save_ts, save_requests=save_requests,
        out_dir=out_dir,
    )
