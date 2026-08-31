"""负载生成：Poisson 到达 + 类别抽样 + burst。同一种子生成完全相同的 trace（CRN 基础）。"""
from __future__ import annotations

import numpy as np


def gen_trace(wl, duration: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    names = np.array([c.name for c in wl.classes])
    shares = np.array([c.share for c in wl.classes])
    trace = []
    t = float(rng.exponential(1.0 / wl.lam))
    while t < duration:
        hit = bool(rng.random() < wl.hit_ratio)
        cls = str(rng.choice(names, p=shares))
        trace.append((t, cls, hit))
        t += float(rng.exponential(1.0 / wl.lam))
    return trace


def gen_burst(burst, seed: int) -> list:
    t0, n, dur, cls = burst
    rng = np.random.default_rng(seed + 777)
    times = np.sort(rng.uniform(t0, t0 + dur, size=int(n)))
    return [(float(t), cls, True) for t in times]


def gen_full_trace(wl, duration: float, seed: int, burst) -> list:
    trace = gen_trace(wl, duration, seed)
    if burst is not None:
        trace = sorted(trace + gen_burst(burst, seed))
    return trace
