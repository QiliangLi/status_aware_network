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


def gen_session_trace(wl, duration: float, seed: int, sessions) -> list:
    """会话型负载：会话 Poisson 到达，会话内多轮复用同一前缀（首轮 miss，后续 hit）。"""
    srate, mean_turns, gap_mean = sessions
    rng = np.random.default_rng(seed + 555)
    names = np.array([c.name for c in wl.classes])
    shares = np.array([c.share for c in wl.classes])
    trace = []
    t = float(rng.exponential(1.0 / srate))
    while t < duration:
        cls = str(rng.choice(names, p=shares))
        turns = min(20, 1 + int(rng.geometric(1.0 / max(1.0, mean_turns))))
        tt = t
        for k in range(turns):
            if tt >= duration:
                break
            trace.append((tt, cls, k > 0))
            tt += float(rng.exponential(gap_mean))
        t += float(rng.exponential(1.0 / srate))
    return sorted(trace)


def gen_full_trace_session(wl, duration: float, seed: int, burst, sessions) -> list:
    trace = gen_session_trace(wl, duration, seed, sessions)
    if burst is not None:
        trace = sorted(trace + gen_burst(burst, seed))
    return trace


def gen_drift_trace(wl, duration: float, seed: int, drift) -> list:
    """热度漂移负载：每 period 秒将类份额向量轮转一位（Zipf 式冷热轮换）。"""
    period = drift[0]
    rng = np.random.default_rng(seed + 888)
    names = np.array([c.name for c in wl.classes])
    shares = np.array([c.share for c in wl.classes])
    trace = []
    t = float(rng.exponential(1.0 / wl.lam))
    while t < duration:
        phase = int(t // period) % len(shares)
        w = np.roll(shares, phase)
        cls = str(rng.choice(names, p=w / w.sum()))
        hit = bool(rng.random() < wl.hit_ratio)
        trace.append((t, cls, hit))
        t += float(rng.exponential(1.0 / wl.lam))
    return trace
