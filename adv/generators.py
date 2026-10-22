"""攻击方生成器库（14 个 + 变异元生成器）。

设计出处：攻击方 Agent 20261022 清单（G1~G14），类常量已按仓库公式实算。
统一返回 (params, payload)：
- 单层 payload = {"kind": "single", "specs": [...], "B": b, "b_ref": br, "guard_w": gw}
- 两层 payload = {"kind": "two", "trace": [(t, cls)...], "classes": {...}, "two_kwargs": {...}}
"""
from __future__ import annotations

import random
from fractions import Fraction as F
from typing import Dict, List, Tuple

from sim.cq.types import RequestSpec
from sim.experiments.e26_ab import E26_PROFILE, make_T0

# ---------------------------------------------------------------------------
# 类常量（攻击方清单 §0，实算验证过）
# ---------------------------------------------------------------------------
CLS = {
    "A_std": (42883, 256),    # d=29.18, T0≈49.62ms
    "B_std": (9399, 4096),    # d=1.35,  T0≈229.04ms
    "C_A":   (0, 2657),       # 零读短算 T0≈48.17ms
    "C_B":   (0, 8110),       # 零读长算 T0≈228.73ms
    "H65":   (53940, 100),    # d=64.94（门槛毒丸）
    "L_vic": (2048, 2657),    # d=0.98 轻读短算受害者
    "Bf":    (150000, 2048),  # d=4.16 超长背景流
}


def _poisson_clumps(rng, lam: float, n: int, k: int, t0: float = 0.0) -> List[float]:
    """簇泊松：每簇 k 条同刻，簇间隔 Exp(k/lam)。"""
    times, t = [], t0
    while len(times) < n:
        t += rng.expovariate(lam / k)
        times.extend([t] * min(k, n - len(times)))
    return times


def _poisson(rng, lam: float, n: int, t0: float = 0.0) -> List[float]:
    times, t = [], t0
    while len(times) < n:
        t += rng.expovariate(lam)
        times.append(t)
    return times


def build_specs(events: List[Tuple[float, str]], dl: Dict[str, dict],
                b_ref: float = 120.0) -> List[RequestSpec]:
    """events=[(t, cls)]；dl[cls]={'alpha':x} 或 {'slack':s}（deadline=arr+T0+s）。"""
    events = sorted(events, key=lambda e: (e[0], e[1]))
    proto = [RequestSpec(i, float(t), *CLS[c], c, F(1), F(1))
             for i, (t, c) in enumerate(events)]
    T0 = make_T0(proto, E26_PROFILE, F(int(b_ref)), F(200))
    out = []
    for s in proto:
        d = dl[s.class_id]
        if "alpha" in d:
            deadline = float(s.arrival_s) + d["alpha"] * float(T0[s.rid])
        else:
            deadline = float(s.arrival_s) + float(T0[s.rid]) + d["slack"]
        out.append(RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens,
                               s.class_id, T0[s.rid], deadline))
    return out


def _p(rng, lo, hi):
    return rng.uniform(lo, hi)


def _c(rng, seq):
    return rng.choice(seq)


# ---------------------------------------------------------------------------
# 单层生成器
# ---------------------------------------------------------------------------

def g1_martyr_heavy(rng):
    """G1 零读 martyr 流饿死重流（主杀）。"""
    lam_s = _c(rng, [120, 125, 130, 135]); h_s = _c(rng, [150, 180, 201, 220])
    av = _c(rng, [5, 6, 7]); k = _c(rng, [4, 8])
    n_s, dur = int(lam_s * 3), 3.0
    ev = [(t, "S") for t in _poisson(rng, lam_s, n_s)]
    for tc in _poisson_clumps(rng, 13, 40, k):
        ev.append((tc, "V"))
    # S 用 C_B（零读长算），V 用 A_std
    CLS["S"], CLS["V"] = CLS["C_B"], CLS["A_std"]
    specs = build_specs(ev, {"S": {"slack": h_s}, "V": {"alpha": av}})
    return {"gen": "G1", "lam_s": lam_s, "h_s": h_s, "alpha_v": av, "k": k}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120,
            "guard_w": _c(rng, [0, 0.05, 0.2])}


def g3_zero_victim(rng):
    """G3 零读受害者：带宽全程空闲的纯排序饿死（不变量最强形态）。"""
    lam_s = _c(rng, [120, 125, 130, 135]); h_s = _c(rng, [80, 100, 120])
    k = _c(rng, [4, 8, 16])
    ev = [(t, "S") for t in _poisson(rng, lam_s, int(lam_s * 3))]
    for tc in _poisson_clumps(rng, 16, 48, k):
        ev.append((tc, "V"))
    CLS["S"], CLS["V"] = CLS["C_B"], CLS["C_A"]
    specs = build_specs(ev, {"S": {"slack": h_s}, "V": {"alpha": 4}})
    return {"gen": "G3", "lam_s": lam_s, "h_s": h_s, "k": k}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120,
            "guard_w": _c(rng, [0, 0.05, 0.2])}


def g14_compute_blind(rng):
    """G14 计算瓶颈侧：轻读短算受害者被 martyr worker 饱和杀死。"""
    lam_s = _c(rng, [120, 125, 130, 135]); h_s = _c(rng, [120, 151, 180])
    k = _c(rng, [4, 8])
    ev = [(t, "S") for t in _poisson(rng, lam_s, int(lam_s * 3))]
    for tc in _poisson_clumps(rng, 20, 40, k):
        ev.append((tc, "V"))
    CLS["S"], CLS["V"] = CLS["C_B"], CLS["L_vic"]
    specs = build_specs(ev, {"S": {"slack": h_s}, "V": {"alpha": 4}})
    return {"gen": "G14", "lam_s": lam_s, "h_s": h_s, "k": k}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def g2_p75_cliff(rng):
    """G2 门槛贴边振荡：C 在 1↔4 跳档。"""
    phase = _c(rng, [0.2, 0.3, 0.5]); lam_m = _c(rng, [50, 60, 70])
    bias = _c(rng, [-0.008, -0.004, 0.004, 0.008])
    ev = [(0.01 * i, "H") for i in range(40)]        # 预热钉 P75
    t, dur, flip = 0.5, 3.0, True
    while t < dur:
        lam = lam_m if flip else 20
        n = int(lam * phase)
        for x in _poisson(rng, lam, n, t0=t):
            ev.append((x, "M" if flip else "H"))
        t += phase
        flip = not flip
    if bias:  # 占比偏置：追加一段
        for x in _poisson(rng, lam_m * abs(bias) * 100, 20, t0=t):
            ev.append((x, "H" if bias < 0 else "M"))
    CLS["M"], CLS["H"] = CLS["A_std"], CLS["H65"]
    specs = build_specs(ev, {"M": {"alpha": 4}, "H": {"alpha": 10}})
    return {"gen": "G2", "phase": phase, "lam_m": lam_m, "bias": bias}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def g5_cold_poison(rng):
    """G5 冷启动门槛毒丸：开局 H65 把 P75 钉死 → C=1 锁死紧期限 A 簇。"""
    n_h = _c(rng, [20, 40, 80]); k = _c(rng, [4, 5, 8]); av = _c(rng, [2.5, 3, 3.5])
    ev = [(0.5 / n_h * i, "H") for i in range(n_h)]
    for tc in _poisson_clumps(rng, 33, 100, k, t0=1.2):
        ev.append((tc, "A"))
    CLS["H"], CLS["A"] = CLS["H65"], CLS["A_std"]
    specs = build_specs(ev, {"H": {"alpha": 10}, "A": {"alpha": av}})
    return {"gen": "G5", "n_h": n_h, "k": k, "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def g6_stale_bref(rng):
    """G6 参考带宽过期：B 升到 240/300 而 b_ref 冻结 120 → 配额误伤。"""
    B = _c(rng, [180, 240, 300]); av = _c(rng, [2.5, 3, 3.5]); k = _c(rng, [8, 16])
    ev = []
    for tc in _poisson_clumps(rng, 53, 112, k):
        ev.append((tc, "A"))
    CLS["A"] = CLS["A_std"]
    specs = build_specs(ev, {"A": {"alpha": av}}, b_ref=120)
    return {"gen": "G6", "B": B, "alpha": av, "k": k}, \
           {"kind": "single", "specs": specs, "B": B, "b_ref": 120, "guard_w": 0}


def g7_deadline_inversion(rng):
    """G7 期限倒挂：受害者 α≫流 α 仍被 T0 折扣套利杀死。变体 A 零读流 / B 带读流。"""
    variant = _c(rng, ["A", "B"]); h_s = _c(rng, [120, 150, 180]); av = _c(rng, [4.5, 5, 6])
    if variant == "A":
        ev = [(t, "S") for t in _poisson(rng, 130, 390)]
        for tc in _poisson_clumps(rng, 13, 40, 4):
            ev.append((tc, "V"))
        CLS["S"], CLS["V"] = CLS["C_B"], CLS["A_std"]
        dl = {"S": {"slack": h_s}, "V": {"alpha": av}}
    else:
        lam = _c(rng, [20, 25, 28])
        ev = [(t, "S") for t in _poisson(rng, lam, int(lam * 4))]
        for tc in _poisson_clumps(rng, 13, 40, 4):
            ev.append((tc, "V"))
        CLS["S"], CLS["V"] = CLS["Bf"], CLS["A_std"]
        dl = {"S": {"slack": h_s}, "V": {"alpha": av}}
    specs = build_specs(ev, dl)
    return {"gen": "G7", "variant": variant, "h_s": h_s, "alpha_v": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def g4_saturation_gap(rng):
    """G4 饱和点错位：λ 落在 TFS 容量(≈80/s)与 EDF 容量(≈85/s)之间。"""
    lam = _c(rng, [58, 63, 68, 73, 78]); av = _c(rng, [3.5, 4, 4.5])
    ev = [(t, "A") for t in _poisson(rng, lam, int(lam * 4))]
    CLS["A"] = CLS["A_std"]
    specs = build_specs(ev, {"A": {"alpha": av}})
    return {"gen": "G4", "lam": lam, "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def g9_double_cluster(rng):
    """G9 双簇唤醒迟滞：第二簇卡在第一簇配额占用窗内。"""
    d_ms = _c(rng, [10, 20, 40]); k1 = _c(rng, [2, 3, 4]); a2 = _c(rng, [1.85, 1.95, 2.05, 2.15])
    ev = []
    for cyc in range(5):
        t0 = cyc * 0.6 + 0.5
        ev += [(t0, "A")] * k1
        ev += [(t0 + d_ms / 1000, "V")] * 4
    CLS["A"], CLS["V"] = CLS["A_std"], CLS["A_std"]
    specs = build_specs(ev, {"A": {"alpha": 4}, "V": {"alpha": a2}})
    return {"gen": "G9", "delta_ms": d_ms, "k1": k1, "alpha2": a2}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def g8_periodic(rng):
    """G8 错相周期簇（允许负结果：TFS 对周期负载可能鲁棒）。"""
    pm = _c(rng, [0.9, 1.1, 1.3, 1.7]); k = _c(rng, [4, 5, 6]); av = _c(rng, [3.5, 4])
    ev, t = [], 0.0
    while t < 3.0:
        ev += [(t, "A")] * k
        t += pm * 0.0496
    CLS["A"] = CLS["A_std"]
    specs = build_specs(ev, {"A": {"alpha": av}})
    return {"gen": "G8", "P_ratio": pm, "k": k, "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def gx_random_mix(rng):
    """变异元生成器：随机类混合 + 随机到达形态 + 随机期限模式（多样性轮）。"""
    names = ["A_std", "B_std", "C_A", "C_B", "H65", "L_vic", "Bf"]
    n_cls = rng.randint(2, 4)
    chosen = rng.sample(names, n_cls)
    ev, dl = [], {}
    for c in chosen:
        n = rng.randint(20, 120)
        mode = rng.random()
        if mode < 0.4:
            times = _poisson(rng, n / 3.0, n)
        elif mode < 0.7:
            times = _poisson_clumps(rng, n / 3.0, n, rng.choice([4, 8, 16]))
        else:
            times = [3.0 * i / n for i in range(n)]
        ev += [(t, c) for t in times]
        dl[c] = ({"alpha": rng.uniform(1.2, 10)} if rng.random() < 0.6
                 else {"slack": rng.uniform(0.05, 0.3)})
    specs = build_specs(ev, dl)
    return {"gen": "GX", "classes": chosen, "n": len(ev)}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


# ---------------------------------------------------------------------------
# 两层生成器
# ---------------------------------------------------------------------------

TL_CLASSES_BASE = {
    "A":  {"V": 1.4, "T0": 0.096},
    "B":  {"V": 0.08, "T0": 0.10},
    "C":  {"V": 0.0, "T0": 0.096},
    "L":  {"V": 0.5, "T0": 0.048},   # 轻读洪峰类 d=10.4 < 29.2
}


def _tl(classes_alpha: Dict[str, float]):
    return {k: dict(v, alpha=a) for k, v in TL_CLASSES_BASE.items()
            for a in [classes_alpha.get(k, 4.0)]}


def g10_e3_starvation(rng):
    """G10 两层 E3 就绪优先饿死大读。"""
    lam_c = _c(rng, [60, 80, 100]); av = _c(rng, [4, 5, 6]); bs = _c(rng, [4, 8])
    trace = []
    for tc in _poisson_clumps(rng, 16, 160, 8):
        trace += [(tc, "A")] * 8
    trace += [(t, "C") for t in _poisson(rng, lam_c, int(lam_c * 5))]
    trace.sort(key=lambda e: e[0])
    return {"gen": "G10", "lam_c": lam_c, "alpha_a": av, "bs": bs}, \
           {"kind": "two", "trace": trace, "classes": _tl({"A": av, "C": 4.0}),
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": bs}}


def g11_backpressure_cycle(rng):
    """G11 两层 G5 反压振荡。"""
    k = _c(rng, [12, 16]); lam_c = _c(rng, [30, 50, 70]); bp = _c(rng, [4, 6, 8])
    trace = []
    t = 0.0
    while t < 6.0:
        t += rng.expovariate(1.0)
        trace += [(t, "A")] * k
    trace += [(x, "C") for x in _poisson(rng, lam_c, int(lam_c * 6))]
    trace.sort(key=lambda e: e[0])
    return {"gen": "G11", "k": k, "lam_c": lam_c, "bp": bp}, \
           {"kind": "two", "trace": trace, "classes": _tl({"A": 4.0, "C": 4.0}),
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": 8, "bp_override": bp}}


def g12_light_flood(rng):
    """G12 两层轻流洪峰 + 在途预算穿透（账本差击杀口径）。"""
    lam_l = _c(rng, [60, 80, 100]); av = _c(rng, [3, 4])
    trace = []
    for tc in _poisson_clumps(rng, 16, 120, 8):
        trace += [(tc, "A")] * 8
    trace += [(t, "L") for t in _poisson(rng, lam_l, int(lam_l * 5))]
    trace.sort(key=lambda e: e[0])
    return {"gen": "G12", "lam_l": lam_l, "alpha_a": av}, \
           {"kind": "two", "trace": trace, "classes": _tl({"A": av, "L": 4.0}),
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": 8}}


def g13_issued_amplifier(rng):
    """G13 单层 issued 账本保守代价放大（重读占比/倍率推满）。"""
    lam = _c(rng, [60, 90]); hf = _c(rng, [0.5, 0.66]); bs = _c(rng, [4, 8])
    n_a = int(lam * hf * 5)
    trace = []
    for tc in _poisson_clumps(rng, lam * hf, n_a, 8):
        trace += [(tc, "A")] * 8
    rest = int(lam * (1 - hf) * 5)
    trace += [(t, "B") for t in _poisson(rng, lam * (1 - hf) / 2, rest)]
    trace += [(t, "C") for t in _poisson(rng, lam * (1 - hf) / 2, rest)]
    trace.sort(key=lambda e: e[0])
    return {"gen": "G13", "lam": lam, "hf": hf, "bs": bs}, \
           {"kind": "two", "trace": trace, "classes": _tl({"A": 2.5, "B": 4.0, "C": 4.0}),
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": bs}}


SINGLE_GENS = {
    "G1": g1_martyr_heavy, "G2": g2_p75_cliff, "G3": g3_zero_victim,
    "G4": g4_saturation_gap, "G5": g5_cold_poison, "G6": g6_stale_bref,
    "G7": g7_deadline_inversion, "G8": g8_periodic, "G9": g9_double_cluster,
    "G14": g14_compute_blind, "GX": gx_random_mix,
}
TWO_GENS = {"G10": g10_e3_starvation, "G11": g11_backpressure_cycle,
            "G12": g12_light_flood, "G13": g13_issued_amplifier}
