"""Wave 3 第二代定向攻击生成器（W1~W9，攻击方 Agent 20261023 二代设计）。

专打 v2 补丁：wl 账本单位错配（W1/W9）、est_bw 钳制迟滞（W2/W5）、
bp_loadbased 加载位垄断（W3）、active×captol 波形态（W4）、
非稳态过载 martyr（W6）、E3 就绪锁（W7）、轻读脉冲锁相（W8）。
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from adv.generators import CLS, _poisson, _poisson_clumps, _c, build_specs

KAPPA = 4.096e-6
A_S, B_S, T_L, N_SAT = 2 * 10 ** -7 * 0.8136, 4.66521e-10, 50e-6, 512
# 注：a 用 e26 标定近似值仅为 d 求解器初值；精确 d 在评估时由 TFS 侧公式重算


def d_of_class(h: int, u: int) -> float:
    """与 bw_demand 同式的 d（浮点重算，精确 a 用标定解）。"""
    # E26_PROFILE 精确 a/b 由线性方程解出；此处直接引用生成时的标定值
    a_s = 1.62727e-6
    b_s = 4.66521e-10
    eta = 1.0 if u >= N_SAT else max(1 / 16, u / N_SAT)
    c = T_L + a_s * u / eta + b_s * (u * h + u * (u + 1) / 2)
    return KAPPA * h / c


def solve_h_for_d(target_d: float, u: int = 8) -> int:
    """二分求 h 使 d(h,u)=target（d 对 h 单调增，上界 κ/(b·u)。"""
    lo, hi = 1, 400000
    for _ in range(80):
        mid = (lo + hi) // 2
        if d_of_class(mid, u) < target_d:
            lo = mid
        else:
            hi = mid
    return hi


def w1_superheavy(rng):
    """W1 d>1.1cap 类永久拒载（wl 删掉 max(1,·) 保底）。"""
    d_t = _c(rng, [133, 140, 160, 200]); frac = _c(rng, [0.3, 0.6, 0.9])
    av = _c(rng, [2, 4, 8])
    h = solve_h_for_d(d_t)
    CLS["X"] = (h, 8)
    n_x, n_a = int(90 * frac), int(90 * (1 - frac))
    ev = [(t, "X") for t in _poisson(rng, 30 * frac, n_x)]
    ev += [(t, "A") for t in _poisson(rng, 30 * (1 - frac), n_a)]
    CLS["A"] = CLS["A_std"]
    specs = build_specs(ev, {"X": {"alpha": av}, "A": {"alpha": 4}})
    return {"gen": "W1", "d_target": d_t, "h": h, "frac": frac, "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def w2_cap_ratchet(rng):
    """W2 est_bw 不对称迟滞：一次带宽下陷把 cap 永久锁死。"""
    dip = _c(rng, [20, 30, 60, 90]); dur_ms = _c(rng, [15, 25, 30])
    lam = _c(rng, [45, 55, 70]); av = _c(rng, [3.5, 4])
    ev = [(t, "A") for t in _poisson(rng, lam, int(lam * 3))]
    CLS["A"] = CLS["A_std"]
    specs = build_specs(ev, {"A": {"alpha": av}})
    return {"gen": "W2", "dip": dip, "dur_ms": dur_ms, "lam": lam, "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0,
            "b_sched": [(0, 120), (0.5, dip), (0.5 + dur_ms / 1000, 120)],
            "sigma": 0.0}


def w3_slot_monopoly(rng):
    """W3 两层 bp_loadbased 慢加载垄断。"""
    vt = _c(rng, [4.8, 6.0, 7.2]); k = _c(rng, [2, 4, 8]); av = _c(rng, [3, 3.5, 4, 4.5])
    lead = _c(rng, [0, 0.03, 0.06])
    classes = {
        "T": {"V": vt, "T0": 0.225, "alpha": 4.0},
        "A": {"V": 1.4, "T0": 0.096, "alpha": av},
    }
    trace = [(0.0, "T")] * 4          # 每实例 1 条 trickle
    for cyc in range(8):
        t0 = 0.5 + cyc * 0.6
        trace += [(t0, "T")]
        trace += [(t0 + lead, "A")] * k
    trace.sort(key=lambda e: e[0])
    return {"gen": "W3", "v_trickle": vt, "k": k, "alpha": av, "lead": lead}, \
           {"kind": "two", "trace": trace, "classes": classes,
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": 8}}


def w4_boost_burst(rng):
    """W4 active 末层豁免×提升流优先反转（中置信）。"""
    k = _c(rng, [9, 10, 11]); av = _c(rng, [1.0, 1.05, 1.1, 1.2])
    CLS["M"] = (49560, 627)
    ev = []
    for cyc in range(12):
        ev += [(0.25 * cyc + 0.1, "M")] * k
    specs = build_specs(ev, {"M": {"alpha": av}})
    return {"gen": "W4", "k": k, "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def w5_noise_pierce(rng):
    """W5 σ 噪声打穿 0.9 阈值 → cap 单向棘轮。"""
    sigma = _c(rng, [0.05, 0.10, 0.15]); lam = _c(rng, [70, 78, 85]); av = 4
    ev = [(t, "A") for t in _poisson(rng, lam, int(lam * 4))]
    CLS["A"] = CLS["A_std"]
    specs = build_specs(ev, {"A": {"alpha": av}})
    return {"gen": "W5", "sigma": sigma, "lam": lam, "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0,
            "sigma": float(sigma)}  # 乘性噪声系数（0.10=±10%；20261023 修正单位错误）}


def w6_overload_burst(rng):
    """W6 非稳态过载 martyr：λ_S>m/T0_S=139.9/s 的滚动活流窗口。

    方向修正（20261023）：流要排在受害者之前需 h_S < H_V=(α_V−1)·T0_V
    （攻击方二代清单笔误写了 >；< 才使静态键更小）。"""
    lam_s = _c(rng, [250, 350, 450]); dur = _c(rng, [0.4, 0.5])
    av = _c(rng, [8, 10, 12])
    H_V = (av - 1) * 0.0496                    # A_std T0≈49.6ms
    h_frac = rng.uniform(0.25, 0.8)
    h_s = h_frac * H_V                          # h_S < H_V
    ev = [(t + 0.2, "S") for t in _poisson(rng, lam_s, int(lam_s * dur))]
    for tc in [0.35, 0.45, 0.55]:
        ev += [(tc, "V")] * 4
    CLS["S"], CLS["V"] = CLS["C_B"], CLS["A_std"]
    specs = build_specs(ev, {"S": {"slack": h_s}, "V": {"alpha": av}})
    return {"gen": "W6", "lam_s": lam_s, "dur": dur, "alpha_v": av,
            "h_s_ms": round(h_s * 1000, 1), "H_V_ms": round(H_V * 1000, 1)}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0}


def w7_ready_lock(rng):
    """W7 两层 E3 就绪优先+准入节奏锁死宽期限重流（G10 重参数化）。"""
    lam_c = _c(rng, [100, 120, 140]); av = _c(rng, [5, 6, 8]); bs = _c(rng, [4, 8])
    classes = {
        "A": {"V": 1.4, "T0": 0.048, "alpha": float(av)},
        "C": {"V": 0.0, "T0": 0.048, "alpha": 4.0},
    }
    trace = []
    for tc in _poisson_clumps(rng, 8, 96, 4):
        trace += [(tc, "A")] * 4
    trace += [(t, "C") for t in _poisson(rng, lam_c, int(lam_c * 6))]
    trace.sort(key=lambda e: e[0])
    return {"gen": "W7", "lam_c": lam_c, "alpha_h": av, "bs": bs}, \
           {"kind": "two", "trace": trace, "classes": classes,
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": bs}}


def w8_pulse_lock(rng):
    """W8 轻读脉冲与波释放锁相（低置信，允许负结果）。"""
    k = _c(rng, [90, 110]); duty = _c(rng, [0.4, 0.5, 0.6])
    lam = _c(rng, [30, 40, 50]); av = _c(rng, [1.8, 1.9, 2.0])
    phase = _c(rng, [0, 12, 25])
    period = 0.0496
    ev = []
    t = phase / 1000
    while t < 3.0:
        for i in range(k):
            ev.append((t + i * (period * duty) / k, "L"))
        t += period
    ev += [(x, "A") for x in _poisson(rng, lam, int(lam * 3))]
    CLS["L"], CLS["A"] = CLS["L_vic"], CLS["A_std"]
    specs = build_specs(ev, {"L": {"alpha": 6}, "A": {"alpha": av}})
    return {"gen": "W8", "k": k, "duty": duty, "lam": lam, "alpha": av,
            "phase": phase}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0,
            "alloc": "max_min"}


def w9_serialize_midheavy(rng):
    """W9 d∈(cap/2,cap) 强制单并发串行（G5 毒丸在连续账本下的转世）。"""
    d_t = _c(rng, [80, 90, 100, 110]); k = _c(rng, [4, 6])
    u = 64
    h = solve_h_for_d(d_t, u)
    CLS["X"] = (h, u)
    # deadline 窗：串行第 k 条完成 ≈ k×T0 之内、并行完成之上
    from sim.cq.types import RequestSpec
    from sim.experiments.e26_ab import E26_PROFILE, make_T0
    from fractions import Fraction as F
    proto = [RequestSpec(0, 0.0, h, u, "X", F(1), F(1))]
    T0 = float(make_T0(proto, E26_PROFILE, F(120), F(200))[0])
    par_done = 8 * (KAPPA * h / (120 / k)) + 8 * (T0 - 7 * max(0.0, KAPPA * h / 120 - (T0 - 7 * max(0.0, 0)) / 8))  # 粗近似
    ser_k = k * T0 * 0.95
    dl = ser_k
    ev = []
    for cyc in range(10):
        ev += [(0.3 * cyc + 0.1, "X")] * k
    specs = build_specs(ev, {"X": {"slack": dl}})
    return {"gen": "W9", "d_target": d_t, "h": h, "k": k, "dl_s": round(dl, 4),
            "T0_ms": round(T0 * 1000, 2)}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120, "guard_w": 0,
            "alloc": "max_min"}


W_SINGLE = {"W1": w1_superheavy, "W2": w2_cap_ratchet, "W4": w4_boost_burst,
            "W5": w5_noise_pierce, "W6": w6_overload_burst, "W8": w8_pulse_lock,
            "W9": w9_serialize_midheavy}
W_TWO = {"W3": w3_slot_monopoly, "W7": w7_ready_lock}
