"""对抗二轮生成器：线 A 物理设定攻击（P 族）+ 线 B 两层必要性（B 族）。

设计文档：docs/TFS对抗二轮-存储感知与两层必要性-实验设计-20261026.md。
P 族场景 alloc=max_min_disk（3 盘 × 40）；B 族跑两层参考引擎。
"""
from __future__ import annotations

import random
from fractions import Fraction as F
from typing import Dict, List, Tuple

from adv.generators import CLS, _poisson, _poisson_clumps, _c, build_specs
from sim.cq.storage import disk_plan

# ---------------------------------------------------------------------------
# 哈希倾斜搜索（P 族工具）：找块落点集中于指定盘的 h
# ---------------------------------------------------------------------------

def fracs_of_h(class_id: str, h: int, disks: int = 3) -> Dict[int, float]:
    plan = disk_plan(f"{class_id}:{h}", h, disks)
    tot = sum(nb for _d, _c, nb in plan)
    if tot <= 0:
        return {}
    return {d: nb / tot for d, _c, nb in plan}


def find_skewed_h(rng: random.Random, target_disk: int, min_frac: float,
                  cls: str = "X", tries: int = 800,
                  h_lo: int = 8000, h_hi: int = 120000):
    """搜 h 使 cls 类在 target_disk 的字节占比 ≥ min_frac。

    哈希键含类名——必须用最终类名搜索（冒烟教训：类名不同分布全变）。"""
    best = None
    for _ in range(tries):
        h = rng.randint(h_lo, h_hi)
        fr = fracs_of_h(cls, h)
        f = fr.get(target_disk, 0.0)
        if f >= min_frac:
            return h, fr
        if best is None or f > best[2]:
            best = (h, fr, f)
    return best[:2] if best and best[2] >= min_frac * 0.75 else None


def _mk_cls(name: str, h: int, u: int):
    CLS[name] = (h, u)


# ---------------------------------------------------------------------------
# P 族：物理设定攻击（单层 cq 引擎，alloc=max_min_disk）
# ---------------------------------------------------------------------------

def _u_for_d(h: int, u: int):
    """返回 (h,u) 原样（构造时直接选 u 控制量级）。"""
    return h, u


def p1_hot_poison(rng):
    """P1 钉总账：分散落点的宽松重读流 S 把 v3 总账占满；热盘紧期限类 H 被卡。

    S 用 A_std 天然近似均匀落点（84 块 → 每盘约 1/3），α=8 宽松、λ 使并发
    ~3~4 条 → v3 总账被钉在 ~116/132；H 块集中盘 0（61~75%）、α 紧——
    v3 因总账无余量而压 H；TFS-D 看 H 的盘 0 只有 S 摊入的 ~10 → 放行。
    """
    lam_s = _c(rng, [60, 70, 80])
    min_frac_h = rng.uniform(0.55, 0.7)
    skew = find_skewed_h(rng, 0, min_frac_h, cls="H", tries=4000)
    h_hot = skew[0] if skew else 11875
    u_h = _c(rng, [64, 256])
    k = _c(rng, [4, 8])
    av_h = _c(rng, [2.5, 3, 3.5])
    ev = [(t, "S") for t in _poisson(rng, lam_s, int(lam_s * 4))]
    for tc in _poisson_clumps(rng, 13, 48, k):
        ev.append((tc, "H"))
    _mk_cls("S", 42883, 256)        # A_std：d≈29、天然分散
    _mk_cls("H", h_hot, u_h)
    specs = build_specs(ev, {"S": {"alpha": 8}, "H": {"alpha": av_h}})
    return {"gen": "P1", "h_hot": h_hot, "u_h": u_h,
            "fr_hot": {str(k_): round(float(v), 2)
                       for k_, v in fracs_of_h("H", h_hot).items()},
            "lam_s": lam_s, "k": k, "alpha_h": av_h}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120,
            "guard_w": 0, "alloc": "max_min_disk"}


def p2_background_window(rng):
    """P2 背景流量窗：重读簇到达窗内带宽压低（时间对齐）。"""
    dip = _c(rng, [40, 60, 80]); dur_ms = _c(rng, [60, 100, 150])
    lam = _c(rng, [45, 55, 65]); av = _c(rng, [3, 3.5, 4])
    ev = []
    centers = []
    t = 0.4
    while t < 2.8:
        ev += [(t, "A")] * 8
        centers.append(t)
        t += 0.3
    CLS["A"] = CLS["A_std"]
    # 带宽 schedule：在每个簇心前后压 dip、时长 dur_ms
    sched = [(0, 120)]
    for c0 in centers:
        sched.append((round(max(0.0, c0 - 0.01), 3), dip))
        sched.append((round(max(0.0, c0 - 0.01) + dur_ms / 1000, 3), 120))
    sched.sort(key=lambda x: x[0])
    # 去重时间点（schedule 需严格递增）
    dedup = [sched[0]]
    for tt, vv in sched[1:]:
        if tt <= dedup[-1][0]:
            dedup[-1] = (dedup[-1][0], vv) if vv < dedup[-1][1] else dedup[-1]
        else:
            dedup.append((tt, vv))
    specs = build_specs(ev, {"A": {"alpha": av}})
    return {"gen": "P2", "dip": dip, "dur_ms": dur_ms, "lam": lam,
            "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120,
            "guard_w": 0, "alloc": "max_min_disk", "b_sched": dedup}


def p3_hot_cold_mix(rng):
    """P3 三盘各热一类：v3 总账（132）对三盘容量之和（132）无差别，
    但每类只吃本盘 55~70%——v3 按 d 全额记账（每条 ~d），TFS-D 按热盘
    份额记账 → TFS-D 的可放行并发显著高于 v3。紧期限下兑现差距。"""
    classes = {}
    u_x = _c(rng, [8, 64])
    for dd in range(3):
        sk = find_skewed_h(rng, dd, 0.55, cls=f"X{dd}", tries=4000)
        h = sk[0] if sk else 11875
        _mk_cls(f"X{dd}", h, u_x)
        classes[f"X{dd}"] = {"alpha": _c(rng, [2.5, 3, 3.5])}
    lam_x = _c(rng, [20, 30, 40])
    k = _c(rng, [4, 6, 8])
    ev = []
    for dd in range(3):
        for tc in _poisson_clumps(rng, lam_x, 36, k):
            ev.append((tc, f"X{dd}"))
    specs = build_specs(ev, classes)
    return {"gen": "P3", "u_x": u_x, "lam_x": lam_x, "k": k,
            "skews": {f"X{dd}": {str(kk): round(float(vv), 2)
                                  for kk, vv in fracs_of_h(f"X{dd}",
                                      CLS[f"X{dd}"][0]).items()}
                      for dd in range(3)}}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120,
            "guard_w": 0, "alloc": "max_min_disk"}


def p4_three_hot(rng):
    """P4 三盘对齐簇：三个类分别热不同盘，簇对齐（必要性边界探针）。"""
    hots = []
    for dd in range(3):
        sk = find_skewed_h(rng, dd, 0.6, cls=f"Z{dd}", tries=1500)
        if sk is None:
            sk = find_skewed_h(rng, dd, 0.5, cls=f"Z{dd}", tries=3000)
        hots.append(sk[0] if sk else [42883, 9399, 2048][dd])
    for i, h in enumerate(hots):
        _mk_cls(f"Z{i}", h, 256)
    av = _c(rng, [3, 3.5, 4])
    ev = []
    t = 0.4
    while t < 2.8:
        for i in range(3):
            ev += [(t + 0.002 * i, f"Z{i}")] * 4
        t += 0.35
    dl = {f"Z{i}": {"alpha": av} for i in range(3)}
    specs = build_specs(ev, dl)
    return {"gen": "P4", "hots": hots, "alpha": av}, \
           {"kind": "single", "specs": specs, "B": 120, "b_ref": 120,
            "guard_w": 0, "alloc": "max_min_disk"}


P_SINGLE = {"P1": p1_hot_poison, "P2": p2_background_window,
            "P3": p3_hot_cold_mix, "P4": p4_three_hot}


# ---------------------------------------------------------------------------
# B 族：两层必要性（参考引擎）
# ---------------------------------------------------------------------------

def _tl_classes(overrides: Dict[str, dict]) -> Dict[str, dict]:
    base = {"A": {"V": 1.4, "T0": 0.096}, "B": {"V": 0.08, "T0": 0.10},
            "C": {"V": 0.0, "T0": 0.096}, "L": {"V": 0.5, "T0": 0.048},
            "T": {"V": 6.0, "T0": 0.225}}
    out = {}
    for k, v in base.items():
        out[k] = {**v, "alpha": overrides.get(k, 4.0)}
    return out


def b1_conservatism(rng):
    """B1 单层 issued 账本保守代价放大（G13 参数域细化）。"""
    lam = _c(rng, [60, 75, 90]); hf = _c(rng, [0.5, 0.6, 0.7]); bs = _c(rng, [4, 8])
    dur = 5.0
    trace = []
    n_a = int(lam * hf * dur)
    for tc in _poisson_clumps(rng, lam * hf, n_a, 8):
        trace += [(tc, "A")] * 8
    rest = int(lam * (1 - hf) * dur)
    trace += [(t, "B") for t in _poisson(rng, lam * (1 - hf) / 2, rest)]
    trace += [(t, "C") for t in _poisson(rng, lam * (1 - hf) / 2, rest)]
    trace.sort(key=lambda e: e[0])
    return {"gen": "B1", "lam": lam, "hf": hf, "bs": bs}, \
           {"kind": "two", "trace": trace,
            "classes": _tl_classes({"A": 2.5, "B": 4.0, "C": 4.0}),
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": bs}}


def b2_loadslot_preempt(rng):
    """B2 慢加载重流占加载位 + 紧期限小簇同实例（E2 保护对照）。"""
    n_t = _c(rng, [2, 3, 4]); av_a = _c(rng, [2.0, 2.5, 3.0]); bs = _c(rng, [4, 8])
    trace = []
    t = 0.2
    while t < 5.0:
        trace += [(t, "T")] * n_t          # 慢加载重流（V=6GB）
        trace += [(t + 0.05, "A")] * 4     # 紧期限小簇
        t += 0.8
    trace.sort(key=lambda e: e[0])
    return {"gen": "B2", "n_t": n_t, "alpha_a": av_a, "bs": bs}, \
           {"kind": "two", "trace": trace,
            "classes": _tl_classes({"T": 8.0, "A": av_a}),
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": bs}}


def b3_backpressure_edge(rng):
    """B3 反压边缘：重流簇 + 中等零读流（waiting 深度贴反压阈）。"""
    k = _c(rng, [8, 12, 16]); lam_c = _c(rng, [40, 60, 80]); bp = _c(rng, [4, 6])
    trace = []
    t = 0.0
    while t < 6.0:
        t += rng.expovariate(1.2)
        trace += [(t, "A")] * k
    trace += [(x, "C") for x in _poisson(rng, lam_c, int(lam_c * 6))]
    trace.sort(key=lambda e: e[0])
    return {"gen": "B3", "k": k, "lam_c": lam_c, "bp": bp}, \
           {"kind": "two", "trace": trace,
            "classes": _tl_classes({"A": 3.0, "C": 4.0}),
            "two_kwargs": {"m_inst": 4, "pipe": 120.0, "max_seqs": 8,
                           "bp_override": bp}}


B_TWO = {"B1": b1_conservatism, "B2": b2_loadslot_preempt,
         "B3": b3_backpressure_edge}
