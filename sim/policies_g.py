"""G 系列策略：D1 路由基线/增强 + D2 排序键（设计文档 §3/§4 的公式直译）。

路由策略在请求到达时调用一次，返回 (worker, action, node, tier, est)；
排序策略在入队时计算一次排序键（到达时快照，避免逐迭代重算多抽噪声）。

信息边界（设计 §5）：非 Oracle 策略只读 quote/StorageObservable（I2/I3/I4）与
GPU 可观测（wait_est，计算侧真值可见、全策略对称）；B-Hist 只读自身历史 EWMA
与在途记账（I1）；无策略触碰存储内部真值（hypothetical_*）。
"""
from __future__ import annotations

from dataclasses import dataclass

EPS = 1e-9


@dataclass
class GCtx:
    world: object
    quote: object
    hist: object
    curve: object
    workers: list
    m1: object
    rng: object
    margin: float = 0.0
    guardband: float = 1.2


# ---------------------------------------------------------------- 路由（D1）

class GRouter:
    name = "base"
    needs_obs = False

    def __init__(self, ctx: GCtx):
        self.ctx = ctx
        self.W = ctx.world.n_workers

    # ---- 共享估计 ----
    def gpu_wait(self, w: int) -> float:
        return self.ctx.workers[w].wait_est()

    def _rate(self, w: int) -> float:
        return self.ctx.workers[w].rate(self.ctx.world.env.now)

    def prefill_t(self, tokens: int, w: int = None) -> float:
        t = self.ctx.curve(tokens)
        if w is not None:
            t /= self._rate(w)   # 与引擎 (1-bg) 速率一致，保证取/算两侧估计对称
        return t

    def static_fetch_t(self, gb: float, w: int, node: int, tier: str) -> float:
        """nominal 带宽静态成本（AAFLOW+/B-Moon-S 的 I0 估计）。"""
        r = self.ctx.world.res(node, tier)
        f = self.ctx.world.fabric
        return (self.ctx.world.path_lat[w][node]
                + gb / r.b_total + r.t_base
                + gb / f.b_total + f.t_base)

    def holders(self, req) -> list:
        return sorted(self.ctx.world.dir.holders(req.cls)) if req.hit else []

    def static_source(self, w: int, req) -> tuple:
        """静态源偏好（Dynamo credit 退化形态）：mem 优先、近路径优先。"""
        best = None
        for (n, t) in self.holders(req):
            key = (0 if t == "mem" else 1, self.ctx.world.path_lat[w][n], n)
            if best is None or key < best[0]:
                best = (key, n, t)
        return (best[1], best[2]) if best else (-1, "")

    def _min_wait_worker(self) -> int:
        return min(range(self.W), key=lambda w: self.gpu_wait(w))

    def _miss_prefill(self, req):
        w = self._min_wait_worker()
        return (w, "prefill", -1, "", self.gpu_wait(w) + self.prefill_t(req.N, w))

    def decide(self, req) -> tuple:
        raise NotImplementedError


class RoundRobin(GRouter):
    """B-RR：轮询 worker（Motor RR / Dynamo 无事件兜底）。"""
    name = "rr"

    def __init__(self, ctx):
        super().__init__(ctx)
        self._i = 0

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        w = self._i % self.W
        self._i += 1
        n, t = self.static_source(w, req)
        return (w, "fetch", n, t,
                self.gpu_wait(w) + self.static_fetch_t(req.kv_gb, w, n, t)
                + self.prefill_t(req.U, w))


class LoadAware(GRouter):
    """B-LL：负载最低 worker（Motor 默认 / SGLang cache_aware 退化形态）。"""
    name = "ll"

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        w = self._min_wait_worker()
        n, t = self.static_source(w, req)
        return (w, "fetch", n, t,
                self.gpu_wait(w) + self.static_fetch_t(req.kv_gb, w, n, t)
                + self.prefill_t(req.U, w))


class DynKV(GRouter):
    """B-DynKV：Dynamo kv_router 在 S-G 下的退化（overlap 全 worker 相同 → 纯负载
    + 静态 tier credit 源选择），实现上与 B-LL 同型——运行它是为了实测该退化结论。"""
    name = "dynkv"

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        w = self._min_wait_worker()
        n, t = self.static_source(w, req)
        return (w, "fetch", n, t,
                self.gpu_wait(w) + self.static_fetch_t(req.kv_gb, w, n, t)
                + self.prefill_t(req.U, w))


class _TTCostRouter(GRouter):
    """TTFT 估计式路由骨架（B-Moon-S / B-Hist）：argmin_w [wait + Ê_fetch + suffix]。
    子类提供 fetch_est；动作恒为命中即取（不重算）。"""

    def fetch_est(self, gb: float, w: int, node: int, tier: str) -> float:
        raise NotImplementedError

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        best = None
        for w in range(self.W):
            for (n, t) in self.holders(req):
                c = (self.gpu_wait(w) + self.fetch_est(req.kv_gb, w, n, t)
                     + self.prefill_t(req.U, w))
                if best is None or c < best[0]:
                    best = (c, w, n, t)
        c, w, n, t = best
        return (w, "fetch", n, t, c)


class MoonStatic(_TTCostRouter):
    """B-Moon-S：Mooncake Conductor 形态，传输成本用 nominal 静态带宽（J+I0）。"""
    name = "moons"

    def fetch_est(self, gb, w, n, t):
        return self.static_fetch_t(gb, w, n, t)


class HistTT(_TTCostRouter):
    """B-Hist：同上，但传输成本来自按 (node,tier) 的 EWMA 历史 + 自身在途（J+I1）。
    核心强基线：增强策略必须赢它才构成接口增量（设计 §6 G1 判读）。"""
    name = "hist"

    def fetch_est(self, gb, w, n, t):
        return self.ctx.hist.est(gb, w, n, t)


class HistJoint(GRouter):
    """histj：B-Hist 信息源 + 联合动作（含重算分支）——分离"信息"与"动作空间"两个
    因素（设计 §4.3 组合归因）。"""
    name = "histj"

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        best = None
        for w in range(self.W):
            c_rec = self.gpu_wait(w) + self.prefill_t(req.N, w)
            if best is None or c_rec < best[0]:
                best = (c_rec, w, "recompute", -1, "")
            for (n, t) in self.holders(req):
                c = (self.gpu_wait(w) + self.ctx.hist.est(req.kv_gb, w, n, t)
                     + self.prefill_t(req.U, w))
                if c < best[0]:
                    best = (c, w, "fetch", n, t)
        c, w, a, n, t = best
        return (w, a, n, t, c)


class D1bSource(GRouter):
    """D1b：worker 固定 B-LL，只改源选择——quote 估计最优源 + 近并列随机化防羊群
    （设计 §4.1，双阈值随机化的简化形态）。"""
    name = "d1b"

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        w = self._min_wait_worker()
        cands = []
        for (n, t) in self.holders(req):
            e = self.ctx.quote.estimate(req.kv_gb, w, n, t)["time"]
            cands.append((e, n, t))
        if not cands:
            return self._miss_prefill(req)
        cands.sort()
        best = cands[0][0]
        tol = max(0.10 * best, 0.02)
        near = [c for c in cands if c[0] <= best + tol]
        e, n, t = near[int(self.ctx.rng.integers(len(near)))]
        return (w, "fetch", n, t, self.gpu_wait(w) + e + self.prefill_t(req.U, w))


class _DynActionRouter(GRouter):
    """动态取算骨架（D2b/D1c）：Ê_fetch(quote) vs T_recompute 逐请求比较。"""

    def _decide_at(self, req, w: int):
        rec = self.gpu_wait(w) + self.prefill_t(req.N, w)
        best = (rec, "recompute", -1, "")
        for (n, t) in self.holders(req):
            e = self.ctx.quote.estimate(req.kv_gb, w, n, t)["time"]
            c = self.gpu_wait(w) + e + self.prefill_t(req.U, w)
            if c < best[0]:
                best = (c, "fetch", n, t)
        return best


class StaticAR(_DynActionRouter):
    """staticar（E-StaticAR）：AAFLOW+ 形态——nominal 带宽静态二选一 + FCFS。"""
    name = "staticar"

    def _decide_static(self, req, w: int):
        rec = self.gpu_wait(w) + self.prefill_t(req.N, w)
        best = (rec, "recompute", -1, "")
        for (n, t) in self.holders(req):
            c = (self.gpu_wait(w) + self.static_fetch_t(req.kv_gb, w, n, t)
                 + self.prefill_t(req.U, w))
            if c < best[0]:
                best = (c, "fetch", n, t)
        return best

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        w = self._min_wait_worker()
        c, a, n, t = self._decide_static(req, w)
        return (w, a, n, t, c)


class DynAR(_DynActionRouter):
    """dynar（D2b/D1c）：worker = B-LL，动作 = quote 动态二选一（AAFLOW+ 的实时版）。"""
    name = "dynar"

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        c, a, n, t = self._decide_at(req, self._min_wait_worker())
        return (self._min_wait_worker(), a, n, t, c)


class D1aJoint(_DynActionRouter):
    """d1a：worker × 源 × 动作联合 argmin，全部成本用 quote 动态估计（设计 §4.1
    主增强：把 Dynamo 的静态 tier credit 替换为动态取出耗时）。"""
    name = "d1a"

    def decide(self, req):
        if not req.hit:
            return self._miss_prefill(req)
        best = None
        for w in range(self.W):
            c, a, n, t = self._decide_at(req, w)
            if best is None or c < best[0]:
                best = (c, w, a, n, t)
        c, w, a, n, t = best
        return (w, a, n, t, c)


_ROUTERS = {c.name: c for c in [RoundRobin, LoadAware, DynKV, MoonStatic,
                                HistTT, HistJoint, D1bSource, StaticAR, DynAR,
                                D1aJoint]}


def make_router(name: str, ctx: GCtx):
    return _ROUTERS[name](ctx)


# ---------------------------------------------------------------- 排序（D2）

def order_key(order: str, req, ctx: GCtx) -> tuple:
    """waiting 排序键（到达时一次性计算；设计 §4.2）。

    fcfs   到达序（vLLM 默认）
    lpm    命中长度降序（SGLang lpm）
    prio   外生优先级升序（vLLM/SGLang priority 的合成对照）
    ready  D2a：预计就绪时间 r_i = max(0, Ê_fetch − gpu_wait)
    budget D2c：SLO 预算升序（Budget = slo_ttft − γ·L̂r，Cascade σ 的存储动态版）
    """
    if order == "fcfs":
        return (0, req.arrival)
    if order == "lpm":
        return (-req.H, req.arrival)
    if order == "prio":
        return (req.prio, req.arrival)
    if order == "ready":
        r = 0.0
        if req.action == "fetch":
            est = ctx.quote.estimate(req.kv_gb, req.worker, req.node, req.tier)["time"]
            r = max(0.0, est - ctx.workers[req.worker].wait_est())
        return (r, req.arrival)
    if order == "budget":
        rt = ctx.workers[req.worker].rate(ctx.world.env.now)
        if req.action == "fetch":
            est = ctx.quote.estimate(req.kv_gb, req.worker, req.node, req.tier)["time"]
            lhat = ctx.workers[req.worker].wait_est() + est + ctx.curve(req.U) / rt
        else:
            lhat = ctx.workers[req.worker].wait_est() + ctx.curve(req.N) / rt
        return (req.slo_ttft - ctx.guardband * lhat, req.arrival)
    raise ValueError(f"unknown order {order}")
