"""v2 策略：共享分布式 KV 存储下的联合调度决策。

与《五工作的Scheduler-Engine修改与主流默认实现》文档的映射：
  alwaysfetch2 / default2  -> 主流默认（vLLM 命中即取 + Dynamo KV-credit 路由）
  tensorcast2              -> TensorCast（locality + load 联合路由，不做重算）
  static2                  -> AAFLOW+（静态 nominal 带宽 fetch-vs-recompute）
  partial_static2          -> CacheFlow（静态成本 token 粒度部分取回 + I/O/计算重叠）
  joint2                   -> 本方向（访问成本查询驱动的 worker×副本×动作×F 联合决策）
  coord2                   -> 本方向 + 协同（滞回 + 抖动，避免羊群/振荡，Q5）
  oracle2                  -> 真值上界（读 ground truth hypothetical）
所有策略只读可观测信息（除 oracle2）；Decision 携带执行所需全部信息。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    worker: int
    action: str                  # local | fetch | partial | recompute | prefill
    node: int = -1
    tier: str = ""
    fetch_tokens: int = 0        # partial：取回前缀 token 数
    fetch_gb: float = 0.0
    overlap: bool = True         # partial 是否与 GPU 计算重叠
    cost: float = 0.0            # 决策时的估计成本（诊断用）


@dataclass
class V2Ctx:
    world: object
    quote: object
    margin: float = 0.10
    kv_gb_per_token: float = 0.0
    rng: object = None
    f_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    gpu_obs: object = None              # GpuObservable 列表（None = 真值；仅 oracle 家族保持 None）
    future: object = None               # callable(cls, t, horizon)->int，未来同类命中到达数（仅先知策略用）
    guardband: float = 1.2              # cascade2 保守系数 γ


F_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


class V2Base:
    needs_obs = False

    def __init__(self, ctx: V2Ctx):
        self.ctx = ctx
        self.f_grid = ctx.f_grid or F_GRID

    # ---- 通用估计 ----
    def gpu_wait(self, w: int) -> float:
        obs = self.ctx.gpu_obs
        if obs is not None:
            return obs[w].estimate()
        return self.ctx.world.gpus[w].drain_est()

    def _rate(self, w: int) -> float:
        return max(0.02, 1.0 - self.ctx.world.gpus[w].bg_at(self.ctx.world.env.now))

    def prefill_t(self, tokens: int, w: int = None) -> float:
        t = self.ctx.world.curve(tokens)
        if w is not None:
            t /= self._rate(w)   # 引擎按 (1-bg) 速率执行，估计侧保持一致
        return t

    def suffix_t(self, req, w: int = None) -> float:
        return self.prefill_t(req.suffix_tokens, w)

    def static_fetch_t(self, nbytes: float, w: int, node: int, tier: str) -> float:
        """nominal 带宽 + 静态路径延迟（AAFLOW+ 式静态成本）。"""
        r = self.ctx.world.res(node, tier)
        f = self.ctx.world.fabric
        return (self.ctx.world.path_lat[w][node]
                + nbytes / r.b_total + r.t_base
                + nbytes / f.b_total + f.t_base)

    def dyn_fetch_t(self, nbytes: float, w: int, node: int, tier: str) -> float:
        """访问成本查询（陈旧/EMA/带噪 + 路径延迟 + fabric）。"""
        return self.ctx.quote.estimate(nbytes, w, node, tier)["time"]

    def holders(self, req):
        return sorted(self.ctx.world.dir.holders(req.cls)) if req.hit else []

    def local_holders(self, req):
        if not req.hit:
            return []
        return [w for w in range(self.ctx.world.n_workers)
                if self.ctx.world.locals[w].holds(req.cls)]

    def best_replica(self, req, w: int, est) -> tuple:
        """给定估计函数，返回 (cost, node, tier) 最优副本。"""
        best = None
        for (n, t) in self.holders(req):
            c = est(req.kv_gb, w, n, t)
            if best is None or c < best[0]:
                best = (c, n, t)
        return best or (float("inf"), -1, "")

    def _prefill_dec(self, req):
        w = min(range(self.ctx.world.n_workers), key=lambda i: self.gpu_wait(i))
        return Decision(worker=w, action="prefill", cost=self.gpu_wait(w) + self.prefill_t(req.prompt_tokens, w))

    def decide(self, req) -> Decision:
        raise NotImplementedError


# ---------------- 路由类（不改引擎成本观，只改选 worker/副本） ----------------

class AlwaysFetch2(V2Base):
    """主流引擎侧默认：命中即取，从不重算；worker 取最短 GPU 队列。"""

    def decide(self, req) -> Decision:
        if not req.hit:
            return self._prefill_dec(req)
        locals_ = self.local_holders(req)
        if locals_:
            w = min(locals_, key=lambda i: self.gpu_wait(i))
            return Decision(worker=w, action="local", cost=self.gpu_wait(w) + self.suffix_t(req, w))
        if self.holders(req):
            w = min(range(self.ctx.world.n_workers), key=lambda i: self.gpu_wait(i))
            _, n, t = self.best_replica(req, w, self.static_fetch_t)
            return Decision(worker=w, action="fetch", node=n, tier=t,
                            cost=self.gpu_wait(w) + self.static_fetch_t(req.kv_gb, w, n, t) + self.suffix_t(req, w))
        return self._prefill_dec(req)


class RoundRobin2(V2Base):
    """RR 路由 + 静态二选一引擎（v1 E2 baseline 的 v2 版）。"""

    def __init__(self, ctx):
        super().__init__(ctx)
        self._i = 0
        self._engine = Static2(ctx)

    def decide(self, req) -> Decision:
        W = self.ctx.world.n_workers
        w = self._i % W
        self._i += 1
        return self._engine._engine_at(req, w)


class LoadAware2(V2Base):
    def __init__(self, ctx):
        super().__init__(ctx)
        self._engine = Static2(ctx)

    def decide(self, req) -> Decision:
        w = min(range(self.ctx.world.n_workers), key=lambda i: self.gpu_wait(i))
        return self._engine._engine_at(req, w)


class DefaultServing2(V2Base):
    """Dynamo KV-credit 路由 + vLLM 命中即取：本地缓存命中高 credit，否则 mem tier 优先、近路径优先。"""

    def _credit_pick(self, req, w):
        best = None
        for (n, t) in self.holders(req):
            key = (0 if t == "mem" else 1, self.ctx.world.path_lat[w][n], n)
            if best is None or key < best[0]:
                best = (key, n, t)
        return (best[1], best[2]) if best else (-1, "")

    def decide(self, req) -> Decision:
        if not req.hit:
            return self._prefill_dec(req)
        locals_ = self.local_holders(req)
        if locals_:
            w = min(locals_, key=lambda i: self.gpu_wait(i))
            return Decision(worker=w, action="local", cost=self.gpu_wait(w) + self.suffix_t(req, w))
        w = min(range(self.ctx.world.n_workers), key=lambda i: self.gpu_wait(i))
        n, t = self._credit_pick(req, w)
        if n >= 0:
            return Decision(worker=w, action="fetch", node=n, tier=t,
                            cost=self.gpu_wait(w) + self.static_fetch_t(req.kv_gb, w, n, t) + self.suffix_t(req, w))
        return self._prefill_dec(req)


class TensorCast2(V2Base):
    """locality + load 联合：worker 成本 = 队列 + （本地 0 / 远取静态成本），换 worker 可牺牲 locality。"""

    def decide(self, req) -> Decision:
        if not req.hit:
            return self._prefill_dec(req)
        best = None
        for w in range(self.ctx.world.n_workers):
            wait = self.gpu_wait(w)
            if self.ctx.world.locals[w].holds(req.cls):
                c = wait + self.suffix_t(req)
                dec = Decision(worker=w, action="local", cost=c)
            elif self.holders(req):
                _, n, t = self.best_replica(req, w, self.static_fetch_t)
                if n < 0:
                    continue
                c = wait + self.static_fetch_t(req.kv_gb, w, n, t) + self.suffix_t(req)
                dec = Decision(worker=w, action="fetch", node=n, tier=t, cost=c)
            else:
                continue
            if best is None or dec.cost < best.cost:
                best = dec
        return best or self._prefill_dec(req)


class NearestReplica2(V2Base):
    """副本选择 = mem 优先 + 最近路径（无压力感知）；worker 最短队列；从不重算。E6 的“默认 credit”基线。"""

    def decide(self, req) -> Decision:
        if not req.hit:
            return self._prefill_dec(req)
        locals_ = self.local_holders(req)
        if locals_:
            w = min(locals_, key=lambda i: self.gpu_wait(i))
            return Decision(worker=w, action="local", cost=self.gpu_wait(w) + self.suffix_t(req, w))
        if self.holders(req):
            w = min(range(self.ctx.world.n_workers), key=lambda i: self.gpu_wait(i))
            n, t = DefaultServing2(self.ctx)._credit_pick(req, w)
            return Decision(worker=w, action="fetch", node=n, tier=t,
                            cost=self.gpu_wait(w) + self.static_fetch_t(req.kv_gb, w, n, t) + self.suffix_t(req, w))
        return self._prefill_dec(req)


class RRReplica2(V2Base):
    """多副本间轮询（无压力感知负载均衡）；引擎静态二选一。E6 基线。"""

    def __init__(self, ctx):
        super().__init__(ctx)
        self._i = 0

    def decide(self, req) -> Decision:
        w = min(range(self.ctx.world.n_workers), key=lambda i: self.gpu_wait(i))
        hs = self.holders(req)
        if not req.hit or not hs:
            return Static2(self.ctx)._engine_at(req, w)
        if self.ctx.world.locals[w].holds(req.cls):
            return Decision(worker=w, action="local", cost=self.gpu_wait(w) + self.suffix_t(req, w))
        n, t = hs[self._i % len(hs)]
        self._i += 1
        tf = self.static_fetch_t(req.kv_gb, w, n, t) + self.suffix_t(req, w)
        tr = self.gpu_wait(w) + self.prefill_t(req.prompt_tokens, w)
        if tr < tf * (1.0 - self.ctx.margin):
            return Decision(worker=w, action="recompute", cost=tr)
        return Decision(worker=w, action="fetch", node=n, tier=t, cost=tf)


# ---------------- 成本类（改引擎决策） ----------------

class Static2(V2Base):
    """AAFLOW+：nominal 带宽二选一（fetch vs recompute），含 local 选项，全局选最优 worker。"""

    def _engine_at(self, req, w: int) -> Decision:
        if not req.hit:
            return Decision(worker=w, action="prefill",
                            cost=self.gpu_wait(w) + self.prefill_t(req.prompt_tokens, w))
        wait = self.gpu_wait(w)
        tr = wait + self.prefill_t(req.prompt_tokens, w)
        best_f = None
        if self.ctx.world.locals[w].holds(req.cls):
            best_f = Decision(worker=w, action="local", cost=wait + self.suffix_t(req, w))
        elif self.holders(req):
            c, n, t = self.best_replica(req, w, self.static_fetch_t)
            if n >= 0:
                best_f = Decision(worker=w, action="fetch", node=n, tier=t, cost=c + wait + self.suffix_t(req))
        if best_f is None or tr < best_f.cost * (1.0 - self.ctx.margin):
            return Decision(worker=w, action="recompute", cost=tr)
        return best_f

    def decide(self, req) -> Decision:
        best = None
        for w in range(self.ctx.world.n_workers):
            d = self._engine_at(req, w)
            if best is None or d.cost < best.cost:
                best = d
        return best


class PartialStatic2(V2Base):
    """CacheFlow：静态成本下在 F 网格上选部分取回比例，I/O 与 GPU 队列/计算重叠。"""

    def _cand_costs(self, req, w, fetch_est):
        """返回 [(cost, F, node, tier)]，overlap 模型：max(取回链, GPU队列+剩余计算)。"""
        out = []
        wait = self.gpu_wait(w)
        if self.ctx.world.locals[w].holds(req.cls):
            out.append((wait + self.suffix_t(req), None, -1, ""))
        for (n, t) in self.holders(req):
            for F in self.f_grid:
                if F <= 0.0:
                    continue
                fb = req.kv_gb * F
                ft = fetch_est(fb, w, n, t)
                comp = wait + self.prefill_t(req.prompt_tokens - int(req.cached_prefix_tokens * F), w)
                out.append((max(ft, comp), F, n, t))
        return out

    def _pick(self, req, fetch_est):
        best = None
        tr_cost = None
        for w in range(self.ctx.world.n_workers):
            wait = self.gpu_wait(w)
            tr = wait + self.prefill_t(req.prompt_tokens, w)
            if tr_cost is None or tr < tr_cost:
                tr_cost = tr
            for (c, F, n, t) in self._cand_costs(req, w, fetch_est):
                if best is None or c < best[0]:
                    best = (c, F, n, t, w)
        if not req.hit or best is None:
            w = min(range(self.ctx.world.n_workers), key=lambda i: self.gpu_wait(i))
            return Decision(worker=w, action="prefill", cost=tr_cost)
        c, F, n, t, w = best
        tr = self.gpu_wait(w) + self.prefill_t(req.prompt_tokens, w)
        if tr < c * (1.0 - self.ctx.margin):   # 纯重算明显更优才放弃取回
            return Decision(worker=w, action="recompute", cost=tr)
        if F is None:
            return Decision(worker=w, action="local", cost=c)
        ftok = int(req.cached_prefix_tokens * F)
        return Decision(worker=w, action="partial", node=n, tier=t, fetch_tokens=ftok,
                        fetch_gb=req.kv_gb * F, cost=c)

    def decide(self, req) -> Decision:
        return self._pick(req, self.static_fetch_t)


class DynamicJoint2(PartialStatic2):
    """本方向：访问成本查询驱动的联合决策（worker × 副本 × 动作 × F），含顺序全量 fetch 候选。"""
    needs_obs = True

    def decide(self, req) -> Decision:
        return self._pick(req, self.dyn_fetch_t)

    def _pick(self, req, fetch_est):
        dec = super()._pick(req, fetch_est)
        # 额外候选：顺序全量 fetch（先取回再做 suffix prefill），在取回很快、GPU 队列很长时更优
        if req.hit and self.holders(req):
            for w in range(self.ctx.world.n_workers):
                if self.ctx.world.locals[w].holds(req.cls):
                    c = self.gpu_wait(w) + self.suffix_t(req, w)
                    if c < dec.cost:
                        dec = Decision(worker=w, action="local", cost=c)
                else:
                    c0, n, t = self.best_replica(req, w, fetch_est)
                    if n < 0:
                        continue
                    c = c0 + self.gpu_wait(w) + self.suffix_t(req, w)
                    if c < dec.cost:
                        dec = Decision(worker=w, action="fetch", node=n, tier=t, cost=c)
        if dec.action in ("fetch", "partial"):
            req.quote_est = self.ctx.quote.estimate(
                dec.fetch_gb if dec.action == "partial" else req.kv_gb,
                dec.worker, dec.node, dec.tier)["time"]
        return dec


class DynamicJointSeq2(DynamicJoint2):
    """消融：joint2 但 partial 不与计算重叠（顺序执行）。"""

    def _pick(self, req, fetch_est):
        dec = super()._pick(req, fetch_est)
        if dec.action == "partial":
            # 顺序代价重估：取回 + 剩余计算（无重叠收益）
            seq = (fetch_est(dec.fetch_gb, dec.worker, dec.node, dec.tier)
                   + self.gpu_wait(dec.worker)
                   + self.prefill_t(req.prompt_tokens - dec.fetch_tokens, dec.worker))
            dec.overlap = False
            dec.cost = seq
        return dec


class CoordinatedJoint2(DynamicJoint2):
    """Q5：joint2 + 三重协同 —— (1) 滞回粘住上次副本；(2) 次优副本概率分流；
    (3) 调度器侧在途流量影子项（自己刚路由过去的取回请求会推高目标副本成本，
    无需等存储反馈即可防羊群）。"""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.last_replica = {}      # cls -> (node, tier)
        self.inflight = {}          # (cls, node, tier) -> 在途取回请求数（指数衰减）
        self._last_t = None
        self.hyst = 0.15
        self.dither_p = 0.15
        self.tau = 1.0              # 在途项时间常数（秒）

    def _tick_decay(self):
        import math
        t = self.ctx.world.env.now
        if self._last_t is not None and t > self._last_t:
            decay = math.exp(-(t - self._last_t) / self.tau)
            for k in list(self.inflight):
                self.inflight[k] *= decay
                if self.inflight[k] < 1e-3:
                    del self.inflight[k]
        self._last_t = t

    def decide(self, req) -> Decision:
        self._tick_decay()
        dec = super().decide(req)
        if dec.action not in ("fetch", "partial"):
            return dec
        hs = self.holders(req)
        if len(hs) <= 1:
            return dec
        nbytes = dec.fetch_gb if dec.action == "partial" else req.kv_gb
        comp = (self.gpu_wait(dec.worker)
                + self.prefill_t(req.prompt_tokens - dec.fetch_tokens, dec.worker))
        cands = []
        for (n, t) in hs:
            ft = self.dyn_fetch_t(nbytes, dec.worker, n, t)
            # 在途影子项：每个在途同源取回约占用 nbytes/b_total 的服务时间
            shadow = self.inflight.get((req.cls, n, t), 0.0) * nbytes / self.ctx.world.res(n, t).b_total
            if dec.action == "partial":
                c = max(ft, comp) + shadow
            else:
                c = ft + self.gpu_wait(dec.worker) + self.suffix_t(req, dec.worker) + shadow
            cands.append((c, n, t))
        cands.sort()
        best = cands[0]
        pick = best
        last = self.last_replica.get(req.cls)
        if last is not None:
            for c, n, t in cands:
                if (n, t) == last:
                    if c < best[0] * (1.0 + self.hyst):   # 换路收益不足则粘住
                        pick = (c, n, t)
                    break
        if (pick is best and len(cands) > 1 and self.ctx.rng is not None
                and cands[1][0] < 2.0 * best[0]
                and float(self.ctx.rng.random()) < self.dither_p):
            pick = cands[1]
        self.last_replica[req.cls] = (pick[1], pick[2])
        key = (req.cls, pick[1], pick[2])
        self.inflight[key] = self.inflight.get(key, 0.0) + 1.0
        dec.node, dec.tier = pick[1], pick[2]
        req.quote_est = self.dyn_fetch_t(nbytes, dec.worker, dec.node, dec.tier)
        return dec


class OracleV2(V2Base):
    """真值上界：GPU/存储 hypothetical 精确预测，遍历 worker × 副本 × F。"""

    def _hyp_fetch(self, nbytes, w, node, tier):
        r = self.ctx.world.res(node, tier)
        f = self.ctx.world.fabric
        return (self.ctx.world.path_lat[w][node]
                + r.hypothetical_fetch_time(nbytes) + f.hypothetical_fetch_time(nbytes))

    def decide(self, req) -> Decision:
        world = self.ctx.world
        best = None
        for w in range(world.n_workers):
            wait_hyp = world.gpus[w].hypothetical(req.prompt_tokens)
            tr = wait_hyp
            cand = Decision(worker=w, action="recompute" if req.hit else "prefill", cost=tr)
            if best is None or cand.cost < best.cost:
                best = cand
            if not req.hit:
                continue
            if world.locals[w].holds(req.cls):
                c = world.gpus[w].hypothetical(req.suffix_tokens)
                if c < best.cost:
                    best = Decision(worker=w, action="local", cost=c)
            for (n, t) in self.holders(req):
                for F in self.f_grid:
                    fb = req.kv_gb * F
                    ft = self._hyp_fetch(fb, w, n, t) if F > 0 else 0.0
                    if F >= 1.0:
                        # 顺序全量 fetch
                        c = ft + world.gpus[w].hypothetical(req.suffix_tokens)
                        if c < best.cost:
                            best = Decision(worker=w, action="fetch", node=n, tier=t, cost=c)
                    if F > 0:
                        comp = world.gpus[w].hypothetical(
                            req.prompt_tokens - int(req.cached_prefix_tokens * F))
                        c = max(ft, comp)
                        if c < best.cost:
                            ftok = int(req.cached_prefix_tokens * F)
                            best = Decision(worker=w, action="partial", node=n, tier=t,
                                            fetch_tokens=ftok, fetch_gb=fb, cost=c)
        return best


class ClairvoyantJoint2(OracleV2):
    """先知上界：真值 + 未来视线（H 秒内同类命中到达数，由 simrun 注入 trace 查询）。

    对同步 burst 做 list-scheduling 平衡分配：当前请求 + m 个未来同类请求视为
    依次到达的球，每个球放到（当前真值排队 + 已分配球负载）最小的资源上；
    实际执行第一个球的分配。用于检验"逐请求贪心（即便完美信息）vs 全局分配"的差距。
    """

    horizon = 5.0

    def __init__(self, ctx):
        super().__init__(ctx)
        from collections import deque
        self.assigned = deque()   # (t_expire, key, seconds)

    def _purge(self, t):
        while self.assigned and self.assigned[0][0] <= t:
            self.assigned.popleft()

    def decide(self, req) -> Decision:
        world = self.ctx.world
        t = world.env.now
        self._purge(t)
        m_future = 0
        if self.ctx.future is not None:
            m_future = int(self.ctx.future(req.cls, t, self.horizon))
        balls = min(1 + m_future, 400)   # 防极端内存

        # 候选（完成时间, [(资源键, service 秒)...], Decision）——动作空间与 oracle 对齐
        def options_for_ball():
            opts = []
            for w in range(world.n_workers):
                gkey = ("g", w)
                opts.append((world.gpus[w].hypothetical(req.prompt_tokens),
                             [(gkey, world.curve(req.prompt_tokens))],
                             Decision(worker=w, action="recompute" if req.hit else "prefill")))
                if not req.hit:
                    continue
                if world.locals[w].holds(req.cls):
                    opts.append((world.gpus[w].hypothetical(req.suffix_tokens),
                                 [(gkey, world.curve(req.suffix_tokens))],
                                 Decision(worker=w, action="local")))
                for (n, ti) in self.holders(req):
                    tier_key = ("s", n, ti)
                    b_tier = max(1e-9, world.res(n, ti).b_total)
                    for F in (0.5, 1.0):
                        ftok = int(req.cached_prefix_tokens * F)
                        fb = req.kv_gb * F
                        ft = self._hyp_fetch(fb, w, n, ti) if F > 0 else 0.0
                        loads = [(tier_key, fb / b_tier)]
                        if F >= 1.0:
                            comp = world.gpus[w].hypothetical(req.suffix_tokens)
                            loads.append((gkey, world.curve(req.suffix_tokens)))
                            opts.append((ft + comp, loads,
                                         Decision(worker=w, action="fetch", node=n, tier=ti)))
                        else:
                            comp = world.gpus[w].hypothetical(req.prompt_tokens - ftok)
                            loads.append((gkey, world.curve(req.prompt_tokens - ftok)))
                            opts.append((max(ft, comp), loads,
                                         Decision(worker=w, action="partial", node=n, tier=ti,
                                                  fetch_tokens=ftok, fetch_gb=fb)))
            return opts

        load = {}
        first_dec = None
        for _ in range(balls):
            best = None
            for base, loads, dec in options_for_ball():
                c = base + max((load.get(k, 0.0) for k, _ in loads), default=0.0)
                if best is None or c < best[0]:
                    best = (c, loads, dec)
            _, loads, dec = best
            for k, sec in loads:
                load[k] = load.get(k, 0.0) + sec
            if first_dec is None:
                first_dec = dec
                first_dec.cost = best[0]
        # 记录已分配球（供后续到达的同类请求决策时叠加）
        for key, sec in load.items():
            self.assigned.append((t + self.horizon, key, sec))
        return first_dec


class CascadeLike2(V2Base):
    """问题⑤：Cascade 边界复刻——request 级优化，无跨 worker/跨副本联合。

    对应调研修正后的 Cascade 定位：单实例内 SLO 预算 + 动态 KV 恢复决策。
    建模：worker 按最短 GPU 队列选（不与 KV 放置联合，对应单实例定位）；恢复估计读
    与 joint2 相同的访问成本查询（对等信息，见 E12），差异在于 (a) guardband γ 保守化、
    (b) 动作空间仅 {local, fetch, recompute/prefill}（无 partial/重叠）、(c) 无跨
    worker×副本联合搜索。
    """
    needs_obs = True

    def decide(self, req) -> Decision:
        if not req.hit:
            return self._prefill_dec(req)
        w = min(range(self.ctx.world.n_workers), key=lambda i: self.gpu_wait(i))
        wait = self.gpu_wait(w)
        gamma = getattr(self.ctx, "guardband", 1.2)
        tr = wait + self.prefill_t(req.prompt_tokens, w)
        if self.ctx.world.locals[w].holds(req.cls):
            c_local = wait + self.suffix_t(req, w)
            if tr < c_local * (1.0 - self.ctx.margin):
                return Decision(worker=w, action="recompute", cost=tr)
            return Decision(worker=w, action="local", cost=c_local)
        best = None
        for (n, t) in self.holders(req):
            est = self.ctx.quote.estimate(req.kv_gb, w, n, t)["time"]
            if best is None or est < best[0]:
                best = (est, n, t)
        if best is None:
            return Decision(worker=w, action="prefill", cost=tr)
        est, n, t = best
        # guardband 只作用于不确定的恢复部分（等待与 prefill 时间不加 γ，与 Cascade 语义一致）
        tf = wait + gamma * (est + self.suffix_t(req, w))
        if tf < tr:
            return Decision(worker=w, action="fetch", node=n, tier=t, cost=tf)
        return Decision(worker=w, action="recompute", cost=tr)


V2POLICIES = {
    "alwaysfetch2": AlwaysFetch2,
    "rr2": RoundRobin2,
    "load2": LoadAware2,
    "default2": DefaultServing2,
    "tensorcast2": TensorCast2,
    "nearest2": NearestReplica2,
    "rrrep2": RRReplica2,
    "static2": Static2,
    "partial_static2": PartialStatic2,
    "joint2": DynamicJoint2,
    "joint2_seq": DynamicJointSeq2,
    "coord2": CoordinatedJoint2,
    "cascade2": CascadeLike2,
    "oracle2": OracleV2,
    "clairvoyant2": ClairvoyantJoint2,
}

V2_NEEDS_OBS = {n for n, c in V2POLICIES.items() if c.needs_obs}


def make_v2_policy(name: str, ctx: V2Ctx):
    return V2POLICIES[name](ctx)
