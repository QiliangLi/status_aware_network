"""M1 引擎与 G 系列核心：迭代级调度 + waiting/running 队列 + HBM 容量 + 共享存储取回。

对应设计文档《全局共享KV场景的存储感知调度策略与实验设计-20260908.md》：
- S-G 场景：目录全局可见（MetadataDirectory）、local_cache_gb=0（无本地缓存复用）、
  副本放置冻结；KV 唯一持久来源是共享存储。
- 共同引擎纪律：每迭代先推进全部 decode，剩余 token 预算按 admission 顺序切 prefill
  chunk（vLLM V1 式 chunked prefill + decode 优先，纲领 v1.1 §5.2 主配置）。
- 保守准入（E-NoEvict 语义，纲领 §5.3 第一版）：活动集合按 (N+O) 全程预留 HBM，
  无抢占；所有 D2 排序策略共用此纪律。
- 命中即取：fetch 由 router 在请求到达时提交（设计 §3.2 E-FCFS），取回完成前请求
  停留在 waiting，不阻塞其他 ready 请求（纲领 §3.2）。
- 输出长度 O 为类别确定值（本轮简化：O_hat == O，无预测误差维度）。

单位约定沿用 sim/config.py：时间秒、字节 GB（十进制）、带宽 GB/s。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import simpy

from .config import (GpuConfig, ModelConfig, ObsConfig, PrefixClass,
                     StorageConfig, TopoConfig, WorkloadConfig)
from .gpu import PrefillCurve
from .policies_g import GCtx, make_router, order_key
from .quote import AccessCostQuery
from .topology import World

EPS = 1e-9


# ---------------------------------------------------------------- 配置

@dataclass(frozen=True)
class M1Class:
    """G 系列负载类别：命中 H / 后缀 U / 输出 O 三维独立（N = H + U）。

    prio 为 E-Prio 基线的外生优先级（合成对照，跨策略固定）；
    slo_ttft / slo_tpot 由 g_common 的合成公式在冻结时计算后传入。
    """
    name: str
    H: int
    U: int
    O: int
    share: float
    prio: int = 0
    slo_ttft: float = 1.0
    slo_tpot: float = 0.05


@dataclass(frozen=True)
class M1Config:
    token_budget: int = 4096        # 每迭代 prefill token 预算（chunked prefill）
    decode_t0: float = 0.004        # decode 迭代固定开销（秒，kernel 启动地板）
    decode_t1: float = 0.0008       # 每个并发 decode 请求的边际迭代时间
    hbm_gb: float = 64.0            # 每 worker HBM 容量（保守预留口径）
    margin_s: float = 240.0         # 停止到达后的随访时长（覆盖长 decode）
    gate_poll: float = 0.02         # D2d 门控轮询周期
    # D2d 双上限档位：(level, 并发条数上限, 在途字节上限 GB)；NORMAL 为不限
    gate_levels: tuple = (("NORMAL", math.inf, math.inf),
                          ("WARM", 6.0, 12.0),
                          ("HOT", 3.0, 6.0),
                          ("CRITICAL", 1.0, 2.0))
    guardband: float = 1.2          # D2c 预算排序的 γ（L̂r 保守系数）


@dataclass(frozen=True)
class M1Spec:
    exp: str
    router: str            # rr | ll | dynkv | moons | hist | histj | d1b | dynar | d1a
    order: str             # fcfs | lpm | prio | ready | budget（gate 为 router 独立开关）
    gate: bool = False     # D2d 取回准入门控
    seed: int = 0
    duration: float = 300.0
    warmup: float = 30.0
    lam: float = 2.5
    classes: tuple = ()    # (M1Class, ...)
    topo: "TopoConfig | None" = None
    obs: ObsConfig = ObsConfig()
    m1: M1Config = M1Config()
    model: ModelConfig = ModelConfig()
    gpu: GpuConfig = GpuConfig()
    timeline: bool = False  # 采集时间线（代表运行用）
    fixed_trace: tuple = ()  # 显式 trace [(t, cls)]，测试用（空 = Poisson 生成）


# ---------------------------------------------------------------- 请求

@dataclass
class M1Req:
    rid: int
    cls: str
    arrival: float
    H: int
    U: int
    N: int
    O: int
    O_max: int                 # 公开输出上限（本轮 = O）
    kv_gb: float               # 命中前缀 KV 字节（fetch 传输量）
    prio: int
    slo_ttft: float
    slo_tpot: float
    hit: bool
    # 决策（router 填写）
    action: str = ""           # fetch | recompute | prefill
    worker: int = -1
    node: int = -1
    tier: str = ""
    dec_est: float = -1.0      # 决策时估计成本（诊断）
    # 执行状态
    prefill_need: int = 0      # fetch=U，recompute=N，prefill=N
    pfilled: int = 0
    phase: str = "waiting"     # waiting -> prefill -> decode -> done
    fetch_done: bool = True    # 非 fetch 动作恒 True
    fetch_t0: float = -1.0
    fetch_t1: float = -1.0
    tok: int = 0
    t_first: float = -1.0
    t_done: float = -1.0
    prev_tok_t: float = -1.0
    itl_max: float = 0.0
    okey: tuple = (0, 0.0)     # waiting 排序键（到达时一次性计算）


# ---------------------------------------------------------------- 取回门控（D2d）

class FetchGate:
    """D2d：按 quote 压力档位限制并发 fetch 条数与在途字节（双上限，设计 §4.2）。"""

    def __init__(self, quote: AccessCostQuery, world: World, cfg: M1Config):
        self.quote = quote
        self.world = world
        self.limits = {lv[0]: (lv[1], lv[2]) for lv in cfg.gate_levels}
        self.order = [lv[0] for lv in cfg.gate_levels]
        self.n_inflight = 0
        self.gb_inflight = 0.0

    def _limit_pair(self, node: int, tier: str):
        ti = self.world.res_idx(node, tier)
        fi = len(self.world.resources) - 1
        li = max(self.order.index(self.quote.pressure(ti)),
                 self.order.index(self.quote.pressure(fi)))
        lv = self.order[li]
        c, b = self.limits[lv]
        return c, b

    def allow(self, node: int, tier: str, gb: float) -> bool:
        c, b = self._limit_pair(node, tier)
        if c is math.inf and b is math.inf:
            return True
        return self.n_inflight + 1 <= c + EPS and self.gb_inflight + gb <= b + EPS

    def add(self, gb: float) -> None:
        self.n_inflight += 1
        self.gb_inflight += gb

    def remove(self, gb: float) -> None:
        self.n_inflight -= 1
        self.gb_inflight = max(0.0, self.gb_inflight - gb)


# ---------------------------------------------------------------- 历史估计（I1，B-Hist 基线）

class HistoryEstimator:
    """按 (node,tier) 的 EWMA 传输速率 + 自身在途字节记账（不触碰存储内部真值）。

    无历史样本时回退 nominal 带宽（0.8×min(tier,fabric)），保证不劣于静态成本太多。
    """

    def __init__(self, world: World, alpha: float = 0.3):
        self.world = world
        self.alpha = alpha
        self.rate: dict[tuple, float] = {}
        self.inflight_gb = 0.0

    def update(self, node: int, tier: str, gb: float, secs: float) -> None:
        if secs > 1e-9 and gb > 1e-9:
            r = gb / secs
            old = self.rate.get((node, tier))
            self.rate[(node, tier)] = r if old is None else old + self.alpha * (r - old)

    def est(self, gb: float, w: int, node: int, tier: str) -> float:
        r = self.rate.get((node, tier))
        if r is None:
            r = 0.8 * min(self.world.res(node, tier).b_total, self.world.fabric.b_total)
        r = max(r, 1.0)
        t_base = self.world.res(node, tier).t_base
        return (self.world.path_lat[w][node] + (self.inflight_gb + gb) / r
                + t_base + t_base)

    def observe_fetch(self, node: int, tier: str, gb: float,
                      t0: float, t1: float, path_lat: float) -> None:
        self.update(node, tier, gb, max(0.0, t1 - t0 - path_lat))


# ---------------------------------------------------------------- 引擎（每 worker）

class M1Worker:
    """迭代级引擎：waiting（排序策略）→ running（保守 HBM 准入）→ 迭代推进。

    迭代时间模型：T_iter = [Σ_chunks curve(c) + (d0 + d1·n_decode)] / rate(t)，
    rate = 1 − gpu 背景负载（计算侧真值可见，全部策略对称）。
    """

    def __init__(self, env, wid: int, sysm: "M1System"):
        self.env = env
        self.wid = wid
        self.sys = sysm
        self.cfg = sysm.spec.m1
        self.curve = sysm.curve
        self.kvgb = sysm.kvgb
        self.waiting: list[M1Req] = []
        self.running: list[M1Req] = []
        self.hbm_resv = 0.0
        self._wake_ev = None
        # 资源记账（秒，含 rate 折算前的服务时间）
        self.busy_time = 0.0
        self.prefill_time = 0.0
        self.nec_prefill_time = 0.0
        self.dup_prefill_time = 0.0
        self.decode_time = 0.0
        self.idle_empty = 0.0      # 队列全空的空闲
        self.idle_kv_wait = 0.0    # 有等待请求但全被 fetch 卡住的 GPU 空转
        self.hbm_peak_gb = 0.0
        self.n_admitted = 0
        env.process(self._loop())

    # ---- 外部事件唤醒（到达 / fetch 完成 / 完成） ----
    def wake(self) -> None:
        ev = self._wake_ev
        if ev is not None and not ev.triggered:
            self._wake_ev = None
            ev.succeed()

    def enqueue(self, req: M1Req) -> None:
        self.waiting.append(req)

    def bg_at(self, t: float) -> float:
        return self.sys.gpu_bg[self.wid](t)

    def rate(self, t: float) -> float:
        return max(0.02, 1.0 - self.bg_at(t))

    def wait_est(self) -> float:
        """GPU 排队估计（计算侧真值可见，全部策略共用同一估计器；含速率折算）。"""
        rem_pre = sum(r.prefill_need - r.pfilled for r in self.running
                      if r.phase == "prefill")
        decs = [r for r in self.running if r.phase == "decode"]
        max_iters = max((r.O - r.tok for r in decs), default=0)
        step = self.cfg.decode_t0 + self.cfg.decode_t1 * max(1, len(decs))
        return (self.curve(rem_pre) + step * max_iters) / self.rate(self.env.now)

    # ---- 主循环 ----
    def _loop(self):
        env = self.env
        while True:
            self._admit()
            if not self.running:
                if not self.waiting:
                    idle_kind = "empty"
                else:
                    idle_kind = "kv_wait" if all(
                        r.action == "fetch" and not r.fetch_done for r in self.waiting
                    ) else "empty"
                self._wake_ev = env.event()
                t0 = env.now
                yield self._wake_ev
                dt = env.now - t0
                if idle_kind == "kv_wait":
                    self.idle_kv_wait += dt
                else:
                    self.idle_empty += dt
                continue

            # ---- 构造迭代：decode 全量 + 剩余预算切 prefill（admission 顺序）----
            decs = [r for r in self.running if r.phase == "decode"]
            budget = self.cfg.token_budget
            chunks: list[tuple[M1Req, int]] = []
            for r in self.running:
                if r.phase != "prefill":
                    continue
                need = r.prefill_need - r.pfilled
                c = min(need, budget)
                if c > 0:
                    chunks.append((r, c))
                    budget -= c
                if budget <= 0:
                    break
            t_pre = sum(self.curve(c) for _, c in chunks)
            t_dec = self.cfg.decode_t0 + self.cfg.decode_t1 * len(decs)
            t_iter = (t_pre + t_dec) / self.rate(env.now)
            # 记账（时间归因）
            self.busy_time += t_iter
            self.prefill_time += t_pre / self.rate(env.now)
            self.decode_time += t_dec / self.rate(env.now)
            for r, c in chunks:
                overlap = max(0, min(r.pfilled + c, r.H) - r.pfilled) if r.action == "recompute" else 0
                dup = t_pre * (overlap / c) / self.rate(env.now)
                self.dup_prefill_time += dup
                self.nec_prefill_time += t_pre * (1 - overlap / c) / self.rate(env.now)
            if self.sys.spec.timeline:
                self.sys.timeline_iters.append(
                    (env.now, self.wid, t_pre, t_dec, len(chunks), len(decs)))
            yield env.timeout(t_iter)

            # ---- 推进 ----
            now = env.now
            for r, c in chunks:
                r.pfilled += c
                if r.pfilled >= r.prefill_need:
                    r.phase = "decode"
                    r.tok = 1
                    r.t_first = now
                    r.prev_tok_t = now
                    self.sys.metrics.on_first_token(r)
            for r in decs:
                r.tok += 1
                dt = now - r.prev_tok_t
                r.prev_tok_t = now
                if dt > r.itl_max:
                    r.itl_max = dt
                if r.tok >= r.O:
                    self._complete(r)

    def _admit(self) -> None:
        self.waiting.sort(key=lambda r: r.okey)
        rest = []
        for r in self.waiting:
            if r.action == "fetch" and not r.fetch_done:
                rest.append(r)
                continue
            need_gb = (r.N + r.O_max) * self.kvgb
            if self.hbm_resv + need_gb <= self.cfg.hbm_gb + EPS:
                r.phase = "prefill"
                r.prefill_need = r.U if r.action == "fetch" else r.N
                self.running.append(r)
                self.hbm_resv += need_gb
                self.hbm_peak_gb = max(self.hbm_peak_gb, self.hbm_resv)
                self.n_admitted += 1
                self.sys.metrics.on_admit(r)
            else:
                rest.append(r)
        self.waiting = rest

    def _complete(self, r: M1Req) -> None:
        r.phase = "done"
        r.t_done = self.env.now
        self.running.remove(r)
        self.hbm_resv = max(0.0, self.hbm_resv - (r.N + r.O) * self.kvgb)
        self.sys.metrics.on_complete(r)
        self.wake()


# ---------------------------------------------------------------- 指标

class M1Metrics:
    """逐请求记录 + 到达窗口聚合（cohort：warmup 后到达的全部请求，含未完成）。"""

    def __init__(self):
        self.reqs: list[M1Req] = []
        self.n_first = 0

    def on_admit(self, r: M1Req) -> None:
        pass

    def on_first_token(self, r: M1Req) -> None:
        self.n_first += 1

    def on_complete(self, r: M1Req) -> None:
        self.reqs.append(r)

    def summarize(self, cohort: list[M1Req], window: float) -> dict:
        import numpy as np
        by_cls: dict[str, list[M1Req]] = {}
        for r in cohort:
            by_cls.setdefault(r.cls, []).append(r)
        per_class = {}
        all_ttft, all_att = [], []
        for cls, rs in sorted(by_cls.items()):
            ttfts = [r.t_first - r.arrival for r in rs if r.t_first > 0]
            undone = [r for r in rs if r.t_done <= 0]
            att = 0
            for r in rs:
                ok = r.t_done > 0 and (r.t_first - r.arrival) <= r.slo_ttft + EPS
                if ok and r.O > 1:
                    tpot = (r.t_done - r.t_first) / (r.O - 1)
                    ok = tpot <= r.slo_tpot + EPS
                att += 1 if ok else 0
            tpots = [(r.t_done - r.t_first) / (r.O - 1) for r in rs
                     if r.t_done > 0 and r.O > 1]
            itls = [r.itl_max for r in rs if r.t_done > 0 and r.O > 1]
            n_hit = sum(1 for r in rs if r.hit)
            n_fetch = sum(1 for r in rs if r.action == "fetch")
            def pct(a, q):
                return float(np.percentile(a, q)) if a else float("nan")
            per_class[cls] = dict(
                n=len(rs), n_undone=len(undone),
                ttft_p50=pct(ttfts, 50), ttft_p95=pct(ttfts, 95), ttft_p99=pct(ttfts, 99),
                tpot_mean=float(np.mean(tpots)) if tpots else float("nan"),
                tpot_p95=pct(tpots, 95), itl_p95=pct(itls, 95),
                attain=att / max(1, len(rs)),
                hit_share=n_hit / max(1, len(rs)),
                fetch_of_hit=n_fetch / max(1, n_hit),
            )
            all_ttft += ttfts
            all_att.append(per_class[cls]["attain"])
        done = [r for r in cohort if r.t_done > 0]
        overall = dict(
            n=len(cohort), n_undone=len(cohort) - len(done),
            ttft_p50=pct(all_ttft, 50), ttft_p95=pct(all_ttft, 95), ttft_p99=pct(all_ttft, 99),
            attain_mean=float(np.mean([c["attain"] for c in per_class.values()])) if per_class else float("nan"),
            attain_min=min([c["attain"] for c in per_class.values()], default=float("nan")),
            goodput=sum(1 for r in cohort if r.t_done > 0
                        and (r.t_first - r.arrival) <= r.slo_ttft + EPS
                        and (r.O <= 1 or (r.t_done - r.t_first) / (r.O - 1) <= r.slo_tpot + EPS)) / max(EPS, window),
        )
        return dict(per_class=per_class, overall=overall)


# ---------------------------------------------------------------- 系统

class M1System:
    """装配 World（复用 v2 存储拓扑/可观测视图/quote）+ M1 引擎 + 路由/排序策略。"""

    def __init__(self, spec: M1Spec):
        self.spec = spec
        env = simpy.Environment()
        self.env = env
        # legacy shim：World 只读 topo/gpu/wl/obs/seed/model
        wl = WorkloadConfig(
            lam=spec.lam, hit_ratio=1.0, suffix_tokens=1,
            classes=tuple(PrefixClass(c.name, c.H, c.share) for c in spec.classes),
        )
        self.world = World(env, SimpleNamespace(
            topo=spec.topo, gpu=spec.gpu, wl=wl, obs=spec.obs,
            seed=spec.seed, model=spec.model,
        ))
        self.curve = self.world.curve
        self.kvgb = spec.model.kv_gb_per_token
        self.quote = AccessCostQuery(self.world, spec.obs)
        self.hist = HistoryEstimator(self.world)
        self.gate = FetchGate(self.quote, self.world, spec.m1)
        self.metrics = M1Metrics()
        self.workers = [M1Worker(env, w, self) for w in range(self.world.n_workers)]
        self.gpu_bg = [g.bg_at for g in self.world.gpus]
        self.ctx = GCtx(world=self.world, quote=self.quote, hist=self.hist,
                        curve=self.curve, workers=self.workers, m1=spec.m1,
                        rng=np.random.default_rng([spec.seed, 999, 7]),
                        margin=0.0, guardband=spec.m1.guardband)
        self.router = make_router(spec.router, self.ctx)
        self.timeline_iters: list = []
        self.timeline_res: list = []
        self.cohort: list[M1Req] = []
        self.rid = 0
        self.trace = self._gen_trace()

    def _gen_trace(self) -> list:
        if self.spec.fixed_trace:
            return [(float(t), c) for (t, c) in self.spec.fixed_trace]
        rng = np.random.default_rng([self.spec.seed, 1, 2])
        names = [c.name for c in self.spec.classes]
        shares = np.array([c.share for c in self.spec.classes], dtype=float)
        shares = shares / shares.sum()
        out = []
        t = float(rng.exponential(1.0 / self.spec.lam))
        while t < self.spec.duration:
            cls = str(rng.choice(names, p=shares))
            out.append((t, cls))
            t += float(rng.exponential(1.0 / self.spec.lam))
        return out

    def _class(self, name: str) -> M1Class:
        return next(c for c in self.spec.classes if c.name == name)

    # ---- 到达处理：路由决策 + fetch 提交 + 入队 ----
    def _arrival(self, t: float, cls_name: str) -> None:
        c = self._class(cls_name)
        holders = self.world.dir.holders(c.name) if c.H > 0 else set()
        hit = len(holders) > 0
        self.rid += 1
        req = M1Req(
            rid=self.rid, cls=c.name, arrival=t, H=c.H, U=c.U, N=c.H + c.U, O=c.O,
            O_max=c.O, kv_gb=c.H * self.kvgb, prio=c.prio,
            slo_ttft=c.slo_ttft, slo_tpot=c.slo_tpot, hit=hit,
        )
        w, action, node, tier, est = self.router.decide(req)
        req.worker, req.action, req.node, req.tier, req.dec_est = w, action, node, tier, est
        if t >= self.spec.warmup:
            self.cohort.append(req)
        if action == "fetch":
            req.fetch_done = False
            self.env.process(self._fetch(req))
        # 排序键：到达时一次性计算（避免逐迭代重算导致噪声多抽，CRN 一致）
        req.okey = order_key(self.spec.order, req, self.ctx)
        self.workers[w].enqueue(req)
        self.workers[w].wake()

    def _fetch(self, req: M1Req):
        # D2d 门控（仅 gate=True 的策略变体启用；设计 §4.2 双上限）
        if self.spec.gate:
            while not self.gate.allow(req.node, req.tier, req.kv_gb):
                yield self.env.timeout(self.spec.m1.gate_poll)
        self.gate.add(req.kv_gb)
        self.hist.inflight_gb += req.kv_gb
        req.fetch_t0 = self.env.now
        yield self.env.timeout(self.world.path_lat[req.worker][req.node])
        yield self.world.res(req.node, req.tier).submit(req.rid, req.kv_gb)
        yield self.world.fabric.submit(req.rid, req.kv_gb)
        req.fetch_t1 = self.env.now
        req.fetch_done = True
        self.hist.inflight_gb = max(0.0, self.hist.inflight_gb - req.kv_gb)
        self.hist.observe_fetch(req.node, req.tier, req.kv_gb,
                                req.fetch_t0, req.fetch_t1,
                                self.world.path_lat[req.worker][req.node])
        self.gate.remove(req.kv_gb)
        self.workers[req.worker].wake()

    def _sampler(self):
        while True:
            yield self.env.timeout(0.25)
            if not self.spec.timeline:
                continue
            self.timeline_res.append((self.env.now,
                                      *[len(r.active) for r in self.world.resources],
                                      *[w.hbm_resv for w in self.workers],
                                      *[len(w.waiting) for w in self.workers]))

    def run(self) -> dict:
        env = self.env
        env.process(self._sampler())

        def arrivals():
            for t, cls in self.trace:
                yield env.timeout(t - env.now)
                self._arrival(env.now, cls)
        env.process(arrivals())
        env.run(until=self.spec.duration + self.spec.m1.margin_s)

        # ---- 汇总 ----
        window = self.spec.duration - self.spec.warmup
        m = self.metrics.summarize(self.cohort, window)
        T = self.spec.duration - self.spec.warmup
        workers = []
        for w in self.workers:
            wall = w.busy_time + w.idle_empty + w.idle_kv_wait
            workers.append(dict(
                busy_frac=w.busy_time / max(EPS, wall),
                nec_prefill_s=w.nec_prefill_time, dup_prefill_s=w.dup_prefill_time,
                decode_s=w.decode_time, idle_empty_s=w.idle_empty,
                idle_kv_wait_s=w.idle_kv_wait, hbm_peak_gb=w.hbm_peak_gb,
            ))
        storage = []
        for name, r in zip(self.world.res_names, self.world.resources):
            served = r.bytes_served
            storage.append(dict(name=name, served_gb=served,
                                util_fg=served / max(EPS, r.b_total * T),
                                bg_gbps=r.bg_at(T)))
        n_fetch = sum(1 for r in self.cohort if r.action == "fetch")
        n_rec = sum(1 for r in self.cohort if r.action == "recompute")
        fetch_gb = sum(r.kv_gb for r in self.cohort if r.action == "fetch")
        fetch_t = [r.fetch_t1 - r.fetch_t0 for r in self.cohort
                   if r.action == "fetch" and r.fetch_t1 > 0]
        out = dict(
            exp=self.spec.exp, router=self.spec.router, order=self.spec.order,
            gate=self.spec.gate, seed=self.spec.seed,
            duration=self.spec.duration, warmup=self.spec.warmup, lam=self.spec.lam,
            per_class=m["per_class"], overall=m["overall"],
            workers=workers, storage=storage,
            n_fetch=n_fetch, n_recompute=n_rec, fetch_gb=fetch_gb,
            fetch_time_p50=float(np.percentile(fetch_t, 50)) if fetch_t else float("nan"),
            fetch_time_p95=float(np.percentile(fetch_t, 95)) if fetch_t else float("nan"),
        )
        if self.spec.timeline:
            out["timeline_iters"] = self.timeline_iters
            out["timeline_res"] = self.timeline_res
        return out
