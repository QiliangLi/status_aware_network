"""可观测信息层：报价采样/滞后/噪声、公共客户端账本、估计快照（§6）。

信息边界铁落的类型化：ObservableSnapshot 不携带 world/storage/engine 引用，
普通策略签名只能收到它。物理真值只在 include_oracle=True 的诊断快照中出现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .config import CqScenario
from .profile import batch_compute_s, layer_read_gb


@dataclass(frozen=True)
class PublicRequest:
    rid: int
    arrival_s: object
    h_tokens: int
    u_tokens: int
    deadline_s: object
    T0_s: object


@dataclass(frozen=True)
class PublicWorker:
    worker_id: int
    state: str                 # IDLE | COMPUTE | STALL
    members: Tuple[int, ...]   # 当前批成员（空=空闲）
    layers_done: int           # 已启动计算的层数
    cur_layer_start_s: object  # 正在计算层的开始时刻（无则 None）
    events: Tuple[Tuple, ...]  # 已发生的 (t, kind, layer) 公开事件


@dataclass(frozen=True)
class PublicFlow:
    flow_id: int
    submit_seq: int
    batch_id: int
    layer: int
    V_gb: object
    submit_s: object
    completed: bool
    completed_s: Optional[object]
    served_reported_gb: object


@dataclass(frozen=True)
class Quote:
    sample_s: object
    delivered_s: object
    bw_hat_gbps: object
    reported_bytes: Dict[int, object]
    quality: str = "ideal"


@dataclass
class ObservableSnapshot:
    """普通策略的唯一输入。预测用 est_bw/pred_* 辅助方法。"""

    now: object
    requests: List[PublicRequest]
    workers: List[PublicWorker]
    flows: List[PublicFlow]
    quote: Optional[Quote]
    queued: frozenset
    idle_workers: Tuple[int, ...]
    b_ref: object
    q_max: object
    compute_bias: object = 0
    oracle: Optional[Dict] = None
    _profile_cfg: object = None      # 预测 profile（真值×(1+e)），不暴露真值世界
    _lam: object = None

    # -- 禁止字段扫描（U17）：序列化白名单 --------------------------------
    def public_fields(self):
        return {k: getattr(self, k) for k in
                ("now", "requests", "workers", "flows", "quote", "queued",
                 "idle_workers", "b_ref", "q_max", "compute_bias")}

    # -- 预测辅助（§6.2）---------------------------------------------------
    def est_bw(self):
        """预测带宽：max(1e-6, bw_hat)；history 模式由报价层给 B_ref。"""
        if self.quote is None:
            return max(1e-6, float(self.b_ref))
        return max(1e-6, float(self.quote.bw_hat_gbps))

    def c_hat(self, members_h_u) -> float:
        """预测整批每层计算时长：真值公式 × (1+e)。"""
        from .types import RequestSpec
        specs = [RequestSpec(rid=i, arrival_s=0, h_tokens=h, u_tokens=u,
                             class_id="adhoc", T0_s=1, deadline_s=1)
                 for i, (h, u) in enumerate(members_h_u)]
        return float(batch_compute_s(specs, self._profile_cfg))

    def pred_singleton_s(self, rid: int) -> float:
        """空副本 singleton 预测（预测带宽+预测 profile），不含队列等待。"""
        req = next(r for r in self.requests if r.rid == rid)
        c = self.c_hat([(req.h_tokens, req.u_tokens)])
        V = float(layer_read_gb(req, self._profile_cfg))
        eff = min(self.est_bw(), float(self.q_max))
        tr = V / eff
        L = self._profile_cfg.L
        return tr + L * c + (L - 1) * max(0.0, tr - c)

    def slack_s(self, rid: int) -> float:
        req = next(r for r in self.requests if r.rid == rid)
        return float(req.deadline_s) - float(self.now) - self.pred_singleton_s(rid)


class Observable:
    """挂在引擎上的观测层：采样、交付、账本、快照构建。"""

    def __init__(self, scenario: CqScenario, numeric=float, seed: int = 0):
        self.scn = scenario
        self.num = numeric
        cfg = scenario.observation
        self.mode = cfg.mode
        self.period = float(cfg.sample_period_s)
        self.lag = float(cfg.delivery_lag_s)
        self.period_n = numeric(cfg.sample_period_s)
        self.lag_n = numeric(cfg.delivery_lag_s)
        self.sigma = float(cfg.sigma)
        self.compute_bias = float(cfg.compute_bias)
        self.b_ref = float(scenario.storage.b_ref_gbps)
        self.seed = seed
        self.eng = None
        self.sample_index = 0
        self.last_sample_t = None
        self.delivered: Optional[Quote] = None
        self.pending_deliveries: List[Tuple[float, Quote]] = []
        self.ledger: Dict[int, dict] = {}     # flow_id -> 公开信息
        self.worker_events: Dict[int, list] = {i: [] for i in range(scenario.m_workers)}
        # 预测 profile：真值参数 × (1+e)
        from dataclasses import replace
        from fractions import Fraction as _F
        p = scenario.profile
        k = _F(1) + _F(cfg.compute_bias).limit_denominator(10**9)
        self.profile_hat = replace(
            p, t_launch_s=p.t_launch_s * k, a_s_per_token=p.a_s_per_token * k,
            b_s_per_pair=p.b_s_per_pair * k)

    def attach(self, engine):
        self.eng = engine

    # -- 事件源 -------------------------------------------------------------
    def next_event_time(self, t):
        cands = []
        nxt = self._next_sample_time(t)
        if nxt is not None:
            cands.append(nxt)
        for dat, _q in self.pending_deliveries:
            if dat > t:
                cands.append(dat)
        return min(cands) if cands else None

    def _next_sample_time(self, t):
        if self.period <= 0:
            return None
        if self.last_sample_t is None:
            return t if t == 0 else t  # t=0 首份在下一次 on_instant 发布
        return self.last_sample_t + self.period_n

    def _noise_factor(self) -> float:
        if self.sigma <= 0:
            return 1.0
        import numpy as np
        rng = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([self.seed, self.sample_index])))
        z = float(rng.standard_normal())
        import math
        return math.exp(self.sigma * z - self.sigma * self.sigma / 2)

    def _sample(self, t):
        eng = self.eng
        w = eng.w
        if self.mode == "history":
            bw = self.b_ref
        else:
            bw = float(w.storage.b_at(t)) * self._noise_factor()
        rep = {}
        for f in w.storage.flows.values():
            rep[f.flow_id] = f.V_gb - f.remaining_gb
        q = Quote(sample_s=self.num(t), delivered_s=self.num(t + self.lag_n),
                  bw_hat_gbps=self.num(bw), reported_bytes=rep)
        self.sample_index += 1
        self.last_sample_t = t
        self.pending_deliveries.append((t + self.lag_n, q))

    # -- 引擎钩子 -------------------------------------------------------------
    def on_instant(self, t):
        # 采样（首个事件时刻 t=0 即发布）
        while (self.period > 0 and
               (self.last_sample_t is None or
                self.last_sample_t + self.period_n <= t)):
            base = 0 if self.last_sample_t is None else self.last_sample_t + self.period_n
            self._sample(min(base, t) if base > t else base)
        # 交付
        still = []
        for dat, q in self.pending_deliveries:
            if dat <= t:
                self.delivered = q
            else:
                still.append((dat, q))
        self.pending_deliveries = still

    def on_flow_submit(self, f, t):
        self.ledger[f.flow_id] = {
            "flow_id": f.flow_id, "submit_seq": f.submit_seq,
            "batch_id": f.batch_id, "layer": f.layer, "V_gb": f.V_gb,
            "submit_s": t, "completed": False, "completed_s": None,
            "served_reported_gb": None}

    def on_flow_complete(self, f, t):
        rec = self.ledger.get(f.flow_id)
        if rec is not None:
            rec["completed"] = True
            rec["completed_s"] = t
            rec["served_reported_gb"] = f.V_gb

    def on_arrival(self, rid, t):
        pass

    def on_dispatch(self, b, t):
        evs = self.worker_events[b.worker_id]
        evs.append((t, "dispatch", tuple(b.members)))

    def on_batch_done(self, b, F):
        evs = self.worker_events[b.worker_id]
        evs.append((F, "batch_done", tuple(b.members)))

    # -- 快照 ---------------------------------------------------------------
    def snapshot(self, t, include_oracle=False) -> ObservableSnapshot:
        eng = self.eng
        w = eng.w
        reqs = []
        for rid in sorted(w.requests):
            rr = w.requests[rid]
            if rr.state in ("QUEUED", "ACTIVE"):
                s = rr.spec
                reqs.append(PublicRequest(rid, s.arrival_s, s.h_tokens,
                                          s.u_tokens, s.deadline_s, s.T0_s))
        workers = []
        for i in range(w.m):
            wk = w.workers[i]
            b = w.batches.get(wk.batch_id) if wk.batch_id is not None else None
            members = b.members if b else ()
            layers_done = b.next_layer if b else 0
            cur_start = b.C[b.next_layer - 1] if (b and b.next_layer > 0) else None
            workers.append(PublicWorker(i, wk.state, members, layers_done,
                                        cur_start, tuple(self.worker_events[i])))
        flows = []
        for fid, rec in sorted(self.ledger.items()):
            served = None
            if self.delivered is not None and fid in self.delivered.reported_bytes:
                served = self.delivered.reported_bytes[fid]
            if rec["completed"]:
                served = rec["V_gb"]
            flows.append(PublicFlow(
                fid, rec["submit_seq"], rec["batch_id"], rec["layer"],
                rec["V_gb"], rec["submit_s"], rec["completed"],
                rec["completed_s"], served))
        oracle = None
        if include_oracle:
            oracle = {
                "b_true_gbps": w.storage.b_at(t),
                "flow_remaining": {f.flow_id: f.remaining_gb
                                   for f in w.storage.flows.values()},
                "flow_rate": {f.flow_id: f.rate_gbps
                              for f in w.storage.flows.values()},
                "batch_compute_end": {b.batch_id: b.Z[b.next_layer - 1]
                                      for b in w.batches.values()
                                      if b.next_layer > 0 and b.F_s is None},
            }
        return ObservableSnapshot(
            now=t, requests=reqs, workers=workers, flows=flows,
            quote=self.delivered,
            queued=frozenset(r.spec.rid for r in w.requests.values()
                             if r.state == "QUEUED"),
            idle_workers=tuple(w.idle_workers()),
            b_ref=self.num(self.b_ref), q_max=w.storage.q_max,
            compute_bias=self.num(self.compute_bias), oracle=oracle,
            _profile_cfg=self.profile_hat)
