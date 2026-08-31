"""策略基类与公共工具。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorkerView:
    worker_id: int
    gpu: object            # GpuPool
    storage: object        # SharedKVStorage（所属 backend）
    obs: object = None     # StorageObservable（p2/p3 才有）
    holds_class: bool = True


@dataclass
class PolicyCtx:
    b_total: float
    t_base: float
    curve: object
    margin: float
    signal: str


def candidate_views(req, views):
    """KV-aware 候选：命中时优先持有该 prefix 的 worker。"""
    if req.hit:
        holders = [v for v in views if v.holds_class]
        if holders:
            return holders
    return list(views)


class BasePolicy:
    needs_obs = False

    def __init__(self, ctx: PolicyCtx):
        self.ctx = ctx

    # ---- 代价估计 ----
    def est_recompute(self, req, view):
        rate = max(0.02, 1.0 - view.gpu.bg_at(view.gpu.env.now))
        return view.gpu.remaining_service() / rate + self.ctx.curve(req.prompt_tokens) / rate

    def est_fetch_tail(self, req, view):
        """fetch 之后的 suffix prefill 时间（suffix 很小，忽略排队漂移）。"""
        rate = max(0.02, 1.0 - view.gpu.bg_at(view.gpu.env.now))
        return self.ctx.curve(req.suffix_tokens) / rate

    def decide(self, tf, tr):
        if tr < tf * (1.0 - self.ctx.margin):
            return "recompute"
        return "fetch"

    # ---- 路由：默认 KV-aware + 最短队列 ----
    def route(self, req, views):
        cands = candidate_views(req, views)
        v = min(cands, key=lambda v: v.gpu.remaining_service())
        return v.worker_id, None

    def engine_decide(self, req, view):
        raise NotImplementedError
