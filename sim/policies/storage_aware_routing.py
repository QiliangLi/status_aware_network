"""P3 Storage-Aware Routing：Global Scheduler 用 min(fetch, recompute) 联合选 worker 与动作。"""
from __future__ import annotations

from .base import BasePolicy, candidate_views
from .static_cost import StaticCost


class StorageAwareRoute(BasePolicy):
    needs_obs = True

    def cost(self, req, view):
        tr = self.est_recompute(req, view)
        if not req.hit or not view.holds_class:
            return tr, "prefill"
        tf = view.obs.estimate(req.kv_gb, signal=self.ctx.signal) + self.est_fetch_tail(req, view)
        if tr < tf * (1.0 - self.ctx.margin):
            return tr, "recompute"
        return tf, "fetch"

    def route(self, req, views):
        cands = candidate_views(req, views)
        best = None
        for v in cands:
            c, act = self.cost(req, v)
            if best is None or c < best[0]:
                best = (c, v.worker_id, act)
        return best[1], best[2]

    def engine_decide(self, req, view):
        return StaticCost.engine_decide(self, req, view)
