"""P2 Dynamic Storage-Aware：读取可观测存储视图（陈旧/EMA/噪声）估计 fetch 代价。"""
from __future__ import annotations

from .static_cost import StaticCost


class DynamicCost(StaticCost):
    needs_obs = True

    def est_fetch(self, req, view):
        return view.obs.estimate(req.kv_gb, signal=self.ctx.signal)

    def engine_decide(self, req, view):
        if not req.hit or not view.holds_class:
            return "prefill"
        tf = self.est_fetch(req, view) + self.est_fetch_tail(req, view)
        tr = self.est_recompute(req, view)
        return self.decide(tf, tr)
