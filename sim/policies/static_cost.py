"""P1 Static Cost：只知道 nominal 配置带宽，不知道实时拥塞。"""
from __future__ import annotations

from .base import BasePolicy


class StaticCost(BasePolicy):
    def est_fetch(self, req, view):
        return req.kv_gb / self.ctx.b_total + self.ctx.t_base

    def engine_decide(self, req, view):
        if not req.hit or not view.holds_class:
            return "prefill"
        tf = self.est_fetch(req, view) + self.est_fetch_tail(req, view)
        tr = self.est_recompute(req, view)
        return self.decide(tf, tr)
