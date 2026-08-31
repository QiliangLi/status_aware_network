"""E2 routing baseline：Round-Robin / Load-aware / KV-aware（engine 均为 P1 static cost）。"""
from __future__ import annotations

from .base import BasePolicy, candidate_views
from .static_cost import StaticCost


class RoundRobin(StaticCost):
    def __init__(self, ctx):
        super().__init__(ctx)
        self._i = 0

    def route(self, req, views):
        holders = [v for v in views if v.holds_class] if req.hit else list(views)
        pool = holders or list(views)
        v = pool[self._i % len(pool)]
        self._i += 1
        return v.worker_id, None


class LoadAware(StaticCost):
    def route(self, req, views):
        v = min(views, key=lambda v: v.gpu.remaining_service())
        return v.worker_id, None


class KvAware(StaticCost):
    def route(self, req, views):
        cands = candidate_views(req, views)
        v = min(cands, key=lambda v: v.gpu.remaining_service())
        return v.worker_id, None
