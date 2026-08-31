"""Global Scheduler：持有当前策略，负责路由 + 动作决策的分发。"""
from __future__ import annotations

from dataclasses import replace


class Scheduler:
    def __init__(self, policy, worker_backend, class_backend_map):
        self.policy = policy
        self.worker_backend = worker_backend
        self.class_backend_map = class_backend_map

    def dispatch(self, req, views):
        cbm = self.class_backend_map
        holdings = cbm.get(req.cls, None)
        views2 = []
        for v in views:
            holds = True
            if req.hit and holdings is not None:
                holds = self.worker_backend[v.worker_id] in holdings
            views2.append(replace(v, worker_id=v.worker_id, holds_class=holds))
        wid, action = self.policy.route(req, views2)
        if action is None:
            action = self.policy.engine_decide(req, views2[wid])
        return wid, action
