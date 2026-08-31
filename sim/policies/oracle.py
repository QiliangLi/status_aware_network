"""P4 Oracle：读 ground truth 并精确预测自身插入后的完成时间。"""
from __future__ import annotations

from .base import BasePolicy


class Oracle(BasePolicy):
    def cost(self, req, view):
        tr = view.gpu.hypothetical(req.prompt_tokens)
        if not req.hit or not view.holds_class:
            return tr, "prefill"
        tf = view.storage.hypothetical_fetch_time(req.kv_gb) + self.est_fetch_tail(req, view)
        if tr <= tf:
            return tr, "recompute" if req.hit else "prefill"
        return tf, "fetch"

    def route(self, req, views):
        from .base import candidate_views
        cands = candidate_views(req, views)
        best = None
        for v in cands:
            c, act = self.cost(req, v)
            if best is None or c < best[0]:
                best = (c, v.worker_id, act)
        return best[1], best[2]

    def engine_decide(self, req, view):
        return self.cost(req, view)[1]
