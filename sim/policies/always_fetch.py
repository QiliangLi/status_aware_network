"""P0 Always Fetch：命中即取（sanity baseline）。"""
from __future__ import annotations

from .base import BasePolicy


class AlwaysFetch(BasePolicy):
    def engine_decide(self, req, view):
        if not req.hit or not view.holds_class:
            return "prefill"
        return "fetch"
