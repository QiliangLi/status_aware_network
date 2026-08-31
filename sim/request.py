"""请求对象与构造。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    rid: int
    arrival: float
    cls: str
    hit: bool
    cached_prefix_tokens: int
    suffix_tokens: int
    kv_gb: float
    ttft_slo: float
    prompt_tokens: int = 0
    worker: int = -1
    action: str = ""        # fetch | recompute | prefill(miss/no-hold)
    t_fetch_done: float = -1.0
    t_prefill_done: float = -1.0

    @property
    def ttft(self) -> float:
        return self.t_prefill_done - self.arrival if self.t_prefill_done > 0 else None


def build_request(rid: int, t: float, cls_name: str, hit: bool, wl, model) -> Request:
    tokens = next(c.tokens for c in wl.classes if c.name == cls_name)
    cached = tokens if hit else 0
    return Request(
        rid=rid, arrival=t, cls=cls_name, hit=hit,
        cached_prefix_tokens=cached, suffix_tokens=wl.suffix_tokens,
        kv_gb=cached * model.kv_gb_per_token, ttft_slo=wl.ttft_slo,
        prompt_tokens=tokens + wl.suffix_tokens,
    )
