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
    action: str = ""        # v1: fetch | recompute | prefill；v2 另有 local | partial
    t_fetch_done: float = -1.0
    t_prefill_done: float = -1.0
    # ---- v2 ----
    node: int = -1
    tier: str = ""
    fetch_tokens: int = 0       # partial 动作取回的前缀 token 数
    fetch_gb: float = 0.0
    quote_est: float = -1.0     # 决策时的访问成本查询估计（秒），用于接口误差统计
    local_avail: bool = False   # 到达时是否有 worker 本地缓存持有该前缀
    sid: int = -1               # 会话 id（会话型负载；-1 = 非会话请求）

    @property
    def ttft(self) -> float:
        return self.t_prefill_done - self.arrival if self.t_prefill_done > 0 else None


def build_request(rid: int, t: float, cls_name: str, hit: bool, wl, model, sid: int = -1) -> Request:
    tokens = next(c.tokens for c in wl.classes if c.name == cls_name)
    cached = tokens if hit else 0
    return Request(
        rid=rid, arrival=t, cls=cls_name, hit=hit,
        cached_prefix_tokens=cached, suffix_tokens=wl.suffix_tokens,
        kv_gb=cached * model.kv_gb_per_token, ttft_slo=wl.ttft_slo,
        prompt_tokens=tokens + wl.suffix_tokens, sid=sid,
    )
