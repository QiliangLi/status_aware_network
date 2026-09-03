"""v2 Engine：在共享分布式 KV 存储上执行 local / fetch / partial(可重叠) / recompute / prefill。

fetch 链路 = 路径延迟 -> 存储节点 tier 服务 -> 共享 fabric 传输（三级顺序）。
partial：取回链路与 GPU 剩余 prefill 并行（overlap=True 时完成时间取 max）。
完成后将前缀 KV 写入该 worker 的本地缓存（LRU）。
"""
from __future__ import annotations


class EngineV2:
    def __init__(self, env, worker_id, world, metrics):
        self.env = env
        self.worker_id = worker_id
        self.world = world
        self.metrics = metrics

    def handle(self, req, dec):
        self.env.process(self._run(req, dec))

    # ---------- 取回链路 ----------
    def _chain(self, rid, node, tier, nbytes):
        if nbytes <= 1e-12:
            return
        yield self.env.timeout(self.world.path_lat[self.worker_id][node])
        res = self.world.res(node, tier)
        yield res.submit(rid, nbytes)
        yield self.world.fabric.submit(rid, nbytes)

    # ---------- 主流程 ----------
    def _run(self, req, dec):
        act = dec.action
        gpu = self.world.gpus[self.worker_id]
        if act == "local":
            yield gpu.submit(req.rid, req.suffix_tokens)
        elif act in ("recompute", "prefill"):
            yield gpu.submit(req.rid, req.prompt_tokens)
        elif act == "fetch":
            t0 = self.env.now
            yield self.env.process(self._chain(req.rid, dec.node, dec.tier, req.kv_gb))
            req.t_fetch_done = self.env.now
            self.metrics.on_fetch_chain(req, self.env.now - t0)
            yield gpu.submit(req.rid, req.suffix_tokens)
        elif act == "partial":
            t0 = self.env.now
            nbytes = dec.fetch_gb
            chain = self.env.process(self._chain(req.rid, dec.node, dec.tier, nbytes))
            if dec.overlap:
                g = gpu.submit(req.rid, req.prompt_tokens - dec.fetch_tokens)
                yield chain
                req.t_fetch_done = self.env.now
                self.metrics.on_fetch_chain(req, self.env.now - t0)
                yield g          # env.now 已为两者完成时间的 max
            else:
                yield chain
                req.t_fetch_done = self.env.now
                self.metrics.on_fetch_chain(req, self.env.now - t0)
                yield gpu.submit(req.rid, req.prompt_tokens - dec.fetch_tokens)
        else:
            raise ValueError(f"unknown action {act}")
        if req.hit:
            self.world.locals[self.worker_id].insert(req.cls, req.kv_gb)
        req.t_prefill_done = self.env.now
        self.metrics.on_complete(req, act)
