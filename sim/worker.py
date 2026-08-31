"""Engine（worker 侧）：执行 fetch / recompute / prefill 动作并记录事件。"""
from __future__ import annotations


class Engine:
    def __init__(self, env, worker_id, gpu, storage, metrics):
        self.env = env
        self.worker_id = worker_id
        self.gpu = gpu
        self.storage = storage
        self.metrics = metrics

    def handle(self, req, action):
        self.env.process(self._run(req, action))

    def _run(self, req, action):
        if action == "fetch":
            ev = self.storage.submit(req.rid, req.kv_gb)
            yield ev
            req.t_fetch_done = self.env.now
            self.metrics.on_fetch_done(req.rid, self.env.now)
            g = self.gpu.submit(req.rid, req.suffix_tokens)
            yield g
        else:
            g = self.gpu.submit(req.rid, req.prompt_tokens)
            yield g
        req.t_prefill_done = self.env.now
        self.metrics.on_complete(req, action)
