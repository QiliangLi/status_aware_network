"""GPU 侧可观测层（问题⑤）：与 StorageObservable 对称的陈旧/EMA/带噪视图。

默认 interval=0 且 noise=0 时等价于真值（drain_est），保证 v1/v2 既有结果可复现；
E12 扫描时启用，用于刻画联合决策对 GPU 估计误差的稳健性。
仅非 Oracle 家族策略使用；Oracle/Clairvoyant 直接读真值。
"""
from __future__ import annotations

import simpy


def _ema(old, new, dt, halflife):
    if old is None:
        return new
    a = 1.0 - 0.5 ** (max(dt, 1e-9) / halflife)
    return old + a * (new - old)


class GpuObservable:
    def __init__(self, env: simpy.Environment, gpu, interval: float, ema_halflife: float,
                 noise_sigma: float, rng):
        self.env = env
        self.g = gpu
        self.interval = interval
        self.halflife = ema_halflife
        self.sigma = noise_sigma
        self.rng = rng
        self.wait_ema = None
        if interval > 1e-12:
            env.process(self._loop())

    def _loop(self):
        while True:
            yield self.env.timeout(self.interval)
            t = self.env.now
            self.wait_ema = _ema(self.wait_ema, self.g.drain_est(),
                                 self.interval, self.halflife)

    def estimate(self) -> float:
        """GPU 排队+服务等待时间估计（秒），带乘性噪声。"""
        if self.interval <= 1e-12 or self.wait_ema is None:
            base = self.g.drain_est()
        else:
            base = self.wait_ema
        if self.sigma > 0:
            base *= float(self.rng.lognormal(0.0, self.sigma))
        return base
