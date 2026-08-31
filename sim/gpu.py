"""计算侧：Prefill 时间曲线（查表插值）+ 每 worker 的 FCFS 流体队列（背景负载 reserved-capacity 模型）。"""
from __future__ import annotations

import math
from collections import deque

import simpy

from .storage import next_breakpoint, piecewise_value

EPS = 1e-9
INF = math.inf


class PrefillCurve:
    def __init__(self, table: tuple):
        pts = [(0, 0.0)] + sorted(table)
        self.xs = [p[0] for p in pts]
        self.ys = [p[1] for p in pts]

    def __call__(self, tokens: int) -> float:
        xs, ys = self.xs, self.ys
        if tokens <= xs[0]:
            return ys[0]
        if tokens >= xs[-1]:
            lo = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return ys[-1] + lo * (tokens - xs[-1])
        for i in range(1, len(xs)):
            if tokens <= xs[i]:
                f = (tokens - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + f * (ys[i] - ys[i - 1])
        return ys[-1]


class _Job:
    __slots__ = ("rid", "service", "remaining", "ev")

    def __init__(self, rid: int, service: float, ev):
        self.rid = rid
        self.service = service
        self.remaining = service
        self.ev = ev


class GpuPool:
    """单 worker prefill 队列：队头服务，速率 = 1 - bg_gpu(t)。"""

    def __init__(self, env: simpy.Environment, worker_id: int, curve: PrefillCurve, gpu_cfg):
        self.env = env
        self.worker_id = worker_id
        self.curve = curve
        self.bg_schedule = tuple(sorted(gpu_cfg.bg_schedule))
        self.queue: deque[_Job] = deque()
        self._gen = 0
        self.last_t = env.now
        self.busy_time = 0.0  # 按 rate 加权的占用时间（分子）

    def bg_at(self, t: float) -> float:
        return piecewise_value(self.bg_schedule, t)

    def next_bg_change(self, t: float) -> float:
        return next_breakpoint(self.bg_schedule, t)

    def submit(self, rid: int, tokens: int) -> simpy.Event:
        self.advance_to(self.env.now)
        ev = self.env.event()
        self.queue.append(_Job(rid, self.curve(tokens), ev))
        self._schedule_wakeup()
        return ev

    def advance_to(self, t: float) -> None:
        if t <= self.last_t + 1e-12:
            return
        while self.last_t < t - 1e-12 and self.queue:
            t_bg = self.next_bg_change(self.last_t)
            seg_end = min(t, t_bg)
            rate = 1.0 - self.bg_at(self.last_t)
            if rate <= 1e-9:
                self.last_t = seg_end
                continue
            head = self.queue[0]
            t_done = self.last_t + head.remaining / rate
            step = min(seg_end, t_done)
            dt = step - self.last_t
            head.remaining -= rate * dt
            self.busy_time += rate * dt
            self.last_t = step
            if head.remaining <= EPS:
                self.queue.popleft()
                if head.ev is not None:
                    head.ev.succeed()
        self.last_t = t

    def _next_event_time(self) -> float:
        t = self.env.now
        times = [self.next_bg_change(t)]
        rate = 1.0 - self.bg_at(t)
        if self.queue and rate > 1e-9:
            times.append(t + self.queue[0].remaining / rate)
        return min(times)

    def _schedule_wakeup(self) -> None:
        self._gen += 1
        g = self._gen
        tn = self._next_event_time()
        if tn < INF and tn > self.env.now + 1e-12:
            ev = self.env.timeout(tn - self.env.now)
            ev.callbacks.append(lambda e, g=g: self._wakeup(g))

    def _wakeup(self, g: int) -> None:
        if g != self._gen:
            return
        self.advance_to(self.env.now)
        self._schedule_wakeup()

    # ---------- 策略可见 ----------
    def remaining_service(self) -> float:
        """队列中全部任务按全速服务时间计的剩余量（秒）。"""
        return sum(j.remaining for j in self.queue)

    def drain_est(self) -> float:
        """按当前背景负载折算的排队排空时间估计。"""
        rate = max(0.02, 1.0 - self.bg_at(self.env.now))
        return self.remaining_service() / rate

    # ---------- Oracle：假设现在追加 tokens，精确完成时间 ----------
    def hypothetical(self, tokens: int) -> float:
        t0 = self.env.now
        ahead = [j.remaining for j in self.queue]
        my_service = self.curve(tokens)
        t = t0
        for _ in range(20000):
            t_bg = self.next_bg_change(t)
            rate = 1.0 - self.bg_at(t)
            if rate <= 1e-9:
                if t_bg == INF:
                    return INF
                t = t_bg
                continue
            if not ahead:
                t_done = t + my_service / rate
                if t_done <= t_bg + 1e-15:
                    return t_done - t0
                my_service -= rate * (t_bg - t)
                t = t_bg
                continue
            t_head = t + ahead[0] / rate
            if t_head <= t_bg + 1e-15:
                dt = t_head - t
                ahead = [r - rate * dt for r in ahead]
                ahead = [r for r in ahead if r > EPS]
                t = t_head
            else:
                dt = t_bg - t
                ahead = [r - rate * dt for r in ahead]
                ahead = [r for r in ahead if r > EPS]
                t = t_bg
        return INF
