"""共享 KV 存储：nominal 带宽固定 + 动态背景负载 + processor sharing（含 t_base 门控）。

双世界设计：SharedKVStorage 是 ground truth（仅 Oracle 允许访问内部状态）；
StorageObservable 是计算侧可见的陈旧/平滑/带噪视图，P2/P3 只能读它。
"""
from __future__ import annotations

import math

import simpy

EPS = 1e-9
INF = math.inf


def piecewise_value(schedule, t: float) -> float:
    v = schedule[0][1]
    for tp, vp in schedule:
        if t >= tp - 1e-12:
            v = vp
        else:
            break
    return v


def next_breakpoint(schedule, t: float) -> float:
    best = INF
    for tp, _ in schedule:
        if tp > t + 1e-9:
            best = min(best, tp)
    return best


class Transfer:
    __slots__ = ("rid", "nbytes", "remaining", "gate_until")

    def __init__(self, rid: int, nbytes: float, gate_until: float):
        self.rid = rid
        self.nbytes = nbytes
        self.remaining = float(nbytes)
        self.gate_until = gate_until  # t_base 期间不占份额、不计入分母


class SharedKVStorage:
    def __init__(self, env: simpy.Environment, name, cfg):
        self.env = env
        self.name = name
        self.b_total = cfg.b_total
        self.t_base = cfg.t_base
        self.bg_schedule = tuple(sorted(cfg.bg_schedule))
        self.active: list[Transfer] = []
        self._events: dict[int, simpy.Event] = {}
        self._gen = 0
        self.last_t = env.now
        self.bytes_served = 0.0
        self.n_submitted = 0
        self.n_completed = 0

    # ---------- ground truth ----------
    def bg_at(self, t: float) -> float:
        return piecewise_value(self.bg_schedule, t)

    def cap_at(self, t: float) -> float:
        return max(0.0, self.b_total - self.bg_at(t))

    def next_bg_change(self, t: float) -> float:
        return next_breakpoint(self.bg_schedule, t)

    def stats_now(self) -> dict:
        return dict(
            qdepth=len(self.active),
            inflight=sum(tr.remaining for tr in self.active),
            bytes_served=self.bytes_served,
        )

    # ---------- 提交 ----------
    def submit(self, rid: int, nbytes: float) -> simpy.Event:
        self.advance_to(self.env.now)
        ev = self.env.event()
        self._events[rid] = ev
        self.active.append(Transfer(rid, nbytes, self.env.now + self.t_base))
        self.n_submitted += 1
        self._schedule_wakeup()
        return ev

    # ---------- 流体推进（在完成点/门控点/背景负载断点处切分，精确无近似）----------
    def advance_to(self, t: float) -> None:
        if t <= self.last_t + 1e-12:
            return
        while self.last_t < t - 1e-12:
            t_bg = self.next_bg_change(self.last_t)
            seg_end = min(t, t_bg)
            ready = [tr for tr in self.active if tr.gate_until <= self.last_t + 1e-12]
            if not ready:
                gates = [tr.gate_until for tr in self.active if tr.gate_until > self.last_t]
                self.last_t = min(seg_end, min(gates) if gates else seg_end)
                if self.last_t >= seg_end - 1e-12 and not gates:
                    break
                continue
            cap = self.cap_at(self.last_t)
            if cap <= 1e-12:
                self.last_t = seg_end
                continue
            rate = cap / len(ready)
            t_done = self.last_t + min(tr.remaining for tr in ready) / rate
            step = min(seg_end, t_done)
            dt = step - self.last_t
            for tr in ready:
                tr.remaining -= rate * dt
            self.bytes_served += rate * dt * len(ready)
            self.last_t = step
            if step >= t_done - 1e-12:
                self._complete_ready()
        self.last_t = t

    def _complete_ready(self) -> None:
        done = [tr for tr in self.active if tr.remaining <= EPS]
        if not done:
            return
        self.active = [tr for tr in self.active if tr.remaining > EPS]
        for tr in done:
            self.n_completed += 1
            ev = self._events.pop(tr.rid, None)
            if ev is not None:
                ev.succeed()

    # ---------- 事件调度 ----------
    def _next_event_time(self) -> float:
        t = self.env.now
        times = [self.next_bg_change(t)]
        gates = [tr.gate_until for tr in self.active if tr.gate_until > t + 1e-12]
        if gates:
            times.append(min(gates))
        ready = [tr for tr in self.active if tr.gate_until <= t + 1e-12]
        cap = self.cap_at(t)
        if ready and cap > 1e-12:
            rate = cap / len(ready)
            times.append(t + min(tr.remaining for tr in ready) / rate)
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

    # ---------- Oracle：假设现在插入 nbytes，在已知未来背景负载下算精确完成时间 ----------
    def hypothetical_fetch_time(self, nbytes: float) -> float:
        t0 = self.env.now
        alive = [[tr.remaining, tr.gate_until, False] for tr in self.active]
        alive.append([float(nbytes), t0 + self.t_base, True])
        t = t0
        for _ in range(20000):
            if not alive:
                return INF
            t_bg = self.next_bg_change(t)
            gates = [x[1] for x in alive if x[1] > t + 1e-12]
            t_gate = min(gates) if gates else INF
            ready = [x for x in alive if x[1] <= t + 1e-12]
            cap = self.cap_at(t)
            if not ready or cap <= 1e-12:
                nt = min(t_bg, t_gate)
                if nt == INF:
                    return INF
                t = nt
                continue
            rate = cap / len(ready)
            t_done = t + min(x[0] for x in ready) / rate
            step = min(t_done, t_bg, t_gate)
            dt = step - t
            for x in alive:
                if x[1] <= t + 1e-12:
                    x[0] -= rate * dt
            if step >= t_done - 1e-15:
                finishing = [x for x in alive if x[0] <= EPS]
                for x in finishing:
                    if x[2]:
                        return t_done - t0
            alive = [x for x in alive if x[0] > EPS]
            t = step
        return INF


def _ema(old: float, new: float, dt: float, halflife: float) -> float:
    if old is None:
        return new
    a = 1.0 - 0.5 ** (max(dt, 1e-9) / halflife)
    return old + a * (new - old)


class StorageObservable:
    """计算侧可见视图：按 interval 采样、EMA 平滑、可选乘性噪声；interval=0 表示 live。"""

    def __init__(self, env: simpy.Environment, storage: SharedKVStorage, obs_cfg, mean_hit_gb: float, rng):
        self.env = env
        self.s = storage
        self.cfg = obs_cfg
        self.mean_hit_gb = mean_hit_gb
        self.rng = rng
        self.interval = obs_cfg.interval
        self.signal = obs_cfg.signal
        self.q = 0
        self.inflight = 0.0
        self.util_ema = None
        self.bw_share_ema = None
        self._last_bytes = 0.0
        if self.interval > 1e-12:
            env.process(self._loop())

    def _loop(self):
        while True:
            yield self.env.timeout(self.interval)
            self._sample()

    def _sample(self) -> None:
        t = self.env.now
        st = self.s.stats_now()
        fg = (st["bytes_served"] - self._last_bytes) / self.interval
        self._last_bytes = st["bytes_served"]
        util = min(1.0, (self.s.bg_at(t) + fg) / self.s.b_total)
        self.util_ema = _ema(self.util_ema, util, self.interval, self.cfg.ema_halflife)
        if st["qdepth"] > 0:
            share = self.s.cap_at(t) / st["qdepth"]
            self.bw_share_ema = _ema(self.bw_share_ema, share, self.interval, self.cfg.ema_halflife)
        self.q = st["qdepth"]
        self.inflight = st["inflight"]

    def estimate(self, nbytes: float, signal: str = None, noisy: bool = True) -> float:
        sig = signal or self.signal
        b_total, t_base = self.s.b_total, self.s.t_base
        if self.interval <= 1e-12:
            st = self.s.stats_now()
            cap = self.s.cap_at(self.env.now)
            q, infl = st["qdepth"], st["inflight"]
            if sig == "quote":
                base = (infl + nbytes) / max(cap, 0.5) + t_base
            elif sig == "bw":
                share = cap / q if q > 0 else cap
                base = nbytes / max(share, 0.3) + t_base
            elif sig == "util":
                base = nbytes / max(cap, 0.5) + t_base
            else:  # queue
                base = (q + 1) * self.mean_hit_gb / b_total + t_base
        else:
            cap_est = b_total * max(0.02, 1.0 - (self.util_ema or 0.0))
            if sig == "quote":
                base = (self.inflight + nbytes) / max(cap_est, 0.5) + t_base
            elif sig == "bw":
                share = self.bw_share_ema if self.bw_share_ema else cap_est
                base = nbytes / max(share, 0.3) + t_base
            elif sig == "util":
                base = nbytes / max(cap_est, 0.5) + t_base
            else:  # queue
                base = (self.q + 1) * self.mean_hit_gb / b_total + t_base
        if noisy and self.cfg.noise_sigma > 0:
            base *= float(self.rng.lognormal(0.0, self.cfg.noise_sigma))
        return base
