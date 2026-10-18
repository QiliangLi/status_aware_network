"""共享 KV 存储：nominal 带宽固定 + 动态背景负载 + 带宽分配策略（ps_share 等分 / max_min_fair 水土填充）。

双世界设计：SharedKVStorage 是 ground truth（仅 Oracle 允许访问内部状态）；
StorageObservable 是计算侧可见的陈旧/平滑/带噪视图，P2/P3 只能读它。

带宽分配策略（设计 docs/存储带宽maxmin替换设计-20261016.md）：
- ps_share（旧模型，回归锚/对照臂）：单池聚合传输 + cap/N 等分，逐字保留原实现；
- max_min_fair（新默认）：带需求上限的 max-min fair（水土填充）——
  * 盘数派生 disks = ceil(b_total/disk_bps)，单盘带宽 = b_total/disks ≤ disk_bps
    （默认 40 GB/s 物理基元；fabric 惯例 disk_bps=b_total → 单链路聚合 claim）；
  * 块 = block_tokens 个 token 的 KV，按 (池,类,块号) HRW 一致性哈希到盘；
  * 分配主体为逐块 IO（每块携需求 d）；同请求同盘的块聚合为一个组、分配时
    按块数展开（数学等价于字面逐块对象，单测对拍）；每盘独立 waterfill；
  * 需求 d 语义 = 消费端吸收速率（kv_gb_per_token / prefill_s_per_token，
    装配处推导注入）；Σd ≤ 容量时按需分 + 剩余均分（独占流可得全容量），
    超过时迭代水土填充到不动点；分配在每个事件点（完成/提交/门控/背景断点）
    全量重算——在途传输速率动态重配。
"""
from __future__ import annotations

import hashlib
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


def waterfill(capacity: float, demands: list) -> list:
    """带需求上限的 max-min fair 分配，迭代到不动点；O(n log n)。

    Σd ≤ 容量：按需分 + 剩余均分（分配可超需求，独占者得全容量）；
    Σd > 容量：升序一轮扫描即不动点（小需求定格，剩余均分给未定格者）。
    与 sim/cq/storage.py 的 waterfill 同构（那边 Fraction 精确算术）。
    """
    n = len(demands)
    if n == 0:
        return []
    total = 0.0
    for d in demands:
        total += d
    if total <= capacity:
        bonus = (capacity - total) / n
        return [d + bonus for d in demands]
    alloc = [0.0] * n
    cap = capacity
    remaining = n
    for i in sorted(range(n), key=lambda j: demands[j]):
        share = cap / remaining
        a = demands[i] if demands[i] < share else share
        alloc[i] = a
        cap -= a
        remaining -= 1
    return alloc


def disk_of(pool: str, cls: str, idx: int, disks: int) -> int:
    """HRW（最高随机权重）一致性哈希：sha256 摘要最大的盘。

    确定性、无随机数（CRN 不受影响）；增删盘只重映射少数块（一致性）。
    与 dynamo 侧 storage.disk_of 同一构造；哈希含池标识以解耦各副本倾斜。
    """
    best_d, best_k = 0, b"\x00" * 32
    for d in range(disks):
        k = hashlib.sha256(f"{pool}:{cls}:{idx}:{d}".encode()).digest()
        if k > best_k:
            best_d, best_k = d, k
    return best_d


_PLAN_CACHE: dict = {}
_PLAN_CACHE_MAX = 8192


def disk_plan(pool: str, cls: str, n_tokens: int, disks: int,
              block_tokens: int, gb_per_token: float) -> tuple:
    """类前缀前 n_tokens 个 token 的块 -> 盘分布计划 ((disk, n_io, nbytes), ...)。

    块 = block_tokens 个 token（末块部分字节）；前缀块对齐（partial 取前缀）。
    结果缓存（同 (池,类,长度) 的计划确定不变）。
    """
    key = (pool, cls, n_tokens, disks, block_tokens, gb_per_token)
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        return hit
    if n_tokens <= 0:
        out = ()
    else:
        n_b = max(1, math.ceil(n_tokens / block_tokens - 1e-12))
        counts = [0] * disks
        for idx in range(n_b):
            counts[disk_of(pool, cls, idx, disks)] += 1
        rem = n_tokens - (n_b - 1) * block_tokens
        last_disk = disk_of(pool, cls, n_b - 1, disks)
        out = []
        for d, c in enumerate(counts):
            if c == 0:
                continue
            nb = c * block_tokens * gb_per_token
            if d == last_disk and rem != block_tokens:
                nb -= block_tokens * gb_per_token
                nb += rem * gb_per_token
            out.append((d, c, nb))
        out = tuple(out)
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        _PLAN_CACHE.clear()
    _PLAN_CACHE[key] = out
    return out


class Transfer:
    __slots__ = ("rid", "nbytes", "remaining", "gate_until", "demand", "disk", "n_io")

    def __init__(self, rid: int, nbytes: float, gate_until: float,
                 demand: float = INF, disk: int = 0, n_io: int = 1):
        self.rid = rid
        self.nbytes = nbytes
        self.remaining = float(nbytes)
        self.gate_until = gate_until  # t_base 期间不占份额、不计入分母
        self.demand = demand          # 逐块 IO 需求（GB/s）；INF = 无界（旧模型/后台兼容）
        self.disk = disk
        self.n_io = n_io              # 组内块数（分配展开用；同组块同尺寸同时刻同需求）


class SharedKVStorage:
    def __init__(self, env: simpy.Environment, name, cfg):
        self.env = env
        self.name = name
        self.b_total = cfg.b_total
        self.t_base = cfg.t_base
        self.bg_schedule = tuple(sorted(cfg.bg_schedule))
        # ---- 带宽策略与盘维度（老配置对象缺字段时按旧模型 ps_share 处理）----
        self.bw_policy = getattr(cfg, "bw_policy", "ps_share")
        if self.bw_policy not in ("ps_share", "max_min_fair"):
            raise ValueError(f"bw_policy 必须为 ps_share|max_min_fair，得到 {self.bw_policy!r}")
        _db = getattr(cfg, "disk_bps", None)
        self.disk_bps = (float(_db) if _db is not None
                         else (self.b_total if self.bw_policy == "ps_share" else 40.0))
        if self.disk_bps <= 0:
            raise ValueError("disk_bps 必须为正")
        _bt = getattr(cfg, "block_tokens", None)
        self.block_tokens = int(_bt) if _bt is not None else 512
        if self.block_tokens <= 0:
            raise ValueError("block_tokens 必须为正")
        self.disks = max(1, math.ceil(self.b_total / self.disk_bps - 1e-12))
        self.disk_cap_base = self.b_total / self.disks   # 单盘名义带宽（≤ disk_bps）
        self.active: list[Transfer] = []
        self._events: dict[int, simpy.Event] = {}
        self._pending: dict[int, int] = {}               # rid -> 剩余组数（完成倒计数）
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

    def plan_for(self, cls: str, n_tokens: int, gb_per_token: float):
        """该池上取回 cls 前 n_tokens 个 token 的块计划（max_min 多盘用）。

        ps_share / 单盘池返回 None（聚合语义）。
        """
        if self.bw_policy != "max_min_fair" or self.disks <= 1 or n_tokens <= 0:
            return None
        return disk_plan(str(self.name), cls, n_tokens, self.disks,
                         self.block_tokens, gb_per_token)

    # ---------- 提交 ----------
    def submit(self, rid: int, nbytes: float, demand: float = None, plan=None) -> simpy.Event:
        """提交一次取回。max_min_fair 需求必填（无则按无界处理，供旧模型兼容）；
        多盘池要求携带块计划 plan（防静默聚合退化）。ps_share 忽略 demand/plan。"""
        self.advance_to(self.env.now)
        ev = self.env.event()
        self._events[rid] = ev
        mm = self.bw_policy == "max_min_fair"
        # 需求非正（None/0，直构请求或旧模型未注入）按无界处理，防静默饿死
        d_eff = INF if (demand is None or demand <= 0) else float(demand)
        if mm and self.disks > 1:
            if plan:
                groups = tuple(plan)
            else:
                raise ValueError("max_min_fair 多盘池 submit 必须携带块计划 plan")
        else:
            groups = ((0, 1, nbytes),)
        n_groups = 0
        for (dk, n_io, nb) in groups:
            if nb <= 1e-15:
                continue
            self.active.append(Transfer(rid, nb, self.env.now + self.t_base,
                                        d_eff, dk, max(1, int(n_io))))
            n_groups += 1
        self._pending[rid] = max(1, n_groups)
        self.n_submitted += 1
        self._schedule_wakeup()
        return ev

    # ---------- 流体推进（在完成点/门控点/背景负载断点处切分，精确无近似）----------
    def advance_to(self, t: float) -> None:
        if t <= self.last_t + 1e-12:
            return
        if self.bw_policy == "ps_share":
            self._advance_ps(t)
        else:
            self._advance_mm(t)
        self.last_t = t

    def _advance_ps(self, t: float) -> None:
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

    def _disk_cap(self, t: float) -> float:
        """单盘容量：池级背景负载平摊到盘（盘对齐下 = disk_bps − bg/disks）。"""
        return max(0.0, (self.b_total - self.bg_at(t)) / self.disks)

    def _mm_rates(self, ready: list, t: float) -> list:
        """max_min_fair 逐盘 waterfill：返回与 ready 同序的组速率（GB/s）。

        组内 K 块 = K 个同需求独立 IO，分配按块数展开（组速率 = K × 单块分配，
        与字面逐块对象等价——同组块同尺寸同时刻同需求，必然同分配同时完成）。
        """
        rates = [0.0] * len(ready)
        by_disk: dict[int, list[int]] = {}
        for i, g in enumerate(ready):
            by_disk.setdefault(g.disk, []).append(i)
        for _, idxs in by_disk.items():
            cap = self._disk_cap(t)
            demands = []
            spans = []
            for i in idxs:
                k = ready[i].n_io
                demands.extend([ready[i].demand] * k)
                spans.append((i, k))
            allocs = waterfill(cap, demands)
            pos = 0
            for i, k in spans:
                rates[i] = allocs[pos] * k
                pos += k
        return rates

    def _advance_mm(self, t: float) -> None:
        while self.last_t < t - 1e-12:
            t_bg = self.next_bg_change(self.last_t)
            seg_end = min(t, t_bg)
            ready = [g for g in self.active if g.gate_until <= self.last_t + 1e-12]
            if not ready:
                gates = [g.gate_until for g in self.active if g.gate_until > self.last_t]
                self.last_t = min(seg_end, min(gates) if gates else seg_end)
                if self.last_t >= seg_end - 1e-12 and not gates:
                    break
                continue
            rates = self._mm_rates(ready, self.last_t)
            positive = [(g, r) for g, r in zip(ready, rates) if r > 1e-15]
            if not positive:
                self.last_t = seg_end
                continue
            t_done_abs = self.last_t + min(g.remaining / r for g, r in positive)
            step = min(seg_end, t_done_abs)
            dt = step - self.last_t
            for g, r in zip(ready, rates):
                g.remaining -= r * dt
                self.bytes_served += r * dt
            self.last_t = step
            if step >= t_done_abs - 1e-12:
                self._complete_ready()
        self.last_t = t

    def _complete_ready(self) -> None:
        done = [tr for tr in self.active if tr.remaining <= EPS]
        if not done:
            return
        self.active = [tr for tr in self.active if tr.remaining > EPS]
        for tr in done:
            self.n_completed += 1
            left = self._pending.get(tr.rid, 1) - 1
            if left <= 0:
                self._pending.pop(tr.rid, None)
                ev = self._events.pop(tr.rid, None)
                if ev is not None:
                    ev.succeed()
            else:
                self._pending[tr.rid] = left

    # ---------- 事件调度 ----------
    def _next_event_time(self) -> float:
        t = self.env.now
        times = [self.next_bg_change(t)]
        gates = [tr.gate_until for tr in self.active if tr.gate_until > t + 1e-12]
        if gates:
            times.append(min(gates))
        ready = [tr for tr in self.active if tr.gate_until <= t + 1e-12]
        if ready:
            if self.bw_policy == "ps_share":
                cap = self.cap_at(t)
                if cap > 1e-12:
                    rate = cap / len(ready)
                    times.append(t + min(tr.remaining for tr in ready) / rate)
            else:
                rates = self._mm_rates(ready, t)
                positive = [(g, r) for g, r in zip(ready, rates) if r > 1e-15]
                if positive:
                    times.append(t + min(g.remaining / r for g, r in positive))
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
    def hypothetical_fetch_time(self, nbytes: float, demand: float = None, plan=None) -> float:
        if self.bw_policy != "max_min_fair":
            return self._hyp_ps(nbytes)
        return self._hyp_mm(nbytes, demand, plan)

    def _hyp_ps(self, nbytes: float) -> float:
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

    def _hyp_mm(self, nbytes: float, demand, plan) -> float:
        """max_min 版：克隆逐盘分组 + 插入假设组，逐步 waterfill 模拟。

        插入项完成 = 其全部组完成（取最慢盘）；demand 缺省按无界（供旧调用兼容）。
        """
        t0 = self.env.now
        d_eff = INF if (demand is None or demand <= 0) else float(demand)
        alive = [[g.remaining, g.gate_until, g.disk, g.n_io, g.demand, False]
                 for g in self.active]
        if plan:
            ins = [[nb, t0 + self.t_base, dk, max(1, int(k)), d_eff, True]
                   for (dk, k, nb) in plan if nb > 1e-15]
        else:
            ins = [[float(nbytes), t0 + self.t_base, 0, 1, d_eff, True]]
        n_ins = len(ins)
        alive += ins
        t = t0
        for _ in range(20000):
            if not alive:
                return INF
            t_bg = self.next_bg_change(t)
            gates = [x[1] for x in alive if x[1] > t + 1e-12]
            t_gate = min(gates) if gates else INF
            ready = [x for x in alive if x[1] <= t + 1e-12]
            if not ready:
                nt = min(t_bg, t_gate)
                if nt == INF:
                    return INF
                t = nt
                continue
            # 逐盘 waterfill（与 _mm_rates 同式，含插入组）
            by_disk: dict[int, list] = {}
            for x in ready:
                by_disk.setdefault(x[2], []).append(x)
            rate_of = {}
            for _, xs in by_disk.items():
                cap = self._disk_cap(t)
                if cap <= 1e-12:
                    for x in xs:
                        rate_of[id(x)] = 0.0
                    continue
                demands = []
                for x in xs:
                    demands.extend([x[4]] * x[3])
                allocs = waterfill(cap, demands)
                pos = 0
                for x in xs:
                    rate_of[id(x)] = allocs[pos] * x[3]
                    pos += x[3]
            positive = [(x, rate_of[id(x)]) for x in ready if rate_of[id(x)] > 1e-15]
            if not positive:
                nt = min(t_bg, t_gate)
                if nt == INF:
                    return INF
                t = nt
                continue
            t_done = t + min(x[0] / r for x, r in positive)
            step = min(t_done, t_bg, t_gate)
            dt = step - t
            for x in ready:
                x[0] -= rate_of[id(x)] * dt
            if step >= t_done - 1e-15:
                finishing = [x for x in alive if x[0] <= EPS]
                for x in finishing:
                    if x[5]:
                        n_ins -= 1
                        if n_ins == 0:
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
