"""聚合存储仿真：带宽分配（fcfs 按申请上限 / max_min 水土填充）、字节积分、提交/升速/完成。

不包含任何策略逻辑（§11.1）。区段 [t,t_next) 内速率不变；下一时刻由读取完成、
计算完成、到达、带宽断点、观测更新、控制返回或唤醒中最早者决定。

alloc="max_min"（20261018，docs/存储带宽maxmin替换设计-20261016.md）：流申请率 q
即需求（引擎对 ℓ≥1 层按 V/c 推导、首层与到期提升为 q_max）；allocate 对全部活跃
流做 waterfill(B(t), [q])，每个事件点全量重算——新流提交会即时改变所有在途流的
速率与完成时刻（在途重配）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .types import FlowState


def waterfill(capacity, demands, zero):
    """带申请上限的 max-min fair（水土填充）分配，迭代到不动点。

    Σq ≤ 容量：按需分 + 剩余均分（分配可超申请，独占流得全容量）；
    Σq > 容量：升序一轮扫描即不动点（低申请者定格，剩余均分给高申请者）。
    数值后端无关（Fraction 精确 / float 生产皆可）；demands 为空返回 []。
    与 sim/storage.py 的 waterfill 同构（那边 float 口径）。
    """
    n = len(demands)
    if n == 0:
        return []
    total = zero
    for q in demands:
        total += q
    if total <= capacity:
        bonus = (capacity - total) / n
        return [q + bonus for q in demands]
    alloc = [zero] * n
    cap = capacity
    remaining = n
    for i in sorted(range(n), key=lambda j: demands[j]):
        share = cap / remaining
        a = demands[i] if demands[i] < share else share
        alloc[i] = a
        cap -= a
        remaining -= 1
    return alloc


class StorageSim:
    """B(t) 阶梯带宽 + 流分配器（fcfs 或 max_min）。"""

    def __init__(self, b_schedule, q_max, numeric=float, tol=None, alloc="fcfs"):
        self.num = numeric
        self.zero = numeric(0)
        self.b_schedule = [(numeric(t), numeric(v)) for t, v in b_schedule]
        self.q_max = numeric(q_max)
        self.alloc = alloc
        self.flows: Dict[int, FlowState] = {}
        self.next_seq = 0
        self.integrated_capacity_gb = self.zero   # ∫B dt
        self.actual_integral_gb = self.zero       # ∫Σrate dt
        self.requested_integral_gb = self.zero    # ∫Σq dt
        self.rate_log: List[Tuple] = []           # (seq, t, rate) 每次变化
        # E25 时间序列账本（默认关闭）：advance 每个 [t0,t1) 区段记 (t0,t1,Σrate,Σq)。
        # 仅在 record_intervals=True 时追加；不改变任何物理行为（BU11 对拍）。
        self.record_intervals: bool = False
        self.interval_log: List[Tuple] = []
        self.tol = tol if tol is not None else (0 if numeric is not float else 1e-12)

    # -- 带宽 schedule -----------------------------------------------------
    def b_at(self, t):
        cur = self.b_schedule[0][1]
        for tt, vv in self.b_schedule:
            if tt <= t:
                cur = vv
            else:
                break
        return cur

    def next_break(self, t):
        for tt, _ in self.b_schedule[1:]:
            if tt > t:
                return tt
        return None

    def has_future_positive(self, t):
        """t 之后是否存在正带宽（用于 undrainable 判定）。"""
        if self.b_at(t) > 0:
            return True
        for tt, vv in self.b_schedule:
            if tt > t and vv > 0:
                return True
        return False

    # -- 流生命周期 ---------------------------------------------------------
    def submit(self, flow_id: int, batch_id: int, layer: int, t, V, q, due):
        """创建流并分配序号。submit_seq 由调用顺序（阶段→闭包轮次→worker→层）决定。"""
        f = FlowState(flow_id=flow_id, batch_id=batch_id, layer=layer,
                      submit_seq=self.next_seq, submit_s=t, V_gb=V,
                      remaining_gb=V, q_gbps=q, due_s=due)
        self.next_seq += 1
        self.flows[f.submit_seq] = f
        return f

    def allocate(self, t):
        """带宽分配（每个事件点全量重算，在途流速率即时重配）。

        fcfs：按 submit_seq 升序，rate=min(q, max(0,free))；
        max_min：全部活跃流按申请率 q 水土填充 waterfill(B(t), [q])。
        """
        if self.alloc == "max_min":
            flows = sorted(self.flows.values(), key=lambda f: f.submit_seq)
            allocs = waterfill(self.b_at(t), [f.q_gbps for f in flows], self.zero)
            for f, r in zip(flows, allocs):
                if r != f.rate_gbps:
                    self.rate_log.append((f.submit_seq, t, r))
                    f.rate_gbps = r
            return
        free = self.b_at(t)
        for seq in sorted(self.flows):
            f = self.flows[seq]
            r = min(f.q_gbps, max(self.zero, free))
            if r != f.rate_gbps:
                self.rate_log.append((f.submit_seq, t, r))
                f.rate_gbps = r
            free -= r

    def advance(self, t0, t1):
        """结算 [t0,t1)：速率不变区段的字节积分。"""
        dt = t1 - t0
        if dt <= 0:
            return
        cap = self.b_at(t0)
        self.integrated_capacity_gb += cap * dt
        total_rate = self.zero
        total_q = self.zero
        for f in self.flows.values():
            served = f.rate_gbps * dt
            f.remaining_gb -= served
            total_rate += f.rate_gbps
            total_q += f.q_gbps
        self.actual_integral_gb += total_rate * dt
        self.requested_integral_gb += total_q * dt
        if self.record_intervals:
            self.interval_log.append((t0, t1, total_rate, total_q))

    def completions(self, t):
        """返回 [(seq, t + remaining/rate)]，rate>0 的流。"""
        out = []
        for seq, f in self.flows.items():
            if f.rate_gbps > 0:
                out.append((seq, t + f.remaining_gb / f.rate_gbps))
        return out

    def next_due(self, t):
        """最早未提升且未完成的到期时刻。"""
        best = None
        for f in self.flows.values():
            if f.due_s is not None and not f.boosted and f.due_s > t:
                if best is None or f.due_s < best:
                    best = f.due_s
        return best

    def boost_due(self, t):
        """到期未读完的流提升为 q_max，保留序号（§4.2）。"""
        changed = False
        for f in self.flows.values():
            if (f.due_s is not None and not f.boosted and f.due_s <= t
                    and f.remaining_gb > self._tolV(f)):
                f.q_gbps = self.q_max
                f.boosted = True
                changed = True
        return changed

    def _tolV(self, f):
        return self.tol * max(1.0, float(f.V_gb)) if self.tol else 0

    def is_complete(self, f):
        return f.remaining_gb <= self._tolV(f)

    def remove(self, seq, t):
        f = self.flows.pop(seq)
        f.completed_s = t
        return f
