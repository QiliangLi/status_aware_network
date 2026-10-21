"""聚合存储仿真：带宽分配（fcfs 按申请上限 / max_min 水土填充）、字节积分、提交/升速/完成。

不包含任何策略逻辑（§11.1）。区段 [t,t_next) 内速率不变；下一时刻由读取完成、
计算完成、到达、带宽断点、观测更新、控制返回或唤醒中最早者决定。

alloc="max_min"（20261018，docs/存储带宽maxmin替换设计-20261016.md）：流申请率 q
即需求（引擎对 ℓ≥1 层按 V/c 推导、首层与到期提升为 q_max）；allocate 对全部活跃
流做 waterfill(B(t), [q])，每个事件点全量重算——新流提交会即时改变所有在途流的
速率与完成时刻（在途重配）。

alloc="max_min_disk"（20261021，docs/E26f-盘维度存储实验设计-20261021.md §2）：
B(t) 平摊为 disks 个独立盘容量 B(t)/disks；块按 HRW 一致性哈希落盘（与主线
sim/storage.py::disk_of 同键同落点）；分配主体为 (流,盘) 组 × 块展开——盘 d 的
就绪组按块数展开为 K_d 个同需求 claimant（逐块需求 = q/n_b，聚合需求守恒），
每盘独立 waterfill；组完成即触发重配（事件组级化），流完成 = 全部盘组完成
（最慢盘决定，read_ready 由引擎在流完成时写入）；跨盘无再平衡。流级账本
（remaining/rate/integral/interval_log/PublicFlow 口径）保持流粒度不变。
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .types import FlowState

#: 块大小（token 数；对齐主线与 Mooncake 口径，本设计引入 cq 的常量）
BLOCK_TOKENS = 512

#: HRW 哈希键的池前缀（主线 disk_of 的 pool 位；cq 线单池固定名）
_CQ_POOL = "cq"


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


def waterfill_weighted(capacity, weighted, zero):
    """带权 waterfill：weighted 为 [(demand, count), ...]，与把 demand 重复
    count 次后调用 waterfill 逐位等价（同需求 claimant 必然同分配——waterfill
    对称性；升序扫描中同值连续段要么全按需定格、要么全得同一 share）；
    用于盘内块展开的高效精确实现。返回 [(单 claimant 分配, count), ...]。
    """
    m = len(weighted)
    if m == 0:
        return []
    n = 0
    total = zero
    for d, c in weighted:
        n += c
        total += d * c
    if total <= capacity:
        bonus = (capacity - total) / n
        return [(d + bonus, c) for d, c in weighted]
    out = [zero] * m
    cap = capacity
    remaining = n
    for i in sorted(range(m), key=lambda j: weighted[j][0]):
        d, c = weighted[i]
        share = cap / remaining
        a = d if d < share else share
        out[i] = a
        cap -= a * c
        remaining -= c
    return [(out[i], weighted[i][1]) for i in range(m)]


def disk_of(prefix_id: str, idx: int, disks: int) -> int:
    """HRW（最高随机权重）一致性哈希：sha256 摘要最大的盘。

    键构造 f"{pool}:{prefix_id}:{idx}:{d}"（pool="cq"）与主线
    sim/storage.py::disk_of 同键同落点（跨线一致性由单测对拍锁定）；
    确定性、无随机数（CRN 不受影响）；增删盘只重映射少数块。
    """
    best_d, best_k = 0, b"\x00" * 32
    for d in range(disks):
        k = hashlib.sha256(
            f"{_CQ_POOL}:{prefix_id}:{idx}:{d}".encode()).digest()
        if k > best_k:
            best_d, best_k = d, k
    return best_d


_PLAN_CACHE: dict = {}
_PLAN_CACHE_MAX = 8192


def disk_plan(prefix_id: str, h_tokens: int, disks: int,
              block_tokens: int = BLOCK_TOKENS, kappa=None):
    """前缀 h_tokens 个 token 的块→盘分布计划（Fraction 精确）。

    返回 ((disk, n_blocks, bytes_gb), ...)，按盘升序；块 = block_tokens 个
    token 的前缀切片（末块部分字节）；与层无关（同一块各层同盘）。
    Σbytes = κ·h 恒等；h=0 返回 ()。结果缓存（同 (前缀,长度,盘数) 确定）。
    """
    if kappa is None:
        from .config import ProfileConfig
        kappa = ProfileConfig().kappa_gb_per_token_layer
    key = (prefix_id, h_tokens, disks, block_tokens, kappa)
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        return hit
    if h_tokens <= 0 or disks <= 1:
        out: tuple = ()
    else:
        n_b = max(1, -(-h_tokens // block_tokens))   # ceil，精确整数
        counts = [0] * disks
        for idx in range(n_b):
            counts[disk_of(prefix_id, idx, disks)] += 1
        rem = h_tokens - (n_b - 1) * block_tokens    # 末块 token 数
        last_disk = disk_of(prefix_id, n_b - 1, disks)
        out = []
        for d, c in enumerate(counts):
            if c == 0:
                continue
            nb = kappa * (c * block_tokens)
            if d == last_disk and rem != block_tokens:
                nb -= kappa * (block_tokens - rem)
            out.append((d, c, nb))
        out = tuple(out)
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        _PLAN_CACHE.clear()
    _PLAN_CACHE[key] = out
    return out


def plan_n_blocks(plan: tuple) -> int:
    """计划的总块数 Σ n_blocks。"""
    return sum(c for _d, c, _b in plan)


def plan_bytes(plan: tuple):
    """计划的总字节 Σ bytes。"""
    return sum(b for _d, _c, b in plan)


class _DiskGroup:
    """(流,盘) 组：某流在某盘上的 K_d 个同质块 IO（同尺寸同时刻同需求，
    分配展开为 K_d 个 claimant，必然同分配同时完成——与逐块对象数学等价）。"""

    __slots__ = ("seq", "disk", "n_blocks", "total_gb", "remaining_gb",
                 "rate_gbps")

    def __init__(self, seq, disk, n_blocks, total_gb, zero):
        self.seq = seq
        self.disk = disk
        self.n_blocks = n_blocks
        self.total_gb = total_gb
        self.remaining_gb = total_gb
        self.rate_gbps = zero


class StorageSim:
    """B(t) 阶梯带宽 + 流分配器（fcfs / max_min / max_min_disk）。"""

    def __init__(self, b_schedule, q_max, numeric=float, tol=None, alloc="fcfs",
                 disk_bps=None, block_tokens=BLOCK_TOKENS, kappa=None):
        self.num = numeric
        self.zero = numeric(0)
        self.b_schedule = [(numeric(t), numeric(v)) for t, v in b_schedule]
        self.q_max = numeric(q_max)
        self.alloc = alloc
        # -- 盘模型（仅 max_min_disk；其余模式忽略，保证既有口径逐位不变）--
        self.uses_groups = (alloc == "max_min_disk")
        if self.uses_groups:
            if disk_bps is None:
                raise ValueError("max_min_disk 需要 disk_bps（单盘上限带宽）")
            if kappa is None:
                raise ValueError("max_min_disk 需要 kappa（GB/token/层）")
            b_peak = max(v for _t, v in self.b_schedule)
            r = b_peak / numeric(disk_bps)
            if numeric is float:
                r = r - 1e-12              # float 容差（3.0 不误进 4）
            self.disks = max(1, math.ceil(r))
            self.disk_bps = numeric(disk_bps)
            self.block_tokens = int(block_tokens)
            self.kappa = kappa
        else:
            self.disks = 1
            self.disk_bps = None
            self.block_tokens = int(block_tokens)
            self.kappa = None
        self.flows: Dict[int, FlowState] = {}
        self.groups: Dict[int, List[_DiskGroup]] = {}   # seq -> 每盘组（盘模式）
        self.next_seq = 0
        self.integrated_capacity_gb = self.zero   # ∫B dt
        self.actual_integral_gb = self.zero       # ∫Σrate dt
        self.requested_integral_gb = self.zero    # ∫Σq dt
        self.rate_log: List[Tuple] = []           # (seq, t, rate) 每次变化
        # E25 时间序列账本（默认关闭）：advance 每个 [t0,t1) 区段记 (t0,t1,Σrate,Σq)。
        # 仅在 record_intervals=True 时追加；不改变任何物理行为（BU11 对拍）。
        self.record_intervals: bool = False
        self.interval_log: List[Tuple] = []
        # 盘模式并行账本（不改既有四元组形状）：(t0, t1, [rate_d0..rate_dn])。
        self.disk_interval_log: List[Tuple] = []
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

    # -- 盘容量 -------------------------------------------------------------
    def disk_cap(self, t):
        """单盘容量：池级带宽 B(t) 平摊（与主线公式一致）。"""
        return self.b_at(t) / self.num(self.disks)

    # -- 流生命周期 ---------------------------------------------------------
    def submit(self, flow_id: int, batch_id: int, layer: int, t, V, q, due,
               prefix_id=None, h_tokens=0):
        """创建流并分配序号。submit_seq 由调用顺序（阶段→闭包轮次→worker→层）决定。

        max_min_disk 下 (prefix_id, h_tokens) 必填（缺省报错，防静默聚合）：
        单成员批传 (str, int)；批内多成员传同长序列（计划按成员拼接，逐盘
        块数/字节累加）。块计划由公开确定性函数 disk_plan 重建（任何策略/
        推演可重建同一计划）。
        """
        f = FlowState(flow_id=flow_id, batch_id=batch_id, layer=layer,
                      submit_seq=self.next_seq, submit_s=t, V_gb=V,
                      remaining_gb=V, q_gbps=q, due_s=due)
        self.next_seq += 1
        self.flows[f.submit_seq] = f
        if self.uses_groups:
            members = self._normalize_members(prefix_id, h_tokens)
            plan = []
            for pid, h in members:
                plan.extend(disk_plan(pid, h, self.disks, self.block_tokens,
                                      self.kappa))
            if not plan:
                raise ValueError(
                    "max_min_disk submit 的 (prefix_id, h_tokens) 解析出空块计划"
                    f"（members={members!r}）")
            agg: Dict[int, List] = {}
            for d, c, nb in plan:
                if d not in agg:
                    agg[d] = [d, 0, self.zero]
                agg[d][1] += c
                agg[d][2] += nb
            gs = [_DiskGroup(f.submit_seq, d, c, nb, self.zero)
                  for d, c, nb in (agg[k] for k in sorted(agg))]
            f.n_blocks = sum(g.n_blocks for g in gs)
            self.groups[f.submit_seq] = gs
        return f

    @staticmethod
    def _normalize_members(prefix_id, h_tokens):
        """(prefix_id, h_tokens) 归一为 ((pid, h), ...) 成员序列（h=0 成员无块，
        过滤；全零则返回空由调用方报错）。"""
        if prefix_id is None:
            raise ValueError("max_min_disk 下 submit 必须携带 prefix_id/h_tokens")
        if isinstance(prefix_id, str):
            if not isinstance(h_tokens, int) or h_tokens < 0:
                raise ValueError(
                    f"单成员 (prefix_id, h_tokens) 需要 h_tokens 为非负整数，"
                    f"得到 {h_tokens!r}")
            return ((prefix_id, h_tokens),) if h_tokens > 0 else ()
        try:
            members = tuple(zip(prefix_id, h_tokens))
        except TypeError:
            raise ValueError(
                f"批内多成员需同长序列，得到 prefix_id={prefix_id!r} "
                f"h_tokens={h_tokens!r}")
        out = []
        for pid, h in members:
            if not isinstance(h, int) or h < 0:
                raise ValueError(f"成员 h_tokens 必须为非负整数，得到 {h!r}")
            if h > 0:
                out.append((pid, h))
        return tuple(out)

    def rescale_remaining(self, seq, remaining):
        """盘模式重注入：把流剩余字节按组字节比例分摊到各盘组（推演世界用，
        真实世界不调用）；聚合模式退化为直接赋值。"""
        f = self.flows[seq]
        if not self.uses_groups:
            f.remaining_gb = remaining
            return
        factor = remaining / f.V_gb if f.V_gb != 0 else self.zero
        for g in self.groups[seq]:
            g.remaining_gb = g.total_gb * factor
        f.remaining_gb = sum(g.remaining_gb for g in self.groups[seq])

    def allocate(self, t):
        """带宽分配（每个事件点全量重算，在途流速率即时重配）。

        fcfs：按 submit_seq 升序，rate=min(q, max(0,free))；
        max_min：全部活跃流按申请率 q 水土填充 waterfill(B(t), [q])；
        max_min_disk：每盘独立 waterfill(B(t)/disks, [逐块需求×块数展开])，
        组速率 = 块分配 × 组内块数；流速率 = Σ组速率（账本流粒度不变）。
        """
        if self.alloc == "max_min":
            flows = sorted(self.flows.values(), key=lambda f: f.submit_seq)
            allocs = waterfill(self.b_at(t), [f.q_gbps for f in flows], self.zero)
            for f, r in zip(flows, allocs):
                if r != f.rate_gbps:
                    self.rate_log.append((f.submit_seq, t, r))
                    f.rate_gbps = r
            return
        if self.alloc == "max_min_disk":
            self._allocate_disks(t)
            return
        free = self.b_at(t)
        for seq in sorted(self.flows):
            f = self.flows[seq]
            r = min(f.q_gbps, max(self.zero, free))
            if r != f.rate_gbps:
                self.rate_log.append((f.submit_seq, t, r))
                f.rate_gbps = r
            free -= r

    def _allocate_disks(self, t):
        """盘模式分配：每盘对就绪组做块展开 waterfill（waterfill_weighted
        精确等价逐块展开）；组内逐块需求 = q/n_b 从当前 q 现算——boost 到
        q_max 后自动同比例抬升。"""
        by_disk: Dict[int, List[_DiskGroup]] = {}
        for seq in sorted(self.groups):
            for g in self.groups[seq]:
                by_disk.setdefault(g.disk, []).append(g)
        cap_d = self.disk_cap(t)
        for _d, gs in by_disk.items():
            # 同需求合并（精确等价展开；同组块同需求必同分配）
            weighted: Dict[object, List] = {}
            order: List[object] = []
            for g in gs:
                per = self.flows[g.seq].q_gbps / self.flows[g.seq].n_blocks
                if per not in weighted:
                    weighted[per] = [per, 0]
                    order.append(per)
                weighted[per][1] += g.n_blocks
            allocs = waterfill_weighted(
                cap_d, [tuple(weighted[per]) for per in order], self.zero)
            per_alloc = {per: a for (per, _c), (a, _n) in zip(
                (weighted[per] for per in order), allocs)}
            for g in gs:
                per = self.flows[g.seq].q_gbps / self.flows[g.seq].n_blocks
                r = per_alloc[per] * g.n_blocks
                if r != g.rate_gbps:
                    g.rate_gbps = r
        # 流级速率聚合（账本/观测口径不变）
        for seq, f in self.flows.items():
            gs = self.groups.get(seq)
            r = (sum(g.rate_gbps for g in gs) if gs else self.zero)
            if r != f.rate_gbps:
                self.rate_log.append((seq, t, r))
                f.rate_gbps = r

    def advance(self, t0, t1):
        """结算 [t0,t1)：速率不变区段的字节积分。"""
        dt = t1 - t0
        if dt <= 0:
            return
        cap = self.b_at(t0)
        self.integrated_capacity_gb += cap * dt
        if self.uses_groups:
            total_rate = self.zero
            disk_rates = [self.zero] * self.disks
            for seq, gs in self.groups.items():
                for g in gs:
                    g.remaining_gb -= g.rate_gbps * dt
                    total_rate += g.rate_gbps
                    disk_rates[g.disk] += g.rate_gbps
                # 流剩余 = Σ组剩余（流级账本口径不变；完成=最慢盘组）
                self.flows[seq].remaining_gb = sum(
                    g.remaining_gb for g in gs)
            total_q = self.zero
            for f in self.flows.values():
                total_q += f.q_gbps
            self.actual_integral_gb += total_rate * dt
            self.requested_integral_gb += total_q * dt
            if self.record_intervals:
                self.interval_log.append((t0, t1, total_rate, total_q))
                self.disk_interval_log.append((t0, t1, disk_rates))
            return
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
        """返回 [(seq, t + remaining/rate)]，rate>0 的完成事件。

        聚合模式为流级；盘模式为组级 ((seq, disk), t + remaining/rate)——
        组完成是重配事件，流完成 = 全部组完成（由引擎在组摘除后判定）。
        """
        out = []
        if self.uses_groups:
            for seq in self.groups:
                for g in self.groups[seq]:
                    if g.rate_gbps > 0:
                        out.append(((seq, g.disk),
                                    t + g.remaining_gb / g.rate_gbps))
            return out
        for seq, f in self.flows.items():
            if f.rate_gbps > 0:
                out.append((seq, t + f.remaining_gb / f.rate_gbps))
        return out

    def completed_groups(self):
        """盘模式下已完成（remaining ≤ 容差）的 (seq, disk) 列表。"""
        out = []
        for seq in self.groups:
            for g in self.groups[seq]:
                if g.remaining_gb <= self._tolGB(g):
                    out.append((seq, g.disk))
        return out

    def remove_group(self, seq, disk):
        """摘除一个盘组（组完成）；剩余字节移出流账本。返回被摘组。"""
        gs = self.groups.get(seq)
        if gs is None:
            return None
        for i, g in enumerate(gs):
            if g.disk == disk:
                gs.pop(i)
                self.flows[seq].remaining_gb -= g.remaining_gb
                if not gs:
                    del self.groups[seq]
                return g
        return None

    def next_due(self, t):
        """最早未提升且未完成的到期时刻。"""
        best = None
        for f in self.flows.values():
            if f.due_s is not None and not f.boosted and f.due_s > t:
                if best is None or f.due_s < best:
                    best = f.due_s
        return best

    def boost_due(self, t):
        """到期未读完的流提升为 q_max，保留序号（§4.2）。盘模式下逐块需求
        由 allocate 从当前 q 现算，自动同比例抬升。"""
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

    def _tolGB(self, g):
        return self.tol * max(1.0, float(g.total_gb)) if self.tol else 0

    def is_complete(self, f):
        if self.uses_groups:
            # 流完成 = 全部盘组完成（组已摘除则不计；末组残留 ≤ 容差即完成）
            return f.remaining_gb <= self._tolV(f) and not self.groups.get(
                f.submit_seq)
        return f.remaining_gb <= self._tolV(f)

    def remove(self, seq, t):
        f = self.flows.pop(seq)
        f.completed_s = t
        self.groups.pop(seq, None)
        return f
