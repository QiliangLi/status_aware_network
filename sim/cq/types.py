"""cq 实验族的核心数据类型。

区分三类对象：
- 不可变输入（RequestSpec、HardLimits 等）：来自 trace/配置，运行期只读；
- 运行状态（RequestRuntime、BatchRuntime、FlowState、WorkerState）：物理真值，只属于
  执行内核与显式 Oracle，普通策略不可见（见 sim/cq/observable.py 的信息边界）；
- 可观测快照与动作（ObservableSnapshot、Action、JointAction）：普通策略的唯一接口。

数值约定：时间秒、字节 GB（十进制）、带宽 GB/s。精确模式（E20、四计划金标）
全程使用 fractions.Fraction；生产仿真使用 float64。同一 run 内不混用两种类型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

# 数值类型：精确后端 Fraction，生产后端 float。
Number = "Fraction | float"


def frac(x) -> Fraction:
    """把 int/str/float 安全转为 Fraction；字符串支持 'a/b' 与小数。"""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x) if "/" in x else Fraction(x)
    if isinstance(x, float):
        return Fraction(x).limit_denominator(10**12)
    raise TypeError(f"无法转为 Fraction: {type(x)}")


# ---------------------------------------------------------------------------
# 不可变输入
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestSpec:
    """请求的不可变输入。rid 从 0 起整数；arrival_s>=0, h>=0, u>=1, T0>0,
    deadline>=arrival。"""

    rid: int
    arrival_s: object          # Number（Fraction 或 float）
    h_tokens: int              # 命中前缀 KV 对应 token 数（读取字节来源）
    u_tokens: int              # 新增 token 数（计算量来源）
    class_id: str
    T0_s: object               # 冻结的 singleton 参考时间（Number）
    deadline_s: object         # Number
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    source_timestamp_ms: Optional[int] = None
    input_length: Optional[int] = None
    output_length: Optional[int] = None


@dataclass(frozen=True)
class HardLimits:
    """批的硬可行域：1<=n<=n_max，sum(u)<=token_max，MemPeak<=workspace_gb。"""

    n_max: int = 8
    token_max: int = 8192
    workspace_gb: Fraction = Fraction(16)


@dataclass(frozen=True)
class Action:
    """单个 worker 的动作。DISPATCH 携带升序 rid 成员 tuple；WAIT 携带严格未来
    唤醒时刻。"""

    kind: str                      # 'DISPATCH' | 'WAIT'
    worker_id: int
    members: tuple[int, ...] = ()
    wake_at: object = None         # Number，WAIT 时必填且 > now

    def key(self):
        if self.kind == "DISPATCH":
            return (0, self.members, None)
        return (1, (), self.wake_at)


@dataclass(frozen=True)
class JointAction:
    """一次决策的全部动作，worker 升序、无重复成员、无重复 worker。"""

    actions: tuple[Action, ...]

    def key(self):
        return tuple(a.key() for a in self.actions)


# ---------------------------------------------------------------------------
# 运行状态（物理真值，普通策略不可见）
# ---------------------------------------------------------------------------


@dataclass
class RequestRuntime:
    spec: RequestSpec
    state: str = "FUTURE"          # FUTURE -> QUEUED -> ACTIVE -> DONE
    batch_id: Optional[int] = None
    F_s: Optional[object] = None   # 完成时刻（Number）
    queue_enter_s: Optional[object] = None
    dispatch_s: Optional[object] = None


@dataclass
class FlowState:
    """一个聚合读取流。submit_seq 决定 FCFS 排位；due 到期未读完则 q 提升为
    q_max 且保留原序号，不可撤回重提。"""

    flow_id: int
    batch_id: int
    layer: int
    submit_seq: int
    submit_s: object               # A_{beta,l}
    V_gb: object
    remaining_gb: object
    q_gbps: object
    due_s: Optional[object]        # lookahead 到期时刻；首层为 None（无到期）
    boosted: bool = False
    rate_gbps: object = 0
    completed_s: Optional[object] = None


@dataclass
class BatchRuntime:
    """一个已领取的批。层状态机：wait_read(ℓ) -> wait_prev(ℓ) -> computing(ℓ)。
    C_{β,ℓ+1} 时刻提交下一层流；末层计算结束 Z 即成员共同 F。"""

    batch_id: int
    members: tuple[int, ...]
    worker_id: int
    dispatch_s: object
    L: int
    next_layer: int = 0            # 下一个需要启动计算的层
    read_ready: list = field(default_factory=list)     # 每层读完成时刻或 None
    C: list = field(default_factory=list)              # 每层计算开始
    Z: list = field(default_factory=list)              # 每层计算结束
    c_layers: list = field(default_factory=list)       # 每层计算时长（真值）
    V_layers: list = field(default_factory=list)       # 每层读取 GB
    F_s: Optional[object] = None
    compute_s: object = 0
    stall_s: object = 0
    submitted_layers: int = 0       # 已提交读取流的层数

    def __post_init__(self):
        if not self.read_ready:
            self.read_ready = [None] * self.L
            self.C = [None] * self.L
            self.Z = [None] * self.L
            self.c_layers = [None] * self.L
            self.V_layers = [None] * self.L


@dataclass
class WorkerState:
    """worker 记账：COMPUTE / STALL / IDLE 三态积分。"""

    worker_id: int
    batch_id: Optional[int] = None
    state: str = "IDLE"            # COMPUTE | STALL | IDLE
    state_since: object = 0
    compute_s: object = 0
    stall_s: object = 0
    idle_s: object = 0
    segments: list = field(default_factory=list)  # (start,end,state,batch_id,layer)
    idle_reason: str = ""

    def set_state(self, t, state, batch_id=None, layer=None, idle_reason=""):
        """切换状态并结算区间。"""
        if state == self.state and batch_id == self.batch_id:
            return
        seg_state = self.state
        dt = t - self.state_since
        if seg_state == "COMPUTE":
            self.compute_s = self.compute_s + dt
        elif seg_state == "STALL":
            self.stall_s = self.stall_s + dt
        elif seg_state == "IDLE":
            self.idle_s = self.idle_s + dt
        self.segments.append((self.state_since, t, seg_state, self.batch_id, layer))
        self.state = state
        self.state_since = t
        self.batch_id = batch_id if state != "IDLE" else None
        self.idle_reason = idle_reason if state == "IDLE" else ""
