"""M0 事件级微仿真：单 GPU、单 I/O 的串行服务世界（纲领 v1.1 R0 解析层配套）。

用 simpy 把 §3.1/§3.3 的限定模型按事件推进，用于"解析值与仿真值"的一致性检查。
服务纪律为 FCFS 串行（容量 1），与解析模型"两资源互不干扰、资源内串行"的假设
一致；排队与理想分享场景属于 R0 后续扩展，不在本模块。

产出三样 R0 要求的证据：
- 甘特区间（JobSpan 列表）；
- 逐资源工作账本（busy_time：累计服务秒数）；
- 完成时间（与 sim.m0 解析值对照）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import simpy

from sim.m0 import RecoveryCost


@dataclass(frozen=True)
class JobSpan:
    """一段不可抢占的作业区间（甘特图基本单元）。"""

    resource: str
    job: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class M0Run:
    """一次微仿真的结果：完成时间、甘特区间与逐资源账本。"""

    completion: float
    spans: list[JobSpan] = field(default_factory=list)
    ledger: dict[str, float] = field(default_factory=dict)


class SerialServer:
    """容量 1 的 FCFS 串行资源，记录每段作业的起止时间。"""

    def __init__(self, env: simpy.Environment, name: str):
        self.env = env
        self.name = name
        self.res = simpy.Resource(env, capacity=1)
        self.spans: list[JobSpan] = []

    def serve(self, job: str, duration: float):
        """提交一段作业；多作业同时提交时按 FCFS 排队。"""
        return self._serve(job, duration)

    def _serve(self, job: str, duration: float):
        with self.res.request() as req:
            yield req
            start = self.env.now
            yield self.env.timeout(duration)
            self.spans.append(JobSpan(self.name, job, start, self.env.now))

    @property
    def busy_time(self) -> float:
        """累计服务时间（秒）＝账本中该资源的工作量。"""
        return sum(s.duration for s in self.spans)


def _collect(env: simpy.Environment, servers: list[SerialServer]) -> M0Run:
    spans = [s for server in servers for s in server.spans]
    completion = max((s.end for s in spans), default=0.0)
    ledger = {server.name: server.busy_time for server in servers}
    return M0Run(completion=completion, spans=spans, ledger=ledger)


def run_two_request(
    recompute_times: tuple[float, float],
    bytes_each: float,
    io_bw: float,
    allocation: str,
) -> M0Run:
    """按 §3.3 算例推进两请求世界，动作语义与 sim.m0.two_request_completion 一致。"""
    if allocation == "both_fetch":
        actions = ("fetch", "fetch")
    elif allocation == "both_recompute":
        actions = ("recompute", "recompute")
    elif allocation == "fetch_first":
        actions = ("fetch", "recompute")
    elif allocation == "fetch_last":
        actions = ("recompute", "fetch")
    else:
        raise ValueError(f"未知分配方式: {allocation}")

    env = simpy.Environment()
    io = SerialServer(env, "io")
    gpu = SerialServer(env, "gpu")
    for i, action in enumerate(actions):
        if action == "fetch":
            env.process(io.serve(f"req{i}-fetch", bytes_each / io_bw))
        else:
            env.process(gpu.serve(f"req{i}-recompute", recompute_times[i]))
    env.run()
    return _collect(env, [io, gpu])


def run_fetch_path(case: RecoveryCost) -> M0Run:
    """§3.1 取回路径：ℓ 固定启动 → I/O 传输 B/bw → GPU 剩余 prefill P。"""
    env = simpy.Environment()
    io = SerialServer(env, "io")
    gpu = SerialServer(env, "gpu")

    def flow():
        yield env.timeout(case.ell)
        yield env.process(io.serve("fetch-transfer", case.B / case.bw))
        yield env.process(gpu.serve("suffix-prefill", case.P))

    env.process(flow())
    env.run()
    return _collect(env, [io, gpu])


def run_recompute_path(case: RecoveryCost) -> M0Run:
    """§3.1 重算路径：仅占用 GPU，时长 R。"""
    env = simpy.Environment()
    gpu = SerialServer(env, "gpu")
    env.process(gpu.serve("recompute-full", case.R))
    env.run()
    return _collect(env, [gpu])
