"""M0：解析/小规模模型（纲领 v1.1 §3、R0 的解析层）。

单位约定与 sim.config 一致：时间为秒、字节用 GB（十进制）、带宽 GB/s。
本模块只包含无状态解析函数与 frozen dataclass 参数，不进入仿真事件循环；
用于 R0 的"解析一致"不变量测试与手算核对，不直接作为完整 serving 的延迟公式。

对应纲领 v1.1 的三处推导：
- §2.3 KV 字节一致性检查（复用 ModelConfig.kv_gb_per_token）；
- §3.1 单请求取回/重算交叉边界：F=ℓ+B/b、b*=B/(ΔC-ℓ)、ΔC≤ℓ 时带宽无解；
- §3.3 两请求资源分配算例：异质计算代价使分配有价值。
"""
from __future__ import annotations

from dataclasses import dataclass

from sim.config import ModelConfig


@dataclass(frozen=True)
class RecoveryCost:
    """单请求取回/重算权衡的输入（纲领 §3.1 限定模型：单瓶颈、固定启动成本、无排队）。

    R: 从现有本地状态出发、不使用远端 KV 的 prefill 成本（秒）
    P: 恢复目标前缀后，剩余 prefill 的成本（秒）；关键比较对象是 ΔC=R-P，
       不是完整输入的 prefill 成本
    B: 需要取回的 KV 字节（GB），按块集合去重、对齐后计数
    ell: 传输固定启动成本 ℓ（秒）
    bw: 取回可用带宽（GB/s）
    """

    R: float
    P: float
    B: float
    ell: float
    bw: float

    @property
    def delta_c(self) -> float:
        """这次取回实际避免的 GPU 计算时间 ΔC = R - P。"""
        return self.R - self.P

    def fetch_time(self) -> float:
        """端到端取回时间 F = ℓ + B/bw。"""
        return self.ell + self.B / self.bw

    def fetch_path_time(self) -> float:
        """取回路径总时间 F + P。"""
        return self.fetch_time() + self.P

    def fetch_wins(self) -> bool:
        """取回路径是否严格优于重算路径：F + P < R ⇔ F < ΔC。"""
        return self.fetch_time() < self.delta_c

    def threshold_bandwidth(self) -> float | None:
        """取回划算所需的带宽阈值 b* = B/(ΔC-ℓ)。

        ΔC ≤ ℓ 时取回时间下界为 ℓ，任何带宽都无法胜出，返回 None。
        """
        denom = self.delta_c - self.ell
        if denom <= 0:
            return None
        return self.B / denom


def two_request_completion(
    recompute_times: tuple[float, float],
    bytes_each: float,
    io_bw: float,
    allocation: str,
) -> float:
    """两请求资源分配算例（纲领 §3.3）：返回"两者全部恢复完成时间"（秒）。

    限定模型：两请求同时到达，各需恢复 bytes_each GB 的 KV；I/O 与 GPU 是
    互不干扰的两类资源，各自串行服务；只允许完整取回或完整重算，忽略后缀、
    decode 和传输启动成本。

    recompute_times: 两请求单独重算时长 (t0, t1)（秒）
    bytes_each: 每请求需取回的 KV 字节（GB）
    io_bw: 共享 I/O 带宽（GB/s）
    allocation: "both_fetch"（都取）/ "both_recompute"（都算）/
        "fetch_first"（请求 0 取、请求 1 算）/ "fetch_last"（请求 0 算、请求 1 取）
    """
    if allocation == "both_fetch":
        return 2.0 * bytes_each / io_bw
    if allocation == "both_recompute":
        return recompute_times[0] + recompute_times[1]
    if allocation == "fetch_first":
        fetched, recomputed = 0, 1
    elif allocation == "fetch_last":
        fetched, recomputed = 1, 0
    else:
        raise ValueError(f"未知分配方式: {allocation}")
    io_time = bytes_each / io_bw
    gpu_time = recompute_times[recomputed]
    return max(io_time, gpu_time)


def recovery_case_from_tokens(
    model: ModelConfig,
    miss_tokens: int,
    recompute_full_s: float,
    recompute_suffix_s: float,
    ell: float,
    bw: float,
) -> RecoveryCost:
    """由 token 数构造取回/重算算例：KV 字节按模型张量计数，计算成本由外部给定。

    miss_tokens: 需要取回的未命中 token 数（块对齐前后的差异由调用方处理）
    recompute_full_s: 不使用远端 KV 的完整 prefill 成本 R（秒）
    recompute_suffix_s: 恢复前缀后剩余 prefill 成本 P（秒）
    """
    return RecoveryCost(
        R=recompute_full_s,
        P=recompute_suffix_s,
        B=model.kv_gb_per_token * miss_tokens,
        ell=ell,
        bw=bw,
    )
