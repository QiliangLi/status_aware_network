"""cq 实验族配置：全部 frozen dataclass，单位为秒 / 十进制 GB / GB/s。

十进制配置从字符串转 Fraction，不先经 float（见规格 §4.5）。分段常数带宽用
右连续、严格递增时间点的非负 schedule tuple 表示，最后一项永久延续。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from fractions import Fraction
from typing import Optional, Tuple

from .types import frac


def _parse_schedule(items) -> Tuple[Tuple[Fraction, Fraction], ...]:
    """解析 ((t0,v),(t1,v1),...) 带宽 schedule；要求 t=0 起点严格递增、值非负。"""
    out = []
    prev_t = None
    for i, (t, v) in enumerate(items):
        tf, vf = frac(t), frac(v)
        if vf < 0:
            raise ValueError(f"带宽不能为负: {v}")
        if i == 0:
            if tf != 0:
                raise ValueError("schedule 必须从 t=0 开始")
        else:
            if tf <= prev_t:
                raise ValueError("schedule 时间点必须严格递增")
        out.append((tf, vf))
        prev_t = tf
    return tuple(out)


@dataclass(frozen=True)
class ProfileConfig:
    """synthetic-v1 计算画像（规格 §5.1）。

    L 层；kappa GB/token/layer；c_{βℓ} = t_launch + aN/η(N) + bA，
    N=Σu, A=Σ[u h + u(u+1)/2], η(N)=min(1,max(1/16,N/N_sat))。
    no_speedup=True 时 η≡1 且 t_launch=0，任意批纯计算等于 singleton 之和。
    """

    profile_id: str = "synthetic-v1"
    L: int = 32
    kappa_gb_per_token_layer: Fraction = Fraction(4096, 10**9)  # 4.096e-6
    N_sat: int = 512
    t_launch_s: Fraction = Fraction(50, 10**6)
    a_s_per_token: Fraction = Fraction(2, 10**7)
    b_s_per_pair: Fraction = Fraction(1, 10**10)
    eta_floor: Fraction = Fraction(1, 16)
    no_speedup: bool = False
    mem_base_gb: Fraction = Fraction(1, 2)
    mem_per_token_gb: Fraction = Fraction(16 * 4096 * 2, 10**9)

    def effective(self):
        """no_speedup 对照：η≡1、launch=0，保持其余参数。"""
        if not self.no_speedup:
            return self
        return ProfileConfig(
            profile_id=self.profile_id + "|no_speedup", L=self.L,
            kappa_gb_per_token_layer=self.kappa_gb_per_token_layer,
            N_sat=self.N_sat, t_launch_s=Fraction(0),
            a_s_per_token=self.a_s_per_token, b_s_per_pair=self.b_s_per_pair,
            eta_floor=self.eta_floor, no_speedup=True,
            mem_base_gb=self.mem_base_gb, mem_per_token_gb=self.mem_per_token_gb)

    def to_json(self):
        d = asdict(self)
        d["kappa_gb_per_token_layer"] = str(self.kappa_gb_per_token_layer)
        for k in ("t_launch_s", "a_s_per_token", "b_s_per_pair", "eta_floor",
                  "mem_base_gb", "mem_per_token_gb"):
            d[k] = str(getattr(self, k))
        return d


@dataclass(frozen=True)
class StorageConfig:
    """聚合存储：B(t) schedule + 连接上限 q_max（GB/s）。"""

    b_schedule: Tuple[Tuple[Fraction, Fraction], ...] = ((Fraction(0), Fraction(80)),)
    q_max_gbps: Fraction = Fraction(200)
    b_ref_gbps: Fraction = Fraction(80)   # T0 与 history 观测使用的参考带宽

    def __post_init__(self):
        object.__setattr__(self, "b_schedule", _parse_schedule(self.b_schedule))

    def b_at(self, t) -> Fraction:
        """右连续阶梯：最后一个 time<=t 的项的值。"""
        cur = self.b_schedule[0][1]
        for tt, vv in self.b_schedule:
            if tt <= t:
                cur = vv
            else:
                break
        return cur

    def next_break(self, t):
        """严格大于 t 的下一个带宽断点；无则 None。"""
        for tt, _ in self.b_schedule[1:]:
            if tt > t:
                return tt
        return None

    def to_json(self):
        return {
            "b_schedule": [[str(t), str(v)] for t, v in self.b_schedule],
            "q_max_gbps": str(self.q_max_gbps),
            "b_ref_gbps": str(self.b_ref_gbps),
        }


@dataclass(frozen=True)
class ObservationConfig:
    """报价接口：默认 5ms 采样、立即交付、无噪声（规格 §6.1/§6.3）。"""

    mode: str = "current"           # history | current | current_oracle
    sample_period_s: Fraction = Fraction(5, 1000)
    delivery_lag_s: Fraction = Fraction(0)
    sigma: Fraction = Fraction(0)
    compute_bias: Fraction = Fraction(0)   # c_hat=(1+e)c_true，普通策略一致


@dataclass(frozen=True)
class ControllerConfig:
    """控制成本三模式：zero / fixed / measured（规格 §6.5）。"""

    cost_mode: str = "zero"
    fixed_delay_s: Fraction = Fraction(0)
    search_budget_s: Fraction = Fraction(2, 1000)
    max_forecast_events: int = 20000


@dataclass(frozen=True)
class SearchConfig:
    """MPC/local 搜索参数（规格 §7 固定默认值）。"""

    K_req: int = 32
    K_batch: int = 24
    J_width: int = 16
    beam_width: int = 8
    H: int = 2
    scenarios: Tuple[Tuple[Fraction, Fraction], ...] = ()   # 多情景容量因子；空=单情景
    gate_mode: str = "zero"         # zero | calibrated
    gate_values: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)   # M,S,T,R


@dataclass(frozen=True)
class CohortConfig:
    """在线 cohort 与排空（规格 §10.1）。有限工作集 warmup=0、全体计入。"""

    warmup_s: Fraction = Fraction(0)
    arrival_stop_s: Optional[Fraction] = None   # None=全部到达后
    drain_budget_s: Fraction = Fraction(600)


@dataclass(frozen=True)
class CqScenario:
    """一个场景的全部物理与 workload 配置（不含策略/窗口/成本模式）。"""

    scenario_name: str
    profile: ProfileConfig
    storage: StorageConfig
    limits: "HardLimits"
    m_workers: int = 4
    observation: ObservationConfig = ObservationConfig()
    controller: ControllerConfig = ControllerConfig()
    search: SearchConfig = SearchConfig()
    cohort: CohortConfig = CohortConfig()
    source_kind: str = "synthetic"          # synthetic | mooncake
    trace_key: str = ""                     # mooncake 文件名或合成标识
    lam0: Optional[Fraction] = None         # 结构组冻结的参考到达率
    lam: Optional[Fraction] = None          # 本次目标到达率
    alpha_slo: Fraction = Fraction(4)
    guard_w: Fraction = Fraction(0)         # 老化保护阈值（训练冻结后填入）

    def to_json(self):
        from .types import HardLimits
        return {
            "scenario_name": self.scenario_name,
            "profile": self.profile.to_json(),
            "storage": self.storage.to_json(),
            "limits": {"n_max": self.limits.n_max,
                       "token_max": self.limits.token_max,
                       "workspace_gb": str(self.limits.workspace_gb)},
            "m_workers": self.m_workers,
            "observation": asdict(self.observation),
            "controller": asdict(self.controller),
            "search": asdict(self.search),
            "cohort": {"warmup_s": str(self.cohort.warmup_s),
                       "arrival_stop_s": None if self.cohort.arrival_stop_s is None
                       else str(self.cohort.arrival_stop_s),
                       "drain_budget_s": str(self.cohort.drain_budget_s)},
            "source_kind": self.source_kind,
            "trace_key": self.trace_key,
            "lam0": None if self.lam0 is None else str(self.lam0),
            "lam": None if self.lam is None else str(self.lam),
            "alpha_slo": str(self.alpha_slo),
            "guard_w": str(self.guard_w),
        }
