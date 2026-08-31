"""仿真配置：全部参数以 dataclass 定义，单位约定时间为秒、字节用 GB(十进制)、带宽 GB/s。"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 80
    kv_heads: int = 8
    head_dim: int = 128
    bytes_per_elem: int = 2  # BF16

    @property
    def kv_gb_per_token(self) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_elem / 1e9


@dataclass(frozen=True)
class StorageConfig:
    b_total: float = 100.0
    t_base: float = 0.005
    # 分段常数背景负载 ((t0, GB/s), (t1, GB/s), ...)，t < t0 取首段
    bg_schedule: tuple = ((0.0, 0.0),)


@dataclass(frozen=True)
class GpuConfig:
    # (tokens, prefill 秒)，查表 + 分段线性插值；为占位值，待真实 GPU profile 替换
    prefill_table: tuple = (
        (4096, 0.030), (8192, 0.055), (16384, 0.105), (32768, 0.205), (65536, 0.410),
    )
    bg_schedule: tuple = ((0.0, 0.0),)


@dataclass(frozen=True)
class PrefixClass:
    name: str
    tokens: int
    share: float


@dataclass(frozen=True)
class WorkloadConfig:
    lam: float = 5.0
    hit_ratio: float = 0.7
    suffix_tokens: int = 1024
    ttft_slo: float = 1.0
    classes: tuple = (
        PrefixClass("A", 32768, 0.40),
        PrefixClass("B", 16384, 0.20),
        PrefixClass("C", 8192, 0.15),
        PrefixClass("D", 65536, 0.25),
    )

    @property
    def mean_hit_tokens(self) -> float:
        return sum(c.tokens * c.share for c in self.classes)


@dataclass(frozen=True)
class ObsConfig:
    interval: float = 0.05        # 状态采样间隔；0 表示 live（每次读取实时真值）
    ema_halflife: float = 0.1
    noise_sigma: float = 0.1      # 乘性 lognormal 噪声
    signal: str = "quote"         # util | queue | bw | quote


@dataclass(frozen=True)
class PolicyConfig:
    margin: float = 0.10          # 滞回：切换到 recompute 需估计优势超过该比例


@dataclass(frozen=True)
class RunSpec:
    exp: str
    policy: str
    seed: int
    duration: float = 400.0
    warmup: float = 20.0
    margin: float = 60.0          # duration 之后留给在途请求完成的仿真余量
    storages: tuple = (StorageConfig(),)          # 每_backend一个
    worker_backend: tuple = (0, 0, 0, 0)          # worker -> backend
    class_backends: tuple = ()                    # ((cls_name, (backend_ids...)),)，空=全部backend都持有
    gpu: GpuConfig = GpuConfig()
    wl: WorkloadConfig = WorkloadConfig()
    obs: ObsConfig = ObsConfig()
    pol: PolicyConfig = PolicyConfig()
    burst: tuple = None                           # (t0, n, dur_s, cls_name)
    window: tuple = None                          # (t0, t1) 窗口指标，如 E4 burst 窗口
    collect_ts: bool = False
    save_requests: bool = False
    save_ts: bool = False
    out_dir: str = None
    model: ModelConfig = ModelConfig()

    def class_backend_map(self) -> dict:
        all_b = tuple(sorted(set(self.worker_backend)))
        return dict(self.class_backends) if self.class_backends else {c.name: all_b for c in self.wl.classes}

    def fg_demand_gbps(self) -> float:
        """前台 fetch 需求的解析估计（GB/s），用于坐标轴的 offered load 换算。"""
        return self.wl.lam * self.wl.hit_ratio * self.wl.mean_hit_tokens * self.model.kv_gb_per_token


def stable(v: float) -> tuple:
    return ((0.0, v),)


def square(mean: float, amp: float, period: float, until: float) -> tuple:
    """方波背景负载：从低电平 mean-amp 开始，每 period/2 切换一次。"""
    segs, t, i = [], 0.0, 0
    while t < until + period:
        segs.append((t, max(0.0, mean - amp if i % 2 == 0 else mean + amp)))
        t += period / 2
        i += 1
    return tuple(segs)


def with_bg_storage(spec: RunSpec, bg_schedule: tuple) -> RunSpec:
    st = replace(spec.storages[0], bg_schedule=bg_schedule)
    return replace(spec, storages=(st,))


def with_bg_gpu(spec: RunSpec, bg_schedule: tuple) -> RunSpec:
    return replace(spec, gpu=replace(spec.gpu, bg_schedule=bg_schedule))
