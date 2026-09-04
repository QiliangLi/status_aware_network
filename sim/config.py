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
class NodeConfig:
    """存储节点：内存/固态盘两个服务层（各自独立带宽与背景负载），容量用于容量压力状态。"""
    name: str
    mem: StorageConfig = StorageConfig()
    ssd: StorageConfig = StorageConfig(b_total=30.0)
    cap_gb: float = 512.0


@dataclass(frozen=True)
class TopoConfig:
    """v2 共享分布式 KV 存储拓扑（对应架构图：计算节点+本地缓存 / 高速网络 / 多存储节点+元数据目录）。"""
    n_workers: int = 4
    nodes: tuple = (NodeConfig("n0"), NodeConfig("n1"), NodeConfig("n2"))
    fabric: StorageConfig = StorageConfig(b_total=120.0)
    # 路径延迟矩阵 W×N（秒），worker->node 的静态路径差异
    path_lat: tuple = ((0.002, 0.004, 0.008),) * 4
    local_cache_gb: float = 12.0
    # 初始副本放置：((cls, ((node_idx, "mem"|"ssd"), ...)), ...)；未列出的类视为无副本（只能重算/prefill）
    replicas: tuple = ()
    # 复制控制器（Q4）：None 表示不启用运行时复制
    ctrl: "CtrlConfig | None" = None
    # 每 worker 的 GPU 背景负载（分段常数表）；空 = 全部用 GpuConfig.bg_schedule
    gpu_bgs: tuple = ()
    # t=0 时预置进 worker 本地缓存的 (worker, cls)
    seed_local: tuple = ()
    # 预取/回写控制器（问题②）：None 表示不启用
    prefetch: "PrefetchConfig | None" = None
    # 本地缓存放置：lru（被动）| coord（类->偏好worker一致性哈希准入）
    cache_mode: str = "lru"


@dataclass(frozen=True)
class PrefetchConfig:
    """Q4 预取（问题②）：会话续写预取 + 重算后回写。"""
    mode: str = "gated"          # none | always | gated | predictive | session
    writeback: bool = False      # 重算完成后把完整 KV 异步回写到低压共享存储节点
    protect: bool = False        # 问题⑫：活跃会话类驱逐豁免 + 预取准入检查


@dataclass(frozen=True)
class CtrlConfig:
    interval: float = 0.5        # 控制器评估周期（秒）
    hot_util: float = 0.85       # 进入 HOT 的利用率阈值（对可观测 util）
    exit_util: float = 0.65      # 退出 HOT 的阈值（滞回）
    hold_s: float = 2.0          # 持续该状态多久才触发
    min_demand: float = 1.0      # 类的滚动到达率（req/s）高于此才值得复制
    predictive: bool = False     # True: 用 EMA 斜率提前触发（更低阈值）
    max_replicas: int = 3        # 单类最大副本数
    # ---- 问题④扩展 ----
    cross_tier: bool = False     # 目标层含 ssd（跨层降级）
    cold_demand: float = 0.5     # 类需求低于此且源在 mem -> 迁移而非复制（回收快层容量）
    cap_evict: float = 0.90      # 节点容量压力超过此值触发淘汰
    evict_target: float = 0.75   # 淘汰到此值以下
    cooldown_s: float = 30.0     # 同一类两次复制/迁移的最小间隔（防漂移 churn）
    evict_demand: float = 0.2    # 仅淘汰滚动需求低于此值的类


@dataclass(frozen=True)
class ObsConfig:
    interval: float = 0.05        # 状态采样间隔；0 表示 live（每次读取实时真值）
    ema_halflife: float = 0.1
    noise_sigma: float = 0.1      # 乘性 lognormal 噪声
    signal: str = "quote"         # util | queue | bw | quote
    # GPU 侧观测（问题⑤）：默认 0/0 = 真值可见（v1 既定简化，向后兼容），E12 扫描时启用
    gpu_interval: float = 0.0
    gpu_noise: float = 0.0
    gpu_ema: float = 0.1


@dataclass(frozen=True)
class PolicyConfig:
    margin: float = 0.10          # 滞回：切换到 recompute 需估计优势超过该比例
    guardband: float = 1.2         # cascade2 保守系数 γ（Lr = γ × 估计值）


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
    sessions: tuple = None                        # (session_rate, mean_turns, gap_mean) 会话型负载
    drift: tuple = None                           # (period,) 类热度漂移周期（秒）
    topo: "TopoConfig | None" = None              # v2 共享分布式存储拓扑；None = v1 单/多 backend 模式
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
