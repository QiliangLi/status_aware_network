"""计算画像：synthetic-v1 真值、内存峰值、singleton 参考时间 T0 与预测器。

纯函数模块，不读 world（规格 §11.1）。所有层共享同一 c；η 仅作用于 aN 项。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, Optional, Sequence, Tuple

from .config import ProfileConfig
from .types import HardLimits, RequestSpec


def eta(n_tokens: int, cfg: ProfileConfig) -> Fraction:
    """η(N)=min(1,max(eta_floor,N/N_sat))；no_speedup 时恒 1。"""
    if cfg.no_speedup:
        return Fraction(1)
    if n_tokens >= cfg.N_sat:
        return Fraction(1)
    lo = max(n_tokens, 1)
    v = Fraction(lo, cfg.N_sat)
    return max(cfg.eta_floor, min(Fraction(1), v))


def batch_compute_s(members: Sequence[RequestSpec], cfg: ProfileConfig) -> Fraction:
    """整批每层计算时长 c_{βℓ}（所有层相同）。

    c = t_launch + aN/η(N) + bA；N=Σu，A=Σ[u h + u(u+1)/2]，只算真实 token
    与有效 attention 配对，不按最长序列 padding。
    """
    N = sum(r.u_tokens for r in members)
    A = sum(r.u_tokens * r.h_tokens + r.u_tokens * (r.u_tokens + 1) // 2
            for r in members)
    e = eta(N, cfg)
    t_launch = 0 if cfg.no_speedup else cfg.t_launch_s
    return t_launch + cfg.a_s_per_token * N / e + cfg.b_s_per_pair * A


def mem_peak_gb(members: Sequence[RequestSpec], cfg: ProfileConfig) -> Fraction:
    """批工作区内存峰值：0.5 + L*κ*Σ(h+u) + mem_per_token*N（GB，十进制）。"""
    L = cfg.L
    kv = sum(r.h_tokens + r.u_tokens for r in members)
    N = sum(r.u_tokens for r in members)
    return (cfg.mem_base_gb + L * cfg.kappa_gb_per_token_layer * kv
            + cfg.mem_per_token_gb * N)


def layer_read_gb(r: RequestSpec, cfg: ProfileConfig) -> Fraction:
    """v_{iℓ}=κ*h_i：命中前缀的 KV 计内存但不计远端读取以外的部分；h=0 不建流。"""
    return cfg.kappa_gb_per_token_layer * r.h_tokens


def is_feasible(members: Sequence[RequestSpec], limits: HardLimits,
                cfg: ProfileConfig) -> bool:
    """统一硬可行域纯函数：1<=n<=n_max、Σu<=token_max、MemPeak<=workspace。"""
    n = len(members)
    if n < 1 or n > limits.n_max:
        return False
    if sum(r.u_tokens for r in members) > limits.token_max:
        return False
    if mem_peak_gb(members, cfg) > limits.workspace_gb:
        return False
    return True


def singleton_K(r: RequestSpec, cfg: ProfileConfig, b_ref: Fraction,
                q_max: Fraction) -> Fraction:
    """singleton 纯服务时间 K（空系统、B_ref/q_max 上限、无队列）。

    K = V/min(B_ref,q_max) + L*c + (L-1)*max(0, V/min(B_ref,q_max) - c)，
    即恒定层画像下的解析 singleton（规格 §5.2）。
    """
    V = layer_read_gb(r, cfg)
    c = batch_compute_s([r], cfg)
    eff = min(b_ref, q_max)
    t_read = V / eff
    L = cfg.L
    return t_read + L * c + (L - 1) * max(Fraction(0), t_read - c)


def make_T0(specs: Sequence[RequestSpec], cfg: ProfileConfig,
            b_ref: Fraction, q_max: Fraction) -> Dict[int, Fraction]:
    """批量生成 T0 参考时间（独立 singleton 重放解析式）。"""
    return {r.rid: singleton_K(r, cfg, b_ref, q_max) for r in specs}


# ---------------------------------------------------------------------------
# 类别表（§5.1 四类 + 校验点 h=2048/u=128）
# ---------------------------------------------------------------------------

CLASS_TABLE: Tuple[Tuple[str, int, int], ...] = (
    ("HS_US", 512, 128),
    ("HS_UL", 512, 2048),
    ("HL_US", 8192, 128),
    ("HL_UL", 8192, 2048),
)


def class_of(class_id: str) -> Tuple[int, int]:
    for cid, h, u in CLASS_TABLE:
        if cid == class_id:
            return h, u
    raise KeyError(class_id)


# ---------------------------------------------------------------------------
# 最近邻预测器（E19 §5.4 首版替身；合成真值模式下直接公式查询）
# ---------------------------------------------------------------------------


@dataclass
class ProfilePredictor:
    """按形状最近邻预测整批每层计算时长。特征 log1p(n,N,Σu²,Σhu,max u,max h)，
    逐维标准化后欧氏距离；精确键命中直接用训练中位数；缺覆盖记 OOD。"""

    cfg: ProfileConfig
    train: Dict[Tuple[int, ...], Fraction] = field(default_factory=dict)
    _mean = None
    _std = None

    def fit(self, shapes: Dict[Tuple[Tuple[int, int], ...], Fraction]):
        import math
        self.train = {self._shape_key(s): v for s, v in shapes.items()}
        feats = [self._features(k) for k in self.train]
        d = len(feats[0])
        self._mean = [sum(f[i] for f in feats) / len(feats) for i in range(d)]
        self._std = []
        for i in range(d):
            var = sum((f[i] - self._mean[i]) ** 2 for f in feats) / len(feats)
            self._std.append(var ** 0.5 if var > 0 else 1.0)

    @staticmethod
    def _shape_key(members: Sequence[Tuple[int, int]]):
        return tuple(sorted(members))

    @staticmethod
    def _features(shape_key: Tuple[Tuple[int, int], ...]):
        import math
        n = len(shape_key)
        N = sum(u for _, u in shape_key)
        su2 = sum(u * u for _, u in shape_key)
        shu = sum(h * u for h, u in shape_key)
        mu = max(u for _, u in shape_key)
        mh = max(h for h, _ in shape_key)
        return tuple(math.log1p(x) for x in (n, N, su2, shu, mu, mh))

    def predict(self, members: Sequence[Tuple[int, int]]):
        """返回 (c_hat, is_exact, nn_distance)。合成真值模式直接公式查询。"""
        key = self._shape_key([(h, u) for h, u in members])
        if not self.train:
            # 合成真值：直接按公式计算（§5.4，误差应为 0）
            specs = [RequestSpec(rid=i, arrival_s=Fraction(0), h_tokens=h,
                                 u_tokens=u, class_id="adhoc",
                                 T0_s=Fraction(1), deadline_s=Fraction(1))
                     for i, (h, u) in enumerate(key)]
            return batch_compute_s(specs, self.cfg), True, 0.0
        if key in self.train:
            return self.train[key], True, 0.0
        f = self._features(key)
        best, bestd = None, None
        for k2 in self.train:
            f2 = self._features(k2)
            d = sum((a - b) ** 2 for a, b in zip(f, f2)) ** 0.5
            if bestd is None or (d, k2) < (bestd, best or k2):
                best, bestd = k2, d
        return self.train[best], False, bestd


@dataclass(frozen=True)
class SplitHasher:
    """形状规范键 SHA256 转整数 mod 5：0 留出、1-4 训练（§5.4）。"""

    @staticmethod
    def split_of(shape_key: Tuple[Tuple[int, int], ...]) -> int:
        import hashlib
        canon = json_key = (";".join(f"{h},{u}" for h, u in sorted(shape_key)))
        h = hashlib.sha256(canon.encode()).digest()
        return int.from_bytes(h[:8], "big") % 5
