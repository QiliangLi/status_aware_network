"""cq 实验公共框架：场景构建、Mooncake 窗口装载、策略工厂、输出目录与绘图入口。

输出路径：results/cq/<stage>/<exp>/…（简化为 stage/exp 维度目录，manifest 内含
完整 scenario 配置与哈希）。图默认留 results，仅显式 publish 复制到 docs/figures。
"""
from __future__ import annotations

import json
import os
import time
from fractions import Fraction as F
from typing import Dict, List, Optional, Sequence, Tuple

from sim.cq.config import (CqScenario, CohortConfig, ControllerConfig,
                           ObservationConfig, ProfileConfig, SearchConfig,
                           StorageConfig)
from sim.cq.metrics import summarize
from sim.cq.observable import Observable
from sim.cq.policies import (SIMPLE_POLICIES, GuardedEDF, make_edf, make_fcfs,
                             make_lpm, make_pair, make_slack, make_spt)
from sim.cq.profile import make_T0, singleton_K
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.trace import (TRACE_LABEL, TimeBlock, block_rows, import_mooncake,
                          split_blocks, trace_hash)
from sim.cq.types import HardLimits, RequestSpec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACE_DIR_DEFAULT = os.path.join(REPO, "mooncake_trace")
RESULTS_DIR = os.path.join(REPO, "results", "cq")

MECH_LIMITS = HardLimits(8, 8192, F(16))            # 四类机制组
TRACE_WIDE_LIMITS = HardLimits(8, 262144, F(128))    # 真实 trace 能力组

# §3.1 全部普通策略（7 简单/局部 + MPC）
ALL_POLICIES = ["cq_fcfs", "cq_lpm", "cq_spt", "cq_edf", "cq_slack",
                "cq_pair", "cq_local", "cq_mpc"]


def build_policy(pid: str, theta: str = "S", c_ref: float = 0.01,
                 H: int = 2):
    if pid == "cq_local":
        return MPCPolicy(pid="cq_local", local=True, H=1, theta=theta,
                         c_ref=c_ref)
    if pid == "cq_mpc":
        return MPCPolicy(pid="cq_mpc", H=H, theta=theta, c_ref=c_ref)
    if pid == "cq_current_oracle":
        return MPCPolicy(pid="cq_current_oracle", H=1, theta=theta,
                         c_ref=c_ref)
    fn = SIMPLE_POLICIES[pid]
    return fn()


class MooncakeSource:
    """一个 Mooncake 文件的冻结导入 + 窗口装载（λ0 从训练区间冻结）。"""

    def __init__(self, fname: str, trace_dir: str = TRACE_DIR_DEFAULT):
        self.fname = fname
        self.path = os.path.join(trace_dir, fname)
        self.imp = import_mooncake(self.path, fname)
        self.label = TRACE_LABEL[fname]
        train_rows = [r for r in self.imp.rows
                      if self.imp.build_end_ms <= r.timestamp_ms
                      < self.imp.train_end_ms]
        self.train_rows = train_rows
        self.eval_rows = [r for r in self.imp.rows
                          if r.timestamp_ms >= self.imp.train_end_ms]
        self.train_blocks = split_blocks(self.imp.rows, self.imp.build_end_ms,
                                         self.imp.train_end_ms, 5)
        self.eval_blocks = split_blocks(self.imp.rows, self.imp.train_end_ms,
                                        max(r.timestamp_ms for r in self.imp.rows) + 1, 15)
        # λ0 = min(80/E[V_layer], m/E[K_singleton])，m=4
        prof = ProfileConfig()
        hus = [self.imp.h_u_of(r)[:2] for r in train_rows]
        if hus:
            e_v = sum(F(prof.kappa_gb_per_token_layer) * h for h, _u in hus) / len(hus)
            ks = []
            for h, u in hus:
                rs = RequestSpec(0, 0, h, u, "x", 1, 1)
                ks.append(F(singleton_K(rs, prof, F(80), F(200))))
            e_k = sum(ks) / len(ks)
            self.lam0 = min(F(80) / e_v, F(4) / e_k)
        else:
            self.lam0 = F(1)

    def window_specs(self, block_id: int, lam: F, alpha: F = F(4),
                     duration_cap: Optional[float] = None,
                     limits: HardLimits = TRACE_WIDE_LIMITS
                     ) -> Tuple[List[RequestSpec], float]:
        """装载一个时间块窗口：缩放到目标负载 λ，T0/SLO 用 B_ref=80 冻结。"""
        blocks = self.train_blocks + self.eval_blocks
        blk = next(b for b in blocks if b.block_id == block_id)
        rows = block_rows(self.imp.rows, blk)
        if not rows:
            return [], 0.0
        arrs, d_sim = _scaled(rows, blk, lam)
        prof = ProfileConfig()
        specs = []
        for i, r in enumerate(rows):
            if duration_cap is not None and arrs[i] > duration_cap:
                break
            h, u, full = self.imp.h_u_of(r)
            specs.append(RequestSpec(
                rid=i, arrival_s=arrs[i], h_tokens=h, u_tokens=u,
                class_id="mooncake", T0_s=1, deadline_s=1,
                source_file=self.fname, source_line=r.source_line,
                source_timestamp_ms=r.timestamp_ms,
                input_length=r.input_length, output_length=r.output_length))
        if not specs:
            return [], d_sim
        T0 = make_T0(specs, prof, F(80), F(200))
        out = []
        for s in specs:
            t0 = T0[s.rid]
            out.append(RequestSpec(
                rid=s.rid, arrival_s=s.arrival_s, h_tokens=s.h_tokens,
                u_tokens=s.u_tokens, class_id=s.class_id, T0_s=t0,
                deadline_s=s.arrival_s + alpha * t0,
                source_file=s.source_file, source_line=s.source_line,
                source_timestamp_ms=s.source_timestamp_ms,
                input_length=s.input_length, output_length=s.output_length))
        return out, d_sim


def _scaled(rows, blk, lam):
    from sim.cq.trace import scaled_arrivals
    return scaled_arrivals(rows, blk, lam, numeric=float)


def default_scenario(B_gbps: float = 80.0, m: int = 4, alpha: F = F(4),
                     limits: HardLimits = MECH_LIMITS,
                     profile: Optional[ProfileConfig] = None,
                     cost_mode: str = "zero",
                     guard_w: float = 0.0) -> CqScenario:
    prof = profile or ProfileConfig()
    return CqScenario(
        scenario_name=f"cq_B{B_gbps:g}_a{alpha}",
        profile=prof,
        storage=StorageConfig(b_schedule=((F(0), F(int(B_gbps))),),
                              q_max_gbps=F(200), b_ref_gbps=F(80)),
        limits=limits, m_workers=m,
        controller=ControllerConfig(cost_mode=cost_mode),
        alpha_slo=alpha, guard_w=F(guard_w) if guard_w else F(0))


def run_one(scenario: CqScenario, specs, pid: str, theta: str,
            c_ref: float, seed: int = 0, H: int = 2,
            warmup: Optional[float] = None) -> dict:
    """跑一个 (scenario, specs, policy) 并返回 summary + trace 哈希。"""
    pol = build_policy(pid, theta, c_ref, H)
    eng = run_case(scenario, specs, pol, numeric=float, seed=seed)
    s = summarize(eng, scenario, warmup=warmup)
    s["policy"] = pid
    s["theta"] = theta
    s["trace_hash"] = trace_hash(specs)[:16]
    s["scenario"] = scenario.scenario_name
    s["cost_mode"] = scenario.controller.cost_mode
    return s


def c_ref_of(specs, scenario) -> float:
    """C_ref = train_median(K_singleton)（此处以窗口集合的中位数近似训练值，
    正式冻结时以训练区间集合替换）。"""
    import numpy as np
    if not specs:
        return 0.01
    ks = [float(singleton_K(s, scenario.profile, scenario.storage.b_ref_gbps,
                            scenario.storage.q_max_gbps)) for s in specs]
    return float(np.median(ks))


def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1,
                  default=lambda o: str(o))


def out_dir(stage: str, exp: str) -> str:
    d = os.path.join(RESULTS_DIR, stage, exp)
    os.makedirs(d, exist_ok=True)
    return d


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC",
                                       "Songti SC", "Arial Unicode MS",
                                       "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def print_progress(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
