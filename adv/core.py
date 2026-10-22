"""对抗迭代评估核心：真实 cq 引擎上的单层评估、击杀判定、轮次日志。

信息边界：评估器是真值侧（引擎跑完后读 w.requests 全量真值），策略本身
只经 ObservableSnapshot（sim/cq 引擎既有边界，本模块不破坏）。

单位：秒 / 十进制 GB / GB/s（与 sim/config.py 一致）。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from sim.cq.config import CqScenario, StorageConfig
from sim.cq.policies import GuardedEDFBW, make_edf, make_fcfs, make_tfs
from sim.cq.simrun import run_case
from sim.cq.types import RequestSpec
from sim.experiments.e26_ab import E26_LIMITS, E26_PROFILE, M_WORKERS

ADV_ROOT = os.path.join("results", "adv")  # results/ gitignored，JSONL 落此处


def build_scenario(b_gbps: float = 120.0, m_workers: int = M_WORKERS,
                   alpha: float = 4.0, alloc: str = "fcfs",
                   b_ref: float = None, guard_w: float = 0.0,
                   b_schedule=None, observation=None) -> CqScenario:
    """对抗场景：默认与 E26f 母实验同物理（B=120、q_max=200、b_ref=120、L=8、n_max=1）。

    b_ref 允许与物理 B 脱钩（G6 参考带宽过期攻击）；guard_w 启用老化锚点。"""
    from fractions import Fraction as F
    kw = {}
    if observation is not None:
        kw["observation"] = observation
    return CqScenario(
        scenario_name=f"adv_B{b_gbps:g}", profile=E26_PROFILE,
        storage=StorageConfig(
            b_schedule=b_schedule or ((F(0), F(int(b_gbps))),),
            q_max_gbps=F(200),
            b_ref_gbps=F(int(b_ref if b_ref else b_gbps)),
            alloc=alloc),
        limits=E26_LIMITS, m_workers=m_workers, alpha_slo=F(int(alpha)),
        guard_w=F(guard_w) if guard_w else F(0),
        source_kind="synthetic", trace_key="adv", **kw)


# ---------------------------------------------------------------------------
# 指标提取（引擎真值侧）
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    n: int
    n_met: int
    slo: float                      # 总 SLO（F_s ≤ deadline 的占比）
    cls_slo: Dict[str, float]       # 分类 SLO
    ttft_mean: float
    ttft_p95: float
    makespan: float
    busy_frac: float                # worker 忙碌占比（派发到批完成）
    per_cls_n: Dict[str, int] = field(default_factory=dict)
    per_cls_met: Dict[str, int] = field(default_factory=dict)
    wall_s: float = 0.0


def metrics_of(engine, specs: Sequence[RequestSpec]) -> RunMetrics:
    w = engine.w
    n = len(specs)
    met = 0
    cls_n: Dict[str, int] = {}
    cls_met: Dict[str, int] = {}
    ttfts: List[float] = []
    fmax = 0.0
    for s in specs:
        rr = w.requests[s.rid]
        f = float(rr.F_s) if rr.F_s is not None else float("inf")
        ok = f <= float(s.deadline_s)
        met += ok
        cls_n[s.class_id] = cls_n.get(s.class_id, 0) + 1
        cls_met[s.class_id] = cls_met.get(s.class_id, 0) + ok
        if rr.F_s is not None:
            ttfts.append(f - float(s.arrival_s))
            fmax = max(fmax, f)
    # worker 忙碌占比：批派发到末成员完成
    busy = 0.0
    for b in w.batches.values():
        ends = [float(w.requests[rid].F_s) for rid in b.members
                if w.requests[rid].F_s is not None]
        if ends:
            busy += max(ends) - float(b.dispatch_s)
    cap = w.m * max(fmax, 1e-9)
    ttfts.sort()
    p95 = ttfts[int(0.95 * (len(ttfts) - 1))] if ttfts else float("nan")
    return RunMetrics(
        n=n, n_met=met, slo=met / max(n, 1),
        cls_slo={c: cls_met[c] / cls_n[c] for c in cls_n},
        ttft_mean=sum(ttfts) / len(ttfts) if ttfts else float("nan"),
        ttft_p95=p95, makespan=fmax, busy_frac=busy / cap,
        per_cls_n=cls_n, per_cls_met=cls_met)


def evaluate(specs, scenario, policies: Dict[str, object]) -> Dict[str, RunMetrics]:
    """CRN：同一 specs 逐策略跑。返回 {策略名: 指标}。"""
    out: Dict[str, RunMetrics] = {}
    for name, pol in policies.items():
        t0 = time.time()
        eng = run_case(scenario, specs, pol)
        m = metrics_of(eng, specs)
        m.wall_s = time.time() - t0
        out[name] = m
    return out


# ---------------------------------------------------------------------------
# 击杀判定
# ---------------------------------------------------------------------------

KILL_TOTAL_PP = 5.0        # 总 SLO 劣于最优基线 ≥5pp
KILL_CLASS_PP = 25.0       # 某类 SLO 劣于该类最优基线 ≥25pp 且最优基线 ≥60%
KILL_CLASS_FLOOR = 0.60
IDLE_STARVE = 0.15         # 饿死判定：类 SLO=0 且系统空闲 ≥15%
FILL_ZERO_READ = 0.95      # 填空律：零读类 SLO 低于该值且系统有 ≥10% 空闲


@dataclass
class Verdict:
    kill: bool
    kind: str = ""                  # total | class | starve | fill | none
    detail: str = ""
    margin_pp: float = 0.0


def zero_read_classes(specs) -> set:
    return {s.class_id for s in specs if s.h_tokens == 0}


def judge_tfs(res: Dict[str, RunMetrics], specs,
              tfs_key: str = "tfs", baseline_keys=("fcfs", "edf", "base")) -> Verdict:
    """单层判定：TFS 对阵基线 + 结构不变量。"""
    tfs = res[tfs_key]
    bl = {k: res[k] for k in baseline_keys if k in res}
    best_total = max(b.slo for b in bl.values())
    if tfs.slo < best_total - KILL_TOTAL_PP / 100.0 and best_total >= 0.30:
        winner = max(bl, key=lambda k: bl[k].slo)
        return Verdict(True, "total",
                       f"总SLO {tfs.slo:.3f} vs {winner} {bl[winner].slo:.3f}",
                       (best_total - tfs.slo) * 100)
    for c in tfs.cls_slo:
        tfs_c = tfs.cls_slo.get(c, 1.0)
        bl_c = [b.cls_slo.get(c, 0.0) for b in bl.values()]
        best_c = max(bl_c)
        if (tfs_c <= best_c - KILL_CLASS_PP / 100.0 and best_c >= KILL_CLASS_FLOOR):
            winner = max(bl, key=lambda k: bl[k].cls_slo.get(c, 0.0))
            return Verdict(True, "class",
                           f"类{c} SLO {tfs_c:.3f} vs {winner} {bl_c[0]:.3f}~{best_c:.3f}",
                           (best_c - tfs_c) * 100)
        if tfs_c <= 0.02 and best_c >= 0.30 and tfs.busy_frac < 1 - IDLE_STARVE:
            return Verdict(True, "starve",
                           f"类{c} 饿死(0%)且空闲 {1-tfs.busy_frac:.0%}", (best_c - tfs_c) * 100)
    # 20261023 Wave3 修正：填空律击杀需可行性护栏——至少一个基线在该类做到
    # ≥95%，否则是过载无解（W6 在 λ=450/s 时全策略 14% 被误判击杀）。
    for c in zero_read_classes(specs):
        z = tfs.cls_slo.get(c, 1.0)
        best_z = max((b.cls_slo.get(c, 0.0) for b in bl.values()), default=0.0)
        if z < FILL_ZERO_READ and tfs.busy_frac < 0.90 and best_z >= 0.95:
            return Verdict(True, "fill",
                           f"零读类{c} SLO {z:.1%}（填空律破，基线可达 {best_z:.0%}）",
                           (1 - z) * 100)
    return Verdict(False, "none", "", 0.0)


# ---------------------------------------------------------------------------
# 轮次日志（JSONL）
# ---------------------------------------------------------------------------

def log_round(path: str, rec: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def base_policies(tfs_factory=None):
    """基线三件 + 当前 TFS 版本。tfs_factory 每轮新造（策略有自持历史）。"""
    return {
        "fcfs": make_fcfs(),
        "edf": make_edf(),
        "base": GuardedEDFBW(),
        "tfs": tfs_factory() if tfs_factory else make_tfs(),
    }
