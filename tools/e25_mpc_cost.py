"""E25 MPC 调度计算复杂度：扩展实验 + 实测曲线（图 J）。

1) 扩展实验：FW 有限工作集（Conversation 窗口9 请求形状），n∈{16..512}，
   MPC θ=T 理想上限档 vs FCFS，逐决策计时并记录队列深度与评分候选数；
2) 输出 cq_fig_e25_mpc_cost.png：左=决策耗时 P50/P95 vs 队列规模（含
   FCFS 对照），右=单次决策耗时 vs 该决策评分的候选数（散点+斜率）。
   用于报告 §5.7：复杂度公式 O((1+|A|)·E_drain) 的实证（事件预算 2 万
   封顶 → 耗时随 n 增长后饱和而非爆炸）。
"""
from __future__ import annotations

import os
import sys
import time
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.getcwd())

from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.types import RequestSpec
from sim.experiments.cq_common import (TRACE_WIDE_LIMITS, MooncakeSource,
                                       build_policy, c_ref_of,
                                       default_scenario)
from sim.cq.profile import make_T0

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "docs", "figures", "cq_fig_e25_mpc_cost.png")
NS = [16, 32, 64, 128, 256, 512]


class TimedMPC(MPCPolicy):
    """逐决策计时：记录 (队列深度, 耗时 s, 本次决策新评分的候选数)。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log = []
        self._prev = 0

    def decide(self, snap, scn):
        t0 = time.perf_counter()
        act = super().decide(snap, scn)
        self.log.append((len(snap.queued), time.perf_counter() - t0,
                         self.n_scored - self._prev))
        self._prev = self.n_scored
        return act


class TimedSimple:
    """简单策略计时壳。"""

    def __init__(self, pid, c_ref):
        self.inner = build_policy(pid, "T", c_ref, 1)
        self.pid = pid
        self.log = []

    def decide(self, snap, scn):
        t0 = time.perf_counter()
        act = self.inner.decide(snap, scn)
        self.log.append((len(snap.queued), time.perf_counter() - t0, 0))
        return act


def specs_n(src, n):
    blk = src.windows[9]
    rows = [r for r in src.imp.rows
            if blk.start_ms <= r.timestamp_ms < blk.end_ms][:n]
    specs = []
    for i, r in enumerate(rows):
        h, u, _ = src.imp.h_u_of(r)
        specs.append(RequestSpec(rid=i, arrival_s=0.0, h_tokens=h, u_tokens=u,
                                 class_id="mooncake", T0_s=1, deadline_s=1))
    prof = __import__("sim.experiments.e25_factor", fromlist=["PROFILES"]).PROFILES["default"]
    T0 = make_T0(specs, prof, F(80), F(200))
    return [RequestSpec(s.rid, 0.0, s.h_tokens, s.u_tokens, s.class_id,
                        T0[s.rid], float(T0[s.rid]) * 4.0) for s in specs]


def main():
    src = MooncakeSource("conversation_trace.jsonl")
    scn = default_scenario(B_gbps=80.0, alpha=F(4), limits=TRACE_WIDE_LIMITS)
    data = {"mpc": {}, "fcfs": {}}
    pts = []          # (scored, ms) 逐决策散点
    for n in NS:
        specs = specs_n(src, n)
        c_ref = c_ref_of(specs, scn)
        for kind, pol in (("mpc", TimedMPC(pid="m", H=1, theta="T",
                                           c_ref=c_ref, budget_s=3600.0)),
                          ("fcfs", TimedSimple("cq_fcfs", c_ref))):
            eng = run_case(scn, specs, pol, numeric=float, seed=9)
            ms = [d[1] * 1000 for d in pol.log] or [0.0]
            data[kind][n] = dict(
                p50=float(np.median(ms)), p95=float(np.percentile(ms, 95)),
                dec=len(pol.log),
                scored_per_dec=float(np.mean([d[2] for d in pol.log])) if kind == "mpc" else 0.0,
                total_s=sum(ms) / 1000.0)
            if kind == "mpc":
                pts += [(d[2], d[1] * 1000) for d in pol.log]
            print(f"n={n:4d} {kind:5s} 决策{len(pol.log):3d}次 P50={data[kind][n]['p50']:8.2f}ms "
                  f"P95={data[kind][n]['p95']:8.2f}ms 每决策评分{data[kind][n]['scored_per_dec']:5.1f} "
                  f"决策总耗时{data[kind][n]['total_s']:6.2f}s")
    # ---- 图 J ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.plot(NS, [data["mpc"][n]["p50"] for n in NS], "o-", label="MPC P50")
    ax.plot(NS, [data["mpc"][n]["p95"] for n in NS], "s--", label="MPC P95")
    ax.plot(NS, [data["fcfs"][n]["p50"] for n in NS], "^-", color="#55A868",
            label="FCFS P50（对照）")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("队列规模 n（FW 同时到达请求数）")
    ax.set_ylabel("单次决策耗时 (ms)")
    ax.set_title("决策耗时 vs 队列规模（B=80，事件预算 2 万）")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=9)
    ax2 = axes[1]
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax2.scatter(xs, ys, s=12, alpha=.55, color="#4C72B0")
        if sum(xs) > 0 and len(xs) > 2:
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, icept = np.linalg.lstsq(A, ys, rcond=None)[0]
            xr = np.linspace(0, max(xs), 50)
            ax2.plot(xr, slope * xr + icept, "r--", lw=1.2,
                     label=f"线性拟合：{slope:.1f} ms/候选")
            ax2.legend(fontsize=9)
    ax2.set_xlabel("该次决策评分的候选动作数")
    ax2.set_ylabel("单次决策耗时 (ms)")
    ax2.set_title("耗时分解：~常数(基础排空) + 每候选增量")
    ax2.grid(alpha=.3)
    fig.suptitle("图J｜MPC 调度计算复杂度实测（Conversation 形状，理想上限档）")
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print(f"-> {FIG}")


if __name__ == "__main__":
    main()
