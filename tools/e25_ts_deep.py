"""E25 时间序列深挖：带宽利用率折线（图 H）与深挖读数。

重跑代表性 OL 格点（窗口 9、基线 ρ=0.6/α=4，B∈{80,20}，fcfs/spt），
用扩展聚合器（rows.workers 每 NPU 分桶三态）生成：
- cq_fig_e25_ts_bandwidth.png：行=B{80,20}、列=三文件，fcfs/spt 利用率折线 + B(t) 阶梯
并打印深挖读数（峰值利用率时刻、该时刻各 NPU 状态、队列水平）。
注：图 G（cq_fig_e25_ts_per_npu_B*，每 NPU 计算/等待两线制）由
tools/e25_redraw.py --all 生成；本工具不再写图 G，避免旧三色堆叠版覆盖。
轨迹与正式矩阵同种子同配置（确定性一致），仅时间序列扩展字段为新。
"""
from __future__ import annotations

import gzip
import json
import os
from fractions import Fraction as F

import numpy as np

from sim.cq.timeseries import aggregate_run, anchor_T, save_ts
from sim.experiments.cq_common import (TRACE_WIDE_LIMITS, MooncakeSource,
                                       build_policy, c_ref_of,
                                       default_scenario, print_progress,
                                       setup_matplotlib)
from sim.cq.simrun import run_case
from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "results", "cq", "eval", "e25", "ts_deep")
WINDOW = 9
POLICIES = ("cq_fcfs", "cq_spt")


def main():
    plt = setup_matplotlib()
    os.makedirs(D, exist_ok=True)
    cache = {}          # (file,B,policy) -> ts dict
    for fname, _n, _m, _s in MOONCAKE_FILES:
        src = MooncakeSource(fname)
        specs, d_sim = src.window_specs(WINDOW, src.lam0 * F(str(0.6)), F(4),
                                        duration_cap=None,
                                        limits=TRACE_WIDE_LIMITS)
        assert len(specs) >= 16, fname
        for B in (80.0, 20.0):
            scn = default_scenario(B_gbps=B, alpha=F(4), limits=TRACE_WIDE_LIMITS)
            c_ref = c_ref_of(specs, scn)
            for pid in POLICIES:
                pol = build_policy(pid, "S", c_ref, 1)
                eng = run_case(scn, specs, pol, numeric=float, seed=WINDOW,
                               record_intervals=True)
                ts = aggregate_run(eng, anchor_T(eng.w.t), cell=f"deep|{fname}|B{B:g}",
                                   policy=pid, mode="OL")
                save_ts(os.path.join(D, f"{fname}_B{B:g}_{pid}.json.gz"), ts)
                cache[(fname, B, pid)] = ts
                print_progress(f"deep {fname[:12]} B={B:g} {pid}: "
                               f"T_end={ts['meta']['T_end_s']:.1f}s")
    files = [f for f, _n, _m, _s in MOONCAKE_FILES]

    # ---- 图 G 已移至 tools/e25_redraw.py（两线制），此处不再生成 ----

    # ---- 图 H：带宽利用率折线（fcfs vs spt + B(t) 阶梯） ----
    fig, axes = plt.subplots(2, len(files), figsize=(5.4 * len(files), 6.4),
                             squeeze=False)
    for ri, B in enumerate((80.0, 20.0)):
        for ci, f in enumerate(files):
            ax = axes[ri][ci]
            for pid, color, ls in (("cq_fcfs", "#4C72B0", "-"),
                                   ("cq_spt", "#DD8452", "--")):
                ts = cache[(f, B, pid)]
                xs, util, rate = [], [], []
                for r in ts["rows"]:
                    if r["width_s"] <= 0 or r["capacity_gb"] <= 0:
                        continue
                    xs.append(r["t_start_s"])
                    util.append(r["served_gb"] / r["capacity_gb"])
                    rate.append(r["served_gb"] / r["width_s"])
                ax.plot(xs, util, color=color, ls=ls, lw=1.3,
                        label=f"{pid.replace('cq_', '')} 利用率")
                pk = int(np.argmax(util)) if util else 0
                if util:
                    ax.annotate(f"峰值{util[pk]:.0%}@{xs[pk]:.0f}s",
                                (xs[pk], util[pk]), fontsize=7.5,
                                xytext=(4, -10), textcoords="offset points",
                                color=color)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(f"B={B:g} GB/s\n利用率", fontsize=9)
            ax2 = ax.twinx()
            sched = cache[(f, B, "cq_fcfs")]["meta"]["b_schedule"]
            st = [t for t, _v in sched]
            sv = [v for _t, v in sched]
            x_end = max(xs, default=0)
            xs_step = [t for t in st if t <= x_end] + [x_end]
            ys_step = [sv[min(i, len(sv) - 1)] for i in range(len(xs_step))]
            ax2.step(xs_step, ys_step, where="post", color="k", ls=":", lw=.9)
            ax2.set_ylabel("B(t) GB/s", fontsize=9)
            if ri == 0:
                ax.set_title(TRACE_LABEL[f], fontsize=11)
            if ri == 1:
                ax.set_xlabel("仿真时间 (s)")
            if ci == 0:
                ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("图H｜存储带宽利用率随 trace 重放的变化（OL 窗口9、ρ=0.6、α=4；"
                 "实线=FCFS、虚线=SPT、点线=可用带宽 B(t)）")
    fig.tight_layout()
    fig.savefig(os.path.join(REPO, "docs", "figures",
                             "cq_fig_e25_ts_bandwidth.png"), dpi=130)
    plt.close(fig)

    # ---- 深挖读数：峰值桶时刻的 NPU 状态与队列 ----
    print("\n=== 深挖读数（每个 (文件,B) 的利用率峰值桶） ===")
    for B in (80.0, 20.0):
        for f in files:
            ts = cache[(f, B, "cq_fcfs")]
            rows = [r for r in ts["rows"] if r["width_s"] > 0 and r["capacity_gb"] > 0]
            pk = max(rows, key=lambda r: r["served_gb"] / r["capacity_gb"])
            u = pk["served_gb"] / pk["capacity_gb"]
            per = [f"{w[0]/pk['width_s']:.0%}/{w[1]/pk['width_s']:.0%}"
                   for w in pk["workers"]]
            print(f"B{B:g} {TRACE_LABEL[f]:18s} 峰值桶 t≈{pk['t_start_s']:.0f}s "
                  f"util={u:.0%} 队列={pk['queue_mean']:.0f} "
                  f"各NPU算/等={','.join(per)}")
    print_progress("deep figures -> docs/figures (bandwidth×1；图G 由 e25_redraw.py 生成)")


if __name__ == "__main__":
    main()
