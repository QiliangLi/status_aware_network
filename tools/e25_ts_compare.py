"""E25 方案对比时序图（图 I 系列，Gantt 式）：行与 §4 各表逐行对应。

设计契约（20260919 定稿，v4.3 入库为正式生成器）：
- FW 场景 6 行：FCFS / EDF（排序类代表，spt/edf/slack 同结果）/ MPC θ=M/S/T/R；
- OL 场景 5 行：去 θ=M（OL 未跑 θ=M，M 无定义）——与 §4 表行数一一对应；
- 每行左侧 4 个子图 = NPU0..3 的四条泳道 Gantt：横轴仿真时间，逐桶按
  计算/等待/闲置的时间占比切分着色（蓝=计算、橙=等待读、米灰底=闲置）；
- 每行右侧窄列 = 存储带宽实际占用（GB/s，黑虚线 = 可用带宽 B）；
- 行标签旁标注该策略统计（算% / 等% / 队列峰），图例含三种状态。

直接读取已归档的 E25.1 时间序列（ts/<cell_id>/<policy>[_θ].json.gz），
不重跑仿真。覆盖：OL 基线/B=20/ρ=0.3/ρ=1.1 与 FW 基线 × 3 trace 共 15 张；
同时打印每行统计供报告图注引用。
"""
from __future__ import annotations

import gzip
import json
import os
import sys

sys.path.insert(0, os.getcwd())

from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_ROOT = os.path.join(REPO, "results", "cq", "eval", "e25", "ts")
FIG_DIR = os.path.join(REPO, "docs", "figures")
WINDOW = 9

C_COMP, C_STALL, C_IDLE = "#4C72B0", "#DD8452", "#E8E5DC"
DPI = 125
# 画布与既有正式图一致：FW 1860x2016、OL 1860x1704
FIGSIZE = {"FW": (14.88, 16.128), "OL": (14.88, 13.632)}

ROWS_FW = [("cq_fcfs", "", "FCFS（基线）"),
           ("cq_edf", "", "EDF（排序类代表）"),
           ("cq_mpc", "_M", "MPC θ=M"),
           ("cq_mpc", "_S", "MPC θ=S"),
           ("cq_mpc", "_T", "MPC θ=T"),
           ("cq_mpc", "_R", "MPC θ=R")]
ROWS_OL = [r for r in ROWS_FW if r[1] != "_M"]

CELLS = []
for f, _n, _m, _s in MOONCAKE_FILES:
    CELLS += [
        dict(tag="OLbase", mode="OL", file=f, B=80.0, rho=0.6, cap="C",
             profile="default", title_extra="OL 基线（B=80, ρ=0.6）"),
        dict(tag="OLB20", mode="OL", file=f, B=20.0, rho=0.6, cap="C",
             profile="default", title_extra="OL 带宽受限（B=20, ρ=0.6）"),
        dict(tag="OLr03", mode="OL", file=f, B=80.0, rho=0.3, cap="C",
             profile="default", title_extra="OL 低负载（B=80, ρ=0.3）"),
        dict(tag="OLr11", mode="OL", file=f, B=80.0, rho=1.1, cap="C",
             profile="default", title_extra="OL 过载（B=80, ρ=1.1）"),
        dict(tag="FWbase", mode="FW", file=f, B=80.0, rho=None, cap="C",
             profile="default", title_extra="FW 有限工作集（128 条同时到达）"),
    ]


def cell_id(c):
    rho = "-" if c["rho"] is None else f"{c['rho']:g}"
    stem = c["file"].replace("_trace.jsonl", "").replace(".jsonl", "")
    return (f"{stem}|{c['mode']}|B{c['B']:g}|rho{rho}|a4|{c['cap']}|"
            f"{c['profile']}|w{WINDOW}")


def load_ts(c, pid, suffix):
    p = os.path.join(TS_ROOT, cell_id(c), f"{pid}{suffix}.json.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def row_stats(ts):
    """行旁统计素材：算/等占比（对三态总量）、队列时间均值峰值。"""
    rows = [r for r in ts["rows"] if r["width_s"] > 0]
    tot = sum(r["compute_s"] + r["stall_s"] + r["idle_s"] for r in rows)
    q = [r["queue_mean"] for r in rows if r["queue_mean"] is not None]
    return (sum(r["compute_s"] for r in rows) / tot,
            sum(r["stall_s"] for r in rows) / tot,
            max(q) if q else 0)


def draw_gantt(ax, ts, wi):
    """单 NPU 泳道 Gantt：全底米灰=闲置，逐桶叠蓝（计算）/橙（等待读）。"""
    rows = [r for r in ts["rows"] if r["width_s"] > 0]
    t0 = min(r["t_start_s"] for r in rows)
    t1 = max(r["t_start_s"] + r["width_s"] for r in rows)
    ax.broken_barh([(t0, t1 - t0)], (wi, 0.86), facecolors=C_IDLE,
                   edgecolor="none", zorder=1)
    blue_xr, orange_xr = [], []
    for r in rows:
        c, s, _i = r["workers"][wi]
        if c > 0:
            blue_xr.append((r["t_start_s"], c))
        if s > 0:
            orange_xr.append((r["t_start_s"] + c, s))
    if blue_xr:
        ax.broken_barh(blue_xr, (wi, 0.86), facecolors=C_COMP,
                       edgecolor="none", zorder=2)
    if orange_xr:
        ax.broken_barh(orange_xr, (wi, 0.86), facecolors=C_STALL,
                       edgecolor="none", zorder=3)
    ax.set_ylim(-0.1, 4.0)
    ax.set_yticks([j + 0.43 for j in range(4)])
    ax.set_yticklabels([f"NPU{j}" for j in range(4)], fontsize=8)
    ax.tick_params(axis="x", labelsize=7.5)


def draw_bandwidth(ax, ts, first_row, last_row):
    rows = [r for r in ts["rows"] if r["width_s"] > 0]
    xs = [r["t_start_s"] for r in rows]
    rate = [r["served_gb"] / r["width_s"] for r in rows]
    b = float(ts["meta"]["b_schedule"][0][1])
    ax.fill_between(xs, rate, color="#8fb8e0", alpha=.45, zorder=1)
    ax.plot(xs, rate, color=C_COMP, lw=1.0, zorder=2)
    ax.axhline(b, color="k", ls="--", lw=1.0, zorder=3)
    ax.text(0.985, b, f"B={b:g} GB/s", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=8.5)
    ax.set_ylim(0, b * 1.15)
    ax.tick_params(labelsize=7.5)
    if first_row:
        ax.set_title("存储带宽占用 (GB/s)", fontsize=10)
    if last_row:
        ax.set_xlabel("仿真时间 (s)", fontsize=9)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    made = 0
    for c in CELLS:
        rows_def = ROWS_FW if c["mode"] == "FW" else ROWS_OL
        tss = []
        for pid, suffix, label in rows_def:
            ts = load_ts(c, pid, suffix)
            if ts is None:
                break
            tss.append((pid, label, ts))
        if len(tss) < len(rows_def):
            print(f"skip {c['tag']} {c['file'][:12]}: ts 缺失")
            continue
        n = len(tss)
        fig, axes = plt.subplots(n, 5, figsize=FIGSIZE[c["mode"]],
                                 squeeze=False, dpi=DPI)
        for ri, (pid, label, ts) in enumerate(tss):
            first, last = ri == 0, ri == n - 1
            for wi in range(4):
                ax = axes[ri][wi]
                draw_gantt(ax, ts, wi)
                if first:
                    ax.set_title(f"NPU{wi}", fontsize=10)
                if last:
                    ax.set_xlabel("仿真时间 (s)", fontsize=9)
            draw_bandwidth(axes[ri][4], ts, first, last)
            comp, stall, qmax = row_stats(ts)
            axes[ri][0].set_ylabel(
                f"{label}\n算{comp:.0%} 等{stall:.0%}\n队列峰{qmax:.0f}",
                fontsize=10)
        handles = [Patch(facecolor=C_COMP, label="计算"),
                   Patch(facecolor=C_STALL, label="等待读（STALL）"),
                   Patch(facecolor=C_IDLE, label="闲置")]
        fig.legend(handles=handles, loc="upper right", ncol=3,
                   frameon=False, fontsize=10.5,
                   bbox_to_anchor=(0.99, 0.995))
        fig.suptitle(f"图I｜方案对比时序（Gantt）：{TRACE_LABEL[c['file']]} · "
                     f"{c['title_extra']}（窗口{WINDOW}；行与 §4 表逐行对应）",
                     fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.4)
        out = os.path.join(FIG_DIR,
                           f"cq_fig_e25_cmp_{c['tag']}_{c['file'].split('_')[0]}.png")
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        made += 1
        print(os.path.basename(out))
        for _pid, label, ts in tss:
            comp, stall, qmax = row_stats(ts)
            print(f"   {label:12s} 算{comp:.0%} 等{stall:.1%} 队列峰{qmax:.0f}")
    print(f"共生成 {made} 张 -> docs/figures")


if __name__ == "__main__":
    main()
