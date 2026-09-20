"""E25 方案对比时序图（图 I 系列，Gantt 式）：行与 §4 各表逐行对应。

设计契约（20260920 v2，修复"子图大面积空白/文字重叠"问题）：
- FW 场景 6 行：FCFS / EDF（排序类代表，spt/edf/slack 同结果）/ MPC θ=M/S/T/R；
- OL 场景 5 行：去 θ=M（OL 未跑 θ=M，M 无定义）——与 §4 表行数一一对应；
- 每行 = 一张含 4 条 NPU 泳道的 Gantt（纵轴 NPU0..3、横轴仿真时间，
  逐桶按 计算/等待/闲置 时间占比切分着色：蓝=计算、橙=等待读、灰=闲置，
  闲置为可见灰色泳道底）+ 右侧窄列 = 存储带宽实际占用（GB/s）；
- 行标签旁标注该策略统计（算% / 等% / 队列峰）；
- 各行横轴独立自适应——行的右端即该策略排空时刻（M 差异直接可见）。

场景覆盖（×3 trace 共 18 张）：OL 基线 / B=20 / α=8 / ρ=0.3 / ρ=1.1 与 FW 基线。
直接读取已归档的 E25.1 时间序列（ts/<cell_id>/<policy>[_θ].json.gz），
不重跑仿真。`--audit` 模式不落盘，只做文字 bbox 重叠检查（QA 用）。
"""
from __future__ import annotations

import gzip
import json
import os
import sys

from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.getcwd())

from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_ROOT = os.path.join(REPO, "results", "cq", "eval", "e25", "ts")
FIG_DIR = os.path.join(REPO, "docs", "figures")
WINDOW = 9

C_COMP, C_STALL, C_IDLE = "#4C72B0", "#DD8452", "#C7C7C7"
DPI = 125
# 画布与正式图一致：FW 1860x2016、OL 1860x1704
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
        dict(tag="OLbase", mode="OL", file=f, B=80.0, rho=0.6, alpha=4,
             title_extra="OL 基线（B=80, ρ=0.6, α=4）"),
        dict(tag="OLB20", mode="OL", file=f, B=20.0, rho=0.6, alpha=4,
             title_extra="OL 带宽受限（B=20, ρ=0.6）"),
        dict(tag="OLa8", mode="OL", file=f, B=80.0, rho=0.6, alpha=8,
             title_extra="OL 松期限（α=8）"),
        dict(tag="OLr03", mode="OL", file=f, B=80.0, rho=0.3, alpha=4,
             title_extra="OL 低负载（B=80, ρ=0.3）"),
        dict(tag="OLr11", mode="OL", file=f, B=80.0, rho=1.1, alpha=4,
             title_extra="OL 过载（B=80, ρ=1.1）"),
        dict(tag="FWbase", mode="FW", file=f, B=80.0, rho=None, alpha=4,
             title_extra="FW 有限工作集（128 条同时到达）"),
    ]


def cell_id(c):
    rho = "-" if c["rho"] is None else f"{c['rho']:g}"
    stem = c["file"].replace("_trace.jsonl", "").replace(".jsonl", "")
    return (f"{stem}|{c['mode']}|B{c['B']:g}|rho{rho}|a{c['alpha']:g}|C|"
            f"default|w{WINDOW}")


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


def draw_gantt(ax, ts):
    """一张含 n 条 NPU 泳道的 Gantt：灰底=闲置，逐桶叠蓝（计算）/橙（等待读）。
    泳道数从 ts 元数据自适应（m=2/4/8 均可）。"""
    n = int(ts["meta"].get("m_workers", 4))
    rows = [r for r in ts["rows"] if r["width_s"] > 0]
    t0 = min(r["t_start_s"] for r in rows)
    t1 = max(r["t_start_s"] + r["width_s"] for r in rows)
    for j in range(n):
        y = (j + 0.055, 0.89)
        ax.broken_barh([(t0, t1 - t0)], y, facecolors=C_IDLE,
                       edgecolor="none", zorder=1)
        blue_xr, orange_xr = [], []
        comp_j = stall_j = 0.0
        for r in rows:
            c, s, _i = r["workers"][j]
            comp_j += c
            stall_j += s
            if c > 0:
                blue_xr.append((r["t_start_s"], c))
            if s > 0:
                orange_xr.append((r["t_start_s"] + c, s))
        if blue_xr:
            ax.broken_barh(blue_xr, y, facecolors=C_COMP,
                           edgecolor="none", zorder=2)
        if orange_xr:
            ax.broken_barh(orange_xr, y, facecolors=C_STALL,
                           edgecolor="none", zorder=3)
        # 泳道右端标注该 NPU 的利用率（计算占比；等待 ≥1% 时附注）
        label = f"{comp_j / (t1 - t0):.0%}"
        if stall_j / (t1 - t0) >= 0.01:
            label += f"/等{stall_j / (t1 - t0):.0%}"
        ax.text(t0 + (t1 - t0) * 0.988, j + 0.5, label, ha="right",
                va="center", fontsize=8.5, color="#1a1a1a", zorder=4)
    ax.set_ylim(-0.06, n + 0.06)
    ax.set_yticks([j + 0.5 for j in range(n)])
    ax.set_yticklabels([f"NPU{j}" for j in range(n)], fontsize=10)
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune="upper"))
    ax.grid(axis="x", alpha=.18, lw=.6)
    ax.set_axisbelow(True)


def draw_bandwidth(ax, ts, first_row, last_row):
    rows = [r for r in ts["rows"] if r["width_s"] > 0]
    xs = [r["t_start_s"] for r in rows]
    rate = [r["served_gb"] / r["width_s"] for r in rows]
    b = float(ts["meta"]["b_schedule"][0][1])
    ax.fill_between(xs, rate, color="#8fb8e0", alpha=.45, zorder=1)
    ax.plot(xs, rate, color=C_COMP, lw=1.0, zorder=2)
    ax.axhline(b, color="k", ls="--", lw=1.0, zorder=3)
    ax.set_ylim(0, b * 1.18)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.text(0.03, 0.965, f"B={b:g} GB/s", transform=ax.transAxes,
            va="top", ha="left", fontsize=9)
    ax.grid(axis="y", alpha=.18, lw=.6)
    if first_row:
        ax.set_title("存储带宽占用 (GB/s)", fontsize=11)
    if last_row:
        ax.set_xlabel("仿真时间 (s)", fontsize=10)
    if first_row:
        ax.set_ylabel("GB/s", fontsize=9.5)


def build_figure(plt, c, legend_patches):
    """构建一张图 I（不落盘）。返回 (fig, [(label, stats), ...])。"""
    rows_def = ROWS_FW if c["mode"] == "FW" else ROWS_OL
    tss = []
    for pid, suffix, label in rows_def:
        ts = load_ts(c, pid, suffix)
        if ts is None:
            break
        tss.append((label, ts))
    if len(tss) < len(rows_def):
        return None, None
    n = len(tss)
    fig = plt.figure(figsize=FIGSIZE[c["mode"]], dpi=DPI)
    gs = fig.add_gridspec(n, 2, width_ratios=[3.1, 1.0], hspace=0.34,
                          wspace=0.12, left=0.085, right=0.985,
                          top=0.885, bottom=0.028)
    stat_rows = []
    for ri, (label, ts) in enumerate(tss):
        first, last = ri == 0, ri == n - 1
        axg = fig.add_subplot(gs[ri, 0])
        draw_gantt(axg, ts)
        if first:
            axg.set_title("4 NPU 泳道（纵轴=NPU，横轴=仿真时间）", fontsize=11)
        if last:
            axg.set_xlabel("仿真时间 (s)", fontsize=10)
        axb = fig.add_subplot(gs[ri, 1])
        draw_bandwidth(axb, ts, first, last)
        comp, stall, qmax = row_stats(ts)
        stat_rows.append((label, comp, stall, qmax))
        axg.set_ylabel(f"{label}\n算{comp:.0%} 等{stall:.0%}\n队列峰{qmax:.0f}",
                       fontsize=10.5)
    fig.suptitle(f"图I｜{TRACE_LABEL[c['file']]} · {c['title_extra']}"
                 f"（窗口{WINDOW}；行与 §4 表逐行对应，行右端=该策略排空时刻）",
                 fontsize=13.5, y=0.985)
    fig.legend(handles=legend_patches, loc="upper center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.958))
    return fig, stat_rows


def audit_text_overlap(fig):
    """QA：收集全部可见文字 artist 的窗口 bbox，报告两两重叠对（空=通过）。

    刻度标签若落在所属 axes 视野之外（locator 生成了但不绘制），不计入。
    """
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    items = []
    for ax in fig.axes:
        ax_bb = ax.get_window_extent(ren)
        for t in ax.texts:
            if t.get_text() and t.get_window_extent(ren).overlaps(ax_bb):
                items.append(t)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text() and t.get_window_extent(ren).overlaps(ax_bb):
                items.append(t)
        if ax.get_title():
            items.append(ax.title)
        for t in (ax.xaxis.label, ax.yaxis.label):
            if t.get_text():
                items.append(t)
    if getattr(fig, "_suptitle", None) is not None:
        items.append(fig._suptitle)
    for leg in fig.legends:
        items.append(leg)
    boxes = [it.get_window_extent(ren) for it in items]
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].overlaps(boxes[j]):
                out.append((str(items[i])[:28], str(items[j])[:28]))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    patches = [Patch(facecolor=C_COMP, label="计算"),
               Patch(facecolor=C_STALL, label="等待读（STALL）"),
               Patch(facecolor=C_IDLE, label="闲置")]
    audit_only = "--audit" in sys.argv
    made = 0
    for c in CELLS:
        fig, stat_rows = build_figure(plt, c, patches)
        if fig is None:
            print(f"skip {c['tag']} {c['file'][:12]}: ts 缺失")
            continue
        if audit_only:
            ov = audit_text_overlap(fig)
            tag = "OK" if not ov else f"重叠 {len(ov)} 对: {ov[:3]}"
            print(f"[audit] {c['tag']}_{c['file'].split('_')[0]}: {tag}")
            plt.close(fig)
            continue
        out = os.path.join(
            FIG_DIR, f"cq_fig_e25_cmp_{c['tag']}_{c['file'].split('_')[0]}.png")
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        made += 1
        print(os.path.basename(out))
        for label, comp, stall, qmax in stat_rows:
            print(f"   {label:12s} 算{comp:.0%} 等{stall:.1%} 队列峰{qmax:.0f}")
    if not audit_only:
        print(f"共生成 {made} 张 -> docs/figures")


if __name__ == "__main__":
    main()
