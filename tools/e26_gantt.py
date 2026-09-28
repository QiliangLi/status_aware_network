"""E26 图1：分类甘特图（大纲 §6 图1）。

每张图 = 一个代表格点 × 4 策略（FCFS / EDF / local θ=S / mpc θ=S），
横轴=仿真时间，纵轴=NPU0..31，四色：A 计算（蓝）、B 计算（橙）、
IO 等待（红）、无请求（浅灰）。桶内四段时长无法恢复先后次序，按
[等待, A算, B算, 闲置] 固定顺序从桶起点向右排（读先于算是主序）。

数据源：results/cq/<stage>/e26/<cell>|s<rep>/<pid><suffix>.gantt.json.gz
（sim/experiments/e26_ab.py 的 gantt_collect 落盘）。--audit 仅查文字重叠。
"""
from __future__ import annotations

import gzip
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.environ.get("E26_STAGE", "eval")
ROOT = os.path.join(REPO, "results", "cq", STAGE, "e26")
FIG_DIR = os.path.join(REPO, "docs", "figures")
DPI = 125

C_A, C_B, C_WAIT, C_IDLE = "#4C72B0", "#DD8452", "#C44E52", "#C7C7C7"
ROWS = [("cq_fcfs", "", "FCFS（基线）"),
        ("cq_edf", "", "EDF（顺序类代表）"),
        ("cq_local", "_S", "local θ=S"),
        ("cq_mpc", "_S", "mpc θ=S")]


def load_gantt(cell: str, rep: int, pid: str, suffix: str):
    p = os.path.join(ROOT, f"{cell}|s{rep}", f"{pid}{suffix}.gantt.json.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def draw_row(ax, g, t_max: float):
    meta, buckets = g["meta"], g["buckets"]
    K, m = meta["K"], meta["m_workers"]
    anchor = meta["T_anchor_s"]
    delta = anchor / K
    # 每泳道：灰底全长 + 逐桶三色段（桶内顺序 [等待, A算, B算]）
    stats = [0.0, 0.0, 0.0, 0.0]      # A算, B算, 等, 闲
    for j in range(m):
        idle_bg, wait_x, a_x, b_x = [], [], [], []
        for i, bk in enumerate(buckets):
            row = bk[j]
            a, b, w, idle = row
            if a <= 0 and b <= 0 and w <= 0 and idle <= 0:
                continue
            t0 = i * delta
            cur = t0
            if w > 0:
                wait_x.append((cur, w))
                cur += w
            if a > 0:
                a_x.append((cur, a))
                cur += a
            stats[0] += a
            if b > 0:
                b_x.append((cur, b))
                cur += b
            stats[1] += b
            stats[2] += w
            stats[3] += idle
        ax.broken_barh([(0, t_max)], (j + 0.06, 0.88), facecolors=C_IDLE,
                       edgecolor="none", zorder=1)
        for xr, col, z in ((a_x, C_A, 3), (b_x, C_B, 3), (wait_x, C_WAIT, 2)):
            if xr:
                ax.broken_barh(xr, (j + 0.06, 0.88), facecolors=col,
                               edgecolor="none", zorder=z)
    ax.set_ylim(-0.05, m + 0.05)
    ax.set_yticks([0, m - 1])
    ax.set_yticklabels(["NPU0", f"NPU{m - 1}"], fontsize=8)
    ax.set_xlim(0, t_max)
    ax.grid(axis="x", alpha=.18, lw=.6)
    ax.set_axisbelow(True)
    tot = sum(stats) or 1.0
    return [x / tot for x in stats]


def build_figure(plt, cell: str, rep: int, title_extra: str = ""):
    gs_ = [(label, load_gantt(cell, rep, pid, suf)) for pid, suf, label in ROWS]
    if any(g is None for _l, g in gs_):
        missing = [l for l, g in gs_ if g is None]
        print(f"skip {cell} s{rep}: 缺 {missing}")
        return None, None
    t_max = max(max(
        (len(g["buckets"]) - 1) * g["meta"]["T_anchor_s"] / g["meta"]["K"]
        for _l, g in gs_), 1.0)
    # 横轴统一到各策略实际 T_end 的最大值（桶数据只到 T_anchor 与溢出桶）
    for _l, g in gs_:
        K = g["meta"]["K"]
        last = max((i for i, bk in enumerate(g["buckets"])
                    if any(v > 0 for row in bk for v in row)), default=0)
        t_max = max(t_max, (last + 1) * g["meta"]["T_anchor_s"] / K)
    n = len(gs_)
    fig = plt.figure(figsize=(14.6, 3.15 * n + 0.9), dpi=DPI)
    from matplotlib.gridspec import GridSpec
    gspec = GridSpec(n, 1, hspace=0.30, left=0.085, right=0.985,
                     top=0.90, bottom=0.045)
    stat_rows = []
    for ri, (label, g) in enumerate(gs_):
        ax = fig.add_subplot(gspec[ri, 0])
        fr = draw_row(ax, g, t_max)
        stat_rows.append((label, fr))
        ax.set_ylabel(
            f"{label}\nA算{fr[0]:.0%} B算{fr[1]:.0%}\nIO等{fr[2]:.0%} 闲{fr[3]:.0%}",
            fontsize=10)
        if ri == n - 1:
            ax.set_xlabel("仿真时间 (s)", fontsize=10)
    fig.suptitle(f"图1｜E26 分类甘特：{cell}（rep{rep}{title_extra}；"
                 f"蓝=A计算 橙=B计算 红=IO等待 灰=无请求；横轴统一，"
                 f"行右端即该策略活动终点）", fontsize=13, y=0.965)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=C_A, label="A 计算"),
                        Patch(facecolor=C_B, label="B 计算"),
                        Patch(facecolor=C_WAIT, label="IO 等待"),
                        Patch(facecolor=C_IDLE, label="无请求")],
               loc="upper center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.938))
    return fig, stat_rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e25_ts_compare import audit_text_overlap

    audit_only = "--audit" in sys.argv
    # 代表格点（默认 rep0）：由调用方通过参数/环境扩展
    cells = []
    for arg in sys.argv[1:]:
        if arg.startswith("--cell="):
            cell, rep = arg[len("--cell="):].split("@")
            cells.append((cell, int(rep)))
    made = 0
    for cell, rep in cells:
        fig, stat_rows = build_figure(plt, cell, rep)
        if fig is None:
            continue
        out = os.path.join(FIG_DIR, f"cq_fig_e26_gantt_{cell}_s{rep}.png")
        if audit_only:
            ov = audit_text_overlap(fig)
            print(f"[audit] {cell}_s{rep}: " +
                  ("OK" if not ov else f"重叠 {len(ov)} 对: {ov[:3]}"))
            plt.close(fig)
            continue
        fig.savefig(out, dpi=DPI)
        plt.close(fig)
        made += 1
        print(os.path.basename(out))
        for label, fr in stat_rows:
            print(f"   {label:16s} A算{fr[0]:.0%} B算{fr[1]:.0%} "
                  f"IO等{fr[2]:.0%} 闲{fr[3]:.0%}")
    if not audit_only:
        print(f"共生成 {made} 张 -> docs/figures")


if __name__ == "__main__":
    main()
