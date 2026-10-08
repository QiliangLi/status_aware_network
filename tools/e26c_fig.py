"""E26c 图工具：每格点两张图（甘特四行 + 带宽时序四行），行=fcfs/edf/local/mpc。

用法：
  .venv/bin/python tools/e26c_fig.py --cell=D2_b1.0_a4 [--audit]
甘特：蓝=A算 橙=B算 红=IO等 灰=无请求（NPU0..31 泳道）。
带宽：10ms 分箱需求 Σq（橙）vs 实际 Σrate（蓝）+120 黑虚线（利用率时序）。
数据源：results/cq/eval/e26c/<cell>/<arm>.{gantt.json.gz,intervals.npz}
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("E26C_ROOT") or os.path.join(REPO, "results", "cq", "eval", "e26c")
FIG_DIR = os.path.join(REPO, "docs", "figures")
B_GBPS = 120.0
BIN_S = 0.01

ROWS = [("FCFS", "fcfs"), ("EDF", "edf"),
        ("local v2.1", "localS_v2.1"), ("mpc v2.1", "mpcS_v2.1")]


def binned(arr, bin_s):
    n = int(arr[-1, 1] / bin_s) + 1
    idx = (arr[:, 0] / bin_s).astype(int)
    idx = np.clip(idx, 0, n - 1)
    w = arr[:, 1] - arr[:, 0]
    r = np.bincount(idx, weights=w * arr[:, 2], minlength=n)
    q = np.bincount(idx, weights=w * arr[:, 3], minlength=n)
    c = np.bincount(idx, weights=w, minlength=n)
    safe = np.where(c > 0, c, 1.0)
    return np.arange(n) * bin_s + bin_s / 2, r / safe, q / safe


def build(cell: str, audit_only: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e26_gantt import draw_row, C_A, C_B, C_WAIT, C_IDLE
    from e25_ts_compare import audit_text_overlap
    from matplotlib.patches import Patch

    gants = []
    for label, fn in ROWS:
        p = os.path.join(ROOT, cell, f"{fn}.gantt.json.gz")
        if not os.path.exists(p):
            print(f"缺 {p}")
            return
        gants.append((label, json.load(gzip.open(p))))

    # ---- 图 A：甘特 ----
    t_max = 0
    for _l, g in gants:
        K = g["meta"]["K"]
        last = max((i for i, bk in enumerate(g["buckets"])
                    if any(v > 0 for row in bk for v in row)), default=0)
        t_max = max(t_max, (last + 1) * g["meta"]["T_anchor_s"] / K)
    fig = plt.figure(figsize=(14.6, 3.0 * 4 + 0.9), dpi=125)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 1, hspace=0.32, left=0.12, right=0.985, top=0.87,
                  bottom=0.05)
    stats = []
    for ri, (label, g) in enumerate(gants):
        ax = fig.add_subplot(gs[ri, 0])
        fr = draw_row(ax, g, t_max)
        stats.append((label, fr))
        ax.set_ylabel(f"{label}\nA算{fr[0]:.0%} B算{fr[1]:.0%}\n"
                      f"IO等{fr[2]:.0%} 闲{fr[3]:.0%}", fontsize=10)
        if ri == 3:
            ax.set_xlabel("仿真时间 (s)", fontsize=10)
    fig.suptitle(f"图A｜E26c 分类甘特：{cell}（蓝=A算 橙=B算 红=IO等 灰=无请求）",
                 fontsize=12.5, y=0.97)
    fig.legend(handles=[Patch(facecolor=C_A, label="A 计算"),
                        Patch(facecolor=C_B, label="B 计算"),
                        Patch(facecolor=C_WAIT, label="IO 等待"),
                        Patch(facecolor=C_IDLE, label="无请求")],
               loc="upper center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.925))
    ov1 = audit_text_overlap(fig)
    ftag = os.environ.get("E26C_FIGTAG", "")
    if not audit_only:
        fig.savefig(os.path.join(
            FIG_DIR, f"cq_fig_e26c_gantt_{cell}{ftag}.png"), dpi=125)
    plt.close(fig)

    # ---- 图 B：带宽利用率时序 ----
    fig = plt.figure(figsize=(14.6, 2.5 * 4 + 0.9), dpi=125)
    gs = GridSpec(4, 1, hspace=0.36, left=0.08, right=0.985, top=0.87,
                  bottom=0.06)
    for ri, (label, fn) in enumerate(ROWS):
        p = os.path.join(ROOT, cell, f"{fn}.intervals.npz")
        ax = fig.add_subplot(gs[ri, 0])
        if not os.path.exists(p):
            ax.text(.5, .5, "无数据", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        arr = np.load(p)["intervals"]
        xs, rate, dem = binned(arr, BIN_S)
        ax.plot(xs, dem, color="#DD8452", lw=.9, label="需求 Σ申请率")
        ax.plot(xs, rate, color="#4C72B0", lw=.9, label="实际 Σ速率")
        ax.axhline(B_GBPS, color="k", ls="--", lw=1.1)
        t_hi = xs[-1]
        ax.text(t_hi * 0.995, B_GBPS * 1.06, "B=120", ha="right", fontsize=8.5)
        peak = max(dem.max(), rate.max(), B_GBPS)
        ax.set_ylim(0, min(peak * 1.15, 1000))
        ax.set_xlim(0, t_hi)
        ax.set_ylabel(f"{label}\nGB/s", fontsize=9.5)
        ax.grid(alpha=.2)
        if ri == 0:
            ax.legend(loc="upper right", fontsize=8)
        if ri == 3:
            ax.set_xlabel("仿真时间 (s)", fontsize=10)
        w = arr[:, 1] - arr[:, 0]
        over = float(w[arr[:, 3] > B_GBPS].sum() / w.sum())
        sat = float(w[arr[:, 2] >= 0.9 * B_GBPS].sum() / w.sum())
        ax.set_title(f"需求超订 {over:.0%}｜实际饱和(≥0.9B) {sat:.0%}",
                     fontsize=8.5, loc="right")
    fig.suptitle(f"图B｜E26c 存储带宽利用率时序：{cell}"
                 "（橙=全部流申请率之和 蓝=实际分配速率，蓝≤黑虚线）",
                 fontsize=12.5, y=0.97)
    ov2 = audit_text_overlap(fig)
    if not audit_only:
        fig.savefig(os.path.join(
            FIG_DIR, f"cq_fig_e26c_bw_{cell}{ftag}.png"), dpi=125)
    plt.close(fig)
    print(f"{cell}: 甘特重叠 {len(ov1)} 对，带宽重叠 {len(ov2)} 对")
    for label, fr in stats:
        print(f"  {label:12s} A算{fr[0]:.1%} B算{fr[1]:.1%} "
              f"IO等{fr[2]:.1%} 闲{fr[3]:.1%}")


if __name__ == "__main__":
    cell = "D2_b1.0_a4"
    audit = "--audit" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--cell="):
            cell = a[len("--cell="):]
    build(cell, audit)
