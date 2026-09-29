"""E26 图2：带宽需求 vs 实际时序图（大纲 §6 图2）。

每张图 = 一个代表格点 × 4 策略（FCFS / EDF / local θ=S / mpc θ=S）。
每行两栏：左=全程 10ms 分箱的需求 Σq（橙）与实际 Σrate（蓝）+120 上限
黑虚线；右=4s 放大窗的原始毫秒级 step 线（需求尖刺形态）。
数据源：results/cq/<stage>/e26/<cell>|s<rep>/<pid><suffix>.intervals.npz
（sim/experiments/e26_ab.py 落盘的 interval_log）。--audit 仅查文字重叠。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.environ.get("E26_STAGE", "eval")
ROOT = os.environ.get(
    "E26_ROOT") or os.path.join(REPO, "results", "cq", STAGE, "e26")
# 默认=3840 原规模；E26_ROOT=results/cq/eval/e26n240 指向 240 缩放规模
FIG_DIR = os.path.join(REPO, "docs", "figures")
DPI = 125
B_GBPS = 120.0
BIN_S = 0.01          # 左栏分箱 10ms
ZOOM_S = 4.0          # 右栏放大窗

ROWS = [("cq_fcfs", "", "FCFS（基线）"),
        ("cq_edf", "", "EDF（顺序类代表）"),
        ("cq_local", "_S", "local θ=S"),
        ("cq_mpc", "_S", "mpc θ=S")]


def load_intervals(cell: str, rep: int, pid: str, suffix: str):
    p = os.path.join(ROOT, f"{cell}|s{rep}", f"{pid}{suffix}.intervals.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["intervals"]     # (N,4) t0,t1,Σrate,Σq


def binned(arr: np.ndarray, bin_s: float, t_hi: float):
    """区间账本 → 分箱均值序列 (t_center, rate_mean, q_mean)。"""
    edges = np.arange(0.0, t_hi + bin_s, bin_s)
    idx = ((arr[:, 0]) / bin_s).astype(int)
    keep = idx < len(edges) - 1
    w = (arr[:, 1] - arr[:, 0])[keep]
    if w.sum() <= 0:
        return edges[:-1], np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    r = np.bincount(idx[keep], weights=w * arr[:, 2][keep],
                    minlength=len(edges) - 1)
    q = np.bincount(idx[keep], weights=w * arr[:, 3][keep],
                    minlength=len(edges) - 1)
    cnt = np.bincount(idx[keep], weights=w, minlength=len(edges) - 1)
    safe = np.where(cnt > 0, cnt, 1.0)
    return edges[:-1] + bin_s / 2, r / safe, q / safe


def draw_row(axes, arr, t_hi: float, zoom_t0: float):
    ax, axz = axes
    xs, rate, dem = binned(arr, BIN_S, t_hi)
    ax.plot(xs, dem, color="#DD8452", lw=.9, label="需求 Σ申请率（含 boost）")
    ax.plot(xs, rate, color="#4C72B0", lw=.9, label="实际 Σ速率")
    ax.axhline(B_GBPS, color="k", ls="--", lw=1.1)
    ax.text(t_hi * 0.995, B_GBPS * 1.06, "B=120", ha="right", fontsize=8.5)
    peak = max(dem.max(), rate.max(), B_GBPS)
    ax.set_ylim(0, min(peak * 1.15, 1000))
    ax.set_xlim(0, t_hi)
    ax.grid(alpha=.2)
    seg = arr[(arr[:, 1] > zoom_t0) & (arr[:, 0] < zoom_t0 + ZOOM_S)]
    if len(seg):
        xs2_r, ys2_r, xs2_q, ys2_q = [], [], [], []
        for t0, t1, r, q in seg[:12000]:
            xs2_r += [t0, t1]
            ys2_r += [r, r]
            xs2_q += [t0, t1]
            ys2_q += [q, q]
        axz.plot(xs2_q, ys2_q, color="#DD8452", lw=.8)
        axz.plot(xs2_r, ys2_r, color="#4C72B0", lw=.8)
    axz.axhline(B_GBPS, color="k", ls="--", lw=1.1)
    axz.set_xlim(zoom_t0, zoom_t0 + ZOOM_S)
    peak2 = max(np.max(seg[:, 3]) if len(seg) else 1.0,
                np.max(seg[:, 2]) if len(seg) else 1.0, B_GBPS)
    axz.set_ylim(0, min(peak2 * 1.15, 1000))
    axz.grid(alpha=.2)
    # 过载占比（Σq>120 时长占比）
    w = arr[:, 1] - arr[:, 0]
    over = float((w[arr[:, 3] > B_GBPS].sum() / w.sum())) if w.sum() > 0 else 0.0
    sat = float((w[arr[:, 2] >= 0.9 * B_GBPS].sum() / w.sum())) if w.sum() > 0 else 0.0
    return over, sat


def build_figure(plt, cell: str, rep: int, zoom_frac: float = 0.25,
                 title_extra: str = ""):
    arrs = [(label, load_intervals(cell, rep, pid, suf))
            for pid, suf, label in ROWS]
    missing = [l for l, a in arrs if a is None]
    arrs = [(l, a) for l, a in arrs if a is not None]
    if not arrs:
        print(f"skip {cell} s{rep}: 无 intervals 数据")
        return None, None
    t_hi = max(float(a[-1, 1]) for _l, a in arrs)
    zoom_t0 = t_hi * zoom_frac
    n = len(arrs)
    fig = plt.figure(figsize=(14.6, 2.75 * n + 0.9), dpi=DPI)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n, 2, width_ratios=[2.6, 1.0], hspace=0.34, wspace=0.14,
                  left=0.07, right=0.985, top=0.90, bottom=0.06)
    stats = []
    for ri, (label, a) in enumerate(arrs):
        over, sat = draw_row((fig.add_subplot(gs[ri, 0]),
                              fig.add_subplot(gs[ri, 1])), a, t_hi, zoom_t0)
        stats.append((label, over, sat))
        fig.axes[-2].set_ylabel(f"{label}\n超订{over:.0%} 饱和{sat:.0%}",
                                fontsize=9.5)
        if ri == 0:
            fig.axes[-2].set_title(f"全程（{BIN_S*1000:.0f}ms 分箱）", fontsize=10)
            fig.axes[-1].set_title(f"{ZOOM_S:g}s 放大（原始毫秒级）", fontsize=10)
            fig.axes[-2].legend(loc="upper right", fontsize=8)
        if ri == n - 1:
            fig.axes[-2].set_xlabel("仿真时间 (s)", fontsize=10)
            fig.axes[-1].set_xlabel("仿真时间 (s)", fontsize=10)
    fig.suptitle(f"图2｜E26 带宽需求 vs 实际：{cell}（rep{rep}{title_extra}；"
                 f"橙=全部流申请率之和 蓝=实际分配速率之和，蓝永不超过黑虚线）",
                 fontsize=13, y=0.965)
    return fig, stats


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e25_ts_compare import audit_text_overlap

    audit_only = "--audit" in sys.argv
    cells = []
    for arg in sys.argv[1:]:
        if arg.startswith("--cell="):
            cell, rep = arg[len("--cell="):].split("@")
            cells.append((cell, int(rep)))
    made = 0
    for cell, rep in cells:
        fig, stats = build_figure(plt, cell, rep)
        if fig is None:
            continue
        out = os.path.join(FIG_DIR, f"cq_fig_e26_bw_{cell}_s{rep}.png")
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
        for label, over, sat in stats:
            print(f"   {label:16s} 需求超订时长 {over:.1%} 饱和时长 {sat:.1%}")
    if not audit_only:
        print(f"共生成 {made} 张 -> docs/figures")


if __name__ == "__main__":
    main()
