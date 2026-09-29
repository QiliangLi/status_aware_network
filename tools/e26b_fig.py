"""E26b 图：MPC 修正前后对照甘特（fcfs / edf / mpc 旧口径 / mpc_v2 四行）。

数据源：results/cq/eval/e26b/<cell>/<label>.gantt.json.gz。
用法：.venv/bin/python tools/e26b_fig.py [--cell=D2_b1.0_a4] [--rows=a,b,c,d] [--audit]
"""
from __future__ import annotations

import gzip
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("E26B_ROOT") or os.path.join(REPO, "results", "cq", "eval", "e26b")
FIG_DIR = os.path.join(REPO, "docs", "figures")

DEFAULT_ROWS = [("FCFS（基线）", "fcfs"),
                ("EDF（顺序类）", "edf"),
                ("mpc θ=S 旧口径", "mpcS_old"),
                ("mpc θ=S v2（bw_edf+组合器）", "mpcS_v2")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e26_gantt import draw_row, C_A, C_B, C_WAIT, C_IDLE
    from e25_ts_compare import audit_text_overlap
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec

    cell, rows = "D2_b1.0_a4", DEFAULT_ROWS
    for arg in sys.argv[1:]:
        if arg.startswith("--cell="):
            cell = arg[len("--cell="):]
        if arg.startswith("--rows="):
            rows = [(x, x) for x in arg[len("--rows="):].split(",")]
    audit_only = "--audit" in sys.argv

    gs_ = []
    for label, fn in rows:
        p = os.path.join(ROOT, cell, f"{fn}.gantt.json.gz")
        if not os.path.exists(p):
            print(f"缺 {p}"); return
        gs_.append((label, json.load(gzip.open(p))))
    t_max = 0
    for _l, g in gs_:
        K = g["meta"]["K"]
        last = max((i for i, bk in enumerate(g["buckets"])
                    if any(v > 0 for row in bk for v in row)), default=0)
        t_max = max(t_max, (last + 1) * g["meta"]["T_anchor_s"] / K)
    n = len(gs_)
    fig = plt.figure(figsize=(14.6, 3.15 * n + 0.9), dpi=125)
    gspec = GridSpec(n, 1, hspace=0.30, left=0.13, right=0.985,
                     top=0.87, bottom=0.05)
    for ri, (label, g) in enumerate(gs_):
        ax = fig.add_subplot(gspec[ri, 0])
        fr = draw_row(ax, g, t_max)
        ax.set_ylabel(f"{label}\nA算{fr[0]:.0%} B算{fr[1]:.0%}\n"
                      f"IO等{fr[2]:.0%} 闲{fr[3]:.0%}", fontsize=10)
        if ri == n - 1:
            ax.set_xlabel("仿真时间 (s)", fontsize=10)
    fig.suptitle(f"图E26b｜MPC 修正前后对照：{cell}（240 规模）——"
                 "v2 行应呈『少量蓝条 A + 大片橙 B、红海消失』的错峰形态", fontsize=12.5, y=0.97)
    fig.legend(handles=[Patch(facecolor=C_A, label="A 计算"),
                        Patch(facecolor=C_B, label="B 计算"),
                        Patch(facecolor=C_WAIT, label="IO 等待"),
                        Patch(facecolor=C_IDLE, label="无请求")],
               loc="upper center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.925))
    if audit_only:
        ov = audit_text_overlap(fig)
        print("文字重叠对:", len(ov), ov[:3])
        return
    out = os.path.join(FIG_DIR, f"cq_fig_e26b_gantt_{cell}.png")
    ov = audit_text_overlap(fig)
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print(f"saved {out} 文字重叠对={len(ov)}")
    for label, g in gs_:
        tot = [0.0] * 4
        for bk in g["buckets"]:
            for row in bk:
                for i in range(4):
                    tot[i] += row[i]
        s = sum(tot)
        print(f"  {label}: A算{tot[0]/s:.1%} B算{tot[1]/s:.1%} "
              f"IO等{tot[2]/s:.1%} 闲{tot[3]/s:.1%}")


if __name__ == "__main__":
    main()
