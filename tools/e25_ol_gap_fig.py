"""E25Q OL 差距遍历配图：13 格点总览 + 领先格点逐窗形态。

图（仓库根文档《实验结果-E25Q-…》配套）：
(a) 全部搜索格点按 SLO 配对中位排序的条形图（颜色=trace，条端标 TTFT 中位）；
(b) ρ0.6/α8 领先格点 20 窗逐窗 SLO 差（全正=稳健差距）；
(c) ρ0.9/α4 的 20 窗逐窗 TTFT 差（中位塌掉、仅 w14 极值级联）。
数据：results/cq/eval/e25_q2/ol_gap_search*.json。
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "docs", "figures")
DPI = 125
TCOL = {"too": "#DD8452", "con": "#4C72B0", "syn": "#55A868"}
TLAB = {"too": "ToolAgent", "con": "Conversation", "syn": "Synthetic"}


def load_db():
    db = {}
    for p in sorted(glob.glob(os.path.join(REPO, "results/cq/eval/e25_q2",
                                           "ol_gap_search*.json"))):
        for k, v in json.load(open(p)).items():
            db[k] = v
    return db


def cell_label(k):
    tr, _, B, rho, m, a, th = k.split("|")
    return (f"{TLAB[tr[:3]][:3]} B{B[1:]} ρ{rho[3:]} m{m[1:]} α{a[1:]}"
            f" θ{th[2:]}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    db = load_db()
    fig = plt.figure(figsize=(15.5, 10.8), dpi=DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)

    # (a) 全格点 SLO 中位条形图
    ax = fig.add_subplot(gs[0, :])
    items = sorted(db.items(), key=lambda kv: kv[1]["median_d_slo_pp"])
    ys = np.arange(len(items))
    colors = [TCOL[k.split("|")[0][:3]] for k, _ in items]
    meds = [v["median_d_slo_pp"] for _, v in items]
    ax.barh(ys, meds, color=colors, alpha=.85)
    ax.set_yticks(ys)
    ax.set_yticklabels([cell_label(k) for k, _ in items], fontsize=9)
    for y, (k, v) in zip(ys, items):
        ax.text(v["median_d_slo_pp"] + (0.12 if v["median_d_slo_pp"] >= 0 else -0.12),
                y, f"TTFT中位{v['median_d_ttft_pct']:+.1f}%",
                va="center", ha="left" if v["median_d_slo_pp"] >= 0 else "right",
                fontsize=8, color="dimgrey")
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("mpc 相对 local 的 SLO 配对中位差（pp，正=mpc 更好）")
    ax.set_title("(a) 全部 13 个搜索格点：SLO 差距排序（条端灰字=TTFT 中位差；"
                 "橙色 ToolAgent 统治榜单，Synthetic 全部归零）", fontsize=10.5)
    ax.set_xlim(min(meds) - 2.6, max(meds) + 2.6)
    ax.grid(alpha=.25, axis="x")

    # (b) ρ0.6α8 的 20 窗 SLO 差
    ax = fig.add_subplot(gs[1, 0])
    v = db["too|OL|B80|rho0.6|m2|a8|thS"]
    ws = sorted(v["per_window"], key=int)
    s = [v["per_window"][w]["d_slo_pp"] for w in ws]
    ax.bar([int(w) for w in ws], s, color="#DD8452", alpha=.85)
    ax.axhline(0, color="k", lw=.8)
    ax.axhline(np.median(s), color="k", ls="--", lw=1)
    ax.text(0.3, np.median(s) + .18, f"中位 {np.median(s):+.1f}pp", fontsize=9)
    ax.set_xlabel("窗口编号（整份文件等切 20 块）")
    ax.set_ylabel("SLO 差（pp）")
    ax.set_title("(b) 最稳健 case（ρ0.6/m2/B80/α8）20 窗验证：\n"
                 "全部 20 窗 SLO 差为正——差距贯穿整份文件", fontsize=10)
    ax.set_ylim(-1, 7.6)
    ax.grid(alpha=.25, axis="y")

    # (c) ρ0.9α4 的 20 窗 TTFT 差
    ax = fig.add_subplot(gs[1, 1])
    v = db["too|OL|B80|rho0.9|m2|a4|thS"]
    ws = sorted(v["per_window"], key=int)
    t = [v["per_window"][w]["d_ttft_pct"] for w in ws]
    ax.bar([int(w) for w in ws], t, color="#C44E52", alpha=.85)
    ax.axhline(0, color="k", lw=.8)
    ax.annotate("w14：−41.8%\n（单窗级联极值）", xy=(14, -41.8), xytext=(9.2, -30),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=.8))
    ax.set_xlabel("窗口编号")
    ax.set_ylabel("TTFT 差（%，正=mpc 更快）")
    ax.set_title("(c) 单窗极值 case（ρ0.9/m2/B80/α4）20 窗验证：\n"
                 "其余 19 窗 |差|≤7.6%、中位≈0——只有 w14 一次级联", fontsize=10)
    ax.set_ylim(-48, 12)
    ax.grid(alpha=.25, axis="y")

    fig.suptitle("OL 模式 mpc vs local 差距遍历：格点总览与两个领先 case 的窗口形态"
                 "（差距=同窗口同策略配对；正 SLO=mpc 按时率更高）", fontsize=12)
    out = os.path.join(FIG, "cq_fig_e25q_ol_gap.png")
    return fig, out


if __name__ == "__main__":
    from e25_ts_compare import audit_text_overlap
    import matplotlib.pyplot as plt
    fig, out = main()
    fig.savefig(out, bbox_inches="tight")
    n = len(audit_text_overlap(fig))
    plt.close(fig)
    print(f"{out}  文字重叠对: {n}")
