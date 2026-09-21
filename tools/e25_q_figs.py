"""三问深答（20260921）配图：两图。

图1 cq_fig_e25q_bw_3views.png —— Q2 主图（ToolAgent OL ρ0.6 w9 FCFS）：
  (a) 桶均实际利用率（=图 E/H 口径）：B=80 vs B=20 —— 用户看到的曲线；
  (b) 瞬时口径（interval 级，B=80 run）：需求 Σq vs 实际 Σrate（对数轴），
      截取 40s 放大段 —— 需求尖刺远超 80、被削顶到 80、只持续毫秒级；
  (c) 过载段（Σq>B 连续段）时长 CCDF：B=80 vs B80 —— 毫秒级 vs 秒级。

图2 cq_fig_e25q_satB.png —— Q3 主图：从 B=80 运行的桶均需求分布反推
  sat(B)=过载占比随 B 的曲线（3 trace × ρ{0.6,0.9}），灰色带=建议间歇窗口
  （10%–40%），标注既有格点实测 sat。
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "docs", "figures")
TS_DIRS = [os.path.join(REPO, "results/cq/eval/e25/ts"),
           os.path.join(REPO, "results/cq/eval/e25_regime/ts")]
DPI = 125
TCOL = {"conversation": "#4C72B0", "toolagent": "#DD8452",
        "synthetic": "#55A868"}
TLAB = {"conversation": "Conversation", "toolagent": "ToolAgent",
        "synthetic": "Synthetic"}


def load_rows(trace, B, rho):
    for td in TS_DIRS:
        for pat in (f"{trace}|OL|B{B:g}|rho{rho:g}|a4|C|default|w9",
                    f"{trace}|OL|B{B:g}|rho{rho:g}|a4|m4|w9"):
            p = os.path.join(td, pat, "cq_fcfs.json.gz")
            if os.path.exists(p):
                return json.load(gzip.open(p))["rows"]
    return None


def run_instant(trace_key, fname, B, rho):
    """重跑 FCFS 拿 interval_log（毫秒级速率区段）。"""
    from sim.cq.simrun import run_case
    from sim.experiments.cq_common import (TRACE_DIR_DEFAULT,
                                           TRACE_WIDE_LIMITS, MooncakeSource,
                                           build_policy, default_scenario)
    src = MooncakeSource(fname, TRACE_DIR_DEFAULT)
    scn = default_scenario(B_gbps=B, m=4, alpha=F(4), limits=TRACE_WIDE_LIMITS)
    lam = src.lam0 * F(str(rho))
    specs, d_sim = src.window_specs(9, lam, F(4), duration_cap=None,
                                    limits=TRACE_WIDE_LIMITS)
    pol = build_policy("cq_fcfs", "S", 0.01, H=1)
    eng = run_case(scn, specs, pol, numeric=float, seed=9,
                   record_intervals=True)
    return [(float(t0), float(t1), float(r), float(q))
            for t0, t1, r, q in eng.w.storage.interval_log]


def episodes(log, b):
    out, cur = [], 0.0
    for t0, t1, r, q in log:
        w = t1 - t0
        if w <= 0:
            continue
        if q > b:
            cur += w
        else:
            if cur > 0:
                out.append(cur)
            cur = 0.0
    if cur > 0:
        out.append(cur)
    return np.array(out) * 1000.0   # ms


def fig1():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    log80 = run_instant("toolagent", "toolagent_trace.jsonl", 80.0, 0.6)
    log20 = run_instant("toolagent", "toolagent_trace.jsonl", 20.0, 0.6)

    fig = plt.figure(figsize=(15.5, 9.6), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.34, wspace=0.22)

    # (a) 桶均实际利用率（图 E/H 口径）
    ax = fig.add_subplot(gs[0, :])
    for B, c in ((80, "#4C72B0"), (20, "#C44E52")):
        rows = load_rows("toolagent", B, 0.6)
        rows = [r for r in rows if r["capacity_gb"] > 0 and r["width_s"] > 0]
        xs = [r["t_start_s"] + r["width_s"] / 2 for r in rows]
        util = [r["served_gb"] / r["capacity_gb"] for r in rows]
        ax.plot(xs, util, color=c, lw=1.2, label=f"B={B} GB/s")
    ax.axhline(1.0, color="#C44E52", ls="--", lw=.8, alpha=.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("仿真时间 (s)")
    ax.set_ylabel("桶均实际利用率 served/B")
    ax.set_title("(a) 图 E/H 画的就是这条：桶均（~0.55s/桶）实际传输÷容量——"
                 "构造上 ≤100%；B=20 峰顶到 100%，B=80 峰仅 ~50%", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.25)

    # (b) 瞬时需求 vs 实际（B=80，40s 放大段）
    ax = fig.add_subplot(gs[1, 0])
    t_lo = 20.0
    seg = [(t0, t1, r, q) for t0, t1, r, q in log80 if t1 > t_lo and t0 < t_lo + 40]
    ts_q, v_q = [], []
    ts_r, v_r = [], []
    for t0, t1, r, q in seg:
        ts_q += [t0, t1]
        v_q += [q, q]
        ts_r += [t0, t1]
        v_r += [r, r]
    ax.plot(ts_q, v_q, color="#DD8452", lw=1.0, label="瞬时需求 Σq（申请）")
    ax.plot(ts_r, v_r, color="#4C72B0", lw=1.0, label="瞬时实际 Σrate（分配后）")
    ax.axhline(80, color="k", ls="--", lw=1.0)
    ax.text(t_lo + 0.5, 90, "B=80", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1000)
    ax.set_xlim(t_lo, t_lo + 40)
    ax.set_xlabel("仿真时间 (s)——截取 20–60s 放大")
    ax.set_ylabel("速率 (GB/s，对数轴)")
    ax.set_title("(b) B=80 的瞬时口径：需求尖刺经常冲到 200–800，\n"
                 "但被削顶到 80 且只持续毫秒级（桶均后被抹平）", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.25, which="both")

    # (c) 过载段时长 CCDF
    ax = fig.add_subplot(gs[1, 1])
    for log, B, c in ((log80, 80, "#4C72B0"), (log20, 20, "#C44E52")):
        eps = episodes(log, float(B))
        if len(eps) == 0:
            continue
        xs = np.sort(eps)
        ccdf = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ccdf, where="post", color=c, lw=1.4,
                label=f"B={B}（n={len(eps)} 段，Σ时长 {eps.sum()/1000:.1f}s）")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1e4)
    ax.set_ylim(0, 1.02)
    ax.axvspan(0.01, 5, color="#4C72B0", alpha=.08)
    ax.axvspan(100, 1e4, color="#C44E52", alpha=.08)
    ax.text(0.05, 0.45, "毫秒级尖刺\n（B=80 主体）\n桶均不可见", fontsize=9,
            color="#4C72B0")
    ax.text(700, 0.45, "百毫秒~秒级\n（B=20 主体）\n桶均可见=间歇过载",
            fontsize=9, color="#C44E52")
    ax.set_xlabel("过载段时长（ms，对数轴）——Σq>B 的连续时段")
    ax.set_ylabel("CCDF（占比）")
    ax.set_title("(c) 过载段时长分布：B=80 与 B=20 差 3~4 个数量级", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.25, which="both")

    fig.suptitle("带宽的三个口径：桶均实际（图 E/H）· 瞬时需求 · 过载段时长"
                 "（ToolAgent OL ρ=0.6 m=4 w9 FCFS）", fontsize=12)
    out = os.path.join(FIG, "cq_fig_e25q_bw_3views.png")
    return fig, out


def fig2():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=DPI)
    bs = np.array([5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80], dtype=float)
    for ax, rho in zip(axes, (0.6, 0.9)):
        for tr in ("conversation", "toolagent", "synthetic"):
            rows = load_rows(tr, 80, rho)
            if rows is None:
                continue
            rows = [r for r in rows if r["width_s"] > 0]
            dem = np.array([r["requested_gb"] / r["width_s"] for r in rows])
            w = np.array([r["width_s"] for r in rows])
            sat = np.array([(dem > b) @ w / w.sum() for b in bs])
            ax.plot(bs, 100 * sat, color=TCOL[tr], lw=1.6, marker="o", ms=3.5,
                    label=TLAB[tr])
        ax.axhspan(10, 40, color="grey", alpha=.15)
        ax.text(52, 24, "建议间歇窗口\n（过载 10–40%）", fontsize=9.5,
                ha="center", color="dimgrey")
        ax.axvline(80, color="k", ls=":", lw=.9)
        ax.text(78, 88, "B=80 基线", fontsize=8.5, rotation=90, ha="right")
        ax.set_ylim(0, 100)
        ax.set_xlim(4, 85)
        ax.set_xlabel("总带宽 B (GB/s)")
        ax.set_ylabel("过载占比（桶均需求>B 的时间，%）")
        ax.set_title(f"ρ={rho}", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=.25)
    fig.suptitle("从 B=80 运行的需求分布反推：sat(B) 配置窗口（OL m=4 w9 FCFS，"
                 "α=4；降 B 是纯供给侧旋钮——λ0 锚的 80 不随 B 漂移）", fontsize=12)
    out = os.path.join(FIG, "cq_fig_e25q_satB.png")
    return fig, out


def fig3():
    """Q2 白话版主图：需求 vs 实际 两条线 + B 上限，B=80/B=20 并排。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=DPI, sharey=False)
    t_lo, t_hi = 10.0, 70.0
    for ax, B, ylim in ((axes[0], 80.0, 120.0), (axes[1], 20.0, 45.0)):
        log = run_instant("toolagent", "toolagent_trace.jsonl", B, 0.6)
        seg = [(t0, t1, r, q) for t0, t1, r, q in log
               if t1 > t_lo and t0 < t_hi]
        for which, col in (("q", "#DD8452"), ("r", "#4C72B0")):
            xs, ys = [], []
            for t0, t1, r, q in seg:
                v = q if which == "q" else r
                xs += [t0, t1]
                ys += [v, v]
            lab = "想要多快（需求 Σ申请率）" if which == "q" else "实际在传（分配后 Σ速率）"
            ax.plot(xs, ys, color=col, lw=0.9, label=lab)
        ax.axhline(B, color="k", ls="--", lw=1.4)
        ax.text(t_hi - 1.0, B * 1.05, f"带宽上限 B={B:g}", fontsize=10,
                ha="right")
        ax.set_ylim(0, ylim)
        ax.set_xlim(t_lo, t_hi)
        ax.set_xlabel("仿真时间 (s)——截取 10–70s")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("速率 (GB/s)")
    axes[0].set_title("B=80（管子粗）：实际=想要，峰值 40 多是需求自己的高度；\n"
                      "想要偶发冲到 200–800 但只持续几毫秒，线性轴上画成细刺",
                      fontsize=10.5)
    axes[0].annotate("想要冲高被压回上限\n（毫秒级，图上仅见细刺）",
                     xy=(31.5, 105), xytext=(38, 105), fontsize=9,
                     arrowprops=dict(arrowstyle="->", lw=.8))
    axes[1].set_title("B=20（管子细）：想要>20 的时段，实际被压平在 20——\n"
                      "蓝色贴着虚线走的段=间歇性过载；平均想要只有 ~8",
                      fontsize=10.5)
    axes[1].annotate("蓝色贴住虚线=过载段\n（持续几百毫秒，看得见）",
                     xy=(20.5, 20.5), xytext=(27, 32), fontsize=9,
                     arrowprops=dict(arrowstyle="->", lw=.8))
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("同一个负载（ToolAgent OL ρ=0.6，w9，FCFS）：橙色=请求们想读多快，"
                  "蓝色=存储实际在传多快——蓝色永远不超过黑虚线", fontsize=11.5)
    out = os.path.join(FIG, "cq_fig_e25q_demand_vs_actual.png")
    return fig, out


if __name__ == "__main__":
    from e25_ts_compare import audit_text_overlap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _save(fig, path):
        fig.savefig(path, bbox_inches="tight")
        n = len(audit_text_overlap(fig))
        plt.close(fig)
        print(f"{path}  文字重叠对: {n}")

    _save(*fig1())
    _save(*fig2())
    _save(*fig3())
