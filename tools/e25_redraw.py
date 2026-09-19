"""E25 汇总图生成器（图 A 箱线图 / 图 C 条形图 / 图 B 因子响应 / 图 G 每 NPU）。

- 图 A：FW 基线四目标箱线图——每策略一箱（3 文件 × 5 窗 = 15 个值），
  箱体=四分位距、中线=中位数、须=全距、灰点=离群窗口；MPC/local 行
  按"指标对应 θ"取数（M 列←θ=M、SLO←θ=S、TTFT←θ=T、归一化←θ=R）；
- 图 C：相对 FCFS 的平均 TTFT 改善%条形图（OL 基线 ρ=0.6、θ=T 口径、
  5 窗中位），每个 trace 一个子图、条上数字直读；
- 图 B（因子响应）/ 图 G（每 NPU 两线制）为既有设计，`--all` 时一并重绘；
- 图 I（方案对比 Gantt 时序）由 tools/e25_ts_compare.py 生成，不在本文件。

从归档 records/ts 直接生成，覆盖 docs/figures 下同名 cq_fig_e25_*。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_ROOT = os.path.join(REPO, "results", "cq", "eval", "e25", "ts")
FIG = os.path.join(REPO, "docs", "figures")

C_COMP, C_STALL, C_IDLE = "#4C72B0", "#DD8452", "#B0B0B0"
PCOL = {"cq_fcfs": "#000000", "cq_lpm": "#8C8C8C", "cq_spt": "#4C72B0",
        "cq_edf": "#DD8452", "cq_slack": "#55A868", "cq_pair": "#8172B3",
        "cq_local": "#937860", "cq_mpc": "#DA8BC3"}
# 图 C 的三 trace 主题色（与正式图一致）
TCOL = {"conversation_trace.jsonl": "#819CC7",
        "toolagent_trace.jsonl": "#E7A885",
        "synthetic_trace.jsonl": "#E5ADD4"}
DPI = 125


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["PingFang SC", "Heiti SC", "Songti SC", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11.5,
        "axes.titlesize": 13, "axes.labelsize": 12, "figure.titlesize": 15,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5})
    return plt


def load_records():
    return json.load(open(os.path.join(REPO, "results/cq/eval/e25/e25_records.json"),
                          encoding="utf-8"))


def cell_ts(mode, fname, B, rho, pid, suffix="", w=9, cap="C", prof="default"):
    stem = fname.replace("_trace.jsonl", "").replace(".jsonl", "")
    rho_s = "-" if rho is None else f"{rho:g}"
    cid = f"{stem}|{mode}|B{B:g}|rho{rho_s}|a4|{cap}|{prof}|w{w}"
    p = os.path.join(TS_ROOT, cid, f"{pid}{suffix}.json.gz")
    if not os.path.exists(p):
        return None
    import gzip
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def lines_of(ts, wi=None):
    """→ (xs, comp_frac, stall_frac)：每 NPU 或全体（wi=None 时对 m 求和）。"""
    xs, cf, sf = [], [], []
    for r in ts["rows"]:
        if r["width_s"] <= 0:
            continue
        if wi is None:
            c = sum(w[0] for w in r["workers"]); s = sum(w[1] for w in r["workers"])
            m = len(r["workers"])
        else:
            c, s, _i = r["workers"][wi]; m = 1
        xs.append(r["t_start_s"]); cf.append(c / r["width_s"] / m)
        sf.append(s / r["width_s"] / m)
    return xs, cf, sf


# ---------------------------------------------------------------- 图 A（箱线）
def fig_objectives(plt, recs):
    sub = [r for r in recs if r["factor_line"] == "baseline" and r["mode"] == "FW"]
    thm = {"M": "M", "slo_rate": "S", "ttft_mean_lower": "T",
           "ttft_norm_mean_lower": "R"}
    metrics = [("M", "makespan_M", "M：最后完成 (s)"),
               ("slo_rate", "slo_rate", "SLO 满足率"),
               ("ttft_mean_lower", "ttft_mean_lower", "平均 TTFT (s)"),
               ("ttft_norm_mean_lower", "ttft_norm_mean_lower", "平均归一化 TTFT")]
    pols = ["cq_fcfs", "cq_lpm", "cq_spt", "cq_edf", "cq_slack", "cq_pair",
            "cq_local", "cq_mpc"]
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0), dpi=DPI)
    for ax, (mk, key, ylab) in zip(axes.flat, metrics):
        data, pos = [], []
        for pi, p in enumerate(pols):
            vals = []
            for r in sub:
                if r["policy"] != p or r["theta"] != thm[mk]:
                    continue
                v = (r["slo_success"] / r["n_cohort"]) if key == "slo_rate" else r.get(key)
                if v is None and key == "makespan_M":
                    v = r.get("makespan_M_lower")
                if v is not None:
                    vals.append(v)
            if vals:
                data.append(vals); pos.append(pi)
        ax.boxplot(data, positions=pos, widths=.55, patch_artist=True,
                   boxprops=dict(facecolor="#E3E3E3", edgecolor="#555555"),
                   medianprops=dict(color="k", lw=2.0),
                   whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor="#999999",
                                   markeredgecolor="none", alpha=.6))
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([p.replace("cq_", "") for p in pols], rotation=32)
        ax.set_title(ylab)
        ax.grid(alpha=.25, axis="y")
    fig.suptitle("图A｜FW 基线四目标箱线图：每策略一箱（3 文件 × 5 窗共 15 值；"
                 "箱=四分位距、中线=中位数、须=全距、灰点=离群窗口；"
                 "MPC/local 按指标对应 θ 取数）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "cq_fig_e25_objectives.png"), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 图 C（条形）
def fig_pareto(plt, recs):
    """相对 FCFS 的平均 TTFT 改善%（OL 基线 ρ=0.6、θ=T 口径、5 窗中位）。"""
    sub = [r for r in recs if r["factor_line"] == "baseline"
           and r["mode"] == "OL" and r["theta"] == "T"]
    pols = ["cq_lpm", "cq_spt", "cq_edf", "cq_slack", "cq_pair",
            "cq_local", "cq_mpc"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), dpi=DPI, sharey=True)
    for ci, (fname, _n, _m, _s) in enumerate(MOONCAKE_FILES):
        ax = axes[ci]
        rs = [r for r in sub if r["file"] == fname]
        fcfs = [r["ttft_mean_lower"] for r in rs if r["policy"] == "cq_fcfs"]
        base = float(np.median(fcfs)) if fcfs else None
        gains, labels = [], []
        for p in pols:
            vs = [r["ttft_mean_lower"] for r in rs if r["policy"] == p]
            med = float(np.median(vs)) if vs else None
            gains.append((base - med) / base * 100.0 if (base and med) else 0.0)
            labels.append(p.replace("cq_", ""))
        bars = ax.bar(range(len(pols)), gains, color=TCOL[fname],
                      edgecolor="#666666", lw=.6)
        for rect, g in zip(bars, gains):
            ax.annotate(f"{g:+.0f}%", (rect.get_x() + rect.get_width() / 2,
                                       rect.get_height()),
                        xytext=(0, 4 if g >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=10.5)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels(labels, rotation=32)
        ax.set_title(TRACE_LABEL[fname])
        if ci == 0:
            ax.set_ylabel("相对 FCFS 的平均 TTFT 改善 (%)")
        ax.grid(alpha=.25, axis="y")
    fig.suptitle("图C｜相对 FCFS 的平均 TTFT 改善%（OL 基线 ρ=0.6、B=80、θ=T 口径、"
                 "5 窗中位；正值=更快；Synthetic 队列空、全策略与 FCFS 相同）")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(FIG, "cq_fig_e25_pareto.png"), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 图 G（两线制）
def fig_per_npu(plt, fname, B, w=9):
    ts = cell_ts("OL", fname, B, 0.6, "cq_fcfs", "", w=w)
    fig, axes = plt.subplots(4, 3, figsize=(16.5, 10.5), dpi=DPI)
    for ci, (f2, _n, _m, _s) in enumerate(MOONCAKE_FILES):
        t = ts if f2 == fname else cell_ts("OL", f2, B, 0.6, "cq_fcfs", "", w=w)
        for wi in range(4):
            ax = axes[wi][ci]
            xs, cf, sf = lines_of(t, wi)
            ax.plot(xs, cf, color=C_COMP, lw=1.5, label="计算")
            ax.plot(xs, sf, color=C_STALL, lw=1.4, label="等待")
            ax.set_ylim(-0.03, 1.03)
            if wi == 0: ax.set_title(TRACE_LABEL[f2])
            if ci == 0: ax.set_ylabel(f"NPU{wi}")
            if wi == 3: ax.set_xlabel("仿真时间 (s)")
            if wi == 0 and ci == 0: ax.legend(loc="upper right")
    fig.suptitle(f"图G｜每 NPU 的计算/等待两条线（OL 窗口{w}、FCFS、B={B:g} GB/s）")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, f"cq_fig_e25_ts_per_npu_B{B:g}.png"), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------- 图 B（因子响应）
def fig_factor(plt, recs, line, out_name):
    levels_l = {"line_B": ([20.0, 80.0, 320.0], lambda r: r["B"], "B (GB/s)"),
                "line_alpha": ([2, 4, 8], lambda r: r["alpha"], "SLO 倍数 α"),
                "line_rho": ([0.3, 0.6, 0.9, 1.1], lambda r: r["rho"], "负载 ρ")}
    levels, get, xlab = levels_l[line]
    rows = [("SLO 满足率", "slo_rate", "S"), ("平均 TTFT (s)", "ttft_mean_lower", "T"),
            ("平均归一化 TTFT", "ttft_norm_mean_lower", "R"),
            ("排空 M 下界 (s)", "drain_M", "S")]
    sub_all = [r for r in recs if r["factor_line"] in ("baseline", line)
               and r["mode"] == "OL"]
    pols = ["cq_fcfs", "cq_spt", "cq_edf", "cq_mpc"]
    files = [f for f, _n, _m, _s in MOONCAKE_FILES]
    fig, axes = plt.subplots(4, 3, figsize=(16.5, 12.5), dpi=DPI)
    for ci, f in enumerate(files):
        for ri, (ylab, key, th) in enumerate(rows):
            ax = axes[ri][ci]
            for p in pols:
                xs, med, lo, hi = [], [], [], []
                for lv in levels:
                    vs = []
                    for r in sub_all:
                        if r["file"] != f or r["policy"] != p or r["theta"] != th:
                            continue
                        if get(r) != lv:
                            continue
                        v = (r["slo_success"] / r["n_cohort"]) if key == "slo_rate" else r.get(key)
                        if v is not None:
                            vs.append(v)
                    if vs:
                        xs.append(levels.index(lv)); med.append(float(np.median(vs)))
                        lo.append(min(vs)); hi.append(max(vs))
                if xs:
                    ax.plot(xs, med, marker="o", ms=5, lw=1.8,
                            label=p.replace("cq_", ""), color=PCOL[p])
                    ax.fill_between(xs, lo, hi, alpha=.12, color=PCOL[p])
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([str(l) for l in levels])
            if ci == 0: ax.set_ylabel(f"{ylab}", fontsize=11.5)
            if ri == 0: ax.set_title(TRACE_LABEL[f])
            if ri == 3: ax.set_xlabel(xlab)
            ax.grid(alpha=.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(f"图B｜因子响应：{line}（OL；线=窗口中位数、带=min–max；MPC=θ 对应运行）")
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    fig.savefig(os.path.join(FIG, f"cq_fig_e25_factor_{out_name}.png"), dpi=DPI)
    plt.close(fig)


def main():
    plt = plt_setup()
    recs = load_records()
    fig_objectives(plt, recs)
    fig_pareto(plt, recs)
    if "--all" in sys.argv:          # 图 B/G 既有设计，按需重绘
        for B in (80.0, 20.0):
            fig_per_npu(plt, "conversation_trace.jsonl", B)
        for line, out in [("line_B", "B"), ("line_alpha", "alpha"),
                          ("line_rho", "rho")]:
            fig_factor(plt, recs, line, out)
    print("图 A（箱线）/ 图 C（条形）生成完成")


if __name__ == "__main__":
    main()
