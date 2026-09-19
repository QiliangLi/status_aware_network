"""E25 报告图统一重绘（20260919 应用户六点反馈）：

- 全部图：字号加大（轴 11/标题 13/suptitle 15），suptitle 预留空间防重叠；
- 每 NPU 时序改为"计算/等待两条线"（不再三色堆叠）：蓝线=计算占比、
  橙线=等待占比，纵轴 0~1；
- 关键图附"标注版"（_annot 后缀）：程序化加框/箭头/文字，指出看点了；
- 图 I 的 MPC 行显式标 θ（OL/FW 均为 θ=S 运行，与 §4 表同口径）。
从归档 records/ts 直接生成，覆盖 docs/figures 下同名 cq_fig_e25_*。
"""
from __future__ import annotations

import gzip
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
W5 = [0, 4, 9, 14, 19]

C_COMP, C_STALL, C_IDLE = "#4C72B0", "#DD8452", "#B0B0B0"
PCOL = {"cq_fcfs": "#000000", "cq_lpm": "#8C8C8C", "cq_spt": "#4C72B0",
        "cq_edf": "#DD8452", "cq_slack": "#55A868", "cq_pair": "#8172B3",
        "cq_local": "#937860", "cq_mpc": "#DA8BC3"}


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


# ---------------------------------------------------------------- 图 I 系列
def fig_cmp(plt, recs, tag, mode, fname, B, rho, title, annot=False):
    rows = [("cq_fcfs", "", "FCFS（基线）"), ("cq_edf", "", "EDF（排序类代表）"),
            ("cq_mpc", "_S", "MPC（θ=S 运行）")]
    fig, axes = plt.subplots(3, 5, figsize=(18.5, 9.2))
    tss = {}
    for ri, (pid, suf, label) in enumerate(rows):
        ts = cell_ts(mode, fname, B, rho, pid, suf)
        tss[pid] = ts
        for wi in range(4):
            ax = axes[ri][wi]
            xs, cf, sf = lines_of(ts, wi)
            ax.plot(xs, cf, color=C_COMP, lw=1.5, label="计算")
            ax.plot(xs, sf, color=C_STALL, lw=1.4, label="等待")
            ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, .5, 1])
            if ri == 0: ax.set_title(f"NPU{wi}")
            if wi == 0: ax.set_ylabel(label, fontsize=12)
            if ri == 2: ax.set_xlabel("仿真时间 (s)")
            if ri == 0 and wi == 0: ax.legend(loc="upper right")
        ax = axes[ri][4]
        xs, rate = [], []
        for r in ts["rows"]:
            if r["width_s"] > 0:
                xs.append(r["t_start_s"]); rate.append(r["served_gb"] / r["width_s"])
        b = float(ts["meta"]["b_schedule"][0][1])
        ax.fill_between(xs, rate, color="#8fb8e0", alpha=.45)
        ax.plot(xs, rate, color=C_COMP, lw=1.2)
        ax.axhline(b, color="k", ls="--", lw=1.1)
        ax.text(0.985, b, f"B={b:g} GB/s", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=10.5)
        ax.set_ylim(0, b * 1.15)
        if ri == 0: ax.set_title("存储带宽占用 (GB/s)")
        if ri == 2: ax.set_xlabel("仿真时间 (s)")
    # 统计行（写进 caption 素材）
    stats = {}
    for pid, ts in tss.items():
        rowsv = [r for r in ts["rows"] if r["width_s"] > 0]
        tot = sum(r["compute_s"] + r["stall_s"] + r["idle_s"] for r in rowsv)
        q = [r["queue_mean"] for r in rowsv if r["queue_mean"] is not None]
        stats[pid] = (sum(r["compute_s"] for r in rowsv) / tot,
                      sum(r["stall_s"] for r in rowsv) / tot,
                      max(q) if q else 0)
    txt = "   |   ".join(f"{k.split('_')[1]}: 算{v[0]:.0%}/等{v[1]:.0%}/队列峰{v[2]:.0f}"
                         for k, v in stats.items())
    fig.suptitle(f"图I｜{TRACE_LABEL[fname]} · {title}（窗口9；MPC 行为 θ=S 运行）\n{txt}")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if annot:
        _annot_cmp(fig, axes, tss, tag)
    fig.savefig(os.path.join(FIG, f"cq_fig_e25_cmp_{tag}_"
                            f"{fname.split('_')[0]}{'' if not annot else '_annot'}.png"),
                dpi=125)
    plt.close(fig)
    return stats


def _annot_cmp(fig, axes, tss, tag):
    """程序化标注：按数据找关键区加红框与文字。"""
    if tag == "FWbase":
        edf = tss["cq_edf"]
        xs, cf, _ = lines_of(edf, 0)
        if xs:
            t_end = xs[-1]
            t0 = t_end * 0.6
            for wi in range(4):
                ax = axes[1][wi]
                ax.axvspan(t0, t_end, color="#C44E52", alpha=.12)
            axes[1][0].annotate("EDF 长尾：短优先把长请求\n留到最后，NPU 大面积闲置",
                                 xy=(t_end * .8, .55), xytext=(t_end * .25, .75),
                                 fontsize=11, color="#C44E52",
                                 arrowprops=dict(arrowstyle="->", color="#C44E52"))
    if tag == "OLbase":
        mpc = tss["cq_mpc"]
        rowsv = [r for r in mpc["rows"] if r["width_s"] > 0]
        q = [r["queue_mean"] for r in rowsv]
        pk = int(np.argmax(q))
        axes[2][4].annotate(f"MPC 队列峰更快回落（见统计行）",
                            xy=(rowsv[pk]["t_start_s"], rowsv[pk]["served_gb"] / rowsv[pk]["width_s"]),
                            xytext=(rowsv[pk]["t_start_s"] * .35,
                                    float(mpc["meta"]["b_schedule"][0][1]) * .8),
                            fontsize=11, color="#2e8b57",
                            arrowprops=dict(arrowstyle="->", color="#2e8b57"))


# ---------------------------------------------------------------- 图 G（两线制）
def fig_per_npu(plt, fname, B, w=9):
    ts = cell_ts("OL", fname, B, 0.6, "cq_fcfs", "", w=w)
    fig, axes = plt.subplots(4, 3, figsize=(16.5, 10.5))
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
    fig.savefig(os.path.join(FIG, f"cq_fig_e25_ts_per_npu_B{B:g}.png"), dpi=125)
    plt.close(fig)


# ---------------------------------------------------------------- 图 A / C
def fig_objectives(plt, recs, annot=False):
    sub = [r for r in recs if r["factor_line"] == "baseline" and r["mode"] == "FW"]
    th_of = {"M": "M", "slo_rate": "S", "ttft_mean_lower": "T",
             "ttft_norm_mean_lower": "R"}
    thm = {"M": "M", "slo_rate": "S", "ttft_mean_lower": "T",
           "ttft_norm_mean_lower": "R"}
    metrics = [("M", "makespan_M", "M：最后完成 (s)"),
               ("slo_rate", "slo_rate", "SLO 满足率"),
               ("ttft_mean_lower", "ttft_mean_lower", "平均 TTFT (s)"),
               ("ttft_norm_mean_lower", "ttft_norm_mean_lower", "平均归一化 TTFT")]
    fmark = {"conversation_trace.jsonl": "o", "toolagent_trace.jsonl": "s",
             "synthetic_trace.jsonl": "^"}
    fcol = {"conversation_trace.jsonl": "#4C72B0", "toolagent_trace.jsonl": "#DD8452",
            "synthetic_trace.jsonl": "#55A868"}
    pols = ["cq_fcfs", "cq_lpm", "cq_spt", "cq_edf", "cq_slack", "cq_pair",
            "cq_local", "cq_mpc"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10))
    for ax, (mk, key, ylab) in zip(axes.flat, metrics):
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
                    ax.scatter(pi, v, marker=fmark[r["file"]], s=26, alpha=.5,
                               color=fcol[r["file"]])
            if vals:
                med = float(np.median(vals))
                ax.hlines(med, pi - .32, pi + .32, color="k", lw=2.4)
                if annot:
                    ax.annotate(f"{med:.3g}", (pi, med), xytext=(0, 6),
                                textcoords="offset points", ha="center",
                                fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([p.replace("cq_", "") for p in pols], rotation=32)
        ax.set_title(ylab + ("（每列=优化该指标的 θ 运行）" if annot else ""))
        ax.grid(alpha=.25, axis="y")
    fig.suptitle("图A｜FW 基线四目标绝对值：点=文件×窗口（形状/颜色区分），黑线=中位数"
                 + ("；数值为中位数标注版" if annot else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "cq_fig_e25_objectives" + ("_annot" if annot else "") + ".png"), dpi=125)
    plt.close(fig)


def fig_pareto(plt, recs, annot=False):
    sub = [r for r in recs if r["factor_line"] == "baseline" and r["mode"] == "FW"
           and r["theta"] == "M"]
    fmark = {"conversation_trace.jsonl": "o", "toolagent_trace.jsonl": "s",
             "synthetic_trace.jsonl": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    cent = defaultdict(list)
    for r in sub:
        x1 = r.get("makespan_M") or r.get("makespan_M_lower")
        y1 = r["slo_success"] / r["n_cohort"]
        x2 = r["ttft_mean_lower"]; y2 = r["ttft_norm_mean_lower"]
        axes[0].scatter(x1, y1, marker=fmark[r["file"]], s=30, alpha=.6,
                        color=PCOL[r["policy"]])
        axes[1].scatter(x2, y2, marker=fmark[r["file"]], s=30, alpha=.6,
                        color=PCOL[r["policy"]])
        cent[r["policy"]].append((x1, y1, x2, y2))
    for p, vs in cent.items():
        if vs and annot and p in ("cq_fcfs", "cq_edf", "cq_mpc"):
            a = np.mean(vs, axis=0)
            axes[0].annotate(p.replace("cq_", ""), (a[0], a[1]), fontsize=12,
                             fontweight="bold", xytext=(6, 4),
                             textcoords="offset points", color=PCOL[p])
            axes[1].annotate(p.replace("cq_", ""), (a[2], a[3]), fontsize=12,
                             fontweight="bold", xytext=(6, 4),
                             textcoords="offset points", color=PCOL[p])
    axes[0].set_xlabel("M：最后完成 (s)"); axes[0].set_ylabel("SLO 满足率")
    axes[0].set_title("M–SLO 平面（左上=优）")
    axes[1].set_xlabel("平均 TTFT (s)"); axes[1].set_ylabel("平均归一化 TTFT")
    axes[1].set_title("TTFT–归一化 平面（左下=优）")
    for ax in axes: ax.grid(alpha=.3)
    if annot:
        vs_f = np.mean(cent["cq_fcfs"], axis=0); vs_m = np.mean(cent["cq_mpc"], axis=0)
        axes[0].annotate("", xy=(vs_m[0], vs_m[1]), xytext=(vs_f[0], vs_f[1]),
                         arrowprops=dict(arrowstyle="->", lw=2.4, color="#2e8b57"))
        axes[0].text(.98, .02, "绿箭头：FCFS→MPC 的平均位移\n（左上移=双双改善）",
                     transform=ax.transAxes if False else axes[0].transAxes,
                     ha="right", fontsize=11, color="#2e8b57")
    fig.suptitle("图C｜目标冲突散点（FW 基线、θ=M 运行；点=策略×文件×窗口；颜色=策略、形状=文件）"
                 + ("；含 FCFS→MPC 平均位移箭头" if annot else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIG, "cq_fig_e25_pareto" + ("_annot" if annot else "") + ".png"), dpi=125)
    plt.close(fig)


# ---------------------------------------------------------------- 图 B（含标注版）
def fig_factor(plt, recs, line, out_name, annot=False):
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
    fig, axes = plt.subplots(4, 3, figsize=(16.5, 12.5))
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
    fig.savefig(os.path.join(FIG, f"cq_fig_e25_factor_{out_name}"
                            + ("_annot" if annot else "") + ".png"), dpi=125)
    plt.close(fig)


def main():
    plt = plt_setup()
    recs = load_records()
    # 图 I（两线制）+ 标注版两张
    jobs = [("OLbase", "OL", 80.0, 0.6, "OL 基线（B=80, ρ=0.6）"),
            ("OLB20", "OL", 20.0, 0.6, "OL 带宽受限（B=20）"),
            ("OLr03", "OL", 80.0, 0.3, "OL 低负载（ρ=0.3）"),
            ("OLr11", "OL", 80.0, 1.1, "OL 过载（ρ=1.1）"),
            ("FWbase", "FW", 80.0, None, "FW 有限工作集（128 条同时到达，MPC=θ=S）")]
    for tag, mode, B, rho, title in jobs:
        for f, _n, _m, _s in MOONCAKE_FILES:
            fig_cmp(plt, recs, tag, mode, f, B, rho, title,
                    annot=(tag in ("OLbase", "FWbase") and
                           f == "conversation_trace.jsonl"))
    for B in (80.0, 20.0):
        fig_per_npu(plt, "conversation_trace.jsonl", B)
    fig_objectives(plt, recs, annot=False)
    fig_objectives(plt, recs, annot=True)
    fig_pareto(plt, recs, annot=False)
    fig_pareto(plt, recs, annot=True)
    for line, out in [("line_B", "B"), ("line_alpha", "alpha"), ("line_rho", "rho")]:
        fig_factor(plt, recs, line, out, annot=(line == "line_B"))
    print("重绘完成")


if __name__ == "__main__":
    main()
