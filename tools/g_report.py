"""G0/G1/G2 结果报告工具：从 results/g{1,2}_*.json 生成图表与 Markdown 表。

用法：.venv/bin/python tools/g_report.py
输出：docs/figures/fig_g1_*.png、fig_g2_*.png，stdout 打印可嵌入报告的表格。

口径：每格为 5 种子的中位数，误差线为种子间 min–max；配对差值按同种子
（policy − 参照）计算中位数与 percentile bootstrap 95% CI（1000 次重采样）。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.experiments.g_common import G1_POLICIES, G2_POLICIES, slo_table
from sim.experiments.g_common import G1_DEFS as _G1_DEFS

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
FIG = Path("docs/figures")
RNG = np.random.default_rng(20260908)

G1_LABELS = [p[3] for p in G1_POLICIES]
G2_LABELS = [p[3] for p in G2_POLICIES]
COLORS = dict(zip(G1_LABELS + G2_LABELS,
                  ["#8c8c8c", "#937860", "#b5b5b5", "#ccb974", "#8172b2",
                   "#64b5cd", "#c44e52", "#55a868", "#4c72b0", "#7f9fc4",
                   "#cca64c", "#a1c9a1", "#c9a3c9", "#6bd"]))


def load(path):
    if not Path(path).exists():
        return []
    return json.load(open(path))


def agg(runs):
    """(quadrant,label) -> {seed: run}。"""
    d = defaultdict(dict)
    for r in runs:
        d[(r["quadrant"], r["label"])][r["seed"]] = r
    return d


def med_ci(vals, n=1000):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float("nan"), (float("nan"), float("nan"))
    boots = [np.median(RNG.choice(vals, size=len(vals), replace=True)) for _ in range(n)]
    return float(np.median(vals)), (float(np.percentile(boots, 2.5)),
                                    float(np.percentile(boots, 97.5)))


def grid_stats(runs, labels, quadrants, ref_label, metric="ttft_p95"):
    """每象限每策略：中位数、min-max、相对参照的同种子配对差（中位数、CI、胜数）。"""
    d = agg(runs)
    table = {}
    for q in quadrants:
        seeds = sorted(s for (qq, l), m in d.items() if qq == q and l == ref_label
                       for s in m)
        if not seeds:
            continue
        refv = {s: d[(q, ref_label)][s]["overall"][metric] for s in seeds}
        for lab in labels:
            if (q, lab) not in d:
                continue
            vals = [d[(q, lab)][s]["overall"][metric] for s in sorted(d[(q, lab)])]
            med, ci = med_ci(vals)
            diffs, wins = [], 0
            for s in sorted(d[(q, lab)]):
                if s in refv:
                    dv = d[(q, lab)][s]["overall"][metric] - refv[s]
                    diffs.append(dv)
                    wins += dv < 0
            dmed, dci = med_ci(diffs) if diffs else (float("nan"),) * 2
            table[(q, lab)] = dict(med=med, ci=ci, lo=min(vals), hi=max(vals),
                                   dmed=dmed, dci=dci, wins=wins, n=len(diffs))
    return table


def _bars(ax, labels, vals, errs, colors, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.68, color=colors, yerr=errs, capsize=2.5,
           error_kw=dict(lw=0.8, ecolor="#333"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=7.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)


def fig_g1_ttft(runs):
    t = grid_stats(runs, G1_LABELS, ["loose", "io", "gpu", "both"], "B-Hist")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharey=True)
    titles = {"loose": "双松", "io": "I/O 紧 GPU 松", "gpu": "GPU 紧 I/O 松", "both": "双紧"}
    for ax, q in zip(axes.flat, ["loose", "io", "gpu", "both"]):
        labs = [l for l in G1_LABELS if (q, l) in t]
        vals = [t[(q, l)]["med"] for l in labs]
        errs = [[max(0, t[(q, l)]["med"] - t[(q, l)]["lo"]) for l in labs],
                [max(0, t[(q, l)]["hi"] - t[(q, l)]["med"]) for l in labs]]
        _bars(ax, labs, vals, errs, [COLORS.get(l, "#888") for l in labs],
              f"{titles[q]}（P95 TTFT，5 种子中位数）", "秒")
    fig.suptitle("G1：四象限压力下各策略 P95 TTFT（误差线 = 种子 min–max）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig_g1_ttft.png", dpi=150)
    plt.close(fig)


def fig_g1_attain(runs):
    t = grid_stats(runs, G1_LABELS, ["loose", "io", "gpu", "both"], "B-Hist",
                   metric="attain_mean")
    tm = grid_stats(runs, G1_LABELS, ["loose", "io", "gpu", "both"], "B-Hist",
                    metric="attain_min")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharey=True)
    titles = {"loose": "双松", "io": "I/O 紧 GPU 松", "gpu": "GPU 紧 I/O 松", "both": "双紧"}
    for ax, q in zip(axes.flat, ["loose", "io", "gpu", "both"]):
        labs = [l for l in G1_LABELS if (q, l) in t]
        vals = [t[(q, l)]["med"] for l in labs]
        errs = [[max(0, t[(q, l)]["med"] - t[(q, l)]["lo"]) for l in labs],
                [max(0, t[(q, l)]["hi"] - t[(q, l)]["med"]) for l in labs]]
        _bars(ax, labs, vals, errs, [COLORS.get(l, "#888") for l in labs],
              f"{titles[q]}（类别达标率均值；点 = 最差类别）", "达标率")
        x = np.arange(len(labs))
        ax.scatter(x, [tm[(q, l)]["med"] for l in labs], marker="_", s=220,
                   color="k", zorder=3, linewidths=1.2)
        ax.set_ylim(0, 1.02)
    fig.suptitle("G1：类别达标率（条 = 各类均值中位数，黑色短划 = min 类达标率中位数）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig_g1_attain.png", dpi=150)
    plt.close(fig)


def fig_g1_resource(runs):
    d = agg(runs)
    qs = ["io", "both"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    for row, q in enumerate(qs):
        labs = [l for l in G1_LABELS if (q, l) in d]
        nec = [np.mean([sum(w["nec_prefill_s"] for w in d[(q, l)][s]["workers"])
                        for s in d[(q, l)]]) for l in labs]
        dup = [np.mean([sum(w["dup_prefill_s"] for w in d[(q, l)][s]["workers"])
                        for s in d[(q, l)]]) for l in labs]
        dec = [np.mean([sum(w["decode_s"] for w in d[(q, l)][s]["workers"])
                        for s in d[(q, l)]]) for l in labs]
        idle = [np.mean([sum(w["idle_empty_s"] + w["idle_kv_wait_s"]
                             for w in d[(q, l)][s]["workers"])
                         for s in d[(q, l)]]) for l in labs]
        kvw = [np.mean([sum(w["idle_kv_wait_s"] for w in d[(q, l)][s]["workers"])
                        for s in d[(q, l)]]) for l in labs]
        x = np.arange(len(labs))
        ax = axes[row][0]
        ax.bar(x, nec, 0.66, label="必要 prefill", color="#4c72b0")
        ax.bar(x, dup, 0.66, bottom=nec, label="重复重算", color="#c44e52")
        ax.bar(x, dec, 0.66, bottom=np.array(nec) + np.array(dup),
               label="decode", color="#55a868")
        ax.bar(x, idle, 0.66, bottom=np.array(nec) + np.array(dup) + np.array(dec),
               label="空闲", color="#d9d9d9")
        ax.plot(x, np.array(nec) + np.array(dup) + np.array(dec) + np.array(kvw),
                "_", ms=18, color="k", label="其中等 KV 空转（短划）")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=38, ha="right", fontsize=7.5)
        ax.set_title(f"{q} 象限：GPU 时间构成（4 worker 合计，种子均值）", fontsize=10)
        ax.set_ylabel("秒")
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
        ax.legend(fontsize=7, ncol=2)

        ax = axes[row][1]
        fg = [np.mean([d[(q, l)][s]["storage"][4]["util_fg"] for s in d[(q, l)]])
              for l in labs]
        n0 = [np.mean([d[(q, l)][s]["storage"][0]["util_fg"] for s in d[(q, l)]])
              for l in labs]
        n1 = [np.mean([d[(q, l)][s]["storage"][2]["util_fg"] for s in d[(q, l)]])
              for l in labs]
        w = 0.27
        ax.bar(x - w, n0, w, label="n0.mem 前台", color="#8172b2")
        ax.bar(x, n1, w, label="n1.mem 前台", color="#ccb974")
        ax.bar(x + w, fg, w, label="fabric 前台", color="#64b5cd")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=38, ha="right", fontsize=7.5)
        ax.set_title(f"{q} 象限：存储前台带宽利用率（served/(容量×窗口)）", fontsize=10)
        ax.set_ylabel("利用率")
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g1_resource.png", dpi=150)
    plt.close(fig)


def fig_g1_timeline(runs):
    d = agg(runs)
    picks = [("B-Hist", "io"), ("D1a 联合", "io"), ("B-LL", "both"), ("D2b 动态取算", "both")]
    picks = [(lab, q) for (lab, q) in picks
             if (q, lab) in d and d[(q, lab)][0].get("timeline_iters")]
    n = len(picks)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(11.5, 2.5 * n), squeeze=False)
    for row, (lab, q) in enumerate(picks):
        r = d[(q, lab)][0]
        it = np.array(r["timeline_iters"])          # (t, wid, t_pre, t_dec, nch, ndec)
        res = np.array(r["timeline_res"])           # (t, q×5, hbm×4, wait×4)
        ax = axes[row][0]
        bins = np.arange(0, r["duration"] + 1, 2.0)
        pre, dec = it[:, 2], it[:, 3]
        for w in range(4):
            m = it[:, 1] == w
            if m.sum() == 0:
                continue
            ax.hist(it[m, 0], bins=bins, weights=pre[m], alpha=0.35,
                    color="#c44e52", label="prefill" if w == 0 else None)
            ax.hist(it[m, 0], bins=bins, weights=dec[m], alpha=0.35,
                    color="#55a868", label="decode" if w == 0 else None)
        if row == 0:
            ax.legend(fontsize=7)
        ax.set_title(f"{lab} @ {q}：每 2s GPU 迭代时间构成（4 worker 合计）", fontsize=9)
        ax.set_ylabel("秒")
        ax = axes[row][1]
        if len(res):
            ax.plot(res[:, 0], res[:, 4], lw=0.9, color="#64b5cd", label="fabric 在途队列深")
            ax2 = ax.twinx()
            ax2.plot(res[:, 0], res[:, 5:9].sum(axis=1), lw=0.9, color="#b5b5b5",
                     label="Σworker HBM 预留(GB, 右轴)")
            ax2.set_ylabel("GB", fontsize=8)
            ax.set_ylabel("队列深度", fontsize=8)
            if row == 0:
                ax.legend(fontsize=7, loc="upper left")
                ax2.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{lab} @ {q}：存储/容量状态", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g1_timeline.png", dpi=150)
    plt.close(fig)


def fig_g2_policy(runs):
    t = grid_stats(runs, G2_LABELS, ["io", "gpu", "loose"], "E-FCFS(B-Hist)")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), sharey=True)
    for ax, q in zip(axes, ["io", "gpu", "loose"]):
        labs = [l for l in G2_LABELS if (q, l) in t]
        vals = [t[(q, l)]["med"] for l in labs]
        errs = [[max(0, t[(q, l)]["med"] - t[(q, l)]["lo"]) for l in labs],
                [max(0, t[(q, l)]["hi"] - t[(q, l)]["med"]) for l in labs]]
        _bars(ax, labs, vals, errs, [COLORS.get(l, "#888") for l in labs],
              f"{q}（P95 TTFT）", "秒")
    fig.suptitle("G2：H×U×O 八类混合负载下 D2 排序/动作策略 P95 TTFT", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig_g2_policy.png", dpi=150)
    plt.close(fig)


def fig_g2_class(runs, quadrant="gpu"):
    d = agg(runs)
    labs = ["E-FCFS(B-Hist)", "E-LPM", "D2a 就绪排序", "D2c 预算排序"]
    labs = [l for l in labs if (quadrant, l) in d]
    cls_names = sorted(next(iter(d[(quadrant, labs[0])].values()))["per_class"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    x = np.arange(len(cls_names))
    w = 0.8 / len(labs)
    for i, lab in enumerate(labs):
        p95 = [np.median([d[(quadrant, lab)][s]["per_class"][c]["ttft_p95"]
                          for s in d[(quadrant, lab)]]) for c in cls_names]
        at = [np.median([d[(quadrant, lab)][s]["per_class"][c]["attain"]
                         for s in d[(quadrant, lab)]]) for c in cls_names]
        axes[0].bar(x + i * w, p95, w * 0.92, label=lab, color=COLORS.get(lab, "#888"))
        axes[1].bar(x + i * w, at, w * 0.92, label=lab, color=COLORS.get(lab, "#888"))
    for ax, (t, y) in zip(axes, [("P95 TTFT（秒）", "秒"), ("类别达标率", "达标率")]):
        ax.set_xticks(x + w * (len(labs) - 1) / 2)
        ax.set_xticklabels(cls_names, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(f"{t} @ {quadrant}", fontsize=10)
        ax.set_ylabel(y)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    axes[0].legend(fontsize=7.5)
    axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_g2_class.png", dpi=150)
    plt.close(fig)


def fig_g2_action(runs):
    d = agg(runs)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    hs, us = (512, 16384), (512, 8192)
    for ax, q in zip(axes, ["loose", "io"]):
        M = np.zeros((2, 2))
        for i, h in enumerate(hs):
            for j, u in enumerate(us):
                cs = [c for c in d[(q, "D2b 动态取算")][0]["per_class"]
                      if c.startswith(f"h{h//1024}k_u{u//1024}k")]
                M[i, j] = np.mean([np.mean([d[(q, "D2b 动态取算")][s]["per_class"][c]["fetch_of_hit"]
                                            for s in d[(q, "D2b 动态取算")]]) for c in cs])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"U={u//1024}k" for u in us])
        ax.set_yticks([0, 1]); ax.set_yticklabels([f"H={h//1024}k" for h in hs])
        ax.set_title(f"D2b 动态取算 @ {q}：命中请求取回比例", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(FIG / "fig_g2_action.png", dpi=150)
    plt.close(fig)


def md_tables(runs1, runs2):
    out = []
    out.append("\n### G1 表（P95 TTFT 秒 / 达标率；Δ = 同种子配对差 vs B-Hist 中位数 [CI95]）\n")
    t = grid_stats(runs1, G1_LABELS, ["loose", "io", "gpu", "both"], "B-Hist")
    ta = grid_stats(runs1, G1_LABELS, ["loose", "io", "gpu", "both"], "B-Hist",
                    metric="attain_mean")
    d = agg(runs1)
    for q in ["loose", "io", "gpu", "both"]:
        out.append(f"\n**{q}**\n")
        out.append("| 策略 | P95 TTFT | ΔvsB-Hist [CI] | 胜/种子 | 达标率 | min类 | fetch% |")
        out.append("|---|---:|---|---:|---:|---:|---:|")
        for lab in G1_LABELS:
            if (q, lab) not in t:
                continue
            e = t[(q, lab)]
            ea = ta[(q, lab)]
            fh = np.median([d[(q, lab)][s]["n_fetch"] /
                            max(1, d[(q, lab)][s]["n_fetch"] + d[(q, lab)][s]["n_recompute"])
                            for s in d[(q, lab)]])
            out.append(f"| {lab} | {e['med']:.3f} | {e['dmed']:+.3f} "
                       f"[{e['dci'][0]:+.3f},{e['dci'][1]:+.3f}] | {e['wins']}/{e['n']} "
                       f"| {ea['med']:.3f} | "
                       f"{np.median([min(c['attain'] for c in d[(q,lab)][s]['per_class'].values()) for s in d[(q,lab)]]):.2f} "
                       f"| {fh:.2f} |")
    out.append("\n### G2 表（vs E-FCFS(B-Hist)）\n")
    t2 = grid_stats(runs2, G2_LABELS, ["io", "gpu", "loose"], "E-FCFS(B-Hist)")
    ta2 = grid_stats(runs2, G2_LABELS, ["io", "gpu", "loose"], "E-FCFS(B-Hist)",
                     metric="attain_mean")
    d2 = agg(runs2)
    for q in ["io", "gpu", "loose"]:
        out.append(f"\n**{q}**\n")
        out.append("| 策略 | P95 TTFT | ΔvsE-FCFS [CI] | 胜/种子 | 达标率 | min类 | fetch% |")
        out.append("|---|---:|---|---:|---:|---:|---:|")
        for lab in G2_LABELS:
            if (q, lab) not in t2:
                continue
            e, ea = t2[(q, lab)], ta2[(q, lab)]
            fh = np.median([d2[(q, lab)][s]["n_fetch"] /
                            max(1, d2[(q, lab)][s]["n_fetch"] + d2[(q, lab)][s]["n_recompute"])
                            for s in d2[(q, lab)]])
            out.append(f"| {lab} | {e['med']:.3f} | {e['dmed']:+.3f} "
                       f"[{e['dci'][0]:+.3f},{e['dci'][1]:+.3f}] | {e['wins']}/{e['n']} "
                       f"| {ea['med']:.3f} | "
                       f"{np.median([min(c['attain'] for c in d2[(q,lab)][s]['per_class'].values()) for s in d2[(q,lab)]]):.2f} "
                       f"| {fh:.2f} |")
    return "\n".join(out)


def main():
    FIG.mkdir(exist_ok=True, parents=True)
    r1 = load("results/g1_quadrant.json")
    r2 = load("results/g2_mixed.json")
    if r1:
        fig_g1_ttft(r1)
        fig_g1_attain(r1)
        fig_g1_resource(r1)
        fig_g1_timeline(r1)
        print("G1 figures done")
    if r2:
        fig_g2_policy(r2)
        fig_g2_class(r2, "gpu")
        fig_g2_action(r2)
        print("G2 figures done")
    print(md_tables(r1, r2))
    print("\nSLO 档位（冻结公式）：")
    for row in slo_table(_G1_DEFS):
        print(row)


if __name__ == "__main__":
    main()
