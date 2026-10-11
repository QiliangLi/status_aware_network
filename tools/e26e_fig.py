"""E26e 图工具：每格点两张图（甘特 5 行 + 带宽时序 5 行）+ 汇总条形图。

用法：
  .venv/bin/python tools/e26e_fig.py --kind gantt --cell=D1c_k16 [--rep 0] [--audit]
  .venv/bin/python tools/e26e_fig.py --kind bw    --cell=D1c_k4
  .venv/bin/python tools/e26e_fig.py --kind bar
甘特：蓝=A算 橙=B算 红=IO等 灰=无请求（NPU0..31 泳道）。
带宽：10ms 分箱需求 Σq（橙）vs 实际 Σrate（蓝）+120 黑虚线，对数纵轴。
汇总：5 策略 × 2 格点总 SLO 条形；多 seed 格点中位数 ± min/max 带。
数据源：results/cq/eval/e26e/<cell>@s<rep>/<arm>.{gantt.json.gz,intervals.npz}
与 results/cq/eval/e26e/records_part_*.json。
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("E26E_ROOT") or os.path.join(REPO, "results", "cq", "eval", "e26e")
FIG_DIR = os.path.join(REPO, "docs", "figures")
B_GBPS = 120.0
BIN_S = 0.01

ROWS = [("FCFS", "fcfs"), ("EDF", "edf"), ("local v2.1", "localS_v2.1"),
        ("mpc v2.1 θ=S", "mpcS_v2.1"), ("mpc v2.1 θ=T", "mpcT_v2.1")]
POLICY_ORDER = ["fcfs", "edf", "localS_v2.1", "mpcS_v2.1", "mpcT_v2.1"]
CELLS = ["D1c_k4", "D1c_k16"]


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


def _load_gants(cell_dir: str):
    gants = []
    for label, fn in ROWS:
        p = os.path.join(cell_dir, f"{fn}.gantt.json.gz")
        if not os.path.exists(p):
            print(f"缺 {p}")
            return None
        gants.append((label, json.load(gzip.open(p))))
    return gants


def build_gantt(cell: str, rep: int, audit_only: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e26_gantt import draw_row, C_A, C_B, C_WAIT, C_IDLE
    from e25_ts_compare import audit_text_overlap
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec

    tag = f"{cell}@s{rep}"
    gants = _load_gants(os.path.join(ROOT, tag))
    if gants is None:
        return
    t_max = 0
    for _l, g in gants:
        K = g["meta"]["K"]
        last = max((i for i, bk in enumerate(g["buckets"])
                    if any(v > 0 for row in bk for v in row)), default=0)
        t_max = max(t_max, (last + 1) * g["meta"]["T_anchor_s"] / K)
    fig = plt.figure(figsize=(14.6, 3.0 * 5 + 0.9), dpi=125)
    gs = GridSpec(5, 1, hspace=0.34, left=0.12, right=0.985, top=0.88,
                  bottom=0.04)
    stats = []
    for ri, (label, g) in enumerate(gants):
        ax = fig.add_subplot(gs[ri, 0])
        fr = draw_row(ax, g, t_max)
        stats.append((label, fr))
        ax.set_ylabel(f"{label}\nA算{fr[0]:.0%} B算{fr[1]:.0%}\n"
                      f"IO等{fr[2]:.0%} 闲{fr[3]:.0%}", fontsize=10)
        if ri == 4:
            ax.set_xlabel("仿真时间 (s)", fontsize=10)
    fig.suptitle(f"图A｜E26e 分类甘特：{tag}"
                 "（蓝=A算 橙=B算 红=IO等 灰=无请求）", fontsize=12.5, y=0.97)
    fig.legend(handles=[Patch(facecolor=C_A, label="A 计算"),
                        Patch(facecolor=C_B, label="B 计算"),
                        Patch(facecolor=C_WAIT, label="IO 等待"),
                        Patch(facecolor=C_IDLE, label="无请求")],
               loc="upper center", ncol=4, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.925))
    ov = audit_text_overlap(fig)
    if not audit_only:
        fig.savefig(os.path.join(FIG_DIR, f"cq_fig_e26e_gantt_{tag}.png"),
                    dpi=125)
    plt.close(fig)
    print(f"{tag}: 甘特文字重叠 {len(ov)} 对")
    for label, fr in stats:
        print(f"  {label:12s} A算{fr[0]:.1%} B算{fr[1]:.1%} "
              f"IO等{fr[2]:.1%} 闲{fr[3]:.1%}")
    return len(ov)


def build_bw(cell: str, rep: int, audit_only: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e25_ts_compare import audit_text_overlap
    from matplotlib.gridspec import GridSpec

    tag = f"{cell}@s{rep}"
    fig = plt.figure(figsize=(14.6, 2.3 * 5 + 0.9), dpi=125)
    gs = GridSpec(5, 1, hspace=0.40, left=0.08, right=0.985, top=0.88,
                  bottom=0.05)
    for ri, (label, fn) in enumerate(ROWS):
        p = os.path.join(ROOT, tag, f"{fn}.intervals.npz")
        ax = fig.add_subplot(gs[ri, 0])
        if not os.path.exists(p):
            ax.text(.5, .5, "无数据", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        arr = np.load(p)["intervals"]
        xs, rate, dem = binned(arr, BIN_S)
        # 对数纵轴（E26d 读者勘误口径）：需求峰与 120 上限、实际速率全部可见
        dem_p = np.maximum(dem, 0.5)
        rate_p = np.maximum(rate, 0.5)
        ax.plot(xs, dem_p, color="#DD8452", lw=.9,
                label="需求 Σ申请率（可超上限=超订）")
        ax.plot(xs, rate_p, color="#4C72B0", lw=.9,
                label="实际 Σ速率（恒≤上限）")
        ax.axhline(B_GBPS, color="k", ls="--", lw=1.1)
        t_hi = xs[-1]
        ax.text(t_hi * 0.995, B_GBPS * 1.25, "B=120", ha="right", fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylim(0.5, max(dem_p.max(), 10.0) * 1.6)
        ax.set_xlim(0, t_hi)
        ax.set_ylabel(f"{label}\nGB/s（对数轴）", fontsize=9.5)
        ax.grid(alpha=.2, which="both")
        if ri == 0:
            ax.legend(loc="lower right", fontsize=7.5)
        if ri == 4:
            ax.set_xlabel("仿真时间 (s)", fontsize=10)
        w = arr[:, 1] - arr[:, 0]
        over = float(w[arr[:, 3] > B_GBPS].sum() / w.sum())
        sat = float(w[arr[:, 2] >= 0.9 * B_GBPS].sum() / w.sum())
        ax.set_title(f"需求超订 {over:.0%}｜实际饱和(≥0.9B) {sat:.0%}",
                     fontsize=8.5, loc="right")
    fig.suptitle(f"图B｜E26e 存储带宽时序：{tag}"
                 "（橙=Σ申请率·可超订 黑虚线=120 上限 蓝=Σ实际速率·恒≤上限；对数纵轴）",
                 fontsize=12.5, y=0.97)
    ov = audit_text_overlap(fig)
    if not audit_only:
        fig.savefig(os.path.join(FIG_DIR, f"cq_fig_e26e_bw_{tag}.png"),
                    dpi=125)
    plt.close(fig)
    print(f"{tag}: 带宽图文字重叠 {len(ov)} 对")
    return len(ov)


def build_bar(audit_only: bool):
    """汇总条形：5 策略 × 2 格点总 SLO；多 seed 格点中位数 ± min/max 带。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e25_ts_compare import audit_text_overlap

    records = []
    for p in sorted(glob.glob(os.path.join(ROOT, "records_part_*.json"))):
        records.extend(json.load(open(p, encoding="utf-8")))
    # (cell, policy) → 各 rep 的总 SLO
    vals = {}
    for r in records:
        vals.setdefault((r["cell"], r["policy"]), []).append(
            (r["rep"], r["slo_rate"], r["A"]["slo_rate"], r["B"]["slo_rate"]))
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=125)
    x = np.arange(len(POLICY_ORDER))
    w = 0.36
    labels = {"D1c_k4": "k=4（30 簇×4A）", "D1c_k16": "k=16（7×16+8 A）"}
    colors = {"D1c_k4": "#6A9FDA", "D1c_k16": "#2C5F9E"}
    med_a, med_b = {}, {}
    for ci, cell in enumerate(CELLS):
        meds, lo, hi, la, lb = [], [], [], [], []
        for pol in POLICY_ORDER:
            rows = sorted(vals.get((cell, pol), []))
            if not rows:
                meds.append(np.nan), lo.append(0.0), hi.append(0.0)
                la.append(np.nan), lb.append(np.nan)
                continue
            v = np.array([r[1] for r in rows])
            meds.append(float(np.median(v)))
            lo.append(float(np.median(v) - v.min()))
            hi.append(float(v.max() - np.median(v)))
            la.append(float(np.median([r[2] for r in rows])))
            lb.append(float(np.median([r[3] for r in rows]))
                      if rows[0][3] is not None else np.nan)
        off = (ci - 0.5) * w
        ax.bar(x + off, meds, w, yerr=[lo, hi], capsize=3,
               color=colors[cell], label=labels[cell] +
               (f"（n={len(rows)} seed 中位数±min/max）" if len(rows) > 1 else ""))
        for xi, (m, a_, b_) in enumerate(zip(meds, la, lb)):
            if not np.isnan(m):
                ax.text(xi + off, m + 0.015, f"{m:.0%}", ha="center",
                        fontsize=8.5, color=colors[cell])
                ax.text(xi + off, m / 2, f"A {a_:.0%}\nB {b_:.0%}",
                        ha="center", va="center", fontsize=7.2, color="w")
        med_a[cell], med_b[cell] = la, lb
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_v2.1", "\nv2.1") for p in POLICY_ORDER],
                       fontsize=9.5)
    ax.set_ylabel("总 SLO 满足率", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("E26e 汇总：A 簇集中度 k∈{4,16} × 五策略总 SLO\n"
                 "（条内白字=分类 A/B SLO 中位数；D1c：1:1、λ=196、α=4、240 条）",
                 fontsize=11.5)
    fig.tight_layout()
    ov = audit_text_overlap(fig)
    if not audit_only:
        fig.savefig(os.path.join(FIG_DIR, "cq_fig_e26e_bar.png"), dpi=125)
    plt.close(fig)
    print(f"bar: 文字重叠 {len(ov)} 对")
    return len(ov)


def build_util(cell: str, rep: int, audit_only: bool):
    """逐 NPU 利用率：箱线（32 NPU 分布）+ 热图（编号维度）双 panel。

    口径（20261011 用户裁定）：NPU 利用率 = 计算时间(A算+B算)/makespan
    ——只有计算算占用；IO 等（STALL，数据未到位计算单元空转）与空闲均
    不计。含等口径（算+等）仅作 stdout 对照：它度量"worker 槽位被派活"，
    会把 FCFS 的红海伪装成高占用。均衡度量：逐 NPU 利用率分布的
    min/p50/p90/max + CV=σ/μ（CV 越小越均衡），不用全局聚合（无法区分
    "均匀忙"与"少数忙多数闲"）；热图保留编号维度。makespan 取自 records
    （gantt 的 T_anchor 是首臂口径，非首臂不可用）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e25_ts_compare import audit_text_overlap

    tag = f"{cell}@s{rep}"
    # makespan 逐臂取自 records（gantt 的 T_anchor 是首臂口径，非首臂不可用）
    recs = []
    for pat in (os.path.join(ROOT, "records_part_*.json"),
                os.path.join(ROOT, "archive", "records_part_*.json")):
        for p in sorted(glob.glob(pat)):
            recs.extend(json.load(open(p, encoding="utf-8")))
    mk = {r["policy"]: r["makespan_s"] for r in recs
          if r["cell"] == cell and r["rep"] == rep}
    stats = {}
    makespans = {}
    for _label, fn in ROWS:
        p = os.path.join(ROOT, tag, f"{fn}.gantt.json.gz")
        g = json.load(gzip.open(p))
        m = mk.get(fn) or g["meta"]["T_anchor_s"] / 1.1
        makespans[fn] = m
        acc_disp = np.zeros(g["meta"]["m_workers"])   # 含等（对照）
        acc_c = np.zeros(g["meta"]["m_workers"])      # 纯算（主口径）
        for bk in g["buckets"]:
            for wid, (a, b, w, _idle) in enumerate(bk):
                acc_disp[wid] += a + b + w
                acc_c[wid] += a + b
        stats[fn] = (acc_c / m, acc_disp / m)
    labels = [l for l, _ in ROWS]
    data = [stats[fn][0] * 100 for _l, fn in ROWS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.6), dpi=125,
                                   gridspec_kw={"width_ratios": [1, 1.6]})
    bp = ax1.boxplot(data, tick_labels=[l.replace(" v2.1", "\nv2.1") for l in labels],
                     showfliers=True, patch_artist=True, widths=.55,
                     medianprops=dict(color="k"))
    for patch, (_l, fn) in zip(bp["boxes"], ROWS):
        patch.set_facecolor("#9DC3E6" if "θ=S" not in _l else "#2C5F9E")
        patch.set_alpha(.75)
    means = [np.mean(x) for x in data]
    ax1.scatter(range(1, 6), means, marker="D", s=26, color="#C44E52",
                zorder=3, label="均值")
    ax1.set_ylabel("逐 NPU 利用率（计算/makespan）%", fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=.25)
    ax1.legend(fontsize=8.5)
    cvs = [f"CV={np.std(x)/np.mean(x):.2f}" for x in data]
    ax1.set_title("32 NPU 利用率分布（CV 越小越均衡）\n"
                  + "｜".join(f"{l.split(' ')[0]}{c[2:]}" for l, c in
                              zip(labels, cvs)), fontsize=8.6)
    mat = np.array(data)
    im = ax2.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xticks([0, 7, 15, 23, 31])
    ax2.set_xlabel("NPU 编号", fontsize=10)
    ax2.set_title("利用率热图（亮=高）——尾部集中在低编号即此处显形",
                  fontsize=9)
    fig.colorbar(im, ax=ax2, label="利用率 %")
    fig.suptitle(f"E26e 逐 NPU 利用率：{tag}"
                 f"（利用率=计算/makespan，IO 等≠占用；makespan "
                 f"{', '.join(f'{fn.split(chr(95))[0]}={makespans[fn]:.2f}s' for _l, fn in ROWS)}）",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ov = audit_text_overlap(fig)
    if not audit_only:
        fig.savefig(os.path.join(FIG_DIR, f"cq_fig_e26e_util_{tag}.png"),
                    dpi=125)
    plt.close(fig)
    print(f"{tag}: util 图文字重叠 {len(ov)} 对")
    for (_l, fn), (c, d) in zip(ROWS, stats.values()):
        print(f"  {_l:12s} 利用率(纯算) min/p50/p90/max="
              f"{c.min():.0%}/{np.median(c):.0%}/{np.quantile(c,.9):.0%}/"
              f"{c.max():.0%} CV={c.std()/c.mean():.2f} 均值={c.mean():.0%}"
              f"｜含等对照均值={d.mean():.0%} makespan={makespans[fn]:.3f}s")
    return len(ov)


def build_cdf(cell: str, rep: int, audit_only: bool):
    """TTFT 经验 CDF（每方案一曲线，全 cohort 240 条；k4 用 rep0 代表）。

    数据源：requests json.gz（rid, cls, arrival, deadline, F_s, ttft）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    from e25_ts_compare import audit_text_overlap

    tag = f"{cell}@s{rep}"
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=125)
    colors = ["#888888", "#C44E52", "#DD8452", "#4C72B0", "#2C5F9E"]
    for (label, fn), col in zip(ROWS, colors):
        p = os.path.join(ROOT, tag, f"{fn}.requests.json.gz")
        if not os.path.exists(p):
            print(f"缺 {p}")
            continue
        rows = json.load(gzip.open(p))["rows"]
        tt = np.sort([r[5] for r in rows])
        y = (np.arange(1, len(tt) + 1)) / len(tt)
        ax.plot(tt, y, lw=1.6, color=col,
                label=f"{label}（p50={np.quantile(tt,.5):.2f}s "
                      f"p95={np.quantile(tt,.95):.2f}s）")
        d = [r[3] - r[2] for r in rows]        # deadline−arrival=期限宽度
        ax.plot(np.sort(d), y, lw=.7, ls=":", color=col, alpha=.55)
    ax.set_xlabel("TTFT (s)（实线=实际 TTFT；同色点线=期限宽度 deadline−arrival，"
                  "TTFT 曲线在其左侧的部分=SLO 内）", fontsize=9.5)
    ax.set_ylabel("累计分布 CDF", fontsize=10.5)
    ax.set_xlim(0, None)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title(f"E26e TTFT 经验 CDF：{tag}（全 240 条请求；k4 画 rep0 代表）",
                 fontsize=11)
    fig.tight_layout()
    ov = audit_text_overlap(fig)
    if not audit_only:
        fig.savefig(os.path.join(FIG_DIR, f"cq_fig_e26e_cdf_{tag}.png"),
                    dpi=125)
    plt.close(fig)
    print(f"{tag}: CDF 图文字重叠 {len(ov)} 对")
    return len(ov)


if __name__ == "__main__":
    kind, cell, rep, audit = "bar", None, 0, False
    for a in sys.argv[1:]:
        if a.startswith("--kind="):
            kind = a[len("--kind="):]
        elif a.startswith("--cell="):
            cell = a[len("--cell="):]
        elif a.startswith("--rep="):
            rep = int(a[len("--rep="):])
        elif a == "--audit":
            audit = True
    if kind == "gantt":
        build_gantt(cell or CELLS[-1], rep, audit)
    elif kind == "bw":
        build_bw(cell or CELLS[-1], rep, audit)
    elif kind == "util":
        build_util(cell or CELLS[-1], rep, audit)
    elif kind == "cdf":
        build_cdf(cell or CELLS[-1], rep, audit)
    else:
        build_bar(audit)
