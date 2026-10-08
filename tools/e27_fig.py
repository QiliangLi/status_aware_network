"""E27 图工具：例子电池（Ex1-4）与 E26c 四格点 eval 的入库图。

用法：
  .venv/bin/python tools/e27_fig.py            # 全部
  .venv/bin/python tools/e27_fig.py --only=ex  # 仅例子电池
数据源：results/cq/eval/e27/{records_examples,records_eval,e26c_reference}.json
输出：docs/figures/fig_e27_*.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "results", "cq", "eval", "e27")
FIG = os.path.join(REPO, "docs", "figures")

C_FORM, C_ROLL, C_MPC, C_FCFS, C_EDF, C_LOCAL, C_DPO = (
    "#2563eb", "#0891b2", "#dc2626", "#6b7280", "#9ca3af", "#f59e0b", "#16a34a")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fig_ex1(plt, ex):
    r = ex["ex1"]
    rows = [("CPA form", r["cpa_form"]["decide_ms"], C_FORM),
            ("CPA rollout", r["cpa_roll"]["decide_ms"], C_ROLL),
            ("MPC v2.1", r["mpc_v21"]["decide_ms"], C_MPC),
            ("DPO 穷举", r["dpo"]["wall_s"] * 1e3, C_DPO)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = [x[0] for x in rows]
    vals = [x[1] for x in rows]
    ax.bar(names, vals, color=[x[2] for x in rows], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:.2f} ms" if v < 10 else f"{v:.0f} ms",
                ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("单决策墙钟（ms，对数轴）")
    ax.set_title("Ex1 裁决对齐：4A+4B 突发（m=8），四方同选 (4A,4B)、同分 0 miss\n"
                 f"OPT 离线穷举同分（{r['opt']['best'][0]} miss），"
                 f"加速比 form:mpc = {r['mpc_v21']['decide_ms']/r['cpa_form']['decide_ms']:.0f}×")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e27_ex1_time.png"), dpi=150)
    plt.close(fig)


def fig_ex2(plt, ex):
    rows = ex["ex2"]
    ks = [r["k"] for r in rows]
    alive = [r["engine_alive"] for r in rows]
    cpa = [r["cpa_pick"] for r in rows]
    mpc = [r["mpc_pick"] for r in rows]
    dk = [r["k"] for r in rows if "dpo_pick" in r]
    dpo = [r["dpo_pick"] for r in rows if "dpo_pick" in r]
    kstar = rows[0]["formula_kstar"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ks, alive, "o-", color="#111827", label="引擎实测存活数（派出全部 k）")
    ax.plot(ks, cpa, "s--", color=C_FORM, label="CPA 首波派遣数")
    ax.plot(ks, mpc, "^--", color=C_MPC, label="MPC v2.1 首波派遣数")
    if dk:
        ax.plot(dk, dpo, "D", color=C_DPO, ms=6,
                label="DPO 决策点最优（n≤16）")
    ax.axvline(kstar, color=C_FORM, ls=":", alpha=0.7)
    ax.annotate(f"闭式存活阈值 k*={kstar}", (kstar, max(alive) * 0.55),
                rotation=90, fontsize=9, color=C_FORM, ha="right")
    ax.set_xlabel("突发规模 k（全部为 A 类，α=4，空管道同刻）")
    ax.set_ylabel("首波派出条数 / 存活数")
    ax.set_title("Ex2 生存阈值：CPA 派遣数 = min(k, k*) 与引擎存活逐点吻合；\n"
                 "MPC 候选网格 {4,8,∞} 在 k≥17 处回落到 8（锯齿），k=8..14 低于存活数")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e27_ex2_threshold.png"), dpi=150)
    plt.close(fig)


def fig_ex3(plt, ex):
    r = ex["ex3"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8),
                             gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    labels = ["CPA rollout", "MPC v2.1", "DPO 最优"]
    A = [r["cpa_roll"]["A"], r["mpc_v21"]["A"], r["dpo"]["cand"][0]]
    B = [r["cpa_roll"]["B"], r["mpc_v21"]["B"], r["dpo"]["cand"][1]]
    W = [r["cpa_roll"]["WAIT"], r["mpc_v21"]["WAIT"],
         32 - r["dpo"]["cand"][0] - r["dpo"]["cand"][1]]
    x = np.arange(3)
    ax.bar(x, A, 0.5, label="派 A", color=C_FORM)
    ax.bar(x, B, 0.5, bottom=A, label="派 B", color="#f59e0b")
    ax.bar(x, W, 0.5, bottom=np.array(A) + np.array(B), label="WAIT（扣住）",
           color="#d1d5db")
    for i in range(3):
        ax.text(i, 34.5, f"A={A[i]} B={B[i]}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("首波动作构成（32 worker）")
    ax.set_title("Ex3 让路（16A+16B 突发）：首波裁决对比")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax2 = axes[1]
    ax2.bar(["16A+16B\n同刻全派", "16A 先行\nB 延后 0.3s"],
            [r["engine_truth"]["same_moment_alive_A"],
             r["engine_truth"]["a_first_alive_A"]],
            color=["#dc2626", "#16a34a"], width=0.5)
    for i, v in enumerate([r["engine_truth"]["same_moment_alive_A"],
                           r["engine_truth"]["a_first_alive_A"]]):
        ax2.text(i, v + 0.4, f"{v}/16", ha="center", fontsize=10)
    ax2.set_ylabel("A 类存活数（引擎实测）")
    ax2.set_title("两种派法的物理结局（事实 4）")
    ax2.set_ylim(0, 18)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e27_ex3_wave.png"), dpi=150)
    plt.close(fig)


def fig_ex4(plt, ex):
    rows = ex["ex4"]
    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    ax.plot(ns, [r["cpa_form"]["slo"] / r["n"] * 100 for r in rows], "o-",
            color=C_FORM, label="CPA form（零推演）")
    ax.plot(ns, [r["cpa_roll"]["slo"] / r["n"] * 100 for r in rows], "s-",
            color=C_ROLL, label="CPA rollout（top-4）")
    ax.plot(ns, [r["fcfs"]["slo"] / r["n"] * 100 for r in rows], "^--",
            color=C_FCFS, label="FCFS")
    ax.set_xlabel("任务数 n（迷你 D2：两轮 (A,B,B)×n/6）")
    ax.set_ylabel("SLO 满足率 %")
    ax.set_title("Ex4 规模阶梯：SLO")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    ax2.plot(ns, [r["cpa_form"]["wall_s"] for r in rows], "o-", color=C_FORM,
             label="CPA form")
    ax2.plot(ns, [r["cpa_roll"]["wall_s"] for r in rows], "s-", color=C_ROLL,
             label="CPA rollout")
    ax2.plot(ns, [r["fcfs"]["wall_s"] for r in rows], "^--", color=C_FCFS,
             label="FCFS")
    ax2.set_yscale("log")
    ax2.set_xlabel("任务数 n")
    ax2.set_ylabel("整 run 墙钟（s，对数轴）")
    ax2.set_title("Ex4 规模阶梯：端到端墙钟")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle("Ex4（mini-D2，240 规模对照：E26c 同规模 MPC 整 run ≈ 10²~10³ s 量级）",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e27_ex4_scale.png"), dpi=150)
    plt.close(fig)


ARMS = [("FCFS", "fcfs", C_FCFS), ("EDF", "edf", C_EDF),
        ("local v2.1", "localS_v2.1", C_LOCAL), ("mpc v2.1", "mpcS_v2.1", C_MPC),
        ("CPA form", "cpa_form", C_FORM), ("CPA rollout", "cpa_roll", C_ROLL)]


def _join_eval(eval_recs, ref_recs):
    """格点×臂 → 记录（新 CPA 记录 + 既有 e26c 参照）。"""
    out = {}
    for r in ref_recs or []:
        if r.get("n_total") == 240:
            out.setdefault(r["cell"], {})[r["policy"]] = r
    for r in eval_recs or []:
        out.setdefault(r["cell"], {})[r["policy"]] = r
    return out


def fig_eval(plt, joined):
    cells = [c for c in ["D2_b1.0_a4", "D2_b0.75_a4", "D1_r1.1_a4", "D2c"]
             if c in joined]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(cells))
    w = 0.12
    for i, (label, key, col) in enumerate(ARMS):
        vals, walls = [], []
        for c in cells:
            r = joined[c].get(key)
            vals.append(100 * r["slo_rate"] if r else np.nan)
            walls.append(r["wall_s"] if r else np.nan)
        axes[0].bar(x + (i - 2.5) * w, vals, w, label=label, color=col)
        axes[1].bar(x + (i - 2.5) * w, walls, w, label=label, color=col)
        for j, v in enumerate(vals):
            if not np.isnan(v):
                axes[0].text(x[j] + (i - 2.5) * w, v + 1, f"{v:.0f}",
                             ha="center", fontsize=6.5, rotation=90)
    axes[0].set_ylabel("SLO 满足率 %")
    axes[0].set_title("E26c 四格点 240 规模：SLO（CPA 两臂 vs 既有四臂）")
    axes[0].set_xticks(x, cells)
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 115)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("整 run 墙钟（s，对数轴）")
    axes[1].set_title("同 CRN 端到端墙钟")
    axes[1].set_xticks(x, cells)
    axes[1].grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e27_eval.png"), dpi=150)
    plt.close(fig)


def main():
    only = ""
    for a in sys.argv[1:]:
        if a.startswith("--only="):
            only = a.split("=")[1]
    plt = _plt()
    ex = load("records_examples.json")
    if ex:
        fig_ex1(plt, ex)
        fig_ex2(plt, ex)
        fig_ex3(plt, ex)
        fig_ex4(plt, ex)
        print("examples figs done")
    if only != "ex":
        ev = load("records_eval.json")
        ref = load("e26c_reference.json")
        if ev:
            fig_eval(plt, _join_eval(ev, ref))
            print("eval fig done")


if __name__ == "__main__":
    main()
