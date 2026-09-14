"""E23：Mooncake 主收益、负载与 SLO（§9.6）。

主矩阵（正式）：三文件 × B∈{20,80,320} × ρ∈{0.3,0.6,0.9,1.1} × α∈{2,4,8}
× theta∈{S,T,R} × 15 评价块 × 8 策略；冒烟为子集（训练块 0/1、
--duration 截断、4 策略）。配对收益相对 FCFS 与 B*（训练冻结）。
图 fig_e23_gain_overview.png / fig_e23_slo_load.png / fig_e23_backlog.png。
"""
from __future__ import annotations

import os
from fractions import Fraction as F

import numpy as np

from sim.experiments.cq_common import (ALL_POLICIES, TRACE_DIR_DEFAULT,
                                          TRACE_WIDE_LIMITS, MooncakeSource,
                                          build_policy, c_ref_of,
                                          default_scenario, out_dir,
                                          print_progress, run_one, save_json,
                                          setup_matplotlib)
from sim.cq.metrics import (latency_gain, moving_block_bootstrap,
                            paired_median, slo_gain_pp)
from sim.cq.trace import MOONCAKE_FILES

THETA_MAIN = {"S": "slo_success", "T": "ttft_mean_lower",
              "R": "ttft_norm_mean_lower"}


def select_b_star(train_recs, theta):
    """B*：训练格点等权平均名次最小的简单策略（§3.3 简化实现）。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in train_recs:
        if r["theta"] != theta:
            continue
        groups[(r["file"], r["B"], r["rho"], r["alpha"], r["cost_mode"])].append(r)
    names = defaultdict(float)
    counts = defaultdict(int)
    for _k, rs in groups.items():
        key = THETA_MAIN[theta]
        ranked = sorted(rs, key=lambda r: (r[key] if r[key] is not None
                                           else float("inf")))
        for i, r in enumerate(ranked):
            names[r["policy"]] += i
            counts[r["policy"]] += 1
    avg = {p: names[p] / counts[p] for p in names if p != "cq_fcfs"}
    if not avg:
        return "cq_fcfs", "selection_unresolved"
    best = min(avg, key=lambda p: (avg[p], p))
    return best, "ok"


def paired_gains(recs, base_pid, cmp_pid, theta):
    """配对收益（同 file/B/rho/alpha/block 配对）。"""
    from collections import defaultdict
    idx = defaultdict(dict)
    for r in recs:
        idx[(r["file"], r["B"], r["rho"], r["alpha"], r["block"])][r["policy"]] = r
    lat, slo = [], []
    for _k, pols in idx.items():
        if base_pid not in pols or cmp_pid not in pols:
            continue
        b, p = pols[base_pid], pols[cmp_pid]
        if theta == "S":
            slo.append(slo_gain_pp(b["slo_success"] / b["n_cohort"],
                                   p["slo_success"] / p["n_cohort"]))
        else:
            key = THETA_MAIN[theta]
            g = latency_gain(b[key], p[key])
            if g is not None:
                lat.append(g)
    if theta == "S":
        return {"median": paired_median(slo), "values": slo,
                "metric": "slo_gain_pp"}
    return {"median": paired_median(lat), "values": lat, "metric": "latency_gain_pct"}


def run_matrix(stage, trace_dir, duration, thetas, policies, files, blocks,
               rho_list, B_list, alpha=4.0, cost_modes=("zero",)):
    recs = []
    for fname in files:
        src = MooncakeSource(fname, trace_dir)
        lam0 = src.lam0
        for B in B_list:
            for rho in rho_list:
                lam = lam0 * F(rho).limit_denominator(10**9)
                scn = default_scenario(B_gbps=B, alpha=F(int(alpha)),
                                       limits=TRACE_WIDE_LIMITS)
                for blk in blocks:
                    specs, d_sim = src.window_specs(blk, lam, F(int(alpha)),
                                                    duration_cap=duration)
                    if len(specs) < 16:
                        continue
                    c_ref = c_ref_of(specs, scn)
                    warmup = 0.1 * d_sim if d_sim else 0.0
                    for cm in cost_modes:
                        scn2 = default_scenario(B_gbps=B, alpha=F(int(alpha)),
                                                cost_mode=cm,
                                                limits=TRACE_WIDE_LIMITS)
                        for th in thetas:
                            for pid in policies:
                                s = run_one(scn2, specs, pid, th, c_ref,
                                            seed=blk, H=1)
                                s.update({"file": fname, "B": B, "rho": rho,
                                          "alpha": alpha, "block": blk,
                                          "n_req": len(specs), "d_sim": d_sim})
                                recs.append(s)
                    print_progress(
                        f"E23 {fname[:12]} B={B} ρ={rho} blk={blk}: "
                        f"{len(specs)} reqs")
    return recs


def main(seeds, procs=None, duration=150.0, stage="smoke",
         trace_dir=TRACE_DIR_DEFAULT, **kw):
    plt = setup_matplotlib()
    d = out_dir(stage, "e23")
    if stage == "smoke":
        files = ["conversation_trace.jsonl", "synthetic_trace.jsonl"]
        policies = ["cq_fcfs", "cq_spt", "cq_slack", "cq_mpc"]
        thetas = ["S", "T"]
        rho_list = [0.6, 0.9]
        B_list = [20.0, 80.0]
        blocks = [0, 1]
        cost_modes = ("zero",)
    else:
        files = [f for f, _n, _m, _s in MOONCAKE_FILES]
        policies = ALL_POLICIES
        thetas = ["S", "T", "R"]
        rho_list = [0.3, 0.6, 0.9, 1.1]
        B_list = [20.0, 80.0, 320.0]
        blocks = [100 + i for i in range(15)]
        cost_modes = ("zero",)
    recs = run_matrix(stage, trace_dir, duration, thetas, policies, files,
                      blocks, rho_list, B_list, cost_modes=cost_modes)
    save_json(os.path.join(d, "e23_records.json"), recs)
    # B*（训练集内冻结）
    bstars = {}
    for th in thetas:
        bstars[th], note = select_b_star(recs, th)
        bstars[th] = (bstars[th], note)
    save_json(os.path.join(d, "e23_bstar.json"), bstars)
    # 配对收益
    gains = {}
    for th in thetas:
        for pid in policies:
            if pid == "cq_fcfs":
                continue
            gains[f"{pid}|vs_fcfs|{th}"] = paired_gains(recs, "cq_fcfs", pid, th)
            bs = bstars[th][0]
            if bs != pid and bs != "cq_fcfs":
                gains[f"{pid}|vs_B*|{th}"] = paired_gains(recs, bs, pid, th)
    save_json(os.path.join(d, "e23_gains.json"), gains)

    # 图 1：收益总览（相对 FCFS 配对中位数）
    fig, axes = plt.subplots(1, len(thetas), figsize=(6 * len(thetas), 4.4),
                             squeeze=False)
    for ti, th in enumerate(thetas):
        ax = axes[0, ti]
        for pid in [p for p in policies if p != "cq_fcfs"]:
            g = gains.get(f"{pid}|vs_fcfs|{th}")
            if g and g["median"] is not None:
                ax.bar(pid.replace("cq_", ""), g["median"],
                       color="#4C72B0" if g["median"] >= 0 else "#C44E52")
        ax.axhline(0, color="k", lw=.8)
        ax.set_title(f"theta={th}：" +
                     ("SLO 提升 (pp)" if th == "S" else "时延降低 (%)"))
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("E23：相对 FCFS 的配对中位数收益（冒烟=训练块，非正式评价）"
                 if stage == "smoke" else "E23：相对 FCFS 的配对中位数收益")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e23_gain_overview.png"), dpi=130)
    plt.close(fig)

    # 图 2：SLO 负载曲线 + 图 3 backlog 代理（队列规模）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for fname in files:
        sub = [r for r in recs if r["file"] == fname and r["theta"] == "S"
               and r["policy"] == "cq_fcfs" and r["B"] == B_list[0]]
        if not sub:
            continue
        xs = sorted({r["rho"] for r in sub})
        ys = []
        for rho in xs:
            ss = [r["slo_success"] / max(1, r["n_cohort"])
                  for r in sub if r["rho"] == rho]
            ys.append(float(np.mean(ss)) if ss else np.nan)
        axes[0].plot(xs, ys, marker="o", label=fname.split("_")[0][:8])
    axes[0].set_xlabel("ρ（负载 / λ0）"); axes[0].set_ylabel("FCFS SLO 满足率")
    axes[0].set_title(f"SLO-负载曲线（B={B_list[0]:g}，冒烟训练块）")
    axes[0].set_ylim(0, 1.02); axes[0].legend()
    for pid in policies:
        sub = [r for r in recs if r["theta"] == "T" and r["policy"] == pid]
        if sub:
            axes[1].scatter([pid.replace("cq_", "")] * len(sub),
                            [r["ttft_mean_lower"] for r in sub], s=12, alpha=.6)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("平均 TTFT 下界 (s)")
    axes[1].set_title("各策略 TTFT 分布")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e23_slo_load.png"), dpi=130)
    plt.close(fig)
    print_progress(f"E23 done -> {d} ({len(recs)} runs)")
    return {"n_runs": len(recs), "b_star": bstars}


if __name__ == "__main__":
    main([0])
