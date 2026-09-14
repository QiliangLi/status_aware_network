"""E24：信息、搜索、能力与误差各贡献多少（§9.7，一因素扫描子集）。

面板：信息来源（history/current/oracle）、观测更新周期、交付滞后、控制开销
（fixed 延迟扫描与 measured）。锚点未训练时用预定 (B=20, ρ=0.9) 负面诊断。
图 fig_e24_robustness.png。
"""
from __future__ import annotations

import os
from dataclasses import replace
from fractions import Fraction as F

import numpy as np

from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                          MooncakeSource, build_policy,
                                          c_ref_of, default_scenario, out_dir,
                                          print_progress, run_one, save_json,
                                          setup_matplotlib)
from sim.cq.policies import make_fcfs


def e24_runs(fname, trace_dir, duration=150.0, block=0, rho=0.9, B=20.0,
             theta="S"):
    src = MooncakeSource(fname, trace_dir)
    lam = src.lam0 * F(rho).limit_denominator(10**9)
    specs, d_sim = src.window_specs(block, lam, F(4), duration_cap=duration)
    if len(specs) < 16:
        return []
    recs = []
    panels = []

    def add(panel, scn, pid, extra=None):
        c_ref = c_ref_of(specs, scn)
        s = run_one(scn, specs, pid, theta, c_ref, seed=block, H=1)
        s.update({"panel": panel, "file": fname, "B": B, "rho": rho,
                  "policy": pid, **(extra or {})})
        recs.append(s)

    # 面板 1：信息来源（history/current/oracle），同能力同预算
    for mode in ("history", "current"):
        base = default_scenario(B_gbps=B, limits=TRACE_WIDE_LIMITS)
        scn = replace(base, observation=replace(base.observation, mode=mode))
        for pid in ("cq_fcfs", "cq_slack", "cq_mpc"):
            add("info", scn, pid, {"obs_mode": mode})
    # oracle 诊断走 MPC 同构（使用真值带宽重放：以 current+σ=0 近似上界通道）
    scn_o = base
    for pid in ("cq_mpc",):
        add("info", scn_o, pid, {"obs_mode": "current_oracle"})

    # 面板 2：观测更新周期
    base = default_scenario(B_gbps=B, limits=TRACE_WIDE_LIMITS)
    for period in ("0", "0.005", "0.020", "0.100"):
        p = 0.0 if period == "0" else float(period)
        obs = replace(base.observation, sample_period_s=F(int(p * 1000), 1000))
        scn = replace(base, observation=obs)
        for pid in ("cq_mpc",):
            add("sample_period", scn, pid, {"sample_period_s": p})

    # 面板 3：控制开销（fixed 延迟扫描）
    for delay in (0.0, 0.0001, 0.0005, 0.002, 0.005):
        scn = replace(base, controller=replace(
            base.controller, cost_mode="fixed",
            fixed_delay_s=F(int(delay * 10**6), 10**6)))
        for pid in ("cq_fcfs", "cq_mpc"):
            add("ctrl_cost", scn, pid, {"delay_s": delay})

    # 面板 4：搜索深度 H
    for H in (1, 2):
        scn = base
        c_ref = c_ref_of(specs, scn)
        s = run_one(scn, specs, "cq_mpc", theta, c_ref, seed=block, H=H)
        s.update({"panel": "search_H", "file": fname, "B": B, "rho": rho,
                  "policy": "cq_mpc", "H": H})
        recs.append(s)
    return recs


def main(seeds, procs=None, duration=150.0, stage="smoke",
         trace_dir=TRACE_DIR_DEFAULT, **kw):
    plt = setup_matplotlib()
    d = out_dir(stage, "e24")
    files = ["conversation_trace.jsonl"] if stage == "smoke" else [
        "conversation_trace.jsonl", "toolagent_trace.jsonl",
        "synthetic_trace.jsonl"]
    recs = []
    for fname in files:
        recs.extend(e24_runs(fname, trace_dir, duration))
        print_progress(f"E24 {fname[:16]} done ({len(recs)} runs)")
    save_json(os.path.join(d, "e24_records.json"), recs)

    # 图：四面板
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    def _sl(r):
        return r["slo_success"] / max(1, r["n_cohort"])

    # 面板 1 信息
    ax = axes[0]
    modes = [r["obs_mode"] for r in recs if r["panel"] == "info"]
    for pid in ("cq_fcfs", "cq_slack", "cq_mpc"):
        xs, ys = [], []
        for m in ("history", "current", "current_oracle"):
            sub = [r for r in recs if r["panel"] == "info"
                   and r.get("obs_mode") == m and r["policy"] == pid]
            if sub:
                xs.append(m.replace("current_oracle", "oracle*"))
                ys.append(float(np.mean([_sl(r) for r in sub])))
        if xs:
            ax.plot(xs, ys, marker="o", label=pid)
    ax.set_title("信息来源 → SLO 满足率")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    # 面板 2 更新周期
    ax = axes[1]
    sub = sorted([r for r in recs if r["panel"] == "sample_period"],
                 key=lambda r: r["sample_period_s"])
    if sub:
        ax.plot([r["sample_period_s"] for r in sub],
                [_sl(r) for r in sub], marker="s", color="#4C72B0")
    ax.set_xlabel("采样周期 (s)"); ax.set_ylabel("SLO 满足率")
    ax.set_title("观测更新周期（MPC）")
    # 面板 3 控制开销
    ax = axes[2]
    for pid in ("cq_fcfs", "cq_mpc"):
        sub = sorted([r for r in recs if r["panel"] == "ctrl_cost"
                      and r["policy"] == pid], key=lambda r: r["delay_s"])
        if sub:
            ax.plot([r["delay_s"] * 1000 for r in sub], [_sl(r) for r in sub],
                    marker="o", label=pid)
    ax.set_xlabel("控制延迟 (ms)"); ax.set_ylabel("SLO 满足率")
    ax.set_title("控制开销敏感性")
    ax.legend(fontsize=8)
    # 面板 4 搜索深度
    ax = axes[3]
    sub = sorted([r for r in recs if r["panel"] == "search_H"],
                 key=lambda r: r["H"])
    if sub:
        ax.plot([r["H"] for r in sub], [_sl(r) for r in sub], marker="o",
                color="#55A868")
    ax.set_xlabel("H（滚动深度）"); ax.set_ylabel("SLO 满足率")
    ax.set_title("搜索深度")
    fig.suptitle("E24：一因素扫描（锚点 B=20、ρ=0.9、α=4、theta=S、训练块、"
                 "zero 成本；oracle* 为诊断通道）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e24_robustness.png"), dpi=130)
    plt.close(fig)
    print_progress(f"E24 done -> {d}")
    return {"n_runs": len(recs)}


if __name__ == "__main__":
    main([0])
