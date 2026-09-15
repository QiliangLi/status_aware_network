"""E21：组批收益和同步损失（§9.4）。

Mooncake 形状驱动的有限工作集：每块前 128 条、arrival=0、deadline=α·T0。
能力块 A singleton（n_max=1）/ B homogeneous（同 h-u 中位数象限）/ C 自由。
profile 对照：默认 synthetic-v1 vs 无合批加速（η≡1、launch=0）。
图 fig_e21_batch_tradeoff.png：计算秒下降与短请求 TTFT 同屏。
"""
from __future__ import annotations

import os
from fractions import Fraction as F

import numpy as np

from sim.cq.config import (CqScenario, ControllerConfig, ProfileConfig,
                           StorageConfig)
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, MooncakeSource,
                                          build_policy, c_ref_of,
                                          default_scenario, out_dir,
                                          print_progress, save_json,
                                          setup_matplotlib)
from sim.cq.metrics import summarize, request_rows
from sim.cq.policies import SIMPLE_POLICIES
from sim.cq.profile import batch_compute_s, make_T0
from sim.cq.simrun import run_case
from sim.cq.trace import MOONCAKE_FILES
from sim.cq.types import HardLimits, RequestSpec


def finite_workload(src: MooncakeSource, block_id: int, n_req: int = 128,
                    alpha: float = 4.0):
    """Mooncake 形状驱动有限工作集：前 n 条、arrival=0、deadline=α·T0。"""
    blk = src.windows[block_id]   # 统一窗口(0..19,全量打分)
    rows = [r for r in src.imp.rows if blk.start_ms <= r.timestamp_ms < blk.end_ms][:n_req]
    prof = ProfileConfig()
    specs = []
    for i, r in enumerate(rows):
        h, u, _full = src.imp.h_u_of(r)
        specs.append(RequestSpec(rid=i, arrival_s=0.0, h_tokens=h, u_tokens=u,
                                 class_id="mooncake", T0_s=1, deadline_s=1,
                                 source_file=src.fname, source_line=r.source_line,
                                 input_length=r.input_length))
    T0 = make_T0(specs, prof, F(80), F(200))
    out = [RequestSpec(s.rid, 0.0, s.h_tokens, s.u_tokens, s.class_id, T0[s.rid],
                       alpha * T0[s.rid], source_file=s.source_file,
                       source_line=s.source_line, input_length=s.input_length)
           for s in specs]
    return out


def quadrant_guard(specs_by_rid, med_h, med_u):
    """E21-B：同批成员必须同属 (h≤med_h, u≤med_u) 四象限之一。"""
    def extra(members):
        quads = {(s.h_tokens <= med_h, s.u_tokens <= med_u) for s in members}
        return len(quads) == 1
    return extra


def run_capability_block(specs, capability: str, profile: ProfileConfig,
                         policies, theta="T", B=80.0, alpha=4.0):
    """一个能力块 × 一个 profile 下的策略对比。"""
    med_h = float(np.median([s.h_tokens for s in specs]))
    med_u = float(np.median([s.u_tokens for s in specs]))
    if capability == "A_singleton":
        limits = HardLimits(1, 262144, F(128))
        extra = None
    elif capability == "B_homogeneous":
        limits = HardLimits(8, 262144, F(128))
        by = {s.rid: s for s in specs}
        extra = quadrant_guard(by, med_h, med_u)
    else:
        limits = HardLimits(8, 262144, F(128))
        extra = None
    scn = default_scenario(B_gbps=B, limits=limits, profile=profile)
    c_ref = c_ref_of(specs, scn)
    recs = []
    for pid in policies:
        pol = build_policy(pid, theta, c_ref, H=1)
        if extra is not None and pid in SIMPLE_POLICIES:
            pol.feasibility_extra = extra
        eng = run_case(scn, specs, pol, numeric=float)
        s = summarize(eng, scn)
        s["policy"] = pid
        # 短/长 u 分组 TTFT（按训练中位数）
        rows = request_rows(eng)
        short = [r for r in rows if r["u_tokens"] <= med_u]
        long_ = [r for r in rows if r["u_tokens"] > med_u]
        s["ttft_short_mean"] = float(np.mean([r["F_s"] for r in short if r["F_s"]])) if short else None
        s["ttft_long_mean"] = float(np.mean([r["F_s"] for r in long_ if r["F_s"]])) if long_ else None
        s["n_short"] = len(short)
        recs.append(s)
    return recs


def main(seeds, procs=None, duration=150.0, stage="smoke",
         trace_dir=TRACE_DIR_DEFAULT, **kw):
    plt = setup_matplotlib()
    d = out_dir(stage, "e21")
    n_req = 32 if stage == "smoke" else 128
    policies = (["cq_fcfs", "cq_spt", "cq_edf", "cq_mpc"]
                if stage == "smoke" else
                ["cq_fcfs", "cq_lpm", "cq_spt", "cq_edf", "cq_slack",
                 "cq_pair", "cq_local", "cq_mpc"])
    files = ["conversation_trace.jsonl", "synthetic_trace.jsonl"] if stage == "smoke" \
        else [f for f, _n, _m, _s in MOONCAKE_FILES]
    all_recs = []
    for fname in files:
        src = MooncakeSource(fname, trace_dir)
        blocks = [0, 1] if stage == "smoke" else [0, 1, 2, 3, 4]
        for blk in blocks:
            specs = finite_workload(src, blk, n_req)
            if len(specs) < 8:
                continue
            for cap in ("A_singleton", "B_homogeneous", "C_unrestricted"):
                for prof_name, prof in (("default", ProfileConfig()),
                                         ("no_speedup",
                                          ProfileConfig(no_speedup=True))):
                    recs = run_capability_block(specs, cap, prof, policies)
                    for r in recs:
                        r.update({"file": fname, "block": blk,
                                  "capability": cap, "profile": prof_name,
                                  "n_req": len(specs)})
                    all_recs.extend(recs)
    save_json(os.path.join(d, "e21_records.json"), all_recs)

    # 图：C 块 default profile 下，计算秒 vs 短请求 TTFT；能力块对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    sub = [r for r in all_recs
           if r["capability"] == "C_unrestricted" and r["profile"] == "default"
           and r["file"] == files[0]]
    if sub:
        pols = sorted({r["policy"] for r in sub})
        comp = [np.mean([r["device_compute_s"] for r in sub if r["policy"] == p])
                for p in pols]
        tS = [np.mean([r["ttft_short_mean"] or 0 for r in sub if r["policy"] == p])
              for p in pols]
        axes[0].bar(pols, comp, color="#4C72B0")
        axes[0].set_ylabel("设备计算秒（越低=组批收益越大）")
        axes[0].set_title(f"{files[0]} C 块：计算节省")
        axes[1].bar(pols, tS, color="#DD8452")
        axes[1].set_ylabel("短请求平均 TTFT (s)")
        axes[1].set_title("短请求同步损失")
        axes[1].tick_params(axis="x", rotation=30)
        # 计算秒 vs 短 TTFT 散点（权衡地图）
        for p, c, t in zip(pols, comp, tS):
            axes[2].scatter(c, t, s=60)
            axes[2].annotate(p, (c, t), fontsize=8,
                             xytext=(4, 4), textcoords="offset points")
        axes[2].set_xlabel("设备计算秒"); axes[2].set_ylabel("短请求 TTFT (s)")
        axes[2].set_title("组批权衡地图（左下=理想）")
    fig.suptitle("E21：Mooncake 形状驱动有限工作集（α=4, B=80, zero 成本）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e21_batch_tradeoff.png"), dpi=130)
    plt.close(fig)
    print_progress(f"E21 done -> {d} ({len(all_recs)} runs)")
    return {"n_runs": len(all_recs)}


if __name__ == "__main__":
    main([0])
