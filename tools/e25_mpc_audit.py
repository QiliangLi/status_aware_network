"""E25 MPC 收益审计：回答"MPC 相较基线到底有没有收益、为什么此前为零"。

三个根因的验证与量化（20260917）：
  R1 2ms CPU 软预算 < 单次基础排空推演成本 → 候选零评分（n_scored=0）；
  R2 forecast_drain 此前丢弃 WAIT（已修复：_HoldPi 保持唤醒前的空闲）；
  R3 cq_local 的"独立带宽"为占位实现（本审计仅记录，不修）。
实验：小世界（四计划金标）+ 真实面板（3 文件 × B{20,80} × ρ0.9 × θ{T,S}
× 窗口9），对比 fcfs / edf / mpc_std(2ms) / mpc_big(预算≈事件上限)。
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.getcwd())

from sim.cq.search import (MPCPolicy, build_forecast_engine, forecast_drain)
from sim.cq.simrun import run_case
from sim.cq.metrics import summarize
from sim.cq.types import Action, JointAction
from sim.experiments.cq_common import (TRACE_WIDE_LIMITS, MooncakeSource,
                                       build_policy, c_ref_of,
                                       default_scenario)
from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def miniature():
    """四计划金标小世界：大预算下 MPC 是否偏离 π 并改善目标。"""
    from tests.cq_reference import e20_scenario, e20_specs
    print("=== A. 小世界（四计划金标，2 实例/2 层/2S+2L，B=4） ===")
    for theta in ("T", "M", "S"):
        for name, kw in (("std(2ms)", {}), ("big(1s)", {"budget_s": 1.0})):
            pol = MPCPolicy(pid="m", H=1, theta=theta, c_ref=3.625, **kw)
            eng = run_case(e20_scenario(), e20_specs(), pol)
            Fs = [float(rr.F_s) for rr in eng.w.requests.values()]
            waits = sum(1 for d in eng.decisions
                        if any(a["kind"] == "WAIT" for a in d["action"]))
            print(f"  θ={theta} {name:9s}: F={[round(x,3) for x in sorted(Fs)]} "
                  f"Σ={sum(Fs):.3f} M={max(Fs):.3f} scored={pol.n_scored} "
                  f"overrun={pol.n_overrun} WAIT决策={waits}")


def wait_hold_unit():
    """WAIT 保持单元验证：单实例单请求，WAIT δ → F = δ + 独占服务时间。"""
    from tests.cq_reference import e20_scenario, e20_specs
    from sim.cq.engine import CqEngine
    from sim.cq.observable import Observable
    from sim.cq.policies import GuardedEDF
    scn = e20_scenario(m=1)
    specs = [e20_specs(n_long=0, n_short=1)[0]]
    obs = Observable(scn, numeric=float)
    eng = CqEngine(scn, specs, GuardedEDF(), numeric=float,
                   fallback_policy=GuardedEDF(), observable=obs)
    obs.attach(eng)
    eng._physical_step(0.0)
    snap = obs.snapshot(0.0)
    est = build_forecast_engine(snap, scn, GuardedEDF())
    Fs = forecast_drain(est, JointAction((Action("WAIT", 0, (), 0.725),)),
                        [20000])
    F_wait = Fs[0]
    est2 = build_forecast_engine(snap, scn, GuardedEDF())
    Fs2 = forecast_drain(est2, None, [20000])   # 无首动作：π 立即派发
    print(f"=== B. WAIT 保持单元：WAIT@0.725 → F={F_wait:.4f}（期望 0.725+2.25=2.975）；"
          f"立即派发 → F={Fs2[0]:.4f}（期望 2.25）===")
    assert abs(F_wait - 2.975) < 1e-9 and abs(Fs2[0] - 2.25) < 1e-9


def real_panel():
    """真实面板：fcfs / edf / mpc_std / mpc_big 同规格配对对比。"""
    print("\n=== C. 真实面板（窗口9、ρ=0.9、α=4、全窗口） ===")
    rows = []
    for fname, _n, _m, _s in MOONCAKE_FILES:
        src = MooncakeSource(fname)
        specs, d_sim = src.window_specs(9, src.lam0 * F(str(0.9)), F(4),
                                        duration_cap=None,
                                        limits=TRACE_WIDE_LIMITS)
        for B in (20.0, 80.0):
            scn = default_scenario(B_gbps=B, alpha=F(4),
                                   limits=TRACE_WIDE_LIMITS)
            c_ref = c_ref_of(specs, scn)
            for theta in ("T", "S"):
                for pid, pol in (
                        ("fcfs", build_policy("cq_fcfs", theta, c_ref, 1)),
                        ("edf", build_policy("cq_edf", theta, c_ref, 1)),
                        ("mpc_std", MPCPolicy(pid="mpc_std", H=1, theta=theta,
                                              c_ref=c_ref)),
                        ("mpc_big", MPCPolicy(pid="mpc_big", H=1, theta=theta,
                                              c_ref=c_ref, budget_s=30.0)),
                        ("local_big", MPCPolicy(pid="local_big", local=True, H=1,
                                                theta=theta, c_ref=c_ref,
                                                budget_s=30.0))):
                    eng = run_case(scn, specs, pol, numeric=float, seed=9)
                    s = summarize(eng, scn, arrival_stop=d_sim)
                    ctrl = sorted(d["ctrl_s"] for d in eng.decisions) or [0]
                    p95 = ctrl[max(0, int(0.95 * len(ctrl)) - 1)]
                    rows.append({
                        "file": TRACE_LABEL[fname], "B": B, "theta": theta,
                        "pid": pid, "n": s["n_cohort"],
                        "ttft": s["ttft_mean_lower"],
                        "slo": s["slo_success"] / s["n_cohort"],
                        "drainM": s.get("makespan_M_lower"),
                        "p50ctrl": 1000 * ctrl[len(ctrl) // 2],
                        "p95ctrl": 1000 * p95,
                        "scored": getattr(pol, "n_scored",
                                          getattr(pol, "n_scored", 0)),
                        "overrun": getattr(pol, "n_overrun", 0),
                        "decisions": len(eng.decisions),
                    })
                    r = rows[-1]
                    print(f"  {r['file'][:14]:15s} B{r['B']:g} θ={r['theta']} "
                          f"{r['pid']:8s} TTFT={r['ttft']:7.3f}s SLO={r['slo']:.3f} "
                          f"drainM={r['drainM']:7.1f}s ctrlP50/P95={r['p50ctrl']:.1f}/"
                          f"{r['p95ctrl']:.1f}ms scored={r['scored']} "
                          f"ovr={r['overrun']} dec={r['decisions']}")
    # 配对差值表
    print("\n=== D. 配对差（同 格点/θ；正=MPC_big 相对 EDF 改善） ===")
    idx = {}
    for r in rows:
        idx[(r["file"], r["B"], r["theta"], r["pid"])] = r
    for theta in ("T", "S"):
        for B in (20.0, 80.0):
            for f in sorted({r["file"] for r in rows}):
                try:
                    edf = idx[(f, B, theta, "edf")]
                    big = idx[(f, B, theta, "mpc_big")]
                    std = idx[(f, B, theta, "mpc_std")]
                except KeyError:
                    continue
                dt_big = 100 * (edf["ttft"] - big["ttft"]) / edf["ttft"]
                ds_big = 100 * (big["slo"] - edf["slo"])
                dt_std = 100 * (edf["ttft"] - std["ttft"]) / edf["ttft"]
                loc = idx[(f, B, theta, "local_big")]
                dt_loc = 100 * (edf["ttft"] - loc["ttft"]) / edf["ttft"]
                ds_loc = 100 * (loc["slo"] - edf["slo"])
                print(f"  θ={theta} B{B:g} {f[:14]:15s}: big vs edf "
                      f"TTFT {dt_big:+6.2f}% SLO {ds_big:+6.2f}pp | "
                      f"local vs edf TTFT {dt_loc:+6.2f}% SLO {ds_loc:+6.2f}pp | "
                      f"std vs edf TTFT {dt_std:+6.2f}%  "
                      f"(scored big/std/loc={big['scored']}/{std['scored']}/{loc['scored']})")


if __name__ == "__main__":
    miniature()
    wait_hold_unit()
    real_panel()
