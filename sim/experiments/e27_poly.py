"""E27：CPA（闭式管道接纳）多项式算法 vs MPC 暴力枚举+推演——差距度量。

设计文档：docs/CPA闭式管道接纳算法设计-20261008.md。
阶段（--stage）：
  examples 例子电池（A/B 构造例）：
    Ex1 裁决对齐（4A+4B, m=8）：CPA form/rollout vs MPC vs DPO vs OPT
        （动作、分数、单决策墙钟）；
    Ex2 生存阈值（kA-only, k=4..32）：公式 k* vs 引擎实测 vs CPA 选择
        vs MPC 网格可达上限 vs DPO（n≤16）；
    Ex3 让路（16A+16B）：CPA/DPO/MPC 对"同刻全派 vs 扣住 B"的裁决；
    Ex4 规模-时间（n=24..240 迷你 D2）：端到端墙钟与每决策墙钟
        CPA form vs CPA rollout vs FCFS（MPC 用 E26c 既有 records）。
  eval   E26c 四格点 240 规模 × {cpa_form, cpa_roll}，对照既有
         e26c records（fcfs/edf/local/mpc_v2.1，同 CRN 同 seed）。
输出：results/cq/eval/e27/（records_examples.json / records_eval.json）。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from fractions import Fraction as F
from typing import List

import numpy as np

from sim.cq.poly import (CPAPolicy, decision_point_optimum, offline_optimum,
                         _snapshot_of, survival_cap, wave_F)
from sim.cq.policies import make_fcfs, make_edf
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.metrics import summarize
from sim.experiments import e26_ab as e26
from sim.experiments import e26c_qa as e26c
from sim.experiments.cq_common import out_dir, print_progress, save_json

V_A = float(F(4096, 10**9) * e26.A_CLASS[0])
V_B = float(F(4096, 10**9) * e26.B_CLASS[0])
S_A = V_A / 120.0
C_A, C_B = 0.00602, 0.02859
T0_A = V_A / 120.0 + 8 * C_A
DL_A = 4 * T0_A
MPC_BUDGET_S = 3600.0
MPC_EVENTS = 20000


def mk_specs(classes, arrivals=None, alpha=4, m=None):
    arrivals = arrivals or [0.0] * len(classes)
    specs = []
    for i, c in enumerate(classes):
        h, u = e26.A_CLASS if c == "A" else e26.B_CLASS
        specs.append(e26.RequestSpec(i, float(arrivals[i]), h, u, c, F(1), F(1)))
    T0 = e26.make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    out = [e26.RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens,
                           s.class_id, T0[s.rid],
                           float(s.arrival_s) + alpha * float(T0[s.rid]))
           for s in specs]
    scn = e26.e26_scenario(alpha)
    if m is not None:
        scn = replace(scn, m_workers=m)
    return out, scn


def _count_action(act):
    d = sum(1 for a in act.actions if a.kind == "DISPATCH")
    w = sum(1 for a in act.actions if a.kind == "WAIT")
    return d, w


def _pol_decide(pol, snap, scn):
    t0 = time.perf_counter()
    act = pol.decide(snap, scn)
    return act, (time.perf_counter() - t0) * 1e3, pol


# ---------------------------------------------------------------------------
# 例子电池
# ---------------------------------------------------------------------------

def ex1_alignment():
    """Ex1：4A+4B 突发、m=8——五方裁决对齐（CPA×2 / MPC / DPO / OPT）。"""
    specs, scn = mk_specs(["A"] * 4 + ["B"] * 4, m=8)
    _eng, snap = _snapshot_of(scn, specs)
    rec = {"n": len(specs), "m": 8}
    act, ms, pol = _pol_decide(CPAPolicy(mode="form"), snap, scn)
    rec["cpa_form"] = {"dispatch": _count_action(act), "decide_ms": ms}
    act, ms, pol = _pol_decide(CPAPolicy(mode="rollout"), snap, scn)
    rec["cpa_roll"] = {"dispatch": _count_action(act), "decide_ms": ms,
                       "n_rollouts": pol.n_rollouts}
    mpc = MPCPolicy(pid="cq_mpc", H=1, theta="S", budget_s=MPC_BUDGET_S,
                    max_events=MPC_EVENTS, base="bw_edf", composer=True)
    act, ms, pol = _pol_decide(mpc, snap, scn)
    rec["mpc_v21"] = {"dispatch": _count_action(act), "decide_ms": ms,
                      "n_scored": pol.n_scored}
    dpo = decision_point_optimum(specs, scn)
    rec["dpo"] = {"best": [list(dpo["best"][0]), list(dpo["best"][1])],
                  "n_actions": dpo["n_actions"], "wall_s": dpo["wall_s"]}
    opt = offline_optimum(specs, scn, time_budget_s=120)
    rec["opt"] = {"status": opt["status"],
                  "best": list(opt["best_score"]) if opt["best_score"] else None,
                  "wall_s": opt["wall_s"]}
    return rec


def ex2_threshold():
    """Ex2：k 条 A 突发（k=4..32）——引擎存活数 vs 公式 k* vs 各算法选择。"""
    rows = []
    for k in (4, 8, 11, 12, 14, 16, 17, 18, 20, 24, 32):
        specs, scn = mk_specs(["A"] * k)
        eng = run_case(scn, specs, make_fcfs(), numeric=float, seed=0)
        fs = [float(eng.w.requests[i].F_s) for i in range(k)]
        alive = sum(1 for x in fs if x <= DL_A)
        _e, snap = _snapshot_of(scn, specs)
        cpa_act, cpa_ms, _p = _pol_decide(CPAPolicy(mode="form"), snap, scn)
        mpc = MPCPolicy(pid="cq_mpc", H=1, theta="S", budget_s=MPC_BUDGET_S,
                        max_events=MPC_EVENTS, base="bw_edf", composer=True)
        mpc_act, mpc_ms, _m = _pol_decide(mpc, snap, scn)
        row = {
            "k": k, "engine_alive": alive,
            "formula_kstar": survival_cap(0.0, DL_A, S_A, C_A, 8),
            "cpa_pick": _count_action(cpa_act)[0], "cpa_ms": cpa_ms,
            "mpc_pick": _count_action(mpc_act)[0], "mpc_ms": mpc_ms,
            "engine_F_last_ms": max(fs) * 1e3,
            "formula_F_last_ms": wave_F(k, k, S_A, C_A, 8) * 1e3,
        }
        if k <= 16:
            dpo = decision_point_optimum(specs, scn)
            row["dpo_pick"] = sum(dpo["best"][1])
            row["dpo_wall_s"] = dpo["wall_s"]
        rows.append(row)
        print_progress(f"  Ex2 k={k}: alive={alive} cpa={row['cpa_pick']} "
                       f"mpc={row['mpc_pick']} "
                       + (f"dpo={row['dpo_pick']}" if "dpo_pick" in row else ""))
    return rows


def ex3_yield():
    """Ex3：16A+16B——"同刻全派 vs 扣住 B"的裁决 + 两种物理的引擎事实。"""
    specs, scn = mk_specs(["A"] * 16 + ["B"] * 16)
    _e, snap = _snapshot_of(scn, specs)
    rec = {}
    act, ms, pol = _pol_decide(CPAPolicy(mode="rollout"), snap, scn)
    d, w = _count_action(act)
    a_n = sum(1 for a in act.actions if a.kind == "DISPATCH"
              and next(r for r in snap.requests
                       if r.rid == a.members[0]).h_tokens == e26.A_CLASS[0])
    rec["cpa_roll"] = {"A": a_n, "B": d - a_n, "WAIT": w, "decide_ms": ms}
    mpc = MPCPolicy(pid="cq_mpc", H=1, theta="S", budget_s=MPC_BUDGET_S,
                    max_events=MPC_EVENTS, base="bw_edf", composer=True)
    act, ms, _m = _pol_decide(mpc, snap, scn)
    d, w = _count_action(act)
    a_n = sum(1 for a in act.actions if a.kind == "DISPATCH"
              and next(r for r in snap.requests
                       if r.rid == a.members[0]).h_tokens == e26.A_CLASS[0])
    rec["mpc_v21"] = {"A": a_n, "B": d - a_n, "WAIT": w, "decide_ms": ms}
    dpo = decision_point_optimum(specs, scn)
    sc, cand = dpo["best"]
    rec["dpo"] = {"cand": list(cand), "score": list(sc),
                  "n_actions": dpo["n_actions"], "wall_s": dpo["wall_s"]}
    # 引擎事实：同刻全派（FCFS）vs A 先行（B 延后 0.3s）
    eng = run_case(scn, specs, make_fcfs(), seed=0)
    same = sum(1 for i in range(16) if float(eng.w.requests[i].F_s) <= DL_A)
    specs2, scn2 = mk_specs(["A"] * 16 + ["B"] * 16,
                            arrivals=[0.0] * 16 + [0.3] * 16)
    eng2 = run_case(scn2, specs2, make_fcfs(), seed=0)
    hold = sum(1 for i in range(16) if float(eng2.w.requests[i].F_s) <= DL_A)
    rec["engine_truth"] = {"same_moment_alive_A": same, "a_first_alive_A": hold}
    return rec


def ex4_scale(seeds):
    """Ex4：迷你 D2（(A,B,B)×n/3 两轮）规模阶梯——墙钟与每决策开销。"""
    rows = []
    for n in (24, 48, 96, 240):
        per = n // 2 // 3
        classes = (["A", "B", "B"] * per)
        arrivals = [0.0] * len(classes) + [1.0] * len(classes)
        classes = classes + classes
        specs, scn = mk_specs(classes, arrivals=arrivals)
        row = {"n": n}
        for label, pol in (
                ("cpa_form", CPAPolicy(mode="form")),
                ("cpa_roll", CPAPolicy(mode="rollout")),
                ("fcfs", make_fcfs())):
            t0 = time.perf_counter()
            eng = run_case(scn, specs, pol, numeric=float, seed=0)
            wall = time.perf_counter() - t0
            s = summarize(eng, scn)
            h = pol.health() if hasattr(pol, "health") else {}
            row[label] = {"slo": s["slo_success"], "n": s["n_cohort"],
                          "wall_s": wall,
                          "decide_ms_max": h.get("decide_s_max", 0) * 1e3,
                          "decide_ms_mean": (h.get("decide_s_mean") or 0) * 1e3,
                          "n_decides": h.get("n_decides"),
                          "n_rollouts": h.get("n_rollouts", 0)}
        rows.append(row)
        print_progress(f"  Ex4 n={n}: form {row['cpa_form']['slo']}/{n} "
                       f"wall={row['cpa_form']['wall_s']:.1f}s | "
                       f"roll {row['cpa_roll']['slo']}/{n} "
                       f"wall={row['cpa_roll']['wall_s']:.1f}s | "
                       f"fcfs {row['fcfs']['slo']}/{n}")
    return rows


# ---------------------------------------------------------------------------
# E26c 四格点 end-to-end
# ---------------------------------------------------------------------------

def run_eval_cell(cell: str, d: str):
    specs = e26c.build_specs(cell)
    alpha = 4 if cell == "D2c" else int(cell.rsplit("_a", 1)[1])
    scn = e26.e26_scenario(alpha)
    records = []
    for label, pol in (("cpa_form", CPAPolicy(mode="form")),
                       ("cpa_roll", CPAPolicy(mode="rollout", top_k=4))):
        t0 = time.perf_counter()
        eng = run_case(scn, specs, pol, numeric=float, seed=0,
                       record_intervals=True)
        wall = time.perf_counter() - t0
        s = summarize(eng, scn)
        s["slo_rate"] = s["slo_success"] / s["n_cohort"]
        s.update(e26.class_metrics(eng, specs))
        s.update(e26.concurrent_a_quantiles(eng))
        s.update(e26.bw_interval_stats(eng.w.storage.interval_log))
        h = pol.health()
        s.update({"policy": label, "pid": pol.pid, "cell": cell,
                  "alpha": alpha, "n_total": e26c.E26C_TOTAL,
                  "makespan_s": float(eng.w.t), "wall_s": wall,
                  "n_decides": h["n_decides"], "n_searched": h["n_searched"],
                  "n_rollouts": h.get("n_rollouts"),
                  "n_fallback": h["n_fallback"],
                  "decide_ms_max": h["decide_s_max"] * 1e3,
                  "decide_ms_mean": (h["decide_s_mean"] or 0) * 1e3})
        records.append(s)
        print_progress(
            f"E27 {cell:12s} {label:9s} wall={wall:.0f}s "
            f"SLO={s['slo_success']}/{s['n_cohort']} "
            f"A={100*s['A']['slo_rate']:.0f}% B={100*s['B']['slo_rate']:.0f}% "
            f"concA {s['a_conc_p50']:.0f}/{s['a_conc_p90']:.0f}/"
            f"{s['a_conc_max']:.0f} rollouts={h.get('n_rollouts')}")
    return records


def load_e26c_records():
    """既有 E26c records（fcfs/edf/local/mpc，同 CRN）作对照。"""
    for p in ("results/cq/eval/e26c/records_part_main.json",):
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    return []


def main(seeds, procs=None, duration=None, stage="examples", **kw):
    d = out_dir("eval", "e27")
    os.makedirs(d, exist_ok=True)
    if stage in ("examples", "eval"):
        pass
    if stage == "examples":
        out = {}
        print_progress("E27 Ex1 裁决对齐…")
        out["ex1"] = ex1_alignment()
        print_progress("E27 Ex2 生存阈值…")
        out["ex2"] = ex2_threshold()
        print_progress("E27 Ex3 让路…")
        out["ex3"] = ex3_yield()
        print_progress("E27 Ex4 规模-时间…")
        out["ex4"] = ex4_scale(seeds)
        save_json(os.path.join(d, "records_examples.json"), out)
        return {"saved": "records_examples.json"}
    # eval：E26c 四格点
    only = [x for x in os.environ.get("E27_ONLY", "").split(",") if x]
    cells = only or ["D2_b1.0_a4", "D2_b0.75_a4", "D1_r1.1_a4", "D2c"]
    part = os.path.join(d, "records_eval.json")
    records = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else []
    done = {(r["cell"], r["policy"]) for r in records}
    for cell in cells:
        if all((cell, a) in done for a in ("cpa_form", "cpa_roll")):
            continue
        os.makedirs(os.path.join(d, cell), exist_ok=True)
        records.extend(run_eval_cell(cell, d))
        save_json(part, records)
    # 附对照（不重跑，引用既有 e26c records）
    ref = load_e26c_records()
    save_json(os.path.join(d, "e26c_reference.json"), ref)
    print_progress(f"E27 eval done -> {part} (+{len(ref)} e26c 对照行)")
    return {"n_records": len(records)}


if __name__ == "__main__":
    main([0])
