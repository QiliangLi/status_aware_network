"""OL 最大差距 case 的候选级诊断：决策序列分叉点分析（20260921）。

对指定格点/窗口跑 local-S 与 mpc-S（record_decisions 默认开），对齐两份
决策序列，找第一个成员不同的 DISPATCH，报告分叉时刻的系统状态（队列、
活动批、读流申请、Σmin(q_i,B) 是否超 B）与分叉涉及的请求及其最终
TTFT/deadline 结果。输出 results/cq/eval/e25_q2/gap_diag_<tag>.json。
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.getcwd())

from sim.cq.config import ProfileConfig
from sim.cq.metrics import summarize
from sim.cq.profile import singleton_K
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.types import RequestSpec
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                       MooncakeSource, c_ref_of,
                                       default_scenario)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(REPO, "results", "cq", "eval", "e25_q2")
FILES = {"con": "conversation_trace.jsonl",
         "too": "toolagent_trace.jsonl",
         "syn": "synthetic_trace.jsonl"}


def lam0_of(src, m):
    prof = ProfileConfig()
    hus = [(r.hit_tokens, r.u_tokens) for r in src.imp.rows]
    e_v = sum(F(prof.kappa_gb_per_token_layer) * h for h, _u in hus) / len(hus)
    ks = [singleton_K(RequestSpec(0, 0, h, u, "x", 1, 1), prof, F(80), F(200))
          for h, u in hus]
    return min(F(80) / e_v, F(m) / (sum(ks) / len(ks)))


def acts_of(eng):
    """[(t, [(worker, tuple(members), kind, wake)..])]：展开 decisions 的 action。"""
    out = []
    for d in eng.decisions:
        acts = []
        for a in d["action"]:          # _act_json 返回 list[dict]
            if a["kind"] == "DISPATCH":
                acts.append((a.get("worker"), tuple(a.get("members", ())),
                             a["kind"], a.get("wake_at")))
            else:
                acts.append((a.get("worker"), None, a["kind"], a.get("wake_at")))
        out.append((d["now"], acts))
    return out


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else "too"
    B = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
    rho = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    m = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    w = int(sys.argv[5]) if len(sys.argv) > 5 else 14
    alpha = 4

    src = MooncakeSource(FILES[trace], TRACE_DIR_DEFAULT)
    scn = default_scenario(B_gbps=B, m=m, alpha=F(alpha),
                           limits=TRACE_WIDE_LIMITS)
    lam = lam0_of(src, m) * F(str(rho))
    specs, d_sim = src.window_specs(w, lam, F(alpha), duration_cap=None,
                                    limits=TRACE_WIDE_LIMITS)
    c_ref = c_ref_of(specs, scn)
    runs = {}
    for pid in ("cq_local", "cq_mpc"):
        pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1, theta="S",
                        c_ref=c_ref, budget_s=3600.0)
        eng = run_case(scn, specs, pol, numeric=float, seed=w,
                       record_intervals=True)
        s = summarize(eng, scn, arrival_stop=d_sim)
        runs[pid] = {"eng": eng, "sum": s, "acts": acts_of(eng)}

    # 对齐决策序列：同刻逐条比较（local 与 mpc 的决策时刻可能因轨迹不同而
    # 早期一致后期漂移——逐刻比对直到首个成员差异）
    a_l, a_m = runs["cq_local"]["acts"], runs["cq_mpc"]["acts"]
    print(f"决策数 local={len(a_l)} mpc={len(a_m)}")
    div = None
    for (tl, actl), (tm, actm) in zip(a_l, a_m):
        if abs(float(tl) - float(tm)) > 1e-9:
            div = ("时间轴漂移", tl, tm, actl, actm)
            break
        ml = [(wk, mem) for wk, mem, _k, _wa in actl if mem is not None]
        mm = [(wk, mem) for wk, mem, _k, _wa in actm if mem is not None]
        if ml != mm:
            div = ("成员不同", tl, tm, ml, mm)
            break
    if div is None:
        print("前缀完全一致（分叉晚或仅评分不同未换动作）")
    else:
        kind, tl, tm, ml, mm = div
        print(f"首个分叉 @t={float(tl):.3f}s（{kind}）")
        print(f"  local 派发: {ml}")
        print(f"  mpc   派发: {mm}")

    # 汇总两 run 的请求级结果：找出 TTFT 差异最大的请求（分叉的受益/受害者）
    req_l, req_m = runs["cq_local"]["eng"].w.requests, runs["cq_mpc"]["eng"].w.requests
    rows = []
    for rid, rl in req_l.items():
        rm = req_m.get(rid)
        if rm is None or rl.F_s is None or rm.F_s is None:
            continue
        spec = specs[rid] if rid < len(specs) else None
        rows.append((rid, float(rl.F_s), float(rm.F_s),
                     float(spec.arrival_s) if spec else 0,
                     float(spec.deadline_s) if spec else 0,
                     int(spec.h_tokens) if spec else 0,
                     int(spec.u_tokens) if spec else 0))
    rows.sort(key=lambda r: r[2] - r[1], reverse=True)  # mpc 更慢的在前
    print("\nmpc 比 local 慢最多的 5 条请求（rid, local完成, mpc完成, 到达, deadline, h, u）：")
    for r in rows[:5]:
        print(f"  rid{r[0]:4d} {r[1]:8.3f}→{r[2]:8.3f} (arr {r[3]:.2f} dl {r[4]:.2f}) h={r[5]} u={r[6]}")
    print("\nlocal 比 mpc 慢最多的 5 条：")
    for r in sorted(rows, key=lambda r: r[1] - r[2], reverse=True)[:5]:
        print(f"  rid{r[0]:4d} {r[1]:8.3f}→{r[2]:8.3f} (arr {r[3]:.2f} dl {r[4]:.2f}) h={r[5]} u={r[6]}")

    out = {
        "cell": dict(trace=trace, B=B, rho=rho, m=m, window=w, alpha=alpha,
                     theta="S"),
        "summary": {p: {"ttft": runs[p]["sum"].get("ttft_mean_lower"),
                         "slo": (runs[p]["sum"].get("slo_rate_interval")
                                 or [0, 0])[0],
                         "n_decisions": len(runs[p]["acts"])}
                     for p in runs},
        "divergence": None if div is None else
                      {"kind": div[0], "t": float(div[1]),
                       "local": [[w_, list(mm_)] for w_, mm_ in div[3]],
                       "mpc": [[w_, list(mm_)] for w_, mm_ in div[4]]},
        "top_shifted_requests": [
            {"rid": r[0], "f_local": r[1], "f_mpc": r[2], "arr": r[3],
             "dl": r[4], "h": r[5], "u": r[6]} for r in rows[:10]],
    }
    tag = f"gap_diag_{trace}_B{B:g}_r{rho:g}_m{m}_w{w}"
    json.dump(out, open(os.path.join(OUTD, tag + ".json"), "w"),
              ensure_ascii=False, indent=1)
    print("saved:", tag + ".json")


if __name__ == "__main__":
    main()
