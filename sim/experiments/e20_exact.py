"""E20：四请求对拍与最优性检查（§9.3）。

- §11.2 四计划金标重放（不调用调度器）；
- n=4/6/8、n_max=1/2、δ∈{1/2,1/4,1/8}、B=4 网格实例四目标独立求解
  （B&B 为主，n=4 小 δ 用无剪枝枚举对拍）；
- 简单基线与 MPC 在同网格上的 gap（证书口径）。
图：fig_e20_four_plan.png、fig_e20_gap_runtime.png。
"""
from __future__ import annotations

import os
import time
from fractions import Fraction as F

from sim.cq.exact import (E20Instance, ExactSolver, THETA_TUPLES,
                          build_catalog)
from sim.experiments.cq_common import (out_dir, print_progress,
                                          save_json, setup_matplotlib)
from sim.cq.policies import GuardedEDF, make_edf, make_fcfs, make_spt
from sim.cq.simrun import run_case
from sim.cq.config import ControllerConfig
from tests.cq_reference import GOLD, PLANS, e20_scenario, e20_specs, run_plan


def four_plan_block() -> dict:
    """四计划金标重放 + 四目标 J 值。"""
    out = {}
    Fs_all = {}
    for name in ("P1", "P2", "P3", "P4"):
        eng, status = run_plan(PLANS[name])
        g = GOLD[name]
        Fs = {rid: rr.F_s for rid, rr in eng.w.requests.items()}
        ok = ((Fs[0], Fs[1]) == g["S"] and (Fs[2], Fs[3]) == g["L"]
              and eng.w.storage.actual_integral_gb == g["read"])
        Fs_all[name] = Fs
        out[name] = {
            "gold_match": bool(ok), "F": [float(v) for v in Fs.values()],
            "M": float(max(Fs.values())),
            "TTFT_mean": float(sum(Fs.values()) / 4),
            "stall": g["stall"] if isinstance(g["stall"], float) else float(g["stall"]),
            "slo_success": g["slo"]}
    sp = {s.rid: s for s in e20_specs()}
    arrs = {r: F(0) for r in range(4)}
    T0s = {r: sp[r].T0_s for r in range(4)}
    dls = {r: sp[r].deadline_s for r in range(4)}
    picks = {}
    for theta in ("M", "S", "T", "R"):
        best = None
        for name, Fs in Fs_all.items():
            key = THETA_TUPLES[theta](Fs, arrs, T0s, dls, (name,))
            if best is None or key < best[0]:
                best = (key, name)
        picks[theta] = best[1]
    out["four_objective_pick"] = picks
    return out


def grid_block(n: int, delta: F, n_max: int, theta_list=("M", "S", "T", "R"),
               node_budget=400000, time_budget=45.0, enum_cross=False) -> list:
    """一个 (n, δ, n_max) 实例的四目标求解记录。"""
    types = [0, 0, 1, 1] if n == 4 else ([0] * (n // 2 - 3 + 3) + [1] * (n // 2))
    types = ([0] * (n // 2) + [1] * (n // 2)) if n > 4 else [0, 0, 1, 1]
    arrival = (F(0),) * n
    recs = []
    for theta in theta_list:
        inst = E20Instance(types=list(types), arrival=arrival, B=F(4),
                           n_max=n_max, delta=delta)
        t0 = time.time()
        s = ExactSolver(inst, theta, mode="bnb", node_budget=node_budget,
                        time_budget_s=time_budget)
        r = s.solve()
        rec = {"n": n, "delta": str(delta), "n_max": n_max, "theta": theta,
               "status": r["status"], "UB": float(r["UB"]),
               "LB": float(r["LB"]) if r.get("LB") is not None else None,
               "nodes": r["nodes"], "runtime_s": round(time.time() - t0, 2),
               "plan": [(str(t), w, m) for (t, w, m) in r["plan"]]}
        if enum_cross and delta >= F(1):
            e = ExactSolver(inst, theta, mode="enumerate",
                            node_budget=node_budget, time_budget_s=time_budget)
            re_ = e.solve()
            rec["enum_UB"] = float(re_["UB"])
            rec["enum_match"] = re_["UB"] == r["UB"] and not re_["timeout"]
        recs.append(rec)
    return recs


def baseline_gap_block() -> list:
    """简单基线 / MPC 在 E20 主例网格上的 gap（对拍最优）。"""
    from sim.cq.exact import e20_specs, e20_scenario
    recs = []
    types = [0, 0, 1, 1]
    for theta in ("M", "S", "T", "R"):
        inst = E20Instance(types=types, arrival=(F(0),) * 4, B=F(4),
                           n_max=2, delta=F(1, 2))
        s = ExactSolver(inst, theta, mode="bnb", node_budget=400000,
                        time_budget_s=45)
        r = s.solve()
        opt = float(r["UB"])
        for pid in ("cq_fcfs", "cq_spt", "cq_edf", "cq_mpc"):
            from sim.experiments.cq_common import build_policy, c_ref_of
            from sim.experiments.cq_common import default_scenario
            scn = e20_scenario()
            scn = type(scn)(scenario_name="e20b", profile=scn.profile,
                            storage=scn.storage, limits=scn.limits,
                            m_workers=2,
                            controller=ControllerConfig(cost_mode="zero"))
            specs = e20_specs(list(types))
            pol = build_policy(pid, theta, c_ref=0.01, H=1)
            eng = run_case(scn, specs, pol, numeric=float)
            Fs = {rid: float(rr.F_s) if rr.F_s is not None else 1e9
                  for rid, rr in eng.w.requests.items()}
            sp = {x.rid: x for x in specs}
            arrs = {x: float(sp[x].arrival_s) for x in Fs}
            T0s = {x: float(sp[x].T0_s) for x in Fs}
            dls = {x: float(sp[x].deadline_s) for x in Fs}
            J = THETA_TUPLES[theta](Fs, arrs, T0s, dls, (pid,))[0]
            recs.append({"theta": theta, "policy": pid, "J": float(J),
                         "opt": opt, "gap_pct": 100.0 * (float(J) - opt) / opt
                         if opt > 0 else None})
    return recs


def main(seeds, procs=None, duration=150.0, stage="smoke", **kw):
    plt = setup_matplotlib()
    d = out_dir(stage, "e20")
    four = four_plan_block()
    save_json(os.path.join(d, "e20_four_plan.json"), four)
    # 网格求解：冒烟= n4 δ1/2 + n6 δ1/2；正式扩展到 n8/δ1/8
    grid_cfgs = ([("n4", 4, F(1, 2), 2), ("n6", 6, F(1, 2), 2)]
                 if stage == "smoke" else
                 [("n4", 4, F(1, 2), 2), ("n4", 4, F(1, 4), 2),
                  ("n4", 4, F(1, 8), 2), ("n6", 6, F(1, 2), 2),
                  ("n6", 6, F(1, 4), 2), ("n8", 8, F(1, 2), 1),
                  ("n8", 8, F(1, 2), 2), ("n4", 4, F(1, 2), 1)])
    grid = []
    for _tag, n, dl, nm in grid_cfgs:
        grid.extend(grid_block(n, dl, nm, enum_cross=(dl >= F(1, 2) and n == 4)))
    save_json(os.path.join(d, "e20_grid.json"), grid)
    gaps = baseline_gap_block()
    save_json(os.path.join(d, "e20_gap.json"), gaps)

    # 图 1：四计划金标对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = ["P1", "P2", "P3", "P4"]
    Mv = [four[n]["M"] for n in names]
    Tt = [four[n]["TTFT_mean"] for n in names]
    St = [four[n]["stall"] for n in names]
    x = range(len(names))
    axes[0].bar([i - .2 for i in x], Mv, width=.4, label="makespan M", color="#4C72B0")
    axes[0].bar([i + .2 for i in x], Tt, width=.4, label="平均 TTFT", color="#DD8452")
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names)
    axes[0].set_ylabel("秒"); axes[0].legend()
    axes[0].set_title("四计划金标：M 与平均 TTFT（Fraction 严格复现）")
    axes[1].bar(names, St, color="#55A868")
    axes[1].set_ylabel("STALL 设备秒")
    axes[1].set_title("四计划 STALL（P4 比 P1 少 20%）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e20_four_plan.png"), dpi=130)
    plt.close(fig)

    # 图 2：gap 与运行时间
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    thetas = ("M", "S", "T", "R")
    pols = ["cq_fcfs", "cq_spt", "cq_edf", "cq_mpc"]
    for pi, pid in enumerate(pols):
        ys = [next(g["gap_pct"] for g in gaps if g["theta"] == th and g["policy"] == pid)
              for th in thetas]
        axes[0].plot(thetas, ys, marker="o", label=pid)
    axes[0].axhline(0, color="k", lw=.8)
    axes[0].set_ylabel("相对网格最优 gap (%)")
    axes[0].set_title("基线/MPC 相对 B&B 最优的 gap（n=4, δ=1/2, B=4）")
    axes[0].legend()
    rtimes = {}
    for g in grid:
        rtimes.setdefault(g["theta"], []).append(g["runtime_s"])
    for th in thetas:
        axes[1].scatter([th] * len(rtimes.get(th, [])), rtimes.get(th, []),
                        s=40, color="#C44E52")
    axes[1].set_ylabel("B&B 运行时间 (s)")
    axes[1].set_title("求解运行时间与证书状态")
    st = {g["theta"]: g["status"] for g in grid}
    axes[1].set_xlabel("证书: " + ", ".join(f"{k}={v}" for k, v in st.items()))
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e20_gap_runtime.png"), dpi=130)
    plt.close(fig)
    print_progress(f"E20 done -> {d}")
    return {"four_plan": four, "grid_n": len(grid), "gaps": gaps}


if __name__ == "__main__":
    main([0])
