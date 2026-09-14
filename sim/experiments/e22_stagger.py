"""E22：排序与错峰的因果隔离（§9.5）。

两层证据：
A. 固定批确定算例：P1→P4（LL 领取 0→0.5），delay×B 网格，完整提交序号与带宽；
B. Mooncake 固定批工作集：FCFS 预分批目录，fixed_immediate / fixed_order_mpc /
   fixed_wait_mpc 只研究批的选择次序与领取时刻；ΔTTFT=Δqueue+Δcompute+Δstall。
图 fig_e22_timeline.png、fig_e22_wait_decomposition.png。
"""
from __future__ import annotations

import os
from fractions import Fraction as F

import numpy as np

from sim.cq.config import ControllerConfig
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, MooncakeSource,
                                          default_scenario, out_dir,
                                          print_progress, save_json,
                                          setup_matplotlib)
from sim.cq.metrics import request_rows
from sim.cq.observable import Observable
from sim.cq.policies import make_fcfs, fill_batch_checked
from sim.cq.profile import make_T0
from sim.cq.simrun import run_case
from sim.cq.engine import CqEngine
from sim.cq.types import Action, JointAction, RequestSpec


# ---------------------------------------------------------------------------
# A. 确定算例：delay × B 网格
# ---------------------------------------------------------------------------

def fixed_plan_run(delay: F, B: F):
    """E20 主例成员不变，SS t=0 w0、LL t=delay w1；返回每请求 F 与流事件。"""
    from tests.cq_reference import ScriptedPolicy, e20_scenario, e20_specs
    scn = e20_scenario(B=B)
    specs = e20_specs()
    plan = [(F(0), 0, (0, 1)), (delay, 1, (2, 3))]
    pol = ScriptedPolicy(plan)
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, pol, numeric=F, observable=obs)
    obs.attach(eng)
    eng.run()
    Fs = {rid: rr.F_s for rid, rr in eng.w.requests.items()}
    return Fs, obs.ledger, eng


def deterministic_block():
    recs = []
    for B in (F(2), F(4), F(8)):
        for dl in (F(0), F(1, 8), F(1, 4), F(1, 2), F(3, 4), F(1)):
            Fs, led, eng = fixed_plan_run(dl, B)
            recs.append({
                "B": float(B), "delay": float(dl),
                "F_S": [float(Fs[0]), float(Fs[1])],
                "F_L": [float(Fs[2]), float(Fs[3])],
                "M": float(max(Fs.values())),
                "stall": float(sum(wk.stall_s for wk in eng.w.workers.values())),
                "seq_order": [(v["submit_seq"], v["batch_id"], v["layer"],
                               float(v["submit_s"])) for v in
                              sorted(led.values(), key=lambda x: x["submit_seq"])],
            })
    return recs


# ---------------------------------------------------------------------------
# B. Mooncake 固定批工作集
# ---------------------------------------------------------------------------

def fcfs_catalog(specs, limits, profile):
    """按 FCFS 顺序对全部请求预分批，形成固定批目录（singleton 也不拆）。"""
    order = sorted(specs, key=lambda s: (s.arrival_s, s.rid))
    by = {s.rid: s for s in specs}
    catalog = []
    i = 0
    while i < len(order):
        members = [by[order[i].rid]]
        j = i + 1
        while j < len(order):
            trial = members + [by[order[j].rid]]
            from sim.cq.profile import is_feasible
            if is_feasible(trial, limits, profile):
                members.append(by[order[j].rid])
                j += 1
            else:
                break
        catalog.append(tuple(sorted(m.rid for m in members)))
        i = j
    return catalog


class FixedBatchPolicy:
    """固定批目录策略。mode: immediate（目录序）/ order（EDF 重排）/ wait（+WAIT）"""

    def __init__(self, catalog, mode="immediate", wait_s=0.0):
        self.catalog = list(catalog)
        self.mode = mode
        self.wait_s = wait_s
        self.taken = set()

    def decide(self, snap, scn):
        avail = [b for b in self.catalog
                 if not (set(b) & self.taken)
                 and all(r in snap.queued for r in b)]
        if self.mode == "order":
            dl = {}
            for r in snap.requests:
                dl[r.rid] = float(r.deadline_s)
            avail.sort(key=lambda b: (min(dl[x] for x in b), b))
        acts = []
        used = set()
        for wid in sorted(snap.idle_workers):
            pick = None
            for b in avail:
                if not (set(b) & used):
                    pick = b
                    break
            if pick is not None:
                acts.append(Action("DISPATCH", wid, pick))
                used |= set(pick)
                self.taken |= set(pick)
            elif self.mode == "wait":
                acts.append(Action("WAIT", wid, (), float(snap.now) + max(
                    self.wait_s, 1e-4)))
        return JointAction(tuple(acts))


def fixed_workset_block(fname, trace_dir, n_req=32, B=80.0, alpha=4.0):
    """E21 同一输入的固定批目录 + 三种 timing 策略对比。"""
    from sim.experiments.e21_batch import finite_workload
    from sim.cq.types import HardLimits
    src = MooncakeSource(fname, trace_dir)
    specs = finite_workload(src, 0, n_req, alpha)   # 训练块 0
    limits = HardLimits(8, 262144, F(128))
    from sim.cq.profile import ProfileConfig
    prof = ProfileConfig()
    cat = fcfs_catalog(specs, limits, prof)
    out = {}
    base_rows = None
    for mode in ("immediate", "order", "wait"):
        scn = default_scenario(B_gbps=B, limits=limits)
        pol = FixedBatchPolicy(cat, mode, wait_s=float(np.median(
            [float(s.T0_s) for s in specs]) * 0.1))
        eng = run_case(scn, specs, pol, numeric=float)
        rows = {r["rid"]: r for r in request_rows(eng)}
        out[mode] = rows
        if base_rows is None:
            base_rows = rows
    deltas = []
    for rid in sorted(base_rows):
        b = base_rows[rid]
        for mode in ("order", "wait"):
            r = out[mode][rid]
            dq = (r["queue_s"] or 0) - (b["queue_s"] or 0)
            dc = (r["batch_compute_s"] or 0) - (b["batch_compute_s"] or 0)
            ds = (r["batch_stall_s"] or 0) - (b["batch_stall_s"] or 0)
            dT = (r["F_s"] or 0) - (b["F_s"] or 0)
            deltas.append({"rid": rid, "mode": mode, "d_queue": dq,
                           "d_compute": dc, "d_stall": ds, "d_ttft": dT})
            if abs((dq + dc + ds) - dT) > 1e-9:
                raise AssertionError("ΔTTFT 恒等式失败")
            if abs(dc) > 1e-9:
                raise AssertionError("固定成员的 Δcompute 必须为 0")
    return {"catalog_size": len(cat), "deltas": deltas,
            "summary": {m: {"M": max((r["F_s"] or 0) for r in rows.values()),
                            "mean_ttft": float(np.mean(
                                [r["F_s"] for r in rows.values()]))}
                        for m, rows in out.items()}}


def main(seeds, procs=None, duration=150.0, stage="smoke",
         trace_dir=TRACE_DIR_DEFAULT, **kw):
    plt = setup_matplotlib()
    d = out_dir(stage, "e22")
    det = deterministic_block()
    save_json(os.path.join(d, "e22_deterministic.json"), det)
    files = ["conversation_trace.jsonl"] if stage == "smoke" else [
        "conversation_trace.jsonl", "toolagent_trace.jsonl",
        "synthetic_trace.jsonl"]
    fixed = {}
    for f in files:
        n_req = 32 if stage == "smoke" else 128
        fixed[f] = fixed_workset_block(f, trace_dir, n_req)
    save_json(os.path.join(d, "e22_fixed_workset.json"), fixed)

    # 图 1：确定性算例 delay×B 的 M 与 STALL
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    Bs = sorted({r["B"] for r in det})
    for B in Bs:
        sub = sorted([r for r in det if r["B"] == B], key=lambda r: r["delay"])
        axes[0].plot([r["delay"] for r in sub], [r["M"] for r in sub],
                     marker="o", label=f"B={B:g}")
        axes[1].plot([r["delay"] for r in sub], [r["stall"] for r in sub],
                     marker="s", label=f"B={B:g}")
    axes[0].set_xlabel("LL 领取延迟 (s)"); axes[0].set_ylabel("makespan M (s)")
    axes[0].set_title("错峰对总完成时间的影响（成员固定）")
    axes[0].legend()
    axes[1].set_xlabel("LL 领取延迟 (s)"); axes[1].set_ylabel("STALL 设备秒")
    axes[1].set_title("错峰对 STALL 的影响")
    axes[1].legend()
    # P1 vs P4 时间线示意
    from tests.cq_reference import PLANS, run_plan
    eng1, _ = run_plan(PLANS["P1"])
    eng4, _ = run_plan(PLANS["P4"])
    for k, (eng, name) in enumerate([(eng1, "P1"), (eng4, "P4")]):
        ax = axes[2]
        for b in eng.w.batches.values():
            y = b.worker_id
            ax.barh(y, float(b.compute_s), left=float(b.dispatch_s),
                    height=.35, color="#4C72B0", label="compute" if b.batch_id == 0 else None)
            st = float(b.F_s) - float(b.dispatch_s) - float(b.compute_s)
            ax.barh(y, st, left=float(b.dispatch_s), height=.35,
                    color="#DD8452", alpha=.7, label="stall" if b.batch_id == 0 else None)
        ax.set_yticks([0, 1]); ax.set_xlabel("时间 (s)")
    axes[2].set_title("P1 vs P4 执行时间线（蓝=计算，橙=等料）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e22_timeline.png"), dpi=130)
    plt.close(fig)

    # 图 2：固定批工作集的 Δ 分解
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for mi, mode in enumerate(("order", "wait")):
        for fname, blk in fixed.items():
            ds = [x for x in blk["deltas"] if x["mode"] == mode]
            q = float(np.mean([x["d_queue"] for x in ds]))
            st_ = float(np.mean([x["d_stall"] for x in ds]))
            axes[mi].bar([f"{fname.split('_')[0][:8]}\nqueue"], [q],
                         color="#4C72B0", label="Δqueue" if fname == files[0] else None)
            axes[mi].bar([f"{fname.split('_')[0][:8]}\nstall"], [st_],
                         color="#DD8452", label="Δstall" if fname == files[0] else None)
        axes[mi].set_title(f"fixed_{mode} vs immediate（ΔTTFT 分解，均值）")
        axes[mi].axhline(0, color="k", lw=.8)
        axes[mi].legend()
    fig.suptitle("E22-B：等待是否只从 STALL 搬到中央队列（Δcompute≡0 已验证）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e22_wait_decomposition.png"), dpi=130)
    plt.close(fig)
    print_progress(f"E22 done -> {d}")
    return {"det_cells": len(det), "files": list(fixed)}


if __name__ == "__main__":
    main([0])
