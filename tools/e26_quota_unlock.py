"""通用组合器验证：不依赖 A/B 标签，按聚合申请率(V/c)配额构造联合候选。

规则：按 EDF 序逐条尝试加入；若 (已选集合 + 在跑集合) 的 Σ(V/c) 超过配额
quota，则跳过该条、试下一条能装下的（贪心装包）。quota∈{B, 0.75B, 0.5B}
——纯资源口径，任何 trace（含 mooncake 连续分布）同样适用。
"""
import sys, time
sys.path.insert(0, ".")
import os
os.environ["E26_TOTAL"] = "240"
import sim.experiments.e26_ab as e26
from sim.cq.simrun import run_case
from sim.cq.policies import GuardedEDF
from sim.cq.search import MPCPolicy
from sim.cq.types import Action, JointAction
from sim.cq.profile import batch_compute_s, layer_read_gb

class MPCQuotaCand(MPCPolicy):
    """联合点注入"带宽配额形"联合动作（类无关），排评分队首。"""
    def _joint_actions(self, node, node_snap, scn, batches, idle):
        base = super()._joint_actions(node, node_snap, scn, batches, idle)
        if len(idle) <= 2:
            return base
        B_total = float(scn.storage.b_schedule[0][1])
        specs = {r.rid: r for r in node_snap.requests}

        def q_rate(rid):
            r = specs[rid]
            v = float(layer_read_gb(r, scn.profile))
            c = float(batch_compute_s([r], scn.profile))
            return v / c if c > 0 else 0.0

        # 在跑批的聚合申请率（batch=1：每批近似其单条 V/c）
        active_q = sum(q_rate(pw.members[0]) for pw in node_snap.workers if pw.members)
        ordered = sorted((r for r in node_snap.requests if r.rid in node_snap.queued),
                         key=lambda r: (r.deadline_s, r.arrival_s, r.rid))
        wids = sorted(snap_idle := node_snap.idle_workers)
        out = []
        for quota in (B_total, 0.75 * B_total, 0.5 * B_total):
            acts, used, acc = [], set(), active_q
            wi = 0
            for r in ordered:
                if wi >= len(wids):
                    break
                qr = q_rate(r.rid)
                if r.rid in used:
                    continue
                if acc + qr > quota + 1e-9:
                    continue
                acts.append(Action("DISPATCH", wids[wi], (r.rid,)))
                used.add(r.rid); acc += qr; wi += 1
            for w in wids[len(acts):]:
                acts.append(Action("WAIT", w, (), float(node_snap.now) + 0.03))
            out.append(JointAction(tuple(acts)))
        return out + base

for tb in (1.0, 0.75):
    specs = e26.d2_specs(tb, 4)
    scn = e26.e26_scenario(4)
    import numpy as np
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    t0 = time.time()
    pol = MPCQuotaCand(pid="cq_mpc", H=1, theta="S", c_ref=c_ref, budget_s=3600.0)
    eng = run_case(scn, specs, pol)
    ok = sum(1 for rid, rr in eng.w.requests.items()
             if rr.F_s is not None and float(rr.F_s) <= float(specs[rid].deadline_s))
    h = pol.health()
    print(f"tb={tb} mpc_S+通用配额注入: SLO={ok}/{len(specs)} ({100*ok/len(specs):.1f}%) "
          f"决策{h['n_decides']} 偏离{h['n_deviate']} 回退{h['n_fallback']} "
          f"wall={time.time()-t0:.0f}s", flush=True)
