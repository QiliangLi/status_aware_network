"""解锁验证：注入限 A 联合候选后，MPC 是否在突发点自己选错峰（240 D2）。"""
import sys, time
sys.path.insert(0, ".")
import os
os.environ["E26_TOTAL"] = "240"
import sim.experiments.e26_ab as e26
from sim.cq.simrun import run_case
from sim.cq.policies import GuardedEDF
from sim.cq.search import MPCPolicy
from sim.cq.types import Action, JointAction

A_H = e26.A_CLASS[0]

class MPCStaggerCand(MPCPolicy):
    """联合决策点（≥3 空闲 worker）注入 π 动作的限 A 变体（cap∈{3,4,8}），
    排评分队列最前；稳态点行为与 MPCPolicy 完全一致。仅用公开信息。"""
    def _joint_actions(self, node, node_snap, scn, batches, idle):
        base = super()._joint_actions(node, node_snap, scn, batches, idle)
        if len(idle) <= 2:
            return base
        pi_act = self.pi.decide(node_snap, scn)
        specs = {r.rid: r for r in node_snap.requests}
        is_a = lambda rid: specs[rid].h_tokens == A_H
        concA = sum(1 for pw in node_snap.workers
                    if pw.members and is_a(pw.members[0]))
        queued_b = sorted((r for r in node_snap.requests
                           if r.rid in node_snap.queued and not is_a(r.rid)),
                          key=lambda r: (r.arrival_s, r.rid))
        out = []
        for cap in (4, 3, 8):
            acts, used, budget_a, bi = [], set(), cap - concA, 0
            for d in pi_act.actions:
                if d.kind != "DISPATCH":
                    acts.append(d); continue
                if is_a(d.members[0]) and budget_a <= 0:
                    while bi < len(queued_b) and queued_b[bi].rid in used:
                        bi += 1
                    if bi < len(queued_b):
                        acts.append(Action("DISPATCH", d.worker_id,
                                           (queued_b[bi].rid,)))
                        used.add(queued_b[bi].rid); bi += 1
                    else:
                        acts.append(Action("WAIT", d.worker_id, (),
                                           float(node_snap.now) + 0.03))
                else:
                    acts.append(d)
                    used.update(d.members)
                    if is_a(d.members[0]):
                        budget_a -= 1
            out.append(JointAction(tuple(acts)))
        return out + base

for tb in (1.0, 0.75):
    specs = e26.d2_specs(tb, 4)
    scn = e26.e26_scenario(4)
    import numpy as np
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    for name, mk in (("mpc_S", lambda: MPCPolicy(pid="cq_mpc", H=1, theta="S",
                                                 c_ref=c_ref, budget_s=3600.0)),
                     ("mpc_S+注入", lambda: MPCStaggerCand(pid="cq_mpc", H=1, theta="S",
                                                           c_ref=c_ref, budget_s=3600.0))):
        t0 = time.time()
        pol = mk()
        eng = run_case(scn, specs, pol)
        ok = sum(1 for rid, rr in eng.w.requests.items()
                 if rr.F_s is not None and float(rr.F_s) <= float(specs[rid].deadline_s))
        h = pol.health()
        print(f"tb={tb} {name:9s} SLO={ok}/{len(specs)} ({100*ok/len(specs):.1f}) "
              f"决策{h['n_decides']} 搜索{h['n_searched']} 偏离{h['n_deviate']} "
              f"回退{h['n_fallback']} wall={time.time()-t0:.0f}s", flush=True)
