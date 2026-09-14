"""cq 测试公共夹具：E20 主例场景、脚本化计划策略（§11.2）。

E20 主例：两 worker、两层、两个 long（V=4,c=2）+ 两个 short（V=1,c=1），
batch c 为成员之和、V 为字节之和，q_max=B=4，batch 上限 2。
T0_long=5、T0_short=9/4，deadline 分别 15、27/4。
用 no_speedup（η≡1、launch=0）且 a=1、b=0、κ=1 实现：c=u、V=h，
取 short(h=1,u=1)、long(h=4,u=2)。
"""
from __future__ import annotations

from fractions import Fraction as F

from sim.cq.config import (CqScenario, CohortConfig, ControllerConfig,
                           ObservationConfig, ProfileConfig, StorageConfig)
from sim.cq.engine import CqEngine
from sim.cq.observable import Observable
from sim.cq.types import Action, HardLimits, JointAction, RequestSpec

E20_PROFILE = ProfileConfig(
    profile_id="e20-main", L=2, kappa_gb_per_token_layer=F(1),
    N_sat=1, t_launch_s=F(0), a_s_per_token=F(1), b_s_per_pair=F(0),
    no_speedup=True)


def e20_scenario(B=F(4), m=2, limits=None) -> CqScenario:
    return CqScenario(
        scenario_name="e20-main", profile=E20_PROFILE,
        storage=StorageConfig(b_schedule=((F(0), B),), q_max_gbps=F(4),
                              b_ref_gbps=F(4)),
        limits=limits or HardLimits(n_max=2, token_max=8192,
                                    workspace_gb=F(10**9)),
        m_workers=m, controller=ControllerConfig(cost_mode="zero"))


def e20_specs(n_long=2, n_short=2, arrival=None) -> list:
    """S0=0,S1=1,L0=2,L1=3；arrival 默认全 0。"""
    specs = []
    arrival = arrival or (F(0),) * (n_long + n_short)
    for rid in range(n_long + n_short):
        is_short = rid < n_short
        h, u = (1, 1) if is_short else (4, 2)
        V = h * 1
        c = u
        eff = min(F(4), F(4))
        tr = F(V) / eff
        T0 = tr + 2 * c + max(F(0), tr - c)
        if is_short:
            dl = arrival[rid] + F(27, 4)
        else:
            dl = arrival[rid] + 15
        specs.append(RequestSpec(rid=rid, arrival_s=arrival[rid], h_tokens=h,
                                 u_tokens=u, class_id="S" if is_short else "L",
                                 T0_s=T0, deadline_s=dl))
    return specs


class ScriptedPolicy:
    """按 (t, worker, members) 脚本领取；未到脚本时刻的空闲 worker WAIT 到下一
    脚本时刻。用于 §11.2 四计划的固定执行，不调用任何调度逻辑。"""

    def __init__(self, plan):
        self.plan = sorted(plan, key=lambda x: (x[0], x[1]))
        self.log = []

    def decide(self, snap, scn):
        now = snap.now
        taken = set()
        acts = []
        used_w = set()
        for (t, w, members) in self.plan:
            if t <= now and w not in used_w:
                avail = all(r in snap.queued for r in members)
                if avail:
                    acts.append(Action("DISPATCH", w, tuple(members)))
                    used_w.add(w)
                    taken.add((t, w))
        future = [t for (t, w, _m) in self.plan if t > now]
        wake = min(future) if future else now + F(1)
        for w in snap.idle_workers:
            if w not in used_w:
                acts.append(Action("WAIT", w, (), max(wake, now + F(1, 1000))))
        self.log.append((now, tuple(acts)))
        return JointAction(tuple(sorted(acts, key=lambda a: a.worker_id)))


def run_plan(plan, scenario=None, specs=None, numeric=F):
    """跑一个固定计划，返回 (engine, F dict)。"""
    scenario = scenario or e20_scenario()
    specs = specs or e20_specs()
    pol = ScriptedPolicy(plan)
    obs = Observable(scenario, numeric=numeric)
    eng = CqEngine(scenario, specs, pol, numeric=numeric, observable=obs)
    obs.attach(eng)
    status = eng.run()
    return eng, status


# §11.2 金标（Fraction 严格等号）
GOLD = {
    "P1": {"S": (F(5), F(5)), "L": (F(43, 4), F(43, 4)), "M": F(43, 4),
           "TTFT": F(63, 8), "norm": F(787, 360), "slo": 4, "compute": F(12),
           "stall": F(15, 4), "idle": F(23, 4), "read": F(20)},
    "P2": {"S": (F(13, 2), F(13, 2)), "L": (F(41, 4), F(41, 4)), "M": F(41, 4),
           "TTFT": F(67, 8), "norm": F(889, 360), "slo": 4, "compute": F(12),
           "stall": F(19, 4), "idle": F(15, 4), "read": F(20)},
    "P3": {"S": (F(373, 48), F(5021, 576)), "L": (F(373, 48), F(5021, 576)),
           "M": F(5021, 576), "TTFT": F(9497, 1152),
           "norm": F(275413, 103680), "slo": 2, "compute": F(12),
           "stall": F(2585, 576), "idle": F(545, 576), "read": F(20)},
    "P4": {"S": (F(9, 2), F(9, 2)), "L": (11, 11), "M": 11,
           "TTFT": F(31, 4), "norm": F(21, 10), "slo": 4, "compute": F(12),
           "stall": 3, "idle": 7, "read": F(20)},
}

PLANS = {
    "P1": [(F(0), 0, (0, 1)), (F(0), 1, (2, 3))],
    "P2": [(F(0), 0, (2, 3)), (F(0), 1, (0, 1))],
    "P3": [(F(0), 0, (0, 2)), (F(0), 1, (1, 3))],
    "P4": [(F(0), 0, (0, 1)), (F(1, 2), 1, (2, 3))],
}
