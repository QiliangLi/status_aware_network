"""E20 精确参照：Fraction 目录/网格、全动作枚举、B&B 与 LB/UB 证书（§8）。

- 目录含全部合法 singleton/pair（按 n_max），不做候选剪枝；
- 启动网格步长 δ；网格终点由"全部 singleton 按 rid 在 worker0 串行、每次领取
  向上取整到 δ 网格"的 UB 计划最后领取时刻确定，所有策略共用；
- 无剪枝枚举与 B&B 使用同一全动作生成器：按 worker 编号递归 DISPATCH 合法批
  或 WAIT 一格，包含全 WAIT；已占用 worker 不分支；
- B&B 剪枝条件仅 LB_primary >= UB_primary；超时返回 time_limit 证书。
"""
from __future__ import annotations

import heapq
import itertools
import time as _time
from dataclasses import dataclass, field
from fractions import Fraction as F
from typing import Dict, List, Optional, Sequence, Tuple

from .config import CqScenario, ProfileConfig, StorageConfig
from .engine import CqEngine
from .profile import is_feasible, singleton_K
from .types import HardLimits, RequestSpec

E20_PROFILE = ProfileConfig(
    profile_id="e20", L=2, kappa_gb_per_token_layer=F(1), N_sat=1,
    t_launch_s=F(0), a_s_per_token=F(1), b_s_per_pair=F(0), no_speedup=True)

SHORT = (1, 1)   # (h, u)：V=1 GB/层，c=1 s/层
LONG = (4, 2)    # V=4 GB/层，c=2 s/层

THETA_TUPLES = {
    "M": lambda Fs, arrs, T0s, dls, key: (max(Fs.values()),
                                          sum(Fs[r] - arrs[r] for r in Fs), key),
    "S": lambda Fs, arrs, T0s, dls, key: (
        sum(1 for r in Fs if Fs[r] > dls[r]),
        sum(Fs[r] - arrs[r] for r in Fs), key),
    "T": lambda Fs, arrs, T0s, dls, key: (
        sum(Fs[r] - arrs[r] for r in Fs),
        sum(1 for r in Fs if Fs[r] > dls[r]), key),
    "R": lambda Fs, arrs, T0s, dls, key: (
        sum((Fs[r] - arrs[r]) / T0s[r] for r in Fs),
        sum(Fs[r] - arrs[r] for r in Fs), key),
}


def e20_specs(types: Sequence[int], arrival: Optional[Sequence] = None) -> list:
    """types[i]∈{0,1}：0=short、1=long；rid=i。arrival 默认全 0。"""
    specs = []
    n = len(types)
    arrival = arrival or (F(0),) * n
    for i, tp in enumerate(types):
        h, u = SHORT if tp == 0 else LONG
        tr = F(h) / F(4)
        c = F(u)
        T0 = tr + 2 * c + max(F(0), tr - c)
        dl = arrival[i] + (F(27, 4) if tp == 0 else 15)
        specs.append(RequestSpec(rid=i, arrival_s=arrival[i], h_tokens=h,
                                 u_tokens=u, class_id="short" if tp == 0 else "long",
                                 T0_s=T0, deadline_s=dl))
    return specs


def e20_scenario(B=F(4), n_max=2, m=2) -> CqScenario:
    from .config import ControllerConfig
    return CqScenario(
        scenario_name="e20-grid", profile=E20_PROFILE,
        storage=StorageConfig(b_schedule=((F(0), B),), q_max_gbps=F(4),
                              b_ref_gbps=F(4)),
        limits=HardLimits(n_max=n_max, token_max=10**9, workspace_gb=F(10**9)),
        m_workers=m, controller=ControllerConfig(cost_mode="zero"))


@dataclass
class E20Instance:
    types: Sequence[int]
    arrival: Sequence
    B: F = F(4)
    n_max: int = 2
    delta: F = F(1, 2)
    m: int = 2

    def specs(self):
        return e20_specs(self.types, self.arrival)

    def scenario(self):
        return e20_scenario(self.B, self.n_max, self.m)


def build_catalog(inst: E20Instance):
    """全部合法批（升序 rid tuple）+ 每批 singleton K 之和与读取字节之和。"""
    specs = {s.rid: s for s in inst.specs()}
    scn = inst.scenario()
    rids = sorted(specs)
    cat = []
    for k in range(1, inst.n_max + 1):
        for comb in itertools.combinations(rids, k):
            members = [specs[r] for r in comb]
            if is_feasible(members, scn.limits, scn.profile):
                K = sum(singleton_K(specs[r], scn.profile, F(4), F(4))
                        for r in comb)
                V = sum(F(specs[r].h_tokens) for r in comb)   # 每层读取 GB
                cat.append((comb, K, V))
    return cat


def grid_of(inst: E20Instance) -> List[F]:
    """UB 计划：全部 singleton 按 rid 在 worker0 串行、领取向上取整到 δ。"""
    specs = {s.rid: s for s in inst.specs()}
    scn = inst.scenario()
    t = F(0)
    last_take = F(0)
    for rid in sorted(specs):
        g = ((t / inst.delta).__ceil__()) * inst.delta
        last_take = g
        t = g + singleton_K(specs[rid], scn.profile, F(4), F(4))
    end = last_take
    n = (end / inst.delta).__ceil__()
    return [i * inst.delta for i in range(n + 1)]


def _ceil_grid(t, delta):
    return ((t / delta).__ceil__()) * delta


class ExactSolver:
    """枚举（无剪枝）或 B&B（LB 剪枝、best-first）。"""

    def __init__(self, inst: E20Instance, theta: str, mode: str = "enumerate",
                 node_budget: int = 2 * 10**6, time_budget_s: float = 60.0):
        assert theta in THETA_TUPLES
        assert mode in ("enumerate", "bnb")
        self.inst = inst
        self.theta = theta
        self.mode = mode
        self.node_budget = node_budget
        self.time_budget_s = time_budget_s
        self.catalog = build_catalog(inst)
        self.grid = grid_of(inst)
        self.grid_end = self.grid[-1]
        self.nodes = 0
        self.timeout = False
        self.best_key = None
        self.best_plan = None
        self.open_min_lb = None
        self._t0 = 0.0
        self._combos: Dict[Tuple, Tuple] = {c[0]: (c[1], c[2]) for c in self.catalog}
        self._K_single = None

    # -- 基础 ---------------------------------------------------------------
    def _root_engine(self) -> CqEngine:
        eng = CqEngine(self.inst.scenario(), self.inst.specs(), policy=None,
                       numeric=F)
        return eng

    def _leaf_key(self, eng, plan):
        w = eng.w
        Fs = {rid: rr.F_s for rid, rr in w.requests.items()}
        arrs = {rid: F(w.specs[rid].arrival_s) for rid in Fs}
        T0s = {rid: F(w.specs[rid].T0_s) for rid in Fs}
        dls = {rid: F(w.specs[rid].deadline_s) for rid in Fs}
        return THETA_TUPLES[self.theta](Fs, arrs, T0s, dls, tuple(plan))

    def _all_done(self, eng):
        return eng.w.all_done()

    # -- 下界（§8.2）---------------------------------------------------------
    def lower_bound(self, eng, t) -> F:
        w = eng.w
        if w.all_done():
            return self._leaf_key(eng, [])[0]
        scn = eng.scn
        specs = w.specs
        active_rids = {rid for rid, rr in w.requests.items() if rr.state == "ACTIVE"}
        e: Dict[int, F] = {}
        rem_compute_total = F(0)
        for rid, rr in w.requests.items():
            if rr.state == "DONE":
                e[rid] = F(rr.F_s)
            elif rr.state == "ACTIVE":
                b = w.batches[rr.batch_id]
                rem = F(0)
                if b.next_layer > 0 and b.Z[b.next_layer - 1] is not None:
                    rem += max(F(0), F(b.Z[b.next_layer - 1]) - t)
                rem += sum(F(b.c_layers[l2]) for l2 in range(b.next_layer, w.L))
                # 同批成员共享批状态
                e[rid] = t + rem
                rem_compute_total += rem / len(b.members)
            else:
                a = F(specs[rid].arrival_s)
                best_k = None
                for comb, K, V in self.catalog:
                    if rid in comb and not (set(comb) & active_rids):
                        kk = K / len(comb)
                        if best_k is None or kk < best_k:
                            best_k = kk
                e[rid] = max(t, a) + (best_k or F(0))
        arrs = {rid: F(specs[rid].arrival_s) for rid in e}
        T0s = {rid: F(specs[rid].T0_s) for rid in e}
        dls = {rid: F(specs[rid].deadline_s) for rid in e}
        if self.theta == "T":
            return sum(e[r] - arrs[r] for r in e)
        if self.theta == "R":
            return sum((e[r] - arrs[r]) / T0s[r] for r in e)
        if self.theta == "S":
            return F(sum(1 for r in e if e[r] > dls[r]))
        # M：三项取 max
        k_rem_lower = rem_compute_total
        for rid, rr in w.requests.items():
            if rr.state in ("QUEUED", "FUTURE"):
                best_k = None
                for comb, K, V in self.catalog:
                    if rid in comb and not (set(comb) & active_rids):
                        kk = K / len(comb)
                        if best_k is None or kk < best_k:
                            best_k = kk
                k_rem_lower += best_k or F(0)
        v_rem = sum(f.remaining_gb for f in w.storage.flows.values())
        for b in w.batches.values():
            if b.F_s is None:
                for l2 in range(w.L):
                    if b.read_ready[l2] is None and b.V_layers[l2] > 0:
                        v_rem += b.V_layers[l2]
        for rid, rr in w.requests.items():
            if rr.state in ("QUEUED", "FUTURE"):
                best_v = None
                for comb, K, V in self.catalog:
                    if rid in comb and not (set(comb) & active_rids):
                        vv = V / len(comb)
                        if best_v is None or vv < best_v:
                            best_v = vv
                v_rem += best_v or F(0)
        return max(max(e.values()), t + k_rem_lower / w.m, t + v_rem / self.inst.B)

    # -- 动作生成 -------------------------------------------------------------
    def _assignments(self, idle, queued_set):
        """按 worker 编号递归 DISPATCH 合法批或 WAIT 一格（含全 WAIT）。"""
        results = []

        def rec(i, cur, used):
            if i == len(idle):
                results.append(tuple(cur))
                return
            w = idle[i]
            for comb, _K, _V in self.catalog:
                if set(comb) <= queued_set and not (set(comb) & used):
                    cur.append((w, comb))
                    rec(i + 1, cur, used | set(comb))
                    cur.pop()
            cur.append((w, None))   # WAIT
            rec(i + 1, cur, used)
            cur.pop()

        rec(0, [], set())
        return results

    # -- 主过程 ---------------------------------------------------------------
    def solve(self) -> dict:
        self._t0 = _time.perf_counter()
        root = self._root_engine()
        # UB 初始候选：worker0 串行 singleton（必有可行解）
        ub_eng = root.clone_for_search()
        ub_plan = self._ub_plan(ub_eng)
        self.best_key = self._leaf_key(ub_eng, ub_plan)
        self.best_plan = ub_plan
        if self.mode == "enumerate":
            self._rec(root, F(0), [])
            status = "time_limit" if self.timeout else "optimal"
            return {"status": status, "UB": self.best_key[0],
                    "plan": self.best_plan, "nodes": self.nodes,
                    "LB": (self.best_key[0] if not self.timeout else None)}
        open_lb = self._bnb(root)
        status = "time_limit" if self.timeout else "optimal"
        if not self.timeout:
            lb = self.best_key[0]
        else:
            lb = min([x for x in [open_lb] if x is not None] + [self.best_key[0]])
        return {"status": status, "UB": self.best_key[0],
                "LB": lb,
                "plan": self.best_plan, "nodes": self.nodes,
                "open_min_lb": open_lb}

    def _ub_plan(self, eng):
        """全部 singleton 按 rid 在 worker0 串行、领取对齐网格。"""
        plan = []
        t = F(0)
        eng._physical_step(F(0))
        for rid in sorted(eng.w.specs):
            g = _ceil_grid(max(t, eng.w.t), self.inst.delta)
            eng.advance_until(g)
            eng._physical_step(g)
            eng._dispatch(g, 0, (rid,))
            eng._closure(g)
            plan.append((g, 0, (rid,)))
            # 推进到该批完成
            while eng.w.requests[rid].F_s is None and not eng.w.all_done():
                tn = eng._next_event()
                if tn is None:
                    break
                eng._advance(tn)
                eng._physical_step(tn)
            t = eng.w.t
        eng.advance_to_completion()
        return plan

    def _budget_ok(self):
        if self.nodes > self.node_budget:
            self.timeout = True
            return False
        if _time.perf_counter() - self._t0 > self.time_budget_s:
            self.timeout = True
            return False
        return True

    def _rec(self, eng, t, plan):
        if not self._budget_ok():
            return
        self.nodes += 1
        eng._physical_step(t)   # 处理恰在网格点的到达/完成
        if eng.w.all_done():
            key = self._leaf_key(eng, plan)
            if self.best_key is None or key < self.best_key:
                self.best_key, self.best_plan = key, list(plan)
            return
        if t > self.grid_end:
            if eng.w.n_queued > 0:
                return   # 最后一格后仍未领取：分支不可行
            st = eng.advance_to_completion()
            if st == "done":
                key = self._leaf_key(eng, plan)
                if self.best_key is None or key < self.best_key:
                    self.best_key, self.best_plan = key, list(plan)
            return
        idle = eng.w.idle_workers()
        queued = set(eng.w.queued_rids())
        if not idle or not queued:
            child = eng.clone_for_search()
            child.advance_until(t + self.inst.delta)
            self._rec(child, t + self.inst.delta, plan)
            return
        for assignment in self._assignments(idle, queued):
            child = eng.clone_for_search()
            for (w, comb) in assignment:
                if comb is not None:
                    child._dispatch(t, w, comb)
            child._closure(t)
            child.advance_until(t + self.inst.delta)
            self._rec(child, t + self.inst.delta,
                     plan + [(t, w, comb) for (w, comb) in assignment
                             if comb is not None])
        return

    def _bnb(self, root):
        """best-first B&B。返回结束时 OPEN 中的最小 LB（空则 None）。"""
        opened = []
        counter = [0]

        def push(eng, t, plan):
            eng._physical_step(t)
            lb = self.lower_bound(eng, t)
            counter[0] += 1
            heapq.heappush(opened, (lb, len(plan), counter[0], eng, t, plan))

        push(root, F(0), [])
        while opened:
            if not self._budget_ok():
                break
            lb, _d, _nid, eng, t, plan = heapq.heappop(opened)
            self.nodes += 1
            eng._physical_step(t)   # 恰在网格点的到达/完成先于本格决策（幂等）
            if eng.w.all_done():
                key = self._leaf_key(eng, plan)
                if self.best_key is None or key < self.best_key:
                    self.best_key, self.best_plan = key, list(plan)
                continue
            if t > self.grid_end:
                if eng.w.n_queued > 0:
                    continue
                if eng.advance_to_completion() == "done":
                    key = self._leaf_key(eng, plan)
                    if self.best_key is None or key < self.best_key:
                        self.best_key, self.best_plan = key, list(plan)
                continue
            idle = eng.w.idle_workers()
            queued = set(eng.w.queued_rids())
            if not idle or not queued:
                child = eng.clone_for_search()
                child.advance_until(t + self.inst.delta)
                push(child, t + self.inst.delta, plan)
                continue
            for assignment in self._assignments(idle, queued):
                child = eng.clone_for_search()
                for (w, comb) in assignment:
                    if comb is not None:
                        child._dispatch(t, w, comb)
                child._closure(t)
                child.advance_until(t + self.inst.delta)
                newplan = plan + [(t, w, comb) for (w, comb) in assignment
                                  if comb is not None]
                if child.w.all_done():
                    key = self._leaf_key(child, newplan)
                    if self.best_key is None or key < self.best_key:
                        self.best_key, self.best_plan = key, list(newplan)
                    continue
                clb = self.lower_bound(child, t + self.inst.delta)
                if clb >= self.best_key[0]:
                    continue   # 仅 LB>=UB 剪枝
                child._physical_step(t + self.inst.delta)
                counter[0] += 1
                heapq.heappush(opened, (clb, len(newplan), counter[0],
                                        child, t + self.inst.delta, newplan))
        return min((h[0] for h in opened), default=None)
