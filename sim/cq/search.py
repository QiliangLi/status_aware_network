"""在线搜索：候选请求/批、联合动作 beam、cq_local / cq_mpc、gate（§7）。

- 候选构建按到达、T0、deadline、slack 四列表轮询；对每个 anchor 构造同形/
  互补/紧迫三种顺序，逐项尝试加入合法成员并保留每个中间批；
- 联合动作逐 worker 扩展（DISPATCH 候选 + WAIT），局部完整评分保留 J_width；
- MPC 主体按 §7.5 伪码：base=guarded_EDF，H 层 beam，尾评估用同一 π 排空；
- 双预算：CPU 软预算 search_budget_s + 确定性事件预算 max_forecast_events
  （每次 decide 共享；超限返回基础动作并记 fallback）。
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import replace
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from .config import CqScenario
from .engine import CqEngine
from .storage import StorageSim
from .observable import Observable, ObservableSnapshot
from .policies import GuardedEDF, _specs_of, aged_anchor
from .profile import is_feasible, layer_read_gb
from .types import Action, BatchRuntime, JointAction, RequestSpec

ZERO = 0.0


def _g_batch(members_specs, scn) -> float:
    from .profile import batch_compute_s, singleton_K
    bk = float(batch_compute_s(members_specs, scn.profile)) * scn.profile.L
    if bk <= 0:
        return 0.0
    sk = sum(float(singleton_K(s, scn.profile, scn.storage.b_ref_gbps,
                                scn.storage.q_max_gbps)) for s in members_specs)
    return sk / bk


def select_requests(snap, scn, K_req) -> List[int]:
    """四列表轮询各取一个，去重直到 K_req（§7.2）。"""
    reqs = [r for r in snap.requests if r.rid in snap.queued]
    by_arrival = sorted(reqs, key=lambda r: (r.arrival_s, r.rid))
    by_t0 = sorted(reqs, key=lambda r: (r.T0_s, r.arrival_s, r.rid))
    by_dl = sorted(reqs, key=lambda r: (r.deadline_s, r.arrival_s, r.rid))

    def slack_key(r):
        try:
            return (snap.slack_s(r.rid), r.arrival_s, r.rid)
        except Exception:
            return (float(r.deadline_s) - float(snap.now), r.arrival_s, r.rid)
    by_slack = sorted(reqs, key=slack_key)
    chosen: List[int] = []
    seen = set()
    while len(chosen) < K_req:
        progressed = False
        for lst in (by_arrival, by_t0, by_dl, by_slack):
            cand = next((r for r in lst if r.rid not in seen), None)
            if cand is not None:
                seen.add(cand.rid)
                chosen.append(cand.rid)
                progressed = True
                if len(chosen) >= K_req:
                    break
        if not progressed:
            break
    return chosen


def build_batches(snap, scn, pi, K_req, K_batch) -> List[Tuple[int, ...]]:
    """候选批目录（§7.2）：三顺序中间批 + singleton + 老化锚点 + π 批。"""
    specs = _specs_of(snap)
    chosen = select_requests(snap, scn, K_req)
    # 强制补入 π 当前联合动作成员与老化锚点
    pi_act = pi.decide(snap, scn)
    for a in pi_act.actions:
        if a.kind == "DISPATCH":
            for r in a.members:
                if r not in chosen:
                    chosen.append(r)
    anc = aged_anchor(snap, getattr(scn, "guard_w", None))
    if anc is not None and anc not in chosen:
        chosen.append(anc)

    reqs = {r.rid: r for r in snap.requests if r.rid in snap.queued}
    cand: set = set()
    for a in chosen:
        cand.add((a,))   # 所有 singleton
    for a in chosen:
        others = [r for r in chosen if r != a]

        def h_u(rid):
            r = reqs[rid]
            return r.h_tokens, r.u_tokens

        def key_same(r):
            ha, ua = h_u(a)
            h, u = h_u(r)
            return (abs(math.log1p(h) - math.log1p(ha)) +
                    abs(math.log1p(u) - math.log1p(ua)), reqs[r].arrival_s, r)

        def ratio(rid):
            V = float(layer_read_gb(specs[rid], scn.profile))
            K = float(reqs[rid].T0_s)
            if V <= 0 or K <= 0:
                return None
            return V / K

        r_ref = 1.0

        def key_comp(r):
            ri, ra = ratio(r), ratio(a)
            if ri is None or ra is None or ra <= 0:
                return (1, 0.0, reqs[r].arrival_s, r)
            return (0, abs(math.log(ri) + math.log(ra) - 2 * math.log(r_ref)),
                    reqs[r].arrival_s, r)

        def key_urg(r):
            return (reqs[r].deadline_s, reqs[r].arrival_s, r)

        for key in (key_same, key_comp, key_urg):
            members = [a]
            for r in sorted(others, key=key):
                trial = sorted(members + [r])
                ms = [specs[x] for x in trial]
                if is_feasible(ms, scn.limits, scn.profile):
                    members = members + [r]
                    cand.add(tuple(sorted(members)))   # 保留每个中间批
                # 不可行：跳过该成员继续（候选构建允许，区别于基线填批）
        # singleton 已加
    for a_ in pi_act.actions:
        if a_.kind == "DISPATCH" and a_.members:
            cand.add(tuple(sorted(a_.members)))
    if anc is not None:
        for r in chosen:
            if r == anc:
                continue
            trial = tuple(sorted([anc, r]))
            if is_feasible([specs[x] for x in trial], scn.limits, scn.profile):
                cand.add(trial)
    # 预排序：(-G, min_slack, sumV/sumK, members)
    def pre(members):
        ms = [specs[x] for x in members]
        g = _g_batch(ms, scn)
        dls = [float(reqs[x].deadline_s) for x in members]
        V = sum(float(layer_read_gb(s, scn.profile)) for s in ms)
        Ksum = sum(float(reqs[x].T0_s) for x in members)
        vk = V / Ksum if Ksum > 0 else float("inf")
        return (-g, min(dls), vk, members)
    ranked = sorted(cand, key=pre)
    keep = list(ranked[:K_batch])
    # 超出 K_batch 的额度：singleton、老化锚点批与 π 批仍保留（合法）
    extra = [c for c in ranked[K_batch:] if len(c) == 1
             or (anc is not None and anc in c)
             or c in {tuple(sorted(a.members)) for a in pi_act.actions
                      if a.kind == "DISPATCH"}]
    return keep + extra


# ---------------------------------------------------------------------------
# 预测：从快照重建估计世界并排空（§6.2）
# ---------------------------------------------------------------------------

def build_forecast_engine(snap: ObservableSnapshot, scn: CqScenario,
                          pi, est_bw: Optional[float] = None,
                          local: bool = False) -> CqEngine:
    """从公开信息重建估计世界（不复制资源真值对象），挂 π 闭环。"""
    from .config import StorageConfig
    bw = est_bw if est_bw is not None else snap.est_bw()
    if local:
        bw = snap.est_bw()   # local 评分由独立分配实现，见 _LocalStorage
    scn_est = replace(
        scn, storage=StorageConfig(b_schedule=((Fraction(0), Fraction(int(bw * 1e9), 10**9)),),
                                   q_max_gbps=scn.storage.q_max_gbps,
                                   b_ref_gbps=scn.storage.b_ref_gbps),
        cohort=replace(scn.cohort, drain_budget_s=Fraction(10**6)))
    specs = []
    for r in snap.requests:
        specs.append(RequestSpec(rid=r.rid, arrival_s=r.arrival_s,
                                 h_tokens=r.h_tokens, u_tokens=r.u_tokens,
                                 class_id="", T0_s=r.T0_s, deadline_s=r.deadline_s))
    obs = Observable(scn_est, numeric=float)
    eng = CqEngine(scn_est, specs, policy=pi, numeric=float, observable=obs,
                   record_decisions=False)
    obs.attach(eng)
    w = eng.w
    if local:
        # 独立带宽评分世界：替换推演存储（须在流注入前；clone 按类型保留子类）
        w.storage = _IndependentBWStorage(
            ((Fraction(0), Fraction(int(bw * 1e9), 10**9)),),
            scn_est.storage.q_max_gbps, float)
    now = float(snap.now)
    # QUEUED 请求直接入队
    for r in snap.requests:
        rr = w.requests[r.rid]
        if r.rid in snap.queued:
            rr.state = "QUEUED"
            w.n_queued += 1
            w.arrival_idx += 1
        else:
            rr.state = "ACTIVE"
            w.arrival_idx += 1
    # ACTIVE 批重建
    flow_map = {f.flow_id: f for f in snap.flows}
    for pw in snap.workers:
        if not pw.members:
            continue
        c_hat = snap.c_hat([(next(r for r in snap.requests if r.rid == x).h_tokens,
                             next(r for r in snap.requests if r.rid == x).u_tokens)
                            for x in pw.members])
        Vl = sum(float(layer_read_gb(w.specs[x], scn.profile)) for x in pw.members)
        b = BatchRuntime(batch_id=pw.worker_id + 1, members=tuple(pw.members),
                         worker_id=pw.worker_id, dispatch_s=now, L=w.L)
        b.c_layers = [c_hat for _ in range(w.L)]
        b.V_layers = [Vl for _ in range(w.L)]
        ld = pw.layers_done
        for l in range(ld):
            b.read_ready[l] = 0.0
        if ld > 0:
            rem = max(0.0, c_hat - max(0.0, now - float(pw.cur_layer_start_s or now)))
            prev_z = now + rem if ld - 1 == max(0, ld - 1) else now
            b.Z[ld - 1] = now + rem
            b.C[ld - 1] = now
            b.next_layer = ld
            b.compute_s = ld * c_hat
        w.batches[b.batch_id] = b
        wk = w.workers[pw.worker_id]
        wk.batch_id = b.batch_id
        wk.state = "COMPUTE" if (ld > 0 and b.Z[ld - 1] > now) else "STALL"
        wk.state_since = now
        wk.compute_s = 0.0
        wk.stall_s = 0.0
    # 未完成流注入
    for f in snap.flows:
        if f.completed:
            continue
        served = f.served_reported_gb if f.served_reported_gb is not None else 0.0
        remaining = max(0.0, float(f.V_gb) - float(served))
        st = w.storage
        fl = st.submit(f.flow_id, f.batch_id, f.layer, float(f.submit_s),
                       float(f.V_gb), st.q_max, None)
        fl.remaining_gb = remaining
    w.t = now
    w.storage.allocate(now)
    return eng


class _IndependentBWStorage(StorageSim):
    """cq_local 评分消融（20260917 真实实现）：带宽独立——每个流各得
    min(申请率, B̂)，忽略 FCFS 共享竞争。仅用于推演评分世界（物理执行
    仍走共享 FCFS）；作用是隔离"预测时是否建模跨 worker 干扰"的价值：
    带宽不紧张时与共享预测几乎一致，竞争紧张时系统性乐观（低估等待）。
    """

    def allocate(self, t):
        cap = self.b_at(t)
        for f in sorted(self.flows.values(), key=lambda x: x.submit_seq):
            f.rate_gbps = min(f.q_gbps, cap)


class _HoldPi:
    """推演期的等待保持器：wake 时刻之前，π 不得向被 HOLD 的 worker 派批。

    规格 §7.5 要求候选落实后"推进至严格未来的可决策事件"；没有本包装时，
    闭环 π 会在同刻立即向等待中的 worker 派批，使 WAIT 候选的推演退化为
    基础动作（MPC 因此永远选不出等待——20260917 审计发现并修复）。
    """

    def __init__(self, pi, holds: Dict[int, float]):
        self.pi = pi
        self.holds = holds

    def decide(self, snap, scn):
        base = self.pi.decide(snap, scn)
        now = float(snap.now)
        acts = tuple(a for a in base.actions
                     if not (a.worker_id in self.holds
                             and now < float(self.holds[a.worker_id])))
        return JointAction(acts)


def forecast_drain(eng: CqEngine, first_action: Optional[JointAction],
                   event_budget: List[int]) -> Optional[dict]:
    """落实首动作（含 WAIT）后用 π 排空；返回各请求 F（None=超预算/不可排空）。

    首动作经引擎自身的 _apply_action 落实：DISPATCH 正常派批，WAIT 写入
    wake_timers；随后用 _HoldPi 包装 π/fallback，在唤醒时刻前禁止向等待
    worker 派批，唤醒事件（engine._next_event 已含 wake_timers）到达后
    恢复正常闭环。无 WAIT 的动作路径与修复前行为一致。
    """
    holds: Dict[int, float] = {}
    if first_action is not None:
        for a in first_action.actions:
            if a.kind == "WAIT" and a.wake_at is not None:
                holds[a.worker_id] = float(a.wake_at)
        eng._apply_action(eng.w.t, first_action, stale=False)
    if holds:
        eng.policy = _HoldPi(eng.policy, holds)
        eng.fallback = _HoldPi(eng.fallback, holds)
    eng.max_events = event_budget[0]
    eng.events_processed = 0
    status = eng.run(drain_deadline=None)
    if status == "budget_exhausted":
        event_budget[0] = 0
        return None
    if eng.w.undrainable or status not in ("done",):
        return None
    event_budget[0] -= eng.events_processed
    return {rid: (float(rr.F_s) if rr.F_s is not None else None)
            for rid, rr in eng.w.requests.items()}


def score_of(Fs: Dict[int, Optional[float]], snap, theta: str):
    """词典序 score（§7.4）；K=snap 全部未完成请求。"""
    now = float(snap.now)
    reqs = {r.rid: r for r in snap.requests}
    vals = []
    fails = 0
    ttft_sum = 0.0
    norm_sum = 0.0
    maxF = 0.0
    for rid, F in Fs.items():
        if F is None:
            return None
        r = reqs[rid]
        ttft = F - float(r.arrival_s)
        ttft_sum += ttft
        norm_sum += ttft / float(r.T0_s)
        maxF = max(maxF, F)
        if F > float(r.deadline_s):
            fails += 1
    if theta == "M":
        return (maxF, ttft_sum)
    if theta == "S":
        return (fails, ttft_sum)
    if theta == "T":
        return (ttft_sum, fails)
    return (norm_sum, ttft_sum)


# ---------------------------------------------------------------------------
# MPC / local
# ---------------------------------------------------------------------------

class MPCPolicy:
    """cq_mpc（global rollout）/ cq_local（忽略跨 worker 干扰的评分消融）。"""

    def __init__(self, pid: str = "cq_mpc", local: bool = False,
                 H: int = 2, beam_width: int = 8, J_width: int = 16,
                 K_req: int = 32, K_batch: int = 24,
                 budget_s: float = 0.002, max_events: int = 20000,
                 theta: str = "S", gate: float = 0.0, c_ref: float = 0.0):
        self.pid = pid
        self.local = local
        self.H = H
        self.beam_width = beam_width
        self.J_width = J_width
        self.K_req = K_req
        self.K_batch = K_batch
        self.budget_s = budget_s
        self.max_events = max_events
        self.theta = theta
        self.gate = gate
        self.c_ref = c_ref or 0.01
        self.pi = GuardedEDF()
        self.n_fallback = 0
        self.n_overrun = 0
        self.n_scored = 0          # 成功评分的候选动作数（审计用）
        self.n_depth_clamped = 0   # H>1 被按 H=1 执行的决策数（审计用）

    def decide(self, snap, scn) -> JointAction:
        t0 = _time.perf_counter()
        budget = [self.max_events]
        base = self.pi.decide(snap, scn)
        if not snap.idle_workers or not snap.queued:
            return base
        # 决策点原始副本：基础排空与每个候选都从"当前状态"出发。
        # 20260917 审计修复：此前候选从已排空的 base 副本克隆，DISPATCH 验证
        # 必然失败、全体候选得分==基础动作，搜索从未真正分支。
        est0 = build_forecast_engine(snap, scn, self.pi, local=self.local)
        est_base = self._clone_forecast(est0)
        Fs = forecast_drain(est_base, base, budget)
        if Fs is None:
            self.n_fallback += 1
            return base
        base_score = score_of(Fs, snap, self.theta)
        if base_score is None:
            self.n_fallback += 1
            return base
        best = (base_score, base)
        # H>1 的逐决策点分支未实现（原 beam 从排空态展开，无效）：统一按
        # H=1 语义执行并记录，避免静默的假深度。
        if self.H > 1:
            self.n_depth_clamped += 1
        node_snap = est0.observable.snapshot(est0.w.t)
        batches = build_batches(node_snap, scn, self.pi, self.K_req, self.K_batch)
        idle = sorted(node_snap.idle_workers)
        actions = self._joint_actions(est0, node_snap, scn, batches, idle)
        for act in actions:
            if _time.perf_counter() - t0 > self.budget_s or budget[0] <= 0:
                self.n_overrun += 1
                break
            child = self._clone_forecast(est0)
            Fs2 = forecast_drain(child, act, budget)
            if Fs2 is None:
                continue
            sc = score_of(Fs2, snap, self.theta)
            if sc is None:
                continue
            self.n_scored += 1
            if sc < best[0]:
                best = (sc, act)
        # gate：主目标须严格改善（默认 gate=0）
        if best[1] is not base and best[0][0] < base_score[0] - self.gate:
            return best[1]
        return base

    def _clone_forecast(self, eng: CqEngine) -> CqEngine:
        e = eng.clone_for_search()
        e.policy = eng.policy
        obs = Observable(eng.scn, numeric=float)
        obs.attach(e)
        e.observable = obs
        return e

    def _joint_actions(self, node, node_snap, scn, batches, idle):
        """逐 worker 扩展：DISPATCH 候选 + WAIT；保留 J_width 个部分动作。"""
        if not idle:
            return []
        wait_opts = self._wait_options(node_snap, scn)
        partial = [([], 0)]
        used_rids: set = set()
        results = []
        for wi, wid in enumerate(idle):
            new_partial = []
            for (acts, _h) in partial:
                for b in batches:
                    if used_rids_check(acts, b):
                        new_partial.append((acts + [("D", wid, b)], 0))
                for wt in wait_opts:
                    new_partial.append((acts + [("W", wid, wt)], 0))
            new_partial = new_partial[: self.J_width] if len(new_partial) > self.J_width else new_partial
            partial = new_partial
        out = []
        for acts, _h in partial:
            ja = []
            for a in acts:
                if a[0] == "D":
                    ja.append(Action("DISPATCH", a[1], tuple(a[2])))
                else:
                    ja.append(Action("WAIT", a[1], (), a[2]))
            out.append(JointAction(tuple(ja)))
        # 额外保留 π 完整动作与全体同刻 WAIT
        pi_act = self.pi.decide(node_snap, scn)
        out.append(pi_act)
        if wait_opts:
            out.append(JointAction(tuple(Action("WAIT", wid, (), min(wait_opts))
                                         for wid in idle)))
        return out[: max(self.J_width, 8)]

    def _wait_options(self, snap, scn):
        now = float(snap.now)
        # C_ref 在线化(v1.4):已到达请求 T0 的滚动中位数
        import numpy as _np
        t0s = [float(r.T0_s) for r in snap.requests]
        C_ref = float(_np.median(t0s)) if t0s else self.c_ref
        opts = [now + f * C_ref for f in (0.05, 0.1, 0.2)]
        for f in snap.flows:
            if f.completed:
                continue
            served = f.served_reported_gb or 0.0
            rem = max(0.0, float(f.V_gb) - float(served))
            bw = snap.est_bw()
            if bw > 0:
                tdone = float(f.submit_s) + float(f.V_gb) / bw
                if now < tdone <= now + 0.2 * C_ref:
                    opts.append(tdone)
        opts = [o for o in opts if o > now]
        return sorted(set(round(o, 9) for o in opts))


def used_rids_check(acts, b) -> bool:
    for a in acts:
        if a[0] == "D" and set(a[2]) & set(b):
            return False
    return True
