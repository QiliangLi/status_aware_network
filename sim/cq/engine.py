"""cq 物理执行内核：批与层状态机、唯一同刻闭包、worker 记账（规格 §4）。

同刻阶段顺序（§4.4，实现为显式循环，不依赖回调注册顺序）：
 1. 积分结算；处理已完成的读取/计算、请求到达、带宽变更；
 2. 先推进已有活动批（按 worker 升序启动依赖已满足的层并提交下一层读取，
    零字节/零计算引发的同刻事件继续闭包）；
 3. 发布观测、处理控制器返回和统一的新批领取（联合动作按 worker 升序原子
    落实，新批首层流提交晚于步骤 2 的旧批流）；
 4. 到期流提升申请率并重新分配带宽；
 5. 出现新空闲 worker 且有队列时同刻触发下一轮派发（每轮必须实际领取至少
    一个请求），循环至固定点后推进时间。

数值后端：E20/金标用 Fraction（严格相等），生产用 float64（1e-12 容差）。
"""
from __future__ import annotations

import time as _time
from typing import Dict, List, Optional, Tuple

from .config import CqScenario
from .profile import batch_compute_s, is_feasible, layer_read_gb, mem_peak_gb
from .storage import StorageSim
from .types import (Action, BatchRuntime, JointAction, RequestRuntime,
                    RequestSpec, WorkerState)


class WorldState:
    """物理真值容器。只属于执行内核与显式 Oracle；普通策略经 ObservableSnapshot。"""

    def __init__(self, scenario: CqScenario, specs, numeric=float):
        self.num = numeric
        self.zero = numeric(0)
        self.scenario = scenario
        self.L = scenario.profile.L
        self.m = scenario.m_workers
        self.specs: Dict[int, RequestSpec] = {s.rid: s for s in specs}
        order = sorted(specs, key=lambda s: (s.arrival_s, s.rid))
        self.arrival_order: List[int] = [s.rid for s in order]
        self.arrival_times = {s.rid: numeric(s.arrival_s) for s in order}
        self.arrival_idx = 0
        self.requests: Dict[int, RequestRuntime] = {
            s.rid: RequestRuntime(spec=s) for s in order}
        self.t = self.zero
        self.batches: Dict[int, BatchRuntime] = {}
        self.next_batch_id = 0
        self.workers: Dict[int, WorkerState] = {
            i: WorkerState(i, state_since=self.zero) for i in range(self.m)}
        self.storage = StorageSim(scenario.storage.b_schedule,
                                  scenario.storage.q_max_gbps, numeric)
        self.wake_timers: Dict[int, Tuple[object, int]] = {}
        self.wake_gen = 0
        self.n_queued = 0
        self.n_done = 0
        self.undrainable = False

    # -- 查询 ---------------------------------------------------------------
    def queued_rids(self) -> List[int]:
        return sorted([rid for rid, rr in self.requests.items()
                       if rr.state == "QUEUED"])

    def idle_workers(self) -> List[int]:
        return [i for i in range(self.m) if self.workers[i].batch_id is None]

    def next_arrival(self):
        if self.arrival_idx < len(self.arrival_order):
            rid = self.arrival_order[self.arrival_idx]
            return self.arrival_times[rid]
        return None

    def all_done(self) -> bool:
        return self.n_done == len(self.requests)

    def clone(self, keep_segments: bool = False) -> "WorldState":
        """深拷贝（预测/回放用）。默认丢弃 worker 段日志以提速。"""
        import copy as _copy
        w = WorldState.__new__(WorldState)
        w.num, w.zero, w.scenario, w.L, w.m = self.num, self.zero, self.scenario, self.L, self.m
        w.specs = self.specs
        w.arrival_order, w.arrival_times, w.arrival_idx = (
            self.arrival_order, self.arrival_times, self.arrival_idx)
        w.requests = {k: _copy.copy(v) for k, v in self.requests.items()}
        w.t = self.t
        w.batches = {}
        for k, b in self.batches.items():
            nb = _copy.copy(b)
            nb.read_ready = list(b.read_ready)
            nb.C = list(b.C)
            nb.Z = list(b.Z)
            nb.c_layers = b.c_layers
            nb.V_layers = b.V_layers
            w.batches[k] = nb
        w.next_batch_id = self.next_batch_id
        w.workers = {}
        for k, wk in self.workers.items():
            nw = _copy.copy(wk)
            nw.segments = wk.segments if keep_segments else []
            w.workers[k] = nw
        st = StorageSim.__new__(StorageSim)
        st.num, st.zero, st.b_schedule, st.q_max = (
            self.num, self.zero, self.storage.b_schedule, self.storage.q_max)
        st.flows = {seq: _copy.copy(f) for seq, f in self.storage.flows.items()}
        st.next_seq = self.storage.next_seq
        st.integrated_capacity_gb = self.storage.integrated_capacity_gb
        st.actual_integral_gb = self.storage.actual_integral_gb
        st.requested_integral_gb = self.storage.requested_integral_gb
        st.rate_log = []
        st.tol = self.storage.tol
        w.storage = st
        w.wake_timers = dict(self.wake_timers)
        w.wake_gen = self.wake_gen
        w.n_queued, w.n_done = self.n_queued, self.n_done
        w.undrainable = self.undrainable
        return w


class CqEngine:
    """事件推进器。policy.decide(snapshot, scenario) -> JointAction。"""

    def __init__(self, scenario: CqScenario, specs, policy, numeric=float,
                 fallback_policy=None, observable=None, max_events=None,
                 record_decisions=True):
        self.scn = scenario
        self.w = WorldState(scenario, specs, numeric)
        self.num = numeric
        self.zero = self.w.zero
        self.policy = policy
        self.fallback = fallback_policy
        self.observable = observable
        self.max_events = max_events
        self.record = record_decisions
        self.decisions: List[dict] = []
        self.pending = None           # (return_at, action, start_s, meta)
        self.events_processed = 0
        self.status = "init"
        self.budget_exhausted = False
        self.ctrl_time_s = 0.0        # decide 累计 CPU 秒（measured）
        self.n_fallback = 0
        self.n_stale_invalid = 0

    # ------------------------------------------------------------------
    # 事件源
    # ------------------------------------------------------------------
    def _next_event(self):
        t = self.w.t
        cands = []
        for seq, fin in self._completion_times(t):
            cands.append(fin)
        nd = self.w.storage.next_due(t)
        if nd is not None:
            cands.append(nd)
        for b in self.w.batches.values():
            if b.F_s is None and b.next_layer > 0:
                z = b.Z[b.next_layer - 1]
                if z is not None and z > t:
                    cands.append(z)
        na = self.w.next_arrival()
        if na is not None:
            cands.append(na)
        nb = self.w.storage.next_break(t)
        if nb is not None:
            cands.append(nb)
        if self.observable is not None and (
                self.w.n_queued > 0
                or self.w.arrival_idx < len(self.w.arrival_order)):
            # 队列非空才可能有决策；队列为空且无未来到达时策略不会被调用，
            # 跳过采样事件（σ=0 恒 B 下无观测影响）
            ot = self.observable.next_event_time(t)
            if ot is not None:
                cands.append(ot)
        if self.pending is not None and self.pending[0] > t:
            cands.append(self.pending[0])
        for wid, (wat, _g) in self.w.wake_timers.items():
            if wat > t:
                cands.append(wat)
        cands = [c for c in cands if c > t]
        if not cands:
            return None
        return min(cands)

    def _completion_times(self, t):
        return self.w.storage.completions(t)

    # ------------------------------------------------------------------
    # 步骤 1：积分与完成处理
    # ------------------------------------------------------------------
    def _advance(self, t_next):
        t0 = self.w.t
        self.w.storage.advance(t0, t_next)
        for wk in self.w.workers.values():
            pass  # worker 积分在 set_state 惰性结算
        self.w.t = t_next

    def _step(self, t):
        self._physical_step(t)
        # 步骤 3+5：控制器（返回、发起、固定点）
        self._controller_tick(t)

    def _physical_step(self, t):
        """§4.4 的步骤 1/2/4：完成、到达、带宽、旧批闭包、到期提升（无决策）。"""
        st = self.w.storage
        # 已完成的读取
        done_seqs = [seq for seq, f in st.flows.items() if st.is_complete(f)]
        for seq in sorted(done_seqs):
            f = st.remove(seq, t)
            b = self.w.batches.get(f.batch_id)
            if b is not None:
                b.read_ready[f.layer] = t
            if self.observable is not None:
                self.observable.on_flow_complete(f, t)
        if done_seqs:
            st.allocate(t)
        # 已完成的计算：在 _closure 的 (a) 分支处理（含批完成）
        # 请求到达
        while self.w.arrival_idx < len(self.w.arrival_order):
            rid = self.w.arrival_order[self.w.arrival_idx]
            at = self.w.arrival_times[rid]
            if at <= t:
                rr = self.w.requests[rid]
                rr.state = "QUEUED"
                rr.queue_enter_s = at
                self.w.n_queued += 1
                self.w.arrival_idx += 1
                if self.observable is not None:
                    self.observable.on_arrival(rid, t)
            else:
                break
        # 带宽变更在 allocate 中自动生效；此处触发一次重分配
        st.allocate(t)
        if self.observable is not None:
            self.observable.on_instant(t)
        # 步骤 2：旧批闭包
        self._closure(t)
        # 步骤 4：到期提升
        if st.boost_due(t):
            st.allocate(t)

    # -- 精确求解 / 预测用的物理推进（无策略决策） -------------------------
    def advance_until(self, t_target):
        """推进物理（含途中事件处理）直到 world.t == t_target。"""
        while not self.w.all_done():
            tn = self._next_event()
            if tn is None:
                if self.w.storage.flows and not self.w.storage.has_future_positive(self.w.t):
                    self.w.undrainable = True
                self.w.storage.advance(self.w.t, t_target)
                self.w.t = t_target
                return
            if tn >= t_target:
                self.w.storage.advance(self.w.t, t_target)
                self.w.t = t_target
                return
            self._advance(tn)
            self.events_processed += 1
            self._physical_step(tn)
        if self.w.t < t_target:
            self.w.storage.advance(self.w.t, t_target)
            self.w.t = t_target

    def advance_to_completion(self):
        """推进物理直到全部完成或无事件。"""
        while not self.w.all_done():
            tn = self._next_event()
            if tn is None:
                if self.w.storage.flows and not self.w.storage.has_future_positive(self.w.t):
                    self.w.undrainable = True
                self.status = "undrainable" if self.w.undrainable else "stalled_no_event"
                return self.status
            self._advance(tn)
            self.events_processed += 1
            self._physical_step(tn)
        self.status = "done"
        return self.status

    def clone_for_search(self) -> "CqEngine":
        """深拷贝引擎状态用于搜索分支（无策略/观测，共享只读配置）。"""
        e = CqEngine.__new__(CqEngine)
        e.scn = self.scn
        e.w = self.w.clone()
        e.num, e.zero = self.num, self.zero
        e.policy = e.fallback = e.observable = None
        e.max_events = None
        e.record = False
        e.decisions = []
        e.pending = None
        e.events_processed = self.events_processed
        e.status = self.status
        e.budget_exhausted = False
        e.ctrl_time_s = 0.0
        e.n_fallback = e.n_stale_invalid = 0
        return e

    # ------------------------------------------------------------------
    # 步骤 2：旧批闭包（零字节/零计算级联）
    # ------------------------------------------------------------------
    def _closure(self, t):
        w = self.w
        changed = True
        while changed:
            changed = False
            for wid in sorted(w.workers):
                wk = w.workers[wid]
                b = w.batches.get(wk.batch_id) if wk.batch_id is not None else None
                if b is None or b.F_s is not None:
                    continue
                # (a) 正在计算的层是否已到 Z
                if wk.state == "COMPUTE" and b.next_layer > 0:
                    l_cur = b.next_layer - 1
                    z = b.Z[l_cur]
                    if z is not None and z <= t:
                        if l_cur == self.w.L - 1:
                            self._finish_batch(b, t)
                            changed = True
                            continue
                        wk.set_state(t, "STALL", b.batch_id)
                        changed = True
                # (b) 启动下一层计算（依赖：read_ready 且前层 Z<=t）
                l = b.next_layer
                if l < self.w.L and b.read_ready[l] is not None:
                    prev_ok = (l == 0) or (b.Z[l - 1] is not None and b.Z[l - 1] <= t)
                    if prev_ok:
                        R = b.read_ready[l]
                        C = R if l == 0 else max(R, b.Z[l - 1])
                        self._start_layer(wid, b, l, C, t)
                        changed = True

    def _start_layer(self, wid: int, b: BatchRuntime, l: int, C, t):
        """启动层 l 的计算：记录 C/Z，切换 worker COMPUTE，提交下一层流。"""
        c = b.c_layers[l]
        b.C[l] = C
        b.Z[l] = C + c
        b.next_layer = l + 1
        self.w.workers[wid].set_state(t, "COMPUTE", b.batch_id, l)
        b.compute_s = b.compute_s + c
        if l + 1 < self.w.L:
            V_next = b.V_layers[l + 1]
            if V_next > 0:
                if c > 0:
                    q = min(self.w.storage.q_max, V_next / c)
                    due = b.Z[l]
                else:
                    q = self.w.storage.q_max
                    due = None
                f = self.w.storage.submit(b.batch_id * 1000 + l + 1, b.batch_id,
                                          l + 1, C, V_next, q, due)
                if self.observable is not None:
                    self.observable.on_flow_submit(f, t)
                self.w.storage.allocate(t)
            else:
                b.read_ready[l + 1] = C

    def _finish_batch(self, b: BatchRuntime, t):
        F = b.Z[self.w.L - 1]
        b.F_s = F
        wk = self.w.workers[b.worker_id]
        wk.set_state(F, "IDLE")
        b.stall_s = (F - b.dispatch_s) - b.compute_s
        for rid in b.members:
            rr = self.w.requests[rid]
            rr.state = "DONE"
            rr.F_s = F
            self.w.n_done += 1
        self.w.batches[b.batch_id] = b
        if self.observable is not None:
            self.observable.on_batch_done(b, F)

    # ------------------------------------------------------------------
    # 控制器（§6.4）
    # ------------------------------------------------------------------
    def _controller_tick(self, t):
        # 处理返回
        if self.pending is not None and self.pending[0] <= t:
            _ret, act, start_s, meta = self.pending
            self.pending = None
            self._apply_action(t, act, stale=(t > start_s))
        # 发起 + 固定点
        rounds = 0
        while (self.pending is None and self.w.n_queued > 0
               and self.w.idle_workers()):
            if rounds > self.w.m * 4 + 8:
                break
            n0 = self.w.next_batch_id
            self._start_decision(t)
            if self.pending is not None:
                break
            if self.w.next_batch_id == n0:
                break   # 纯 WAIT（或无可行批）：本刻不再派发
            rounds += 1

    def _start_decision(self, t):
        """构造快照并调用策略；zero 同刻返回，fixed/measured 延迟生效。"""
        if self.observable is None:
            raise RuntimeError("无观测层的引擎不能做策略决策（预测模式请注入 π）")
        snap = self.observable.snapshot(t)
        t0 = _time.perf_counter_ns()
        act = self.policy.decide(snap, self.scn)
        el = (_time.perf_counter_ns() - t0) * 1e-9
        self.ctrl_time_s += el
        mode = self.scn.controller.cost_mode
        if mode == "zero":
            delay = 0.0
        elif mode == "fixed":
            delay = self.num(self.scn.controller.fixed_delay_s)
        else:  # measured
            delay = el
        meta = {"decide_s": el, "mode": mode, "start_s": t}
        if self.record:
            self.decisions.append({
                "now": t, "ctrl_s": el, "mode": mode, "action": _act_json(act)})
        if delay <= 0:
            self._apply_action(t, act, stale=False)
        else:
            self.pending = (t + delay, act, t, meta)

    def _apply_action(self, t, act: JointAction, stale: bool):
        """验证→按 worker 升序领取。过期 WAIT 剔除；全 WAIT 过期回退基础动作。"""
        w = self.w
        acts = []
        for a in act.actions:
            if a.kind == "WAIT" and (a.wake_at is None or a.wake_at <= t):
                continue   # 过期 WAIT：删除约束，worker 立即可参与下次决策
            acts.append(a)
        ok = self._validate(t, acts)
        if not ok:
            self.n_stale_invalid += 1
            if self.fallback is not None:
                snap = self.observable.snapshot(t)
                fb = self.fallback.decide(snap, self.scn)
                self.n_fallback += 1
                acts = [a for a in fb.actions
                        if not (a.kind == "WAIT" and a.wake_at is not None
                                and a.wake_at <= t)]
            else:
                acts = []
        # 按 worker 升序落实（联合动作原子性已在验证中保证）
        for a in sorted(acts, key=lambda x: x.worker_id):
            if a.kind == "DISPATCH":
                self._dispatch(t, a.worker_id, a.members)
                w.wake_timers.pop(a.worker_id, None)
            else:
                w.wake_gen += 1
                w.wake_timers[a.worker_id] = (a.wake_at, w.wake_gen)
        # 新领零工作批的同刻闭包（在旧批之后执行）
        self._closure(t)

    def _validate(self, t, acts) -> bool:
        seen = set()
        wset = set()
        for a in acts:
            if a.worker_id in wset or a.worker_id >= self.w.m:
                return False
            wset.add(a.worker_id)
            if a.kind == "DISPATCH":
                if not a.members:
                    return False
                if self.w.workers[a.worker_id].batch_id is not None:
                    return False
                for r in a.members:
                    rr = self.w.requests.get(r)
                    if rr is None or rr.state != "QUEUED" or r in seen:
                        return False
                    seen.add(r)
                specs = [self.w.specs[r] for r in a.members]
                if not is_feasible(specs, self.scn.limits, self.scn.profile):
                    return False
            else:
                if a.wake_at is None or a.wake_at <= t:
                    return False
        return True

    def _dispatch(self, t, worker_id: int, members):
        w = self.w
        rids = tuple(sorted(members))
        specs = [w.specs[r] for r in rids]
        c = batch_compute_s(specs, self.scn.profile)
        Vl = sum(layer_read_gb(s, self.scn.profile) for s in specs)
        b = BatchRuntime(batch_id=w.next_batch_id, members=rids,
                         worker_id=worker_id, dispatch_s=t, L=w.L)
        b.c_layers = [c for _ in range(w.L)]
        b.V_layers = [Vl for _ in range(w.L)]
        b.mem_peak_gb = mem_peak_gb(specs, self.scn.profile)
        w.next_batch_id += 1
        w.batches[b.batch_id] = b
        wk = w.workers[worker_id]
        wk.set_state(t, "STALL", b.batch_id, 0)
        for r in rids:
            rr = w.requests[r]
            rr.state = "ACTIVE"
            rr.batch_id = b.batch_id
            rr.dispatch_s = t
        w.n_queued -= len(rids)
        if self.observable is not None:
            self.observable.on_dispatch(b, t)
        if Vl > 0:
            q = w.storage.q_max
            f = w.storage.submit(b.batch_id * 1000, b.batch_id, 0, t, Vl, q, None)
            if self.observable is not None:
                self.observable.on_flow_submit(f, t)
            w.storage.allocate(t)
        else:
            b.read_ready[0] = t

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, drain_deadline=None):
        # 初始时刻（t=0）：先处理到达/初始观测/首轮决策，再进入事件循环
        self._step(self.w.t)
        while True:
            tn = self._next_event()
            if tn is None:
                if self.w.all_done():
                    self.status = "done"
                else:
                    # 无事件：可能 undrainable（无未来正带宽）或控制器死锁
                    if self.w.storage.flows and not self.w.storage.has_future_positive(self.w.t):
                        self.w.undrainable = True
                        self.status = "undrainable"
                    else:
                        self.status = "stalled_no_event"
                break
            if drain_deadline is not None and tn > drain_deadline:
                self.w.storage.advance(self.w.t, drain_deadline)
                self.w.t = drain_deadline
                self.status = "drain_deadline"
                break
            if tn <= self.w.t:
                tn = self.w.t  # 数值容差保护，避免零时间回退
            else:
                self._advance(tn)
            self.events_processed += 1
            if self.max_events is not None and self.events_processed > self.max_events:
                self.budget_exhausted = True
                self.status = "budget_exhausted"
                break
            self._step(tn)
            if self.w.all_done():
                self.status = "done"
                break
        # 结算 worker 到最终时刻
        for wk in self.w.workers.values():
            wk.set_state(self.w.t, "FINAL")
        self._finalize()
        return self.status

    def _finalize(self):
        """把 worker 段的 FINAL 状态折回（FINAL 仅用于结算最后一段）。"""
        for wk in self.w.workers.values():
            wk.state = "IDLE" if wk.state == "FINAL" else wk.state

    # 供 E22 / 审计使用 -----------------------------------------------------
    def flow_events(self):
        return self.w.storage.rate_log

    def served_gb(self):
        return self.w.storage.actual_integral_gb


def _act_json(act: JointAction):
    return [{"kind": a.kind, "worker": a.worker_id,
             "members": list(a.members),
             "wake_at": str(a.wake_at) if a.wake_at is not None else None}
            for a in act.actions]
