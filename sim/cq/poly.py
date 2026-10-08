"""CPA 闭式管道接纳（Closed-form Pipe Admission）：多项式替代 MPC 的枚举+推演。

设计文档：docs/CPA闭式管道接纳算法设计-20261008.md。三层结构：

- 公式层（wave_F / survival_cap / mixed_wave_F）：对称 cohort 完成时间统一
  闭式 F_p = p·s + (L−1)·w + c（w=max(c, k·s)，s=V/B），O(1) 每候选。
  引擎对拍（探针 20261008）：A-only k=1..32 绝对误差 ≤0.1ms；混合波饱和
  段（k_a·s_a≥c_a）精确（16A+16B 同刻末位 229.3ms 复算吻合），未饱和段
  高估轻流伤害（保守偏向"重者先行、轻流让路"）；
- 决策层（CPAPolicy）：完整对称族 (k_0..k_j)（重桶计数 k_h 全程 0..m′，
  轻 OH桶组合）公式预筛 O(m²) → 每个 k_h 保最优轻桶向量 → top-K 真推演
  复核（mode="rollout"，复用 MPC 的 forecast_drain，π=GuardedEDF-BW 延续
  与 mpc_v2.1 同口径）或纯公式取优（mode="form"，零推演）。对照 MPC 的
  截断网格：E26c D2 突发点其可达上限 concA=11，完整族含 k*=16（探针与
  引擎双重确认的存活阈值）。
- 参照层（decision_point_optimum / offline_optimum）：小规模"理论最优"
  穷举（对称去重），供差距度量（E27），不进生产路径。

信息边界：与 MPC 相同——只读 ObservableSnapshot 公共量 + 自身历史派发
（均为公开事实）；不触碰 world/storage 真值。数值 float64。
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import CqScenario
from .observable import Observable, ObservableSnapshot
from .policies import GuardedEDFBW
from .profile import layer_read_gb
from .types import Action, JointAction, RequestSpec
from .search import build_forecast_engine, forecast_drain, score_of


# ---------------------------------------------------------------------------
# 公式层：对称 cohort 闭式（设计文档 §3 事实 2/4/5）
# ---------------------------------------------------------------------------

def wave_F(p: int, k: int, s: float, c: float, L: int) -> float:
    """同刻派出的 k 条同类请求中第 p 条（1 起）的完成时间（相对波起点）。

    F_p = p·s + (L−1)·max(c, k·s) + c；未饱和（k·s≤c）退化为 p·s+L·c。
    引擎对拍（A-only，k=1..32）：绝对误差 ≤0.1ms。
    """
    w = max(c, k * s)
    return p * s + (L - 1) * w + c


def survival_cap(t_start: float, deadline: float, s: float, c: float,
                 L: int) -> int:
    """存活阈值 k*：波起点 t_start 后，使末位（第 k 条）仍按期的最大 k。

    由 wave_F 反解：k* = ⌊(deadline − t_start − c)/(L·s)⌋。α=4 的 A 类
    空管道 t=0 时 k*=16（引擎实测 16 全活、17 死 5、20 全灭）。
    """
    denom = L * s
    if denom <= 0:
        return 1 << 30
    return max(0, int((deadline - t_start - c) / denom))


def mixed_wave_F(p: int, k_a: int, s_a: float, c_a: float, s_extra: float,
                 L: int) -> float:
    """混合波中第 p 条重请求的完成时间（轻流字节 s_extra=Σk_j·s_j 全额
    计入轮次，保守 β=1；饱和段精确、未饱和段高估轻流伤害，见模块头注）。"""
    w = max(c_a, k_a * s_a + s_extra)
    return p * s_a + (L - 1) * w + c_a


# ---------------------------------------------------------------------------
# 状态抽象：从快照构建对称类桶（公共信息）
# ---------------------------------------------------------------------------

@dataclass
class ClassBucket:
    """对称类桶：同 (V, c) 的在队请求（E26c 工况即 A/B 两桶），桶内 EDF。"""

    key: Tuple[float, float]
    V: float
    c: float
    rids: List[int]
    deadlines: Dict[int, float]


def buckets_of(snap: ObservableSnapshot) -> List[ClassBucket]:
    """公共信息构建对称桶：按 (V,c) 归并、V 降序（桶 0=重桶），桶内 (dl,rid) 升序。"""
    tmp: Dict[Tuple[float, float], List] = {}
    for r in snap.requests:
        if r.rid not in snap.queued:
            continue
        V = float(layer_read_gb(r, snap._profile_cfg))
        c = float(snap.c_hat([(r.h_tokens, r.u_tokens)]))
        key = (round(V, 12), round(c, 12))
        tmp.setdefault(key, []).append(r)
    out = []
    for key, rs in sorted(tmp.items(), key=lambda kv: -kv[0][0]):
        rs.sort(key=lambda r: (float(r.deadline_s), r.rid))
        out.append(ClassBucket(key=key, V=key[0], c=key[1],
                               rids=[r.rid for r in rs],
                               deadlines={r.rid: float(r.deadline_s)
                                          for r in rs}))
    return out


def pipe_backlog_s(snap: ObservableSnapshot) -> float:
    """管道剩余字节（公共账本 served_reported 口径）/ 带宽 = 新 L0 流的等待。"""
    rem = 0.0
    for f in snap.flows:
        if f.completed:
            continue
        served = f.served_reported_gb if f.served_reported_gb is not None else 0.0
        rem += max(0.0, float(f.V_gb) - float(served))
    return rem / max(1e-6, float(snap.est_bw()))


# ---------------------------------------------------------------------------
# 决策层：CPAPolicy
# ---------------------------------------------------------------------------

class CPAPolicy:
    """cq_cpa：完整对称族 + 公式预筛 + top-K 复核（详见模块头注）。

    候选序：桶间 V 降序（重前轻后——轻流层0流按 FCFS 位次切进重流层1+ 流
    之前，重者先行不劣，设计文档 §3 事实 4）；桶内 EDF。未获派发的空闲
    worker 输出 WAIT，唤醒对齐自身重波闭式完成估计（避免 BW 式 c-only
    估计在饱和波中段过早唤醒放轻流切入）。hold 态短路径：决策相关状态
    （队列集/活动批集）不变时直接延续上次 WAIT（层读进度不影响裁决）。
    """

    def __init__(self, pid: str = "cq_cpa", mode: str = "rollout",
                 top_k: int = 4, theta: str = "S",
                 max_events: int = 20000, budget_s: float = 1.0,
                 base: str = "bw_edf"):
        assert mode in ("rollout", "form")
        self.pid = pid
        self.mode = mode
        self.top_k = top_k
        self.theta = theta
        self.max_events = max_events
        self.budget_s = budget_s
        self._d_hist: Dict[int, float] = {}
        self.pi = GuardedEDFBW(hist=self._d_hist)
        self._cohort_fin: Dict[int, float] = {}   # rid -> 自身波闭式完成估计
        self._hold: Optional[Tuple[Tuple, float]] = None   # (签名, wake)
        self.n_decides = 0
        self.n_searched = 0
        self.n_formula_scored = 0
        self.n_rollouts = 0
        self.n_fallback = 0
        self.decide_s_max = 0.0
        self.decide_s_sum = 0.0

    def health(self) -> dict:
        return {"n_decides": self.n_decides, "n_searched": self.n_searched,
                "n_formula_scored": self.n_formula_scored,
                "n_rollouts": self.n_rollouts, "n_fallback": self.n_fallback,
                "decide_s_max": self.decide_s_max,
                "decide_s_mean": (self.decide_s_sum / self.n_decides
                                  if self.n_decides else None)}

    # -- 决策 -------------------------------------------------------------

    def decide(self, snap: ObservableSnapshot, scn: CqScenario) -> JointAction:
        t0 = _time.perf_counter()
        self.n_decides += 1
        base = self.pi.decide(snap, scn)
        if not snap.idle_workers or not snap.queued:
            return base
        sig = (frozenset(snap.queued), tuple(snap.idle_workers),
               tuple(sorted((pw.worker_id, pw.members)
                            for pw in snap.workers if pw.members)))
        now = float(snap.now)
        if self._hold is not None and self._hold[0] == sig \
                and self._hold[1] > now:
            self._tick(t0)
            return JointAction(tuple(
                Action("WAIT", w, (), self._hold[1]) for w in snap.idle_workers))
        self.n_searched += 1
        cands, buckets = self._formula_candidates(snap, scn)
        self.n_formula_scored += len(cands)
        if not cands:
            return base
        if self.mode == "form":
            act = self._action_of(cands[0][1], buckets, snap, scn)
            self._after(t0, act, sig)
            return act
        # rollout 复核（与 mpc_v2.1 同口径：est 世界 + GuardedEDF-BW 延续）
        budget = [self.max_events]
        est0 = build_forecast_engine(snap, scn, self.pi)
        scored: List[Tuple[Tuple, Optional[Tuple[int, ...]]]] = []
        for _sc, cand in cands[: self.top_k]:
            if _time.perf_counter() - t0 > self.budget_s or budget[0] <= 0:
                break
            act = self._action_of(cand, buckets, snap, scn)
            child = self._clone_fc(est0)
            fs = forecast_drain(child, act, budget)
            if fs is None:
                continue
            sc = score_of(fs, snap, self.theta)
            if sc is None:
                continue
            self.n_rollouts += 1
            scored.append((sc, cand))
        if not scored:
            self.n_fallback += 1
            self._tick(t0)
            return base
        if budget[0] > 0:
            child = self._clone_fc(est0)
            fs = forecast_drain(child, base, budget)
            if fs is not None:
                sc_b = score_of(fs, snap, self.theta)
                if sc_b is not None:
                    scored.append((sc_b, None))
        scored.sort(key=lambda x: x[0])
        if scored[0][1] is None:
            self._tick(t0)
            return base
        act = self._action_of(scored[0][1], buckets, snap, scn)
        self._after(t0, act, sig)
        return act

    def _tick(self, t0: float):
        el = _time.perf_counter() - t0
        self.decide_s_max = max(self.decide_s_max, el)
        self.decide_s_sum += el

    def _after(self, t0: float, act: JointAction, sig):
        """记账 + 记录 hold 短路径与自身波完成估计。"""
        self._tick(t0)
        now = 0.0
        if all(a.kind == "WAIT" for a in act.actions) and act.actions:
            wake = max(float(a.wake_at) for a in act.actions)
            self._hold = (sig, wake)
        else:
            self._hold = None

    def _action_of(self, cand: Tuple[int, ...], buckets: List[ClassBucket],
                   snap: ObservableSnapshot, scn: CqScenario) -> JointAction:
        """计数向量 → 联合动作：桶序（重前轻后、桶内 EDF）逐 worker 落实。

        WAIT 唤醒波感知：取"本次派出的 cohort 闭式完成估计 ∪ 在跑 cohort
        估计"的最早者——避免 c-only 估计在饱和波中段过早唤醒放轻流切入
        （设计文档 §3 事实 4：16A+16B 同刻 A 全灭）。"""
        acts = []
        idle = sorted(snap.idle_workers)
        wi = 0
        bw = max(1e-6, float(snap.est_bw()))
        L = snap._profile_cfg.L
        now = float(snap.now)
        new_fins: List[float] = []
        for j, k in enumerate(cand):
            b = buckets[j]
            for p, rid in enumerate(b.rids[:k], start=1):
                if wi >= len(idle):
                    break
                acts.append(Action("DISPATCH", idle[wi], (rid,)))
                fin = now + wave_F(p, k, b.V / bw, b.c, L)
                self._cohort_fin[rid] = fin
                new_fins.append(fin)
                wi += 1
        active_rids = {rid for pw in snap.workers if pw.members
                       for rid in pw.members}
        old_fins = [v for rid, v in self._cohort_fin.items()
                    if rid in active_rids]
        fins = new_fins + old_fins
        wake = (now + min(0.2, max(0.005, min(fins) - now))
                if fins else now + 0.005)
        for w in idle[wi:]:
            acts.append(Action("WAIT", w, (), wake))
        return JointAction(tuple(acts))

    def _clone_fc(self, est0):
        """推演克隆（与 MPCPolicy._clone_forecast 同构：补 policy/observable）。"""
        e = est0.clone_for_search()
        e.policy = est0.policy
        obs = Observable(est0.scn, numeric=float)
        obs.attach(e)
        e.observable = obs
        return e

    # -- 公式预筛 -----------------------------------------------------------

    def _formula_candidates(self, snap: ObservableSnapshot, scn: CqScenario
                            ) -> Tuple[List[Tuple[Tuple, Tuple[int, ...]]],
                                       List[ClassBucket]]:
        """全族公式打分：每个重桶计数 k_h 保最优轻桶向量，按 (misses, ttft) 排序。"""
        bw = max(1e-6, float(snap.est_bw()))
        L = snap._profile_cfg.L
        now = float(snap.now)
        buckets = buckets_of(snap)
        if not buckets:
            return [], buckets
        pipe_free = now + pipe_backlog_s(snap)
        idle = len(snap.idle_workers)
        out = []
        heavy = buckets[0]
        light_caps = [len(b.rids) for b in buckets[1:]]
        for k_h in range(0, min(idle, len(heavy.rids)) + 1):
            best = None
            for lc in (_light_counts(light_caps, idle - k_h)
                       if light_caps else [()]):
                cand = (k_h,) + tuple(lc)
                sc = self._formula_score(cand, buckets, pipe_free, bw, L, now)
                if best is None or sc < best[0]:
                    best = (sc, cand)
            if best is not None:
                out.append(best)
        # 平局裁决：misses 同分时偏好多派遣（排空终须完成，闲置无收益；
        # 且避免"全 WAIT 高分"的活锁）。ttft 仅作第三键。
        out.sort(key=lambda x: (x[0][0], -sum(x[1]), x[0][1]))
        return out, buckets

    def _formula_score(self, cand: Tuple[int, ...],
                       buckets: List[ClassBucket], pipe_free: float,
                       bw: float, L: int, now: float):
        """(misses_est, ttft_est)：首波逐位置闭式存活 + 尾部字节平移阈值。"""
        heavy = buckets[0]
        k_h = cand[0]
        s_h, c_h = heavy.V / bw, heavy.c
        s_extra = sum(buckets[j].V / bw * cand[j] for j in range(1, len(cand)))
        misses = 0
        ttft = 0.0
        for p in range(1, k_h + 1):
            dl = heavy.deadlines[heavy.rids[p - 1]]
            F = pipe_free + mixed_wave_F(p, k_h, s_h, c_h, s_extra, L)
            ttft += F - now
            if F > dl:
                misses += 1
                ttft += F - dl
        a_rest = len(heavy.rids) - k_h
        if a_rest > 0:
            wave_bytes = sum(buckets[j].V * cand[j] for j in range(len(cand)))
            tail_start = pipe_free + wave_bytes * L / bw
            dl_min = min(heavy.deadlines[r] for r in heavy.rids[k_h:])
            k_star = survival_cap(tail_start, dl_min, s_h, c_h, L)
            misses += a_rest - min(a_rest, max(0, k_star))
        for j in range(1, len(cand)):
            b = buckets[j]
            for p in range(1, cand[j] + 1):
                ttft += pipe_free + p * b.V / bw + L * b.c - now
        return (misses, round(ttft, 9))


def _light_counts(caps: Sequence[int], rest: int) -> List[Tuple[int, ...]]:
    """轻桶计数向量枚举：Σ≤rest（桶数 ≤3 时组合数小；>3 桶退化为贪心前缀）。"""
    if not caps:
        return [()]
    if len(caps) > 3:
        out, acc = [], 0
        cur = []
        for cap in caps:
            k = min(cap, max(0, rest - acc))
            cur.append(k)
            acc += k
        return [tuple(cur)]
    out = []
    for k in range(0, min(caps[0], rest) + 1):
        for tail in _light_counts(caps[1:], rest - k):
            out.append((k,) + tail)
    return out


# ---------------------------------------------------------------------------
# 参照层：决策点穷举（DPO）与离线穷举（OPT）
# ---------------------------------------------------------------------------

def _sym_actions(buckets: List[ClassBucket], idle: Sequence[int]):
    """对称去重的全部计数向量（每 worker 派 1 条或全体 WAIT）。

    同桶同期限 rid 可互换；生成 (k_0..k_j) 全网格（含全 0=纯 WAIT）。
    落实顺序与 CPAPolicy._action_of 同构，保证该空间 ⊇ CPA 可达集。
    """
    caps = tuple(len(b.rids) for b in buckets)

    def rec(j, rest, cur):
        if j == len(caps):
            yield tuple(cur)
            return
        for k in range(0, min(caps[j], rest) + 1):
            yield from rec(j + 1, rest - k, cur + [k])

    yield from rec(0, len(idle), [])


def _snapshot_of(scn: CqScenario, specs: Sequence[RequestSpec]):
    """t=0 突发、空系统的初始引擎与快照（无策略决策，仅物理步进入队）。"""
    from .engine import CqEngine
    obs = Observable(scn, numeric=float)
    eng = CqEngine(scn, specs, policy=None, numeric=float, observable=obs,
                   record_decisions=False)
    obs.attach(eng)
    eng._physical_step(0.0)
    snap = eng.observable.snapshot(eng.w.t)
    return eng, snap


def decision_point_optimum(specs: Sequence[RequestSpec], scn: CqScenario,
                           cont_pi=None, theta: str = "S",
                           event_budget: int = 400000):
    """决策点真最优（DPO）：t=0 突发，穷举对称动作 × 真推演（π 延续排空）。

    与 CPA/mpc 同口径（est 世界 + GuardedEDF-BW 延续、θ=S 词典序）。
    返回 {"best": (score, cand)|None, "scores": [...], "n_actions", "wall_s"}。
    """
    pi = cont_pi if cont_pi is not None else GuardedEDFBW()
    _eng, snap = _snapshot_of(scn, specs)
    buckets = buckets_of(snap)
    idle = tuple(snap.idle_workers)
    est0 = build_forecast_engine(snap, scn, pi)
    budget = [event_budget]
    out = []
    t0 = _time.perf_counter()
    for cand in _sym_actions(buckets, idle):
        if sum(cand) == 0:
            act = JointAction(tuple(Action("WAIT", w, (),
                                           float(snap.now) + 0.001)
                                    for w in idle))
        else:
            act = CPAPolicy()._action_of(cand, buckets, snap, scn)
        child = CPAPolicy()._clone_fc(est0)
        fs = forecast_drain(child, act, budget)
        if fs is None:
            continue
        sc = score_of(fs, snap, theta)
        if sc is None:
            continue
        out.append((sc, cand))
    out.sort(key=lambda x: x[0])
    return {"best": out[0] if out else None, "scores": out,
            "n_actions": len(out), "wall_s": _time.perf_counter() - t0}


def offline_optimum(specs: Sequence[RequestSpec], scn: CqScenario,
                    theta: str = "S", node_budget: int = 300000,
                    time_budget_s: float = 600.0):
    """离线真最优（OPT）：全轨迹穷举（对称去重 + 备忘录 + 预算保护）。

    递归：决策点（空闲×非空队列）枚举对称计数向量（含全体 WAIT 至下一
    物理事件）；引擎物理推进；叶=全体 F 的 (misses, ΣTTFT)。WAIT 仅在
    有在算批时允许（全闲等待不优：无进度也不改变到达序列）。
    返回 {"status","best_score","nodes","wall_s","n_memo"}；time_limit
    时 best_score 为已搜最优（上界仍有效，最优性声明失效）。
    """
    from .engine import CqEngine
    t0 = _time.perf_counter()
    ctx = {"nodes": 0, "timeout": False, "best": None, "memo": set()}

    def score_leaf(eng):
        Fs = {rid: float(rr.F_s) for rid, rr in eng.w.requests.items()}
        arrs = {rid: float(eng.w.specs[rid].arrival_s) for rid in Fs}
        dls = {rid: float(eng.w.specs[rid].deadline_s) for rid in Fs}
        return (sum(1 for r in Fs if Fs[r] > dls[r]),
                round(sum(Fs[r] - arrs[r] for r in Fs), 9))

    def advance_to_decision(eng) -> str:
        """推进到下一决策点：批完成或新到达之后（且空闲×非空队列）。

        决策点限制在进度事件（设计文档 §7 局限声明：流完成/计算事件处
        不再决策——好计划本就避免波中插入派遣，受限集上的最优即 OPT
        口径）。返回 done|decision|stuck。
        """
        while True:
            if eng.w.all_done():
                return "done"
            if eng.w.idle_workers() and eng.w.n_queued > 0:
                return "decision"
            pre = (eng.w.n_done, eng.w.arrival_idx)
            tn = eng._next_event()
            if tn is None or tn <= eng.w.t:
                return "stuck"
            eng._advance(tn)
            eng.events_processed += 1
            eng._physical_step(tn)
            if eng.w.all_done():
                return "done"
            if ((eng.w.n_done, eng.w.arrival_idx) != pre
                    and eng.w.idle_workers() and eng.w.n_queued > 0):
                return "decision"

    def lb_misses(eng) -> int:
        """下界：已完成超时数 + 队列/在算中"纯计算最短完成也超期"的请求数。

        OPT 为分析工具（非策略），读引擎真值不破坏信息边界。"""
        m = 0
        for rid, rr in eng.w.requests.items():
            dl = float(eng.w.specs[rid].deadline_s)
            if rr.state == "DONE":
                if float(rr.F_s) > dl:
                    m += 1
                continue
            b = eng.w.batches.get(rr.batch_id) if rr.batch_id is not None else None
            if b is not None:
                rem = sum(float(b.c_layers[l]) for l in range(b.next_layer, eng.w.L))
                if b.next_layer > 0 and b.Z[b.next_layer - 1] is not None:
                    rem += max(0.0, float(b.Z[b.next_layer - 1]) - float(eng.w.t))
                if float(eng.w.t) + rem > dl:
                    m += 1
            else:
                spec = eng.w.specs[rid]
                from .profile import batch_compute_s
                rem = float(batch_compute_s([spec], eng.scn.profile)) * eng.w.L
                if float(eng.w.t) + rem > dl:
                    m += 1
        return m

    def rec(eng):
        if ctx["timeout"]:
            return
        ctx["nodes"] += 1
        if (ctx["nodes"] > node_budget
                or _time.perf_counter() - t0 > time_budget_s):
            ctx["timeout"] = True
            return
        if ctx["best"] is not None and lb_misses(eng) >= ctx["best"][0]:
            return    # B&B：下界剪枝（词典序首项不可能改善）
        st = advance_to_decision(eng)
        if st == "done":
            sc = score_leaf(eng)
            if ctx["best"] is None or sc < ctx["best"]:
                ctx["best"] = sc
            return
        if st == "stuck":
            return
        eng._physical_step(eng.w.t)
        idle = eng.w.idle_workers()
        if not idle or eng.w.n_queued == 0:
            return
        snap = eng.observable.snapshot(eng.w.t)
        buckets = buckets_of(snap)
        any_active = any(pw.members for pw in snap.workers)
        # 公式引导排序：CPA 预筛分决定枚举顺序（好分支先行，配合 LB 剪枝）
        try:
            order, _ = CPAPolicy(mode="form")._formula_candidates(snap, scn)
            rank = {cand: i for i, (_sc, cand) in enumerate(order)}
            cands = sorted(_sym_actions(buckets, idle),
                           key=lambda c: rank.get(c, 1 << 20))
        except Exception:
            cands = list(_sym_actions(buckets, idle))
        for cand in cands:
            if sum(cand) == 0 and not any_active:
                continue    # 全闲纯 WAIT：无进度分支，剪除
            sig = _state_sig(eng, cand)
            if sig in ctx["memo"]:
                continue
            ctx["memo"].add(sig)
            if sum(cand) == 0:
                # 纯等待=推进到下一进度事件（完成/到达）后再决策——"持有"
                child = eng.clone_for_search()
                _attach_obs(child, scn)
                while True:
                    if child.w.all_done():
                        break
                    pre = (child.w.n_done, child.w.arrival_idx)
                    tn = child._next_event()
                    if tn is None or tn <= child.w.t:
                        break
                    child._advance(tn)
                    child.events_processed += 1
                    child._physical_step(tn)
                    if (child.w.n_done, child.w.arrival_idx) != pre:
                        break
                rec(child)
                continue
            # 派遣动作只含 DISPATCH（不占未用 worker，不发唤醒定时器）
            act = _dispatch_only_action(cand, buckets, snap, idle)
            child = eng.clone_for_search()
            child._apply_action(child.w.t, act, stale=False)
            _attach_obs(child, scn)
            rec(child)
        return

    obs = Observable(scn, numeric=float)
    eng = CqEngine(scn, specs, policy=None, numeric=float, observable=obs,
                   record_decisions=False)
    obs.attach(eng)
    eng._physical_step(0.0)
    rec(eng)
    return {"status": "time_limit" if ctx["timeout"] else "optimal",
            "best_score": ctx["best"], "nodes": ctx["nodes"],
            "wall_s": _time.perf_counter() - t0, "n_memo": len(ctx["memo"])}


def _dispatch_only_action(cand, buckets, snap, idle):
    """计数向量 → 仅含 DISPATCH 的联合动作（OPT/DPO 搜索用，无 WAIT 填充）。"""
    acts = []
    wi = 0
    for j, k in enumerate(cand):
        for rid in buckets[j].rids[:k]:
            if wi >= len(idle):
                break
            acts.append(Action("DISPATCH", idle[wi], (rid,)))
            wi += 1
    return JointAction(tuple(acts))


def _attach_obs(eng, scn):
    """给搜索分支引擎挂新观测层（clone 不带 observable，snapshot 需要）。"""
    obs = Observable(scn, numeric=float)
    obs.attach(eng)
    eng.observable = obs


def _state_sig(eng, cand) -> tuple:
    """递归去重签名（对称感知）：时刻量化 + 队列 (类,期限) 多重集 +
    在算批 (类画像,层数,剩余) 多重集 + 候选。同类同期限的 rid 可互换。"""
    t = round(float(eng.w.t), 6)
    specs = eng.w.specs
    q = tuple(sorted((round(float(specs[r].h_tokens) * 4096e-9, 9),
                      float(specs[r].deadline_s))
                     for r in eng.w.queued_rids()))
    active = []
    for b in eng.w.batches.values():
        if b.F_s is not None:
            continue
        V = float(b.V_layers[0]) if b.V_layers else 0.0
        rem = (round(float(b.Z[b.next_layer - 1]) - float(eng.w.t), 6)
               if b.next_layer > 0 and b.Z[b.next_layer - 1] is not None
               else None)
        active.append((round(V, 9), b.next_layer, rem))
    return (t, q, tuple(sorted(active)), cand)
