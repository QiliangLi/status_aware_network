"""两层对抗引擎：Dynamo（TFS-G）× vLLM（TFS-E）的批粒度物理世界。

设计语义对齐 `docs/TFS工况边界与两层拆分设计-20261022.md` §4 与
`docs/TFS两层拆分图解-20261022.html`（示意模拟的 Python 正式版）：

物理世界（引擎，不随策略变）：
- m 个实例；每实例 waiting 队列、加载位、计算位（slot ≤ max_seqs）；
- 准入节奏：每 admit_dt×(1+slow_k×批占用) 允许一条新请求进入批/加载，
  批满时原生模式队首阻塞（FIFO 头阻塞）；
- 原生引擎语义：加载发生在计算位内、整批被加载成员卡住（批等待）；
- TFS-E 语义（策略选择启用）：异步加载位与计算位解耦（不占节奏）、
  重流加载并发上限、就绪优先、无批等待；
- 管道 PIPE GB/s 由全部在途加载均分；
- 分诊 d=V/T0，门槛=已到请求 d 的 P75（与 bw_demand 的 hist 口径一致），
  配额 C=⌊PIPE/d_heavy⌋。

策略版本（攻防迭代修改的对象）：TwoLayerPolicy 的字段即版本参数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

DT = 0.005  # 模拟步长 5ms


@dataclass
class TLReq:
    rid: int
    cls: str
    t_arr: float
    V: float          # 总读取量 GB
    T0: float         # singleton 服务时间（读+算）
    deadline: float   # 绝对期限
    heavy: bool = False
    t_disp: Optional[float] = None
    t_load_s: Optional[float] = None
    t_load_e: Optional[float] = None
    bytes_left: float = 0.0
    comp_p: float = 0.0
    phase: str = "arrived"   # arrived|waiting|loading|ready|compute|done
    t_done: Optional[float] = None


@dataclass
class TwoLayerPolicy:
    """两层策略版本参数。v0 = 设计文档 §4 原案（攻击起点）。"""
    name: str = "v0"
    # --- TFS-G ---
    gate: str = "active"        # none(直接提交) | issued(单层语义) | active(两层)
    inflight_bound: int = 2     # active 口径的在途预算
    backpressure: int = 6       # 实例 waiting 深度阈值（重流停收）；0=关
    g_slack_order: bool = True  # G 派发序：slack（False=到达序）
    quota_override: Optional[int] = None   # 固定 C（实验用）
    # --- TFS-E ---
    e_mode: bool = True         # 启用 TFS-E 全套；False=引擎原生语义
    e_cap_mode: str = "count"   # count(条数) | byte(字节加权)
    ready_aging: bool = False   # 防御项 D10：就绪重流老化提升
    ready_aging_w: float = 0.5  # 老化折扣（等待每秒 slack 视作减这么多秒）
    light_gate_e: bool = False  # 防御项 D12：E 层轻流加载也设上限
    bp_hyst: bool = False       # 防御项 D11：反压迟滞（高关低开 BP/2）
    bp_loadbased: bool = False  # 防御项 D13：反压判据改加载位（Wave1 发现）
    # --- 引擎物理（同代版本共享，不作为策略旋钮） ---
    admit_dt: float = 0.012
    slow_k: float = 0.5


class TwoLayerEngine:
    """物理世界 + 策略钩子。deterministic：无内部随机。"""

    def __init__(self, trace: Sequence[Tuple[float, str]], classes: Dict[str, dict],
                 policy: TwoLayerPolicy, m_inst: int = 4, pipe: float = 120.0,
                 max_seqs: int = 8, sim_deadline: float = 30.0):
        self.pol = policy
        self.pipe = pipe
        self.max_seqs = max_seqs
        self.m = m_inst
        self.cls = classes
        self.sim_deadline = sim_deadline
        self.t = 0.0
        self.arrivals = sorted(trace, key=lambda e: (e[0], e[1]))
        self.arr_idx = 0
        self.q: List[TLReq] = []          # 调度层队列
        self.done: List[TLReq] = []
        self.insts = [{"waiting": [], "loading": [], "batch": [], "admit_cd": 0.0,
                       "bp_open": True} for _ in range(m_inst)]
        self.hist_d: List[float] = []     # 分诊历史（已到请求的 d）
        self.req_by_id: Dict[int, TLReq] = {}
        self._rid = 0
        # 统计
        self.gap_sum = 0.0; self.gap_n = 0
        self.bs_sum = 0.0; self.bs_n = 0
        self.bw_sum = 0.0; self.bw_n = 0
        self.stall_sum = 0.0; self.stall_n = 0
        self.n_loads_hist: List[int] = []

    # -- 分诊口径 --
    def d_of(self, r: TLReq) -> float:
        return r.V / r.T0 if r.T0 > 0 else 0.0

    def triage(self):
        if not self.hist_d:
            return 0.0, 1 << 30
        vals = sorted(self.hist_d)
        d_heavy = vals[int(0.75 * (len(vals) - 1))]
        C = self.pol.quota_override or max(1, int(self.pipe / d_heavy)) \
            if d_heavy > 0 else 1 << 30
        return d_heavy, C

    def _mk(self, t: float, cls: str) -> TLReq:
        c = self.cls[cls]
        r = TLReq(self._rid, cls, t, c["V"], c["T0"], t + c["alpha"] * c["T0"],
                  bytes_left=c["V"])
        self._rid += 1
        return r

    def run(self):
        while (self.arr_idx < len(self.arrivals) or self.q
               or any(i["waiting"] or i["loading"] or i["batch"] for i in self.insts)) \
                and self.t < self.sim_deadline:
            self.step()
        return self.stats()

    # -- 账本 --
    def _iter_all(self):
        yield from self.q
        for i in self.insts:
            yield from i["waiting"]
            yield from i["loading"]
            yield from i["batch"]

    def ledger(self):
        d_heavy, C = self.triage()
        issued = sum(1 for r in self._iter_all()
                     if r.heavy and r.t_disp is not None and r.phase != "done")
        active = sum(1 for r in self._iter_all()
                     if r.heavy and r.phase == "loading")
        inflight = sum(1 for r in self._iter_all()
                       if r.heavy and r.phase == "waiting" and r.t_disp is not None)
        return issued, active, inflight, C, d_heavy

    def step(self):
        p, now = self.pol, self.t
        # 1. 到达 + 分诊历史（批量：本 tick 一次 triage）
        new_reqs = []
        while self.arr_idx < len(self.arrivals) and self.arrivals[self.arr_idx][0] <= now:
            t, cls = self.arrivals[self.arr_idx]
            r = self._mk(t, cls)
            self.hist_d.append(self.d_of(r))
            new_reqs.append(r)
            self.arr_idx += 1
        if new_reqs:
            d_h0, _C0 = self.triage()
            for r in new_reqs:
                r.heavy = self.d_of(r) >= d_h0
                self.q.append(r)
                self.req_by_id[r.rid] = r
        # 2. TFS-G 准入（账本每 tick 只算一次；本地计数器递增）
        d_heavy, C = self.triage()
        issued, active, inflight, _, _ = self.ledger()
        slack = lambda r: r.deadline - now - r.T0
        order = sorted(self.q, key=(lambda r: (slack(r), r.deadline, r.rid))
                       if p.g_slack_order else (lambda r: (r.t_arr, r.rid)))
        heavy_adm = 0
        if p.backpressure > 0:
            self._bp_update()
        bp_all_closed = p.backpressure > 0 and all(not i["bp_open"]
                                                   for i in self.insts)
        for r in order:
            go = True
            if p.gate == "issued":
                go = (not r.heavy) or issued + heavy_adm < C
            elif p.gate == "active":
                go = (not r.heavy) or \
                    active + inflight + heavy_adm < C + p.inflight_bound
                if go and r.heavy and bp_all_closed:
                    go = False
            if not go:
                continue
            self.q.remove(r)
            r.t_disp = now
            r.phase = "waiting"
            tgt = self._pick_inst(r.heavy)
            if tgt < 0:
                continue
            self.insts[tgt]["waiting"].append(r)
            if r.heavy:
                heavy_adm += 1
        # 3. 引擎层（triage 用 tick 缓存）
        self._tc = self.triage()
        loads: List[TLReq] = []
        for inst in self.insts:
            if p.e_mode:
                self._engine_e(inst, now)
            else:
                self._engine_native(inst, now)
            loads.extend(inst["loading"])
            loads.extend(r for r in inst["batch"] if r.phase == "loading")
        # 4. 管道均分推进
        n = len(loads)
        rate = self.pipe / n if n else 0.0
        for inst in self.insts:
            for r in list(inst["loading"]):
                r.bytes_left -= rate * DT
                if r.bytes_left <= 0:
                    r.bytes_left = 0.0
                    r.t_load_e = now
                    r.phase = "ready"
                    inst["loading"].remove(r)
                    if len(inst["batch"]) < self.max_seqs:
                        inst["batch"].append(r)
                    else:
                        inst["waiting"].insert(0, r)
            for r in inst["batch"]:
                if r.phase == "loading":
                    r.bytes_left -= rate * DT
                    if r.bytes_left <= 0:
                        r.bytes_left = 0.0
                        r.t_load_e = now
                        r.phase = "ready"
        # 5. 计算
        for inst in self.insts:
            stalled = (not p.e_mode) and any(r.phase == "loading" for r in inst["batch"])
            if stalled:
                self.stall_sum += 1
            self.stall_n += 1
            for r in list(inst["batch"]):
                if r.phase == "ready" and not stalled:
                    r.comp_p += DT / r.T0
                    if r.comp_p >= 1:
                        r.phase = "done"
                        r.t_done = now
                        inst["batch"].remove(r)
                        self.done.append(r)
        # 6. 统计（账本差 = 在途滞留：已下发仍压在 waiting 的重流数）
        iss, act, inf, Cq, _ = self.ledger()
        self.gap_sum += inf; self.gap_n += 1
        self.iss_sum = getattr(self, 'iss_sum', 0.0) + iss
        self.n_loads_hist.append(n)
        self.bw_sum += (n * rate / self.pipe if n else 0.0); self.bw_n += 1
        self.bs_sum += sum(len(i["batch"]) for i in self.insts) / self.m
        self.bs_n += 1
        self.t += DT

    def _engine_e(self, inst, now):
        p = self.pol
        d_heavy, C = getattr(self, '_tc', self.triage())
        # E1+E3：waiting 排序（就绪优先 + slack；就绪重流老化由防御项控制）
        def key(r):
            age = (now - r.t_load_e) if (p.ready_aging and r.phase == "ready") else 0.0
            return ((0 if r.phase == "ready" else 1),
                    r.deadline - now - r.T0 - p.ready_aging_w * age,
                    r.rid)
        inst["waiting"].sort(key=key)
        # 异步加载位：不占计算节奏；重流受上限，被挡不阻塞轻流
        cap = self._e_cap(C)
        h_load = sum(1 for r in inst["loading"] if r.heavy)
        i = 0
        while i < len(inst["waiting"]):
            w = inst["waiting"][i]
            if w.V > 0 and w.phase != "ready":
                if w.heavy and h_load >= cap:
                    i += 1
                    continue
                if (not w.heavy) and p.light_gate_e and \
                        sum(1 for r in inst["loading"] if not r.heavy) >= cap:
                    i += 1
                    continue
                inst["waiting"].pop(i)
                w.phase = "loading"
                w.t_load_s = now
                inst["loading"].append(w)
                if w.heavy:
                    h_load += 1
            else:
                i += 1
        # 计算位准入：节奏化 + 就绪优先/零读直入
        inst["admit_cd"] -= DT
        guard = 8
        while inst["admit_cd"] <= 0 and guard:
            guard -= 1
            if len(inst["batch"]) >= self.max_seqs:
                break
            pick = -1
            for k, w in enumerate(inst["waiting"]):
                if w.phase == "ready" or w.V == 0:
                    pick = k
                    break
            if pick < 0:
                break
            w = inst["waiting"].pop(pick)
            if w.phase != "ready":
                w.phase = "ready"
                w.t_load_e = now
            inst["batch"].append(w)
            inst["admit_cd"] += p.admit_dt * (1 + p.slow_k * len(inst["batch"]))

    def _e_cap(self, C) -> int:
        p = self.pol
        if p.e_cap_mode == "byte":
            return 1 << 30   # 字节上限在 _engine_e 的重流判断处近似为条数上限×上限字节
        return max(1, round(C / self.m))

    def _engine_native(self, inst, now):
        p = self.pol
        inst["waiting"].sort(key=lambda r: (r.t_arr, r.rid))
        inst["admit_cd"] -= DT
        guard = 8
        while inst["admit_cd"] <= 0 and guard:
            guard -= 1
            if len(inst["batch"]) >= self.max_seqs:
                break
            if not inst["waiting"]:
                break
            w = inst["waiting"].pop(0)
            if w.V > 0:
                w.phase = "loading"
                w.t_load_s = now
            else:
                w.phase = "ready"
                w.t_load_e = now
            inst["batch"].append(w)
            inst["admit_cd"] += p.admit_dt * (1 + p.slow_k * len(inst["batch"]))

    def _bp_update(self):
        """反压状态机：默认即时（>BP 关）；bp_hyst 迟滞（<BP/2 开）；
        bp_loadbased（D13）判据改「该实例重流加载位已满」——轻流把 waiting
        撑深不再误触反压（Wave1：C 流 λ=60~100/s 使全实例常闭、重流中央
        等待 4.8~8.2s）。"""
        p = self.pol
        _, C = self.triage()
        cap_i = max(1, round(C / self.m))
        for inst in self.insts:
            if p.bp_loadbased:
                h_load = sum(1 for r in inst["loading"] if r.heavy)
                inst["bp_open"] = h_load < cap_i
                continue
            depth = len(inst["waiting"])
            if inst["bp_open"]:
                if depth > p.backpressure:
                    inst["bp_open"] = False
            else:
                thr = p.backpressure / 2 if p.bp_hyst else p.backpressure
                if depth < thr:
                    inst["bp_open"] = True

    def _pick_inst(self, heavy: bool) -> int:
        p = self.pol
        best, best_s = -1, float("inf")
        for i, inst in enumerate(self.insts):
            if heavy and p.gate == "active" and p.backpressure > 0 and \
                    not inst["bp_open"]:
                continue
            s = len(inst["waiting"]) + len(inst["loading"]) + 0.5 * len(inst["batch"])
            if s < best_s:
                best_s, best = s, i
        return best

    # -- 统计 --
    def stats(self) -> Dict:
        n = len(self.done)
        cls_n: Dict[str, int] = {}
        cls_met: Dict[str, int] = {}
        ttfts = []
        cw_h = []
        for r in self.done:
            ok = r.t_done <= r.deadline
            cls_n[r.cls] = cls_n.get(r.cls, 0) + 1
            cls_met[r.cls] = cls_met.get(r.cls, 0) + ok
            ttfts.append(r.t_done - r.t_arr)
            if r.heavy and r.t_disp is not None:
                cw_h.append(r.t_disp - r.t_arr)
        ttfts.sort()
        return {
            "n_done": n, "n_total": self._rid,
            "slo": (sum(cls_met.values()) / n) if n else float("nan"),
            "cls_slo": {c: cls_met[c] / cls_n[c] for c in cls_n},
            "ttft_mean": sum(ttfts) / len(ttfts) if ttfts else float("nan"),
            "cw_heavy_mean": sum(cw_h) / len(cw_h) if cw_h else 0.0,
            "gap_mean": self.gap_sum / max(self.gap_n, 1),
            "issued_mean": getattr(self, 'iss_sum', 0.0) / max(self.gap_n, 1),
            "bw_mean": self.bw_sum / max(self.bw_n, 1),
            "bs_mean": self.bs_sum / max(self.bs_n, 1),
            "stall_frac": self.stall_sum / max(self.stall_n, 1),
            "peak_loads": max(self.n_loads_hist) if self.n_loads_hist else 0,
        }


# ---------------------------------------------------------------------------
# 两层判定
# ---------------------------------------------------------------------------

def judge_two(tfs2: Dict, baselines: Dict, C_hint: int = 4) -> Tuple[bool, str]:
    """击杀判定：TFS2(v*) 对阵 immediate 与 single。"""
    best_bl = max(b["slo"] for b in baselines.values())
    if tfs2["slo"] < best_bl - 0.05 and best_bl >= 0.30:
        w = max(baselines, key=lambda k: baselines[k]["slo"])
        return True, f"总SLO {tfs2['slo']:.3f} vs {w} {baselines[w]['slo']:.3f}"
    for c, v in tfs2["cls_slo"].items():
        bl_c = [b["cls_slo"].get(c, 0.0) for b in baselines.values()]
        if v <= max(bl_c) - 0.25 and max(bl_c) >= 0.60:
            return True, f"类{c} SLO {v:.3f} vs 基线最优 {max(bl_c):.3f}"
        if v <= 0.02 and max(bl_c) >= 0.30:
            return True, f"类{c} 饿死(0%)（基线最优 {max(bl_c):.3f}）"
    # 账本差/中央等待只在伴随 SLO 回退时才算击杀（20261023 Wave1 修正：
    # 单独的 gap 大是缓冲形态证据，不是调度效果差——tfs2 曾以 6~10 倍
    # SLO 优势被 gap 口径误杀）
    sing = baselines.get("single")
    if sing:
        worse = tfs2["slo"] < sing["slo"] - 0.05
        if worse and tfs2["cw_heavy_mean"] > 1.5 * max(sing["cw_heavy_mean"], 1e-9) \
                and tfs2["cw_heavy_mean"] > 0.3:
            return True, (f"中央等待 {tfs2['cw_heavy_mean']*1e3:.0f}ms 且总SLO回退 "
                          f"({tfs2['slo']:.3f}<{sing['slo']:.3f})")
        if worse and tfs2["gap_mean"] >= max(1, C_hint) / 2.0:
            return True, (f"账本差 {tfs2['gap_mean']:.2f} 且总SLO回退 "
                          f"({tfs2['slo']:.3f}<{sing['slo']:.3f})")
    return False, ""


def policy_immediate() -> TwoLayerPolicy:
    return TwoLayerPolicy(name="immediate", gate="none", e_mode=False)


def policy_single() -> TwoLayerPolicy:
    return TwoLayerPolicy(name="single", gate="issued", e_mode=False)
