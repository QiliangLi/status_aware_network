"""E27 设计质疑消融探针（20261008）：模仿性/更简方式/批与带宽耦合。

P1 消融（E26c 四格点）：
   BWEDFKstar —— 底座最小改动（重并发上限 C→动态 k*，约 15 行）：
      抬 cap 而不扣轻流 → 16A+16B 同刻 → 事实4 → A 全灭（D2 66.7%，
      比底座低 8.3pp）。证明"派到 k*"必须与"轮次预算扣 B"组合才成立。
   LiteBurst —— 静态突发规则（k* 开波、B 绝对扣到波尾，约 25 行）：
      D2c 46.7%（饱和区 B 被饿死）；D1 81.7% 与 CPA-form 相当。
   结论：CPA-form 的 R2/R3/边距复杂度是被消融证实的"挣来的复杂度"。
P2 batch：mooncake conversation 上 MPC 多成员批占比（CPA/底座只派
   singleton；E26c n_max=1 时批维度不存在）。
P3 带宽分配耦合：q_max<B 时闭式的层0串行假设失效程度。

用法：.venv/bin/python tools/e27_ablation.py
"""

import sys, os, time
sys.path.insert(0, os.path.abspath("."))
from fractions import Fraction as F
from dataclasses import replace

from sim.cq.config import StorageConfig
from sim.cq.observable import ObservableSnapshot
from sim.cq.policies import GuardedEDFBW, bw_demand, make_fcfs
from sim.cq.profile import layer_read_gb
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.metrics import summarize
from sim.cq.types import Action, JointAction
from sim.cq.poly import CPAPolicy, wave_F, survival_cap, ALIVE_MARGIN_S
from sim.experiments import e26_ab as e26
from sim.experiments import e26c_qa as e26c


# ---------------- P1：更简策略 ----------------

class BWEDFKstar(GuardedEDFBW):
    """底座最小改动：重并发上限 C 换成动态 k*（其余逐位不动）。"""

    def _kstar_cap(self, snap):
        L = snap._profile_cfg.L
        bw = max(1e-6, float(snap.est_bw()))
        now = float(snap.now)
        d, d_heavy, C, heavy_run, wake = bw_demand(snap, self._hist)
        queued = [r for r in snap.requests if r.rid in snap.queued]
        k = C
        if queued:
            # 重请求=队列中最大 V（与 CPA 桶 0 语义一致；不用 V/c≥P75——
            # P75 恰落在 A 值上时浮点刀锋等式会失败）
            r = max(queued, key=lambda r: (float(layer_read_gb(
                r, snap._profile_cfg)), -float(r.deadline_s)))
            V = float(layer_read_gb(r, snap._profile_cfg))
            c = float(snap.c_hat([(r.h_tokens, r.u_tokens)]))
            # 重判定用 bw_demand 同源 d（E26c 1:2 配比使 P75 恰落在 A 值
            # 上，c_hat 与 bw_demand 两条浮点路径差 3e-15 即判否）
            if d[r.rid] >= d_heavy:   # 只对真重类启用 k*
                k = max(C, survival_cap(now, float(r.deadline_s),
                                        V / bw, c, L))
        return d, d_heavy, k, heavy_run, wake

    def decide(self, snap, scn):
        d, d_heavy, C, heavy_run, wake = self._kstar_cap(snap)
        anchors = self._aged_anchors(snap, scn)
        order = sorted((r for r in snap.requests if r.rid in snap.queued),
                       key=lambda r: (r.deadline_s, r.arrival_s, r.rid))
        acts, used = [], set()
        acc = heavy_run
        for wid in sorted(snap.idle_workers):
            pick = None
            for a in anchors:
                if a in snap.queued and a not in used:
                    pick = a
                    break
            if pick is None:
                for r in order:
                    if r.rid in used:
                        continue
                    if d[r.rid] >= d_heavy and acc >= C:
                        continue
                    pick = r.rid
                    break
            if pick is None:
                acts.append(Action("WAIT", wid, (), wake))
                continue
            acts.append(Action("DISPATCH", wid, (pick,)))
            used.add(pick)
            if d[pick] >= d_heavy:
                acc += 1
        return JointAction(tuple(acts))


class LiteBurst:
    """静态突发规则（无在飞损伤模型/无边距/无守卫/无 doomed 延后）：
    有重请求在队 → 按 k* 开波、其余全 WAIT 到自身波尾闭式完成时刻；
    无重请求 → 轻流填满。"""

    pid = "lite_burst"

    def __init__(self):
        self._fin = {}

    def decide(self, snap, scn):
        L = snap._profile_cfg.L
        bw = max(1e-6, float(snap.est_bw()))
        now = float(snap.now)
        queued = [r for r in snap.requests if r.rid in snap.queued]
        idle = sorted(snap.idle_workers)
        acts = []
        if not queued or not idle:
            return JointAction(tuple(
                Action("WAIT", w, (), now + 0.005) for w in idle))
        Vmax = max(float(layer_read_gb(r, snap._profile_cfg)) for r in queued)
        heavy = sorted((r for r in queued
                        if float(layer_read_gb(r, snap._profile_cfg)) >= Vmax - 1e-12),
                       key=lambda r: (r.deadline_s, r.rid))
        light = [r for r in queued if r not in heavy]
        if heavy:
            h = heavy[0]
            c = float(snap.c_hat([(h.h_tokens, h.u_tokens)]))
            k = min(len(idle), len(heavy),
                    survival_cap(now, float(h.deadline_s), Vmax / bw, c, L))
            k = max(k, 0)
            if k > 0:
                wave_end = now + wave_F(k, k, Vmax / bw, c, L)
                for w in idle[:k]:
                    acts.append(Action("DISPATCH", w, (heavy[0].rid,)))
                    heavy.pop(0)
                for w in idle[k:]:
                    acts.append(Action("WAIT", w, (), wave_end))
                return JointAction(tuple(acts))
            # k=0（重请求全注定超期）：轻流填满
        for w, r in zip(idle, sorted(light, key=lambda r: (r.deadline_s, r.rid))):
            acts.append(Action("DISPATCH", w, (r.rid,)))
        for w in idle[len(light):]:
            acts.append(Action("WAIT", w, (), now + 0.005))
        return JointAction(tuple(acts))


def p1_ablation():
    cells = ["D2_b1.0_a4", "D2_b0.75_a4", "D1_r1.1_a4", "D2c"]
    print("== P1 消融：更简策略 vs CPA-form/mpc（E26c 四格点） ==")
    print(f"{'格点':13s} {'BW底座':>7s} {'BW-k*':>7s} {'LiteBurst':>9s} "
          f"{'CPA-form':>8s} {'mpc':>6s}")
    mpc = {"D2_b1.0_a4": 78.1, "D2_b0.75_a4": 78.1,
           "D1_r1.1_a4": 90.4, "D2c": 74.2}
    cpa_form = {"D2_b1.0_a4": 82.3, "D2_b0.75_a4": 78.6,
                "D1_r1.1_a4": 80.8, "D2c": 72.9}
    for cell in cells:
        specs = e26c.build_specs(cell)
        scn = e26.e26_scenario(4)
        row = []
        for pol in (GuardedEDFBW(), BWEDFKstar(), LiteBurst()):
            s = summarize(run_case(scn, specs, pol, numeric=float, seed=0), scn)
            row.append(100 * s["slo_success"] / s["n_cohort"])
        print(f"{cell:13s} {row[0]:>6.1f}% {row[1]:>6.1f}% {row[2]:>8.1f}% "
              f"{cpa_form[cell]:>7.1f}% {mpc[cell]:>5.1f}%")


# ---------------- P2：MPC 在 mooncake 上的批使用 ----------------

def p2_batch_usage():
    from sim.experiments.cq_common import (MooncakeSource, TRACE_DIR_DEFAULT,
                                           TRACE_WIDE_LIMITS, default_scenario)
    src = MooncakeSource("conversation_trace.jsonl", TRACE_DIR_DEFAULT)
    scn = default_scenario(B_gbps=80.0, m=4, alpha=F(4), limits=TRACE_WIDE_LIMITS)
    specs, _ = src.window_specs(2, lam=src.lam0 * F(9, 10), alpha=F(4),
                                duration_cap=30.0)
    pol = MPCPolicy(pid="cq_mpc", H=1, theta="S", budget_s=3600.0,
                    max_events=20000, base="bw_edf", composer=True)
    eng = run_case(scn, specs, pol, numeric=float, seed=0)
    s = summarize(eng, scn)
    bs = list(eng.w.batches.values())
    multi = [b for b in bs if len(b.members) > 1]
    n_multi_members = sum(len(b.members) for b in multi)
    sizes = sorted(len(b.members) for b in multi)
    print(f"== P2 batch：mooncake conversation 上 MPC 的批使用 ==")
    print(f"MPC SLO={s['slo_success']}/{s['n_cohort']}; "
          f"批总数={len(bs)} 多成员批={len(multi)} "
          f"({100*len(multi)/max(1,len(bs)):.0f}%); "
          f"多成员批覆盖请求={n_multi_members}/{s['n_cohort']} "
          f"({100*n_multi_members/max(1,s['n_cohort']):.0f}%); "
          f"批大小分布(多成员)={sizes[:20]}")


# ---------------- P3：带宽分配耦合（q_max < B） ----------------

def p3_qmax_probe():
    V_A = float(F(4096, 10**9) * e26.A_CLASS[0])
    sA = V_A / 120.0
    print("== P3 带宽分配耦合：q_max=40（<B=120）× 16 条 A 突发 ==")
    for qmax in (200, 40, 10):
        specs = []
        for i in range(16):
            h, u = e26.A_CLASS
            specs.append(e26.RequestSpec(i, 0.0, h, u, "A", F(1), F(1)))
        T0 = e26.make_T0(specs, e26.E26_PROFILE, F(120), F(200))
        specs = [e26.RequestSpec(s.rid, 0.0, s.h_tokens, s.u_tokens, s.class_id,
                                 T0[s.rid], 4 * float(T0[s.rid])) for s in specs]
        scn0 = e26.e26_scenario(4)
        scn = replace(scn0, storage=StorageConfig(
            b_schedule=((F(0), F(120)),), q_max_gbps=F(qmax), b_ref_gbps=F(120)))
        eng = run_case(scn, specs, make_fcfs(), numeric=float, seed=0)
        fs = sorted(float(eng.w.requests[i].F_s) for i in range(16))
        pred = wave_F(16, 16, sA, 0.00602, 8) * 1e3
        print(f"q_max={qmax:>3}: F_last={fs[-1]*1e3:7.2f}ms "
              f"F_first={fs[0]*1e3:7.2f}ms | 闭式(串行假设)预测 F_last="
              f"{pred:.2f}ms | 偏差={100*(fs[-1]*1e3/pred-1):+.1f}%")


if __name__ == "__main__":
    p1_ablation()
    p3_qmax_probe()
    p2_batch_usage()
