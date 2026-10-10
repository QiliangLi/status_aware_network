"""E26e：A 簇泊松集中度实验 D1c（设计 docs/E26e-A簇泊松集中度扫描-实验设计-20261009.md）。

补上 E26 系列缺失的两个维度：①A:B=1:1（120A+120B，A 需求 138>120 GB/s
供给过剩——"比拼谁死得少"区）；②A 集中度 k∈{4,16}（簇大小=单次涌入队列
的 A 条数）。

负载 D1c（双独立随机流，seed 驱动 CRN）：
  A 簇流——簇心间隔 ~ Exp(k/λ_A) iid，每簇心恰 k 条 A 同刻到达（rid 连续），
  余数入末簇（k=16：7×16+8）；B 泊松流——间隔 ~ Exp(1/λ_B) iid 每次 1 条
  （与 E26c D1 背景一致，唯一被扫描的变量是 A 集中度）。
  λ=196（r1.1）、λ_A=λ_B=98/s、α=4、240 条、到达期≈1.22s。

臂（5，引擎口径与 E26d 完全一致）：fcfs / edf / localS_v2.1 / mpcS_v2.1 /
mpcT_v2.1（base=bw_edf、composer=True、budget_s=3600、事件预算 2 万/决策、H=1）。
矩阵：2 格点 × 5 臂 × 1 seed = 10 runs；seed=400+基号*8+rep（沿用 e26.d1_seed，
k4 基号 0 / k16 基号 1）。条件补测：若任一格点 MPC−FCFS <10pp，该格点补
rep∈{1,2} 取中位数（E26E_REPS）。
输出 results/cq/eval/e26e/（records 分片 + 甘特 json.gz + intervals npz，
格式同 e26d；新增簇冲击响应=每簇心后 50ms 内实际派发的 A 条数）。
环境变量：E26E_ONLY="D1c_k16[,D1c_k4]"（或 cell@rep 精选）、E26E_SHARD、
E26E_REPS（默认 "0"）。
"""
from __future__ import annotations

import gzip
import json
import os
import time
from fractions import Fraction as F

import numpy as np

from sim.cq.policies import make_edf, make_fcfs
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.metrics import summarize
from sim.experiments import e26_ab as e26
from sim.experiments.cq_common import out_dir, print_progress, save_json

MPC_BUDGET_S = 3600.0
MPC_EVENTS = 20000
LAM = 196.0                  # r1.1（供不应求、队列持续加深）
N_A, N_B = 120, 120          # 1:1（A 供给过剩：98/s 到达 vs ~64/s 吞吐上限）
ALPHA = 4
SHOCK_WINDOW_S = 0.05        # 簇冲击响应窗口：簇心后 50ms

ARMS = [("fcfs", "cq_fcfs", None, None, None),
        ("edf", "cq_edf", None, None, None),
        ("localS_v2.1", "cq_local", "S", "bw_edf", True),
        ("mpcS_v2.1", "cq_mpc", "S", "bw_edf", True),
        ("mpcT_v2.1", "cq_mpc", "T", "bw_edf", True)]

# (cell, k, seed 基号)；k16 在前——先行即冒烟（设计 §7-4）
K_LEVELS = (("D1c_k16", 16, 1), ("D1c_k4", 4, 0))


def d1c_seed(base: int, rep: int) -> int:
    """沿用 e26.d1_seed 公式：400 + 基号*8 + rep。"""
    return 400 + base * 8 + rep


def d1c_specs(k: int, seed: int, alpha: int = ALPHA, n_a: int = N_A,
              n_b: int = N_B, lam: float = LAM):
    """D1c 负载：A 簇泊松流（每簇心恰 k 条同刻，余数入末簇）+ B 泊松流。

    返回 (specs, a_centers)：a_centers=A 簇心时刻列表（供簇冲击响应统计）。
    同时刻排序稳定（A 先插入 → A 簇 rid 连续）；T0 由 e26 工况解析式生成，
    deadline=arrival+α·T0。
    """
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    lam_a = lam / 2                       # 1:1 → λ_A=λ_B=lam/2
    t, a_times, centers = 0.0, [], []
    while len(a_times) < n_a:             # A 簇流：间隔 Exp(k/λ_A)
        t += rng.exponential(k / lam_a)
        m = min(k, n_a - len(a_times))    # 末簇补余（k=16: 7×16+8）
        a_times.extend([t] * m)
        centers.append(float(t))
    b_times, t = [], 0.0
    while len(b_times) < n_b:             # B 流：间隔 Exp(1/λ_B)，每次 1 条
        t += rng.exponential(1 / lam_a)
        b_times.append(float(t))
    events = ([(x, "A") for x in a_times] + [(x, "B") for x in b_times])
    events.sort(key=lambda e: e[0])       # 稳定：同刻 A 在前（概率 0 事件）
    specs = [e26.RequestSpec(i, ev[0],
                             *(e26.A_CLASS if ev[1] == "A" else e26.B_CLASS),
                             ev[1], F(1), F(1)) for i, ev in enumerate(events)]
    T0 = e26.make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    out = [e26.RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens,
                           s.class_id, T0[s.rid],
                           float(s.arrival_s) + alpha * float(T0[s.rid]))
           for s in specs]
    return out, centers


def cluster_shock(engine, centers, window_s: float = SHOCK_WINDOW_S):
    """簇冲击响应：每簇心后 window_s 内实际派发的 A 条数。

    按实际派发时刻 BatchRuntime.dispatch_s 统计（zero-delay 决策口径下与
    engine.decisions 的 action 时间戳一致，且涵盖 fallback 落实的派发）。
    """
    specs = engine.w.specs                    # rid → RequestSpec（dict）
    disp = [(float(b.dispatch_s),
             sum(1 for r in b.members if specs[r].class_id == "A"))
            for b in engine.w.batches.values()]
    out = []
    for c in centers:
        out.append(sum(n for t, n in disp if c <= t < c + window_s))
    return out


def run_cell(cell: str, rep: int, d: str, arms=None):
    k, base = next((kk, b) for c, kk, b in K_LEVELS if c == cell)
    seed = d1c_seed(base, rep)
    specs, centers = d1c_specs(k, seed)
    scn = e26.e26_scenario(ALPHA)
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    anchor = None
    records = []
    for label, pid, theta, base_pol, composer in (arms or ARMS):
        if pid == "cq_fcfs":
            pol = make_fcfs()
        elif pid == "cq_edf":
            pol = make_edf()
        else:
            pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1,
                            theta=theta, c_ref=c_ref, budget_s=MPC_BUDGET_S,
                            max_events=MPC_EVENTS, base=base_pol,
                            composer=composer)
        t0 = time.time()
        eng = run_case(scn, specs, pol, numeric=float, seed=rep,
                       record_intervals=True)
        wall = time.time() - t0
        s = summarize(eng, scn)
        s["slo_rate"] = s["slo_success"] / s["n_cohort"]
        s.update(e26.class_metrics(eng, specs))
        s.update(e26.concurrent_a_quantiles(eng))
        s.update(e26.bw_interval_stats(eng.w.storage.interval_log))
        comp, stall = s["device_compute_s"], s["device_stall_s"]
        s["io_wait_share2"] = comp / (comp + stall) if comp + stall > 0 else None
        s["idle_frac"] = s["device_idle_s"] / s["device_total_mh_s"]
        shock = cluster_shock(eng, centers)
        # 逐请求完成记录（TTFT CDF / makespan 精确口径 / deadline 裕量分析用）
        rq = []
        for sp in specs:
            rr = eng.w.requests[sp.rid]
            fs = float(rr.F_s) if rr.F_s is not None else float(eng.w.t)
            rq.append([sp.rid, sp.class_id, float(sp.arrival_s),
                       float(sp.deadline_s), fs, fs - float(sp.arrival_s)])
        h = pol.health() if hasattr(pol, "health") else {}
        s.update({"policy": label, "pid": pid, "theta": theta,
                  "mpc_base": base_pol, "composer": composer, "cell": cell,
                  "k": k, "rep": rep, "seed": seed,
                  "alpha": ALPHA, "n_total": N_A + N_B,
                  "makespan_s": float(eng.w.t), "wall_s": wall,
                  "n_decides": h.get("n_decides"),
                  "n_searched": h.get("n_searched"),
                  "n_scored": h.get("n_scored"),
                  "n_fallback": h.get("n_fallback"),
                  "n_deviate": h.get("n_deviate"),
                  "n_tie": h.get("n_tie"),
                  "n_clusters": len(centers),
                  "cluster_A_disp": shock,
                  "cluster_A_disp_mean": float(np.mean(shock)) if shock else None,
                  "cluster_A_disp_max": int(max(shock)) if shock else 0})
        if anchor is None:
            anchor = float(eng.w.t) * 1.1
        g = e26.gantt_collect(eng, anchor)
        with gzip.open(os.path.join(d, f"{cell}@s{rep}/{label}.gantt.json.gz"),
                       "wt") as f:
            json.dump({"meta": {"cell": cell, "rep": rep, "policy": label,
                                "theta": None, "T_anchor_s": anchor,
                                "K": 200, "m_workers": 32}, "buckets": g},
                      f, separators=(",", ":"))
        arr = np.asarray([(float(a), float(b), float(r_), float(q))
                          for (a, b, r_, q) in eng.w.storage.interval_log])
        np.savez_compressed(
            os.path.join(d, f"{cell}@s{rep}/{label}.intervals.npz"),
            intervals=arr)
        with gzip.open(os.path.join(d, f"{cell}@s{rep}/{label}.requests.json.gz"),
                       "wt") as f:
            json.dump({"meta": {"cell": cell, "rep": rep, "policy": label,
                                "n": len(rq)}, "rows": rq},
                      f, separators=(",", ":"))
        records.append(s)
        print_progress(
            f"E26e {cell}@s{rep} {label:12s} wall={wall:.0f}s "
            f"SLO={s['slo_success']}/{s['n_cohort']} "
            f"A={100*s['A']['slo_rate']:.0f}% B={100*s['B']['slo_rate']:.0f}% "
            f"concA {s['a_conc_p50']:.0f}/{s['a_conc_p90']:.0f}/"
            f"{s['a_conc_max']:.0f} 簇发A均{s['cluster_A_disp_mean']:.1f}/"
            f"峰{s['cluster_A_disp_max']} dev={s['n_deviate']} "
            f"tie={s['n_tie']}")
    return records


def main(seeds, procs=None, duration=None, stage="eval", **kw):
    d = out_dir("eval", "e26e")
    only = [x for x in os.environ.get("E26E_ONLY", "").split(",") if x]
    reps = [int(x) for x in os.environ.get("E26E_REPS", "0").split(",")]
    shard = os.environ.get("E26E_SHARD", "main")
    cells = only or [c for c, _k, _b in K_LEVELS]
    # cells 条目可为 "cell"（全部 rep）或 "cell@rep"（单 rep 精选）
    picked = []
    for ent in cells:
        if "@" in ent:
            cg, rp = ent.split("@", 1)
            picked += [(c, int(rp)) for c, _k, _b in K_LEVELS if c == cg]
        else:
            picked += [(c, r) for c, _k, _b in K_LEVELS if c == ent
                       for r in reps]
    part = os.path.join(d, f"records_part_{shard}.json")
    records = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else []
    done = {(r["cell"], r["rep"], r["policy"]) for r in records}
    # E26E_ARMS：可选臂过滤（单臂粒度分片重跑用；空=全臂）
    arm_sel = [x for x in os.environ.get("E26E_ARMS", "").split(",") if x]
    print_progress(f"E26e eval: {len(picked)} 格点×rep，shard={shard}，"
                   f"arms={arm_sel or '全'}，已完成 run={len(records)}")
    for cell, rep in picked:
        todo = [a[0] for a in ARMS
                if (cell, rep, a[0]) not in done
                and (not arm_sel or a[0] in arm_sel)]
        if not todo:
            continue
        if not todo:
            continue
        os.makedirs(os.path.join(d, f"{cell}@s{rep}"), exist_ok=True)
        recs = run_cell(cell, rep, d, arms=[a for a in ARMS
                                            if a[0] in todo])
        records.extend(recs)
        save_json(part, records)          # 每 cell 完成即落盘（中断最多丢本 cell）
    print_progress(f"E26e done -> {part} ({len(records)} 记录)")
    return {"n_records": len(records)}


if __name__ == "__main__":
    main([0])
