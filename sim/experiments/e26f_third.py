"""E26f：零读第三类 C 请求实验 D1f（设计 docs/E26f-零读第三类实验-20261011.md）。

在 E26e D1c（A:B=1:1、A 簇 k=16）基础上增加第三类 C=零读取纯计算请求
（h=0 → V=0，不占带宽；u 由二分解出使每层计算恰等于 A/B 的每层计算），
两档 C 计算强度构成挤出轻/重对照：
  D1fA_k16：C_A 型（u=2657，每层算 6.02ms=A）——恒占 ~4.7 NPU，存储仍
  唯一瓶颈（计算下限 1.219s<存储 1.713s）；
  D1fB_k16：C_B 型（u=8110，每层算 28.59ms=B）——恒占 ~22.4 NPU（32 的
  70%），计算翻转为主瓶颈（1.896s>1.713s）。
回答四问：C 的填空与排队死 / C 的挤出（轻 vs 重）/ 瓶颈翻转后带宽感知
调度收益的变化 / EDF 期限序被零读墙稀释后的结构（C_A 期限 192.7ms 早于
A 的 198.5ms——无首层读使 T0=8c 更短）。

负载 D1f（三流，单 PCG64(seed) 顺序抽，CRN：同格点全臂同一份）：
  A 簇流——与 E26e D1c 完全同构（簇心间隔 Exp(k/λ)，每簇恰 k=16 条同刻，
  余数入末簇 7×16+8）；B 泊松流——间隔 Exp(1/λ) 每次 1 条；C 泊松流——
  与 B 同构（唯一新变量=存在零读类）。λ_A=λ_B=λ_C=98/s（总 294/s）、
  α=4、三类各 120 条共 360、到达期≈1.22s。
  注：D1f 与 D1c 的 RNG 抽取序列不同，不可逐位对照（设计 §2.2/§9）。

臂（5，引擎口径与 E26d/e 完全一致）：fcfs / edf / localS_v2.1 / mpcS_v2.1 /
mpcT_v2.1（base=bw_edf、composer=True、budget_s=3600、事件预算 2 万、H=1）；
全量以 MPC_PAR=6 执行（fork 候选评分并行，20261011 验证零结果影响）。

seed：400+基号×8+rep（沿 e26.d1_seed；D1fA 基号 2→rep0=416、D1fB 基号
3→rep0=424）。矩阵：2 格点 × 5 臂 × 1 seed = 10 runs；每格点独立判定，
MPC−FCFS 不达 H1 阈值则补 rep∈{1,2} 取中位数。
输出 results/cq/eval/e26f/（records 分片 + 五列甘特 json.gz + intervals npz +
requests json.gz 三类全量）。环境变量：E26F_ONLY="D1fA_k16[,D1fB_k16]"
（或 cell@rep 精选）、E26F_ARMS（单臂分片）、E26F_SHARD、E26F_REPS、
E26F_TOTAL（默认 360，整体缩放兜底）。
"""
from __future__ import annotations

import gzip
import json
import os
import time
from fractions import Fraction as F

import numpy as np

from sim.cq.policies import make_edf, make_fcfs, make_tfs
from sim.cq.profile import batch_compute_s, layer_read_gb
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.metrics import summarize
from sim.cq.timeseries import bucket_intersections
from sim.cq.types import RequestSpec
from sim.experiments import e26_ab as e26
from sim.experiments.cq_common import out_dir, print_progress, save_json

MPC_BUDGET_S = 3600.0
MPC_EVENTS = 20000
ALPHA = 4
LAM_EACH = 98.0                   # λ_A=λ_B=λ_C（每类；总到达率 294/s）
N_EACH = 120                      # 三类各 120（1:1:1，共 360）
SCALE_TOTAL = int(os.environ.get("E26F_TOTAL", str(3 * N_EACH)))
SHOCK_WINDOW_S = 0.05             # 簇冲击响应窗口：簇心后 50ms
K_BUCKETS = 200
# 20261018 存储层重跑（docs/存储带宽maxmin替换设计-20261016.md）：max_min =
# 按申请率 q 的带需求上限水土填充（新流提交即时重配全部在途流速率），默认
# fcfs 保持既有口径；输出目录/records 自动分流到 e26f_mm，互不覆盖。
# 20261021 盘维度第三臂（docs/E26f-盘维度存储实验设计-20261021.md）：
# E26F_ALLOC=max_min_disk → B=120 平摊 3 盘（disk_bps=40 派生）、块 HRW 落盘、
# (流,盘) 组级分配/完成事件；records 增盘统计、npz 增盘速率矩阵，输出目录
# e26f_max_min_disk（EXP_DIR_NAME 规则沿用），前两组结果不覆盖。
ALLOC = os.environ.get("E26F_ALLOC", "fcfs")
EXP_DIR_NAME = "e26f" if ALLOC == "fcfs" else f"e26f_{ALLOC}"

ARMS = [("fcfs", "cq_fcfs", None, None, None),
        ("edf", "cq_edf", None, None, None),
        ("localS_v2.1", "cq_local", "S", "bw_edf", True),
        ("mpcS_v2.1", "cq_mpc", "S", "bw_edf", True),
        ("mpcT_v2.1", "cq_mpc", "T", "bw_edf", True),
        ("tfs", "cq_tfs", None, None, None)]   # 20261015 增补臂(TFS 设计 §4)

# (cell, c_variant, seed 基号)：C 变体决定计算强度档（A=轻挤出/B=重挤出）
C_LEVELS = (("D1fA_k16", "A", 2), ("D1fB_k16", "B", 3))


# ---------------------------------------------------------------------------
# C 类构造：二分整数 u 使 c(u, h=0) 命中目标每层计算（零新自由度，标定
# 复用 E26_PROFILE；与 E26 标定同一公式 batch_compute_s）
# ---------------------------------------------------------------------------

def _c_pure(u: int) -> float:
    rs = RequestSpec(rid=0, arrival_s=F(0), h_tokens=0, u_tokens=int(u),
                     class_id="C", T0_s=F(1), deadline_s=F(1))
    return float(batch_compute_s([rs], e26.E26_PROFILE))


def solve_c_u(target: float) -> int:
    """最小化 |c(u,0)−target| 的整数 u（c 随 u 严格增：a,b>0）。

    整数 u 离散限制下命中容差 ≤2e-6（单测锁定：2657→差 1.03μs、
    8110→差 1.13μs，均为邻点中的最近者）。
    """
    t = float(target)
    lo, hi = 1, 1 << 20
    while lo < hi:                 # 最小 u 使 c(u) ≥ t
        mid = (lo + hi) // 2
        if _c_pure(mid) < t:
            lo = mid + 1
        else:
            hi = mid
    if lo > 1 and abs(_c_pure(lo - 1) - t) <= abs(_c_pure(lo) - t):
        return lo - 1
    return lo


C_A_CLASS = (0, 2657)   # 零读纯算：c=0.006021030s=A 每层算；T0=8c=48.17ms
C_B_CLASS = (0, 8110)   # 零读纯算：c=0.028591131s=B 每层算；T0=8c=228.7ms
C_CLASSES = {"A": C_A_CLASS, "B": C_B_CLASS}


def d1f_seed(base: int, rep: int) -> int:
    """沿 e26.d1_seed 公式：400 + 基号*8 + rep。"""
    return 400 + base * 8 + rep


# ---------------------------------------------------------------------------
# D1f 负载：三流（A 簇 + B 泊松 + C 泊松），单 PCG64 顺序抽
# ---------------------------------------------------------------------------

def d1f_specs(k: int, seed: int, c_class, alpha: int = ALPHA, n: int = N_EACH,
              lam_each: float = LAM_EACH):
    """三类混合负载。返回 (specs, a_centers)。

    A 簇流与 E26e D1c 完全同构（每簇心恰 k 条同刻，余数入末簇）；
    B/C 泊松流与 B 同构（每次 1 条）。合并按时刻稳定排序（同时刻 A 簇
    在前 → A 簇 rid 连续）；T0 由 e26 工况解析式生成（V=0 → T0=8c），
    deadline=arrival+α·T0。
    """
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    t, a_times, centers = 0.0, [], []
    while len(a_times) < n:              # A 簇流：间隔 Exp(k/λ_each)
        t += rng.exponential(k / lam_each)
        m = min(k, n - len(a_times))     # 末簇补余（k=16: 7×16+8）
        a_times.extend([t] * m)
        centers.append(float(t))
    b_times, t = [], 0.0
    while len(b_times) < n:              # B 流：间隔 Exp(1/λ_each)，每次 1 条
        t += rng.exponential(1 / lam_each)
        b_times.append(float(t))
    c_times, t = [], 0.0
    while len(c_times) < n:              # C 流：与 B 同构
        t += rng.exponential(1 / lam_each)
        c_times.append(float(t))
    events = ([(x, "A") for x in a_times] + [(x, "B") for x in b_times]
              + [(x, "C") for x in c_times])
    events.sort(key=lambda e: e[0])      # 稳定：同刻 A 在前
    cls_map = {"A": e26.A_CLASS, "B": e26.B_CLASS, "C": tuple(c_class)}
    specs = [RequestSpec(i, float(ev[0]), *cls_map[ev[1]], ev[1], F(1), F(1))
             for i, ev in enumerate(events)]
    T0 = e26.make_T0(specs, e26.E26_PROFILE, F(120), F(200))
    out = [e26.RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens,
                           s.class_id, T0[s.rid],
                           float(s.arrival_s) + alpha * float(T0[s.rid]))
           for s in specs]
    return out, centers


# ---------------------------------------------------------------------------
# 收集器：五列甘特 / 分类指标（A/B/C）/ 分类并发分布 / 簇冲击（A/B/C）
# ---------------------------------------------------------------------------

def gantt_collect5(engine, T_anchor: float, K: int = K_BUCKETS):
    """逐桶每 worker [A算, B算, C算, 等待, 闲置]（秒）。batch=1 → 批类别=成员类别。"""
    w = engine.w
    cls_of = {bid: w.specs[b.members[0]].class_id
              for bid, b in w.batches.items() if len(b.members) == 1}
    col = {"A": 0, "B": 1, "C": 2}
    delta = T_anchor / K
    n = len(w.workers)
    buckets = [[[0.0] * 5 for _ in range(n)] for _ in range(K + 1)]
    for wk in w.workers.values():
        for (s, e, state, bid, _layer) in wk.segments:
            cls = cls_of.get(bid)
            for i, ov in bucket_intersections(float(s), float(e), delta, K):
                row = buckets[i][wk.worker_id]
                if state == "COMPUTE":
                    row[col.get(cls, 0)] += ov
                elif state == "STALL":
                    row[3] += ov
                else:
                    row[4] += ov
    return buckets


def class_metrics3(engine, specs) -> dict:
    """分 A/B/C 三类 TTFT（均值/p95，下界口径）与 SLO 满足率（E26e 口径扩 C）。"""
    w = engine.w
    out = {}
    for cls in ("A", "B", "C"):
        rids = [s.rid for s in specs if s.class_id == cls]
        ttfts, ok = [], 0
        for rid in rids:
            rr = w.requests[rid]
            s = specs[rid]
            tt = (float(rr.F_s) if rr.F_s is not None
                  else float(w.t)) - float(s.arrival_s)
            ttfts.append(tt)
            if rr.F_s is not None and float(rr.F_s) <= float(s.deadline_s):
                ok += 1
        out[cls] = {
            "n": len(rids), "ttft_mean": float(np.mean(ttfts)) if ttfts else None,
            "ttft_p95": float(np.quantile(ttfts, 0.95)) if ttfts else None,
            "slo_rate": ok / len(rids) if rids else None}
    return out


def concurrent_class_quantiles(engine, cls: str) -> dict:
    """时间加权并发 cls 批数分布 p50/p90/max（E26e A 口径的类参数化版）。"""
    w = engine.w
    cls_of = {bid: w.specs[b.members[0]].class_id
              for bid, b in w.batches.items() if len(b.members) == 1}
    evs = []
    for wk in w.workers.values():
        for (s, e, _state, bid, _layer) in wk.segments:
            if bid is not None and cls_of.get(bid) == cls:
                evs.append((float(s), 1))
                evs.append((float(e), -1))
    evs.sort()
    pfx = cls.lower()
    segs, cnt, prev = [], 0, 0.0
    for t, d in evs:
        if t > prev:
            segs.append((t - prev, cnt))
            prev = t
        cnt += d
    if not segs:
        return {f"{pfx}_conc_p50": None, f"{pfx}_conc_p90": None,
                f"{pfx}_conc_max": 0}
    total = sum(x for x, _ in segs)

    def wq(p):
        acc = 0.0
        for dur, c in sorted(segs, key=lambda x: x[1]):
            acc += dur
            if acc >= p * total:
                return float(c)
        return float(max(c for c, _ in segs))

    return {f"{pfx}_conc_p50": wq(0.5), f"{pfx}_conc_p90": wq(0.9),
            f"{pfx}_conc_max": float(max(c for _, c in segs))}


def npu_util_from_gantt(buckets, m: int, makespan: float) -> dict:
    """逐 NPU 利用率（纯算=a+b+c 列 / makespan）分布统计 + 含等对照。

    口径沿 E26e 20260911 裁定：只有计算算占用，IO 等与闲不计；
    makespan 用本 run 自身 eng.w.t（非首臂甘特 anchor）。
    """
    acc_c = np.zeros(m)
    acc_d = np.zeros(m)
    wait_t = act_t = 0.0
    for bk in buckets:
        for wid, (a, b, c, w_, _idle) in enumerate(bk):
            comp = a + b + c
            acc_c[wid] += comp
            acc_d[wid] += comp + w_
            wait_t += w_
            act_t += comp
    u_c = acc_c / makespan if makespan > 0 else acc_c * 0.0
    return {"util_min": float(u_c.min()), "util_p50": float(np.median(u_c)),
            "util_p90": float(np.quantile(u_c, 0.9)),
            "util_max": float(u_c.max()),
            "util_cv": float(np.std(u_c) / np.mean(u_c)) if u_c.mean() > 0 else None,
            "util_wait_mean": float((acc_d / makespan).mean()) if makespan > 0 else None,
            "io_wait_share_gantt": wait_t / (wait_t + act_t)
            if wait_t + act_t > 0 else None}


def disk_stats(st) -> dict:
    """盘维度统计（仅 max_min_disk；数据源 disk_interval_log，设计 §5）。

    disk_util_mean_d：每盘时间加权均值利用率（分母=区段起点盘容量 B(t0)/disks）；
    disk_util_cv：逐区段盘间 CV 的时间加权均值；cv15_frac：CV>0.15 的时间占比
    （H3 判定口径）；max_disk_frac_{A,B,C}：类计划的最慢盘字节占比（E26f
    合成负载下为类常量，由真实类前缀 disk_plan 算出，单测锁定）。
    """
    from sim.cq.storage import disk_plan
    out = {"disks": st.disks if st.uses_groups else None,
           "disk_bps_gbps": float(st.disk_bps) if st.uses_groups else None,
           "disk_util_cv": None, "disk_cv15_frac": None,
           "max_disk_frac_A": None, "max_disk_frac_B": None,
           "max_disk_frac_C": None}
    for d in range(getattr(st, "disks", 0) if st.uses_groups else 0):
        out[f"disk_util_mean_{d}"] = None
    if not st.uses_groups:
        return out
    wsum = 0.0
    acc = [0.0] * st.disks
    cv_acc = cv15_t = 0.0
    for (t0, t1, rates) in st.disk_interval_log:
        dt = float(t1) - float(t0)
        if dt <= 0:
            continue
        cap_d = float(st.b_at(t0)) / st.disks
        utils = [float(r) / cap_d if cap_d > 0 else 0.0 for r in rates]
        wsum += dt
        for d in range(st.disks):
            acc[d] += utils[d] * dt
        mu = sum(utils) / st.disks
        cv = ((sum((u - mu) ** 2 for u in utils) / st.disks) ** 0.5 / mu
              if mu > 0 else 0.0)
        cv_acc += cv * dt
        if cv > 0.15:
            cv15_t += dt
    for d in range(st.disks):
        out[f"disk_util_mean_{d}"] = acc[d] / wsum if wsum > 0 else None
    out["disk_util_cv"] = cv_acc / wsum if wsum > 0 else None
    out["disk_cv15_frac"] = cv15_t / wsum if wsum > 0 else None
    for cls, (h, _u) in (("A", e26.A_CLASS), ("B", e26.B_CLASS)):
        plan = disk_plan(f"{cls}:{h}", h, st.disks, st.block_tokens, st.kappa)
        tot = sum(b for _d, _c, b in plan)
        out[f"max_disk_frac_{cls}"] = (
            float(max(b for _d, _c, b in plan) / tot) if tot > 0 else None)
    return out


def cluster_shock3(engine, centers, window_s: float = SHOCK_WINDOW_S):
    """簇冲击响应：每簇心后 window_s 内实际派发的 A/B/C 条数（三列观测）。"""
    specs = engine.w.specs
    disp = [(float(b.dispatch_s), specs[b.members[0]].class_id)
            for b in engine.w.batches.values()]
    rows = {"A": [], "B": [], "C": []}
    for c in centers:
        cnt = {"A": 0, "B": 0, "C": 0}
        for t, cl in disp:
            if c <= t < c + window_s:
                cnt[cl] += 1
        for cl in ("A", "B", "C"):
            rows[cl].append(cnt[cl])
    return rows


# ---------------------------------------------------------------------------
# 单 run 执行与落盘（逐臂：records 增量 + 甘特/区间/requests 三件套）
# ---------------------------------------------------------------------------

def run_cell(cell: str, rep: int, d: str, arms=None):
    variant, base = next((v, b) for c, v, b in C_LEVELS if c == cell)
    seed = d1f_seed(base, rep)
    n_each = max(1, SCALE_TOTAL // 3)
    specs, centers = d1f_specs(16, seed, C_CLASSES[variant], n=n_each)
    scn = e26.e26_scenario(ALPHA)
    if ALLOC != "fcfs":
        from dataclasses import replace as _replace
        scn = _replace(scn, storage=_replace(scn.storage, alloc=ALLOC))
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    anchor = None
    records = []
    for label, pid, theta, base_pol, composer in (arms or ARMS):
        if pid == "cq_fcfs":
            pol = make_fcfs()
        elif pid == "cq_edf":
            pol = make_edf()
        elif pid == "cq_tfs":
            pol = make_tfs()
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
        s.update(class_metrics3(eng, specs))
        s.update(concurrent_class_quantiles(eng, "A"))
        s.update(concurrent_class_quantiles(eng, "C"))
        s.update(e26.bw_interval_stats(eng.w.storage.interval_log))
        comp, stall = s["device_compute_s"], s["device_stall_s"]
        s["io_wait_share2"] = comp / (comp + stall) if comp + stall > 0 else None
        s["idle_frac"] = s["device_idle_s"] / s["device_total_mh_s"]
        shock = cluster_shock3(eng, centers)
        # 逐请求完成记录（TTFT CDF / makespan 精确口径 / 期限裕量分析用）
        rq = []
        f_list = []
        for sp in specs:
            rr = eng.w.requests[sp.rid]
            fs = float(rr.F_s) if rr.F_s is not None else float(eng.w.t)
            f_list.append(fs)
            rq.append([sp.rid, sp.class_id, float(sp.arrival_s),
                       float(sp.deadline_s), fs, fs - float(sp.arrival_s)])
        tt_all = [r[5] for r in rq]
        last_arr = max(r[2] for r in rq)
        max_f = max(f_list)
        if anchor is None:
            anchor = float(eng.w.t) * 1.1
        g = gantt_collect5(eng, anchor)
        util = npu_util_from_gantt(g, len(eng.w.workers), float(eng.w.t))
        h = pol.health() if hasattr(pol, "health") else {}
        s.update({"policy": label, "pid": pid, "theta": theta,
                  "mpc_base": base_pol, "composer": composer, "cell": cell,
                  "c_variant": variant, "k": 16, "rep": rep, "seed": seed,
                  "alpha": ALPHA, "n_total": len(specs), "alloc": ALLOC,
                  "makespan_s": float(eng.w.t), "wall_s": wall,
                  "last_arrival_s": last_arr, "max_F_s": max_f,
                  "drain_s": max_f - last_arr,
                  "ttft_all_mean": float(np.mean(tt_all)),
                  "ttft_all_p95": float(np.quantile(tt_all, 0.95)),
                  "n_decides": h.get("n_decides"),
                  "n_searched": h.get("n_searched"),
                  "n_scored": h.get("n_scored"),
                  "n_fallback": h.get("n_fallback"),
                  "n_deviate": h.get("n_deviate"),
                  "n_tie": h.get("n_tie"),
                  "n_overrun": h.get("n_overrun"),
                  "n_clusters": len(centers),
                  "cluster_A_disp": shock["A"],
                  "cluster_B_disp": shock["B"],
                  "cluster_C_disp": shock["C"],
                  "cluster_A_disp_mean": float(np.mean(shock["A"]))
                  if shock["A"] else None,
                  "cluster_A_disp_max": int(max(shock["A"])) if shock["A"] else 0})
        s.update(util)
        s.update(disk_stats(eng.w.storage))
        with gzip.open(os.path.join(d, f"{cell}@s{rep}/{label}.gantt.json.gz"),
                       "wt") as f:
            json.dump({"meta": {"cell": cell, "rep": rep, "policy": label,
                                "theta": None, "T_anchor_s": anchor,
                                "K": K_BUCKETS, "m_workers": 32}, "buckets": g},
                      f, separators=(",", ":"))
        arr = np.asarray([(float(a), float(b), float(r_), float(q))
                          for (a, b, r_, q) in eng.w.storage.interval_log])
        # 盘速率矩阵（仅 max_min_disk；与 intervals 行逐行同区段，E25 工具读
        # intervals 键不受影响）
        if eng.w.storage.uses_groups:
            disk_arr = np.asarray([list(r) for (_a, _b, r)
                                   in eng.w.storage.disk_interval_log],
                                  dtype=float)
        else:
            disk_arr = np.zeros((0, 0))
        np.savez_compressed(
            os.path.join(d, f"{cell}@s{rep}/{label}.intervals.npz"),
            intervals=arr, disk_rates=disk_arr)
        with gzip.open(os.path.join(d, f"{cell}@s{rep}/{label}.requests.json.gz"),
                       "wt") as f:
            json.dump({"meta": {"cell": cell, "rep": rep, "policy": label,
                                "c_variant": variant, "n": len(rq)},
                       "rows": rq},
                      f, separators=(",", ":"))
        records.append(s)
        print_progress(
            f"E26f {cell}@s{rep} {label:12s} wall={wall:.0f}s "
            f"SLO={s['slo_success']}/{s['n_cohort']} "
            f"A={100*s['A']['slo_rate']:.0f}% B={100*s['B']['slo_rate']:.0f}% "
            f"C={100*s['C']['slo_rate']:.0f}% "
            f"concA {s['a_conc_p50']:.0f}/{s['a_conc_p90']:.0f}/"
            f"{s['a_conc_max']:.0f} 簇发A均{s['cluster_A_disp_mean']:.1f}/"
            f"峰{s['cluster_A_disp_max']} dev={s['n_deviate']} "
            f"tie={s['n_tie']}")
    return records


def main(seeds, procs=None, duration=None, stage="eval", **kw):
    d = out_dir("eval", EXP_DIR_NAME)
    only = [x for x in os.environ.get("E26F_ONLY", "").split(",") if x]
    reps = [int(x) for x in os.environ.get("E26F_REPS", "0").split(",")]
    shard = os.environ.get("E26F_SHARD", "main")
    cells = only or [c for c, _v, _b in C_LEVELS]
    # cells 条目可为 "cell"（全部 rep）或 "cell@rep"（单 rep 精选）
    picked = []
    for ent in cells:
        if "@" in ent:
            cg, rp = ent.split("@", 1)
            picked += [(c, int(rp)) for c, _v, _b in C_LEVELS if c == cg]
        else:
            picked += [(c, r) for c, _v, _b in C_LEVELS if c == ent
                       for r in reps]
    part = os.path.join(d, f"records_part_{shard}.json")
    records = json.load(open(part, encoding="utf-8")) if os.path.exists(part) else []
    done = {(r["cell"], r["rep"], r["policy"]) for r in records}
    # E26F_ARMS：可选臂过滤（单臂粒度分片重跑用；空=全臂）
    arm_sel = [x for x in os.environ.get("E26F_ARMS", "").split(",") if x]
    print_progress(f"E26f eval: {len(picked)} 格点×rep，shard={shard}，"
                   f"arms={arm_sel or '全'}，MPC_PAR={os.environ.get('MPC_PAR', '0')}，"
                   f"已完成 run={len(records)}")
    for cell, rep in picked:
        todo = [a[0] for a in ARMS
                if (cell, rep, a[0]) not in done
                and (not arm_sel or a[0] in arm_sel)]
        if not todo:
            continue
        os.makedirs(os.path.join(d, f"{cell}@s{rep}"), exist_ok=True)
        recs = run_cell(cell, rep, d, arms=[a for a in ARMS
                                            if a[0] in todo])
        records.extend(recs)
        save_json(part, records)          # 每 cell 完成即落盘（中断最多丢本 cell）
    print_progress(f"E26f done -> {part} ({len(records)} 记录)")
    return {"n_records": len(records)}


if __name__ == "__main__":
    main([0])
