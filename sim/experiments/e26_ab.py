"""E26：A/B 双类带宽争抢工况（用户指定 case；大纲 docs/E26-AB双类带宽争抢实验大纲-20260928.md）。

工况（大纲 §1，用户给定不得更改）：m=32 全局中央队列、L=8、B=120GB/s、
batch=1（n_max=1）；A=128K 前缀/256 重算（每层 175.65MB/6.02ms）、
B=32K 前缀/4096 重算（每层 38.5MB/28.59ms）；总量 1280A+2560B=3840 条。
到达：D1 泊松（λ 按 ρ_storage 定标 89~196 req/s）与 D2 突发轮次
（96 条/轮×40 轮，T_burst∈{0.75,1.0,1.5}s）。策略 6/格点：
cq_fcfs / cq_edf / cq_local×θ{S,T} / cq_mpc×θ{S,T}（理想上限档
budget_s=3600，事件预算 max_events=E26_MPC_EVENTS）。

计算画像标定（大纲 §3，零引擎改动）：虚拟 h + 重标定 (a,b) 精确命中
(c_A, c_B)，方程解为精确 Fraction（U1 单测对拍容差 1e-9）；语义：虚拟 h
使注意力项吸收部分线性投影计算，对调度有意义的量（每层 c/V、T0、deadline、
读取协议）精确匹配用户给定值——结果文档必须复述此语义说明。

记录：逐 run（格点×α×seed×策略）一行 records；ts（aggregate_run 守恒）、
分类甘特桶数据（逐桶每 worker [A算,B算,等待,闲置]）、瞬时带宽账本
（interval_log，θ=S 口径四个策略）另存。图与表工具见 tools/e26_*。

并行：进程级分片。环境变量 E26_ONLY="tag1,tag2"（全格点标签，如
"D1_r0.9_a8"）限定本进程跑的格点；E26_SHARD=名字 决定 records 分片文件名。
"""
from __future__ import annotations

import gzip
import json
import os
import time
from fractions import Fraction as F
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sim.cq.config import CqScenario, ProfileConfig, StorageConfig
from sim.cq.metrics import summarize
from sim.cq.policies import make_edf, make_fcfs
from sim.cq.profile import make_T0
from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.cq.timeseries import (TsConservationError, aggregate_run,
                               bucket_intersections)
from sim.cq.types import HardLimits, RequestSpec
from sim.experiments.cq_common import (out_dir, print_progress,
                                       save_json, setup_matplotlib)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 工况常量（大纲 §1）
# ---------------------------------------------------------------------------

M_WORKERS = 32
L_LAYERS = 8
B_GBPS = 120.0
Q_MAX_GBPS = 200.0
B_REF_GBPS = 120.0        # T0 参考带宽 = 本场景带宽（单带宽实验）
N_A, N_B = 1280, 2560
N_TOTAL = N_A + N_B
# 规模缩放（20260929 用户决策：总请求降到 ~240 缩短端到端）：E26_TOTAL=240
# → 80A+160B（配比与每层 c/V/T0 不变；D1 λ 档不变=到达强度语义不变；
# D2 每轮仍 96 条，轮数=total//96）。默认 3840=大纲原规模。
SCALE_TOTAL = int(os.environ.get("E26_TOTAL", str(N_TOTAL)))
A_CLASS = (42883, 256)    # (h_tokens, u_tokens)
B_CLASS = (9399, 4096)
C_A_S = F("0.00602")      # 用户给定每层计算（秒）
C_B_S = F("0.02859")
LAMBDA_SAT = 178.0        # E[读取]/条 = 0.6737 GB → 120/0.6737

D2_ROUNDS = max(1, SCALE_TOTAL // 96)

D1_LEVELS = (("D1_r0.5", 89.0), ("D1_r0.7", 125.0), ("D1_r0.9", 160.0),
             ("D1_r1.0", 178.0), ("D1_r1.1", 196.0))
D2_LEVELS = (("D2_b0.75", 0.75), ("D2_b1.0", 1.0), ("D2_b1.5", 1.5))
ALPHAS = (4, 8)
SEEDS = 3                 # D1 复制数（中位数）；D2 轮次确定性 → 1 个 seed

MPC_BUDGET_S = 3600.0     # 理想上限档（E25 20260917 先例）
MPC_LIKE = ("cq_local", "cq_mpc")
POLICY_RUNS = (("cq_fcfs", None), ("cq_edf", None),
               ("cq_local", "S"), ("cq_local", "T"),
               ("cq_mpc", "S"), ("cq_mpc", "T"))
# 策略集开关（E26_POLICIES）：full=大纲 6 run；s4=θ=S 四策略（fcfs/edf/
# local_S/mpc_S，验收判据全为 θ=S 口径）；simple=仅基线。20260928 全量
# 因 8 核机器墙钟约束按 s4 执行（结果文档局限披露），默认保持 full。
_POLICY_SETS = {
    "full": POLICY_RUNS,
    "s4": (("cq_fcfs", None), ("cq_edf", None),
           ("cq_local", "S"), ("cq_mpc", "S")),
    "simple": (("cq_fcfs", None), ("cq_edf", None)),
}
SMOKE_CELL = ("D1_r0.9", 8)          # 大纲 §8：D1/λ160/α8 ×6 策略
K_BUCKETS = 200


# ---------------------------------------------------------------------------
# 计算画像标定（大纲 §3）：精确解线性方程组
# ---------------------------------------------------------------------------

def _attn(h: int, u: int) -> int:
    return u * h + u * (u + 1) // 2


def _calibrate() -> ProfileConfig:
    """解 512a + A_A·b = c_A−t_launch；4096a + A_B·b = c_B−t_launch。

    η_A=256/512=0.5 → aN/η=512a；η_B=1 → aN/η=4096a。解为精确 Fraction，
    c_A/c_B 命中用户给定值到浮点舍入（U1 断言 1e-9）。
    """
    t_launch = F(50, 10 ** 6)
    A_A, A_B = _attn(*A_CLASS), _attn(*B_CLASS)
    r1, r2 = C_A_S - t_launch, C_B_S - t_launch
    b = (8 * r1 - r2) / (8 * A_A - A_B)
    a = (r1 - A_A * b) / 512
    return ProfileConfig(L=L_LAYERS, a_s_per_token=a, b_s_per_pair=b)


E26_PROFILE = _calibrate()

E26_LIMITS = HardLimits(n_max=1, token_max=F(8192), workspace_gb=F(8))


def e26_scenario(alpha: int) -> CqScenario:
    return CqScenario(
        scenario_name=f"e26_ab_B{B_GBPS:g}_a{alpha}",
        profile=E26_PROFILE,
        storage=StorageConfig(b_schedule=((F(0), F(int(B_GBPS))),),
                              q_max_gbps=F(int(Q_MAX_GBPS)),
                              b_ref_gbps=F(int(B_REF_GBPS))),
        limits=E26_LIMITS, m_workers=M_WORKERS,
        alpha_slo=F(int(alpha)), source_kind="synthetic", trace_key="e26_ab")


# ---------------------------------------------------------------------------
# 负载生成（大纲 §4；CRN：同格点同 seed 全策略同一份流）
# ---------------------------------------------------------------------------

def _specs_from(classes: Sequence[str], arrivals, alpha: int) -> List[RequestSpec]:
    specs = []
    for i, cls in enumerate(classes):
        h, u = A_CLASS if cls == "A" else B_CLASS
        specs.append(RequestSpec(rid=i, arrival_s=float(arrivals[i]),
                                 h_tokens=h, u_tokens=u, class_id=cls,
                                 T0_s=F(1), deadline_s=F(1)))
    T0 = make_T0(specs, E26_PROFILE, F(int(B_REF_GBPS)), F(int(Q_MAX_GBPS)))
    return [RequestSpec(s.rid, s.arrival_s, s.h_tokens, s.u_tokens, s.class_id,
                        T0[s.rid], float(arrivals[s.rid]) + float(alpha) * float(T0[s.rid]))
            for s in specs]


def d1_specs(lam: float, seed: int, alpha: int) -> List[RequestSpec]:
    """分批持续到达：全局泊松 + 固定配比多重集的随机置换（总量精确 1:2）。

    seed=格点基号×4+rep（模块常量），同格点内全策略共享（CRN）。
    总量=SCALE_TOTAL（默认 3840；E26_TOTAL 可缩放）。
    """
    n_total = SCALE_TOTAL
    n_a = n_total // 3
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    gaps = rng.exponential(scale=1.0 / lam, size=n_total)
    arrivals = np.cumsum(gaps)
    classes = np.empty(n_total, dtype="<U1")
    perm = rng.permutation(n_total)
    classes[perm[:n_a]] = "A"
    classes[perm[n_a:]] = "B"
    return _specs_from(list(classes), arrivals, alpha)


def d2_specs(t_burst: float, alpha: int) -> List[RequestSpec]:
    """突发轮次：D2_ROUNDS 轮 × 96 条（32A+64B，轮内 (A,B,B)×32 交错），确定性。"""
    order = ["A", "B", "B"] * 32
    classes, arrivals = [], []
    for r in range(D2_ROUNDS):
        arrivals.extend([r * t_burst] * len(order))
        classes.extend(order)
    return _specs_from(classes, arrivals, alpha)


def d1_seed(level_idx: int, rep: int) -> int:
    return 400 + level_idx * 8 + rep


# ---------------------------------------------------------------------------
# 分类收集器：甘特四元组 / 并发 A 分布 / 带宽区间统计（大纲 §6）
# ---------------------------------------------------------------------------

def gantt_collect(engine, T_anchor: float, K: int = K_BUCKETS):
    """逐桶每 worker [A算, B算, 等待, 闲置]（秒）。batch=1 → 批类别=成员类别。"""
    w = engine.w
    cls_of = {bid: w.specs[b.members[0]].class_id
              for bid, b in w.batches.items() if len(b.members) == 1}
    delta = T_anchor / K
    n = len(w.workers)
    buckets = [[[0.0] * 4 for _ in range(n)] for _ in range(K + 1)]
    for wk in w.workers.values():
        for (s, e, state, bid, _layer) in wk.segments:
            cls = cls_of.get(bid)
            for i, ov in bucket_intersections(float(s), float(e), delta, K):
                row = buckets[i][wk.worker_id]
                if state == "COMPUTE":
                    row[0 if cls == "A" else 1] += ov
                elif state == "STALL":
                    row[2] += ov
                else:
                    row[3] += ov
    return buckets


def concurrent_a_quantiles(engine) -> Dict[str, Optional[float]]:
    """时间加权并发 A 批数分布 p50/p90/max（A 批占用 worker 的区段扫描）。"""
    w = engine.w
    cls_of = {bid: w.specs[b.members[0]].class_id
              for bid, b in w.batches.items() if len(b.members) == 1}
    evs = []
    for wk in w.workers.values():
        for (s, e, _state, bid, _layer) in wk.segments:
            if bid is not None and cls_of.get(bid) == "A":
                evs.append((float(s), 1))
                evs.append((float(e), -1))
    evs.sort()
    segs = []          # (时长, 并发数)
    cnt = prev = 0.0
    for t, d in evs:
        if t > prev:
            segs.append((t - prev, cnt))
            prev = t
        cnt += d
    if not segs:
        return {"a_conc_p50": None, "a_conc_p90": None, "a_conc_max": 0}
    total = sum(x for x, _ in segs)

    def wq(p):
        acc, prev_acc = 0.0, 0.0
        for dur, c in sorted(segs, key=lambda x: x[1]):
            acc += dur
            if acc >= p * total:
                return float(c)
            prev_acc = acc
        return float(max(c for c, _ in segs))

    return {"a_conc_p50": wq(0.5), "a_conc_p90": wq(0.9),
            "a_conc_max": float(max(c for _, c in segs))}


def bw_interval_stats(interval_log, b_gbps: float = B_GBPS) -> Dict[str, Optional[float]]:
    """带宽饱和占比（实际≥0.9B）与需求超订占比（Σq>B）。"""
    tot = sat = over = 0.0
    for (t0, t1, r, q) in interval_log:
        width = float(t1) - float(t0)
        if width <= 0:
            continue
        tot += width
        if float(r) >= 0.9 * b_gbps:
            sat += width
        if float(q) > b_gbps:
            over += width
    if tot <= 0:
        return {"bw_sat_frac": None, "bw_over_frac": None}
    return {"bw_sat_frac": sat / tot, "bw_over_frac": over / tot}


def class_metrics(engine, specs) -> Dict[str, dict]:
    """分 A/B 两类 TTFT（均值/p95，下界口径）与 SLO 满足率。"""
    w = engine.w
    out = {}
    for cls in ("A", "B"):
        rids = [s.rid for s in specs if s.class_id == cls]
        ttfts = []
        ok = 0
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


# ---------------------------------------------------------------------------
# 单 run 执行与落盘
# ---------------------------------------------------------------------------

def cell_tag(level_tag: str, alpha: int) -> str:
    return f"{level_tag}_a{alpha}"


def _save_gantt(path: str, buckets, meta: dict):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"meta": meta, "buckets": buckets}, f, separators=(",", ":"))


def _save_intervals(path: str, interval_log):
    arr = np.asarray([(float(t0), float(t1), float(r), float(q))
                      for (t0, t1, r, q) in interval_log], dtype=np.float64)
    np.savez_compressed(path, intervals=arr)


def run_one_cell(level_tag: str, alpha: int, rep: int, d: str,
                 specs: List[RequestSpec], mpc_events: int,
                 policy_set=None) -> List[dict]:
    """跑一个（格点×α×seed）的策略集；返回记录列表。"""
    tag = cell_tag(level_tag, alpha)
    policy_set = policy_set if policy_set is not None else POLICY_RUNS
    scn = e26_scenario(alpha)
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    cid = f"{tag}|s{rep}"
    os.makedirs(os.path.join(d, cid), exist_ok=True)
    anchor = None
    records = []
    for pid, theta in policy_set:
        if pid in MPC_LIKE:
            pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1,
                            theta=theta, c_ref=c_ref, budget_s=MPC_BUDGET_S,
                            max_events=mpc_events)
        else:
            pol = make_fcfs() if pid == "cq_fcfs" else make_edf()
        t0 = time.time()
        eng = run_case(scn, specs, pol, numeric=float, seed=rep,
                       record_intervals=True)
        wall = time.time() - t0
        s = summarize(eng, scn)
        s["slo_rate"] = (s["slo_success"] / s["n_cohort"]) if s["n_cohort"] else None
        s.update(class_metrics(eng, specs))
        s.update(concurrent_a_quantiles(eng))
        s.update(bw_interval_stats(eng.w.storage.interval_log))
        comp, stall = s["device_compute_s"], s["device_stall_s"]
        s["io_wait_share2"] = (comp / (comp + stall)) if comp + stall > 0 else None
        ctrl = sorted(dd["ctrl_s"] for dd in eng.decisions) or [0.0]
        s["ctrl_p50_ms"] = 1000.0 * ctrl[len(ctrl) // 2]
        s["ctrl_p95_ms"] = 1000.0 * ctrl[max(0, int(0.95 * len(ctrl)) - 1)]
        s["n_scored"] = getattr(pol, "n_scored", None)
        s["n_fallback"] = getattr(pol, "n_fallback", None)
        s["n_overrun"] = getattr(pol, "n_overrun", None)
        s["wall_s"] = wall
        s.update({"policy": pid, "theta": theta, "cell": tag, "level": level_tag,
                  "mode": level_tag[:2], "alpha": alpha, "rep": rep,
                  "n_req": len(specs), "n_total": SCALE_TOTAL,
                  "makespan_s": float(eng.w.t),
                  "scenario": scn.scenario_name,
                  "mpc_events": mpc_events if pid in MPC_LIKE else None})
        # ts 聚合（守恒校验；失败记 invalid_ts 不静默）
        if anchor is None:
            anchor = float(eng.w.t) * 1.1
        suffix = f"_{theta}" if pid in MPC_LIKE else ""
        ts_rel = f"{cid}/{pid}{suffix}.ts.json.gz"
        try:
            ts = aggregate_run(eng, anchor, K=K_BUCKETS, cell=cid,
                               policy=pid,
                               theta=(theta if pid in MPC_LIKE else None))
            from sim.cq.timeseries import save_ts
            save_ts(os.path.join(d, ts_rel), ts)
            s["ts_path"], s["delta_s"] = ts_rel, ts["meta"]["delta_s"]
        except TsConservationError as e:
            s["invalid_ts"] = str(e)
            s["ts_path"] = None
        # 甘特桶数据（全部 run）
        g_path = f"{cid}/{pid}{suffix}.gantt.json.gz"
        _save_gantt(os.path.join(d, g_path),
                    gantt_collect(eng, anchor),
                    {"cell": cid, "policy": pid, "theta": theta,
                     "T_anchor_s": anchor, "K": K_BUCKETS,
                     "m_workers": M_WORKERS})
        # 瞬时带宽账本（θ=S 口径四策略，供图 2）
        if theta in (None, "S"):
            _save_intervals(os.path.join(d, f"{cid}/{pid}{suffix}.intervals.npz"),
                            eng.w.storage.interval_log)
        records.append(s)
        print_progress(
            f"E26 {cid} {pid}{suffix or ''} wall={wall:.0f}s "
            f"M={float(eng.w.t):.1f}s SLO={s['slo_success']}/{s['n_cohort']} "
            f"n_scored={s['n_scored']} n_fb={s['n_fallback']}")
    return records


# ---------------------------------------------------------------------------
# 矩阵与 main
# ---------------------------------------------------------------------------

def all_cells() -> List[Tuple[str, str, float, int, int, int]]:
    """(level_tag, mode, 参数值, α, rep, seed) 全矩阵；D2 确定性只跑 rep=0。"""
    cells = []
    for i, (tag, lam) in enumerate(D1_LEVELS):
        for alpha in ALPHAS:
            for rep in range(SEEDS):
                cells.append((tag, "D1", lam, alpha, rep, d1_seed(i, rep)))
    for tag, tb in D2_LEVELS:
        for alpha in ALPHAS:
            cells.append((tag, "D2", tb, alpha, 0, 0))
    return cells


def build_specs(cell) -> List[RequestSpec]:
    tag, mode, param, alpha, rep, seed = cell
    if mode == "D1":
        return d1_specs(param, seed, alpha)
    return d2_specs(param, alpha)


def load_records(d: str) -> List[dict]:
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(d, "records_part*.json"))):
        out.extend(json.load(open(p, encoding="utf-8")))
    return out


def main(seeds, procs=None, duration=None, stage="smoke",
         mpc_events=None, **kw):
    """入口。stage=smoke：大纲 §8 冒烟格点；stage=eval：全矩阵（分片见模块 docstring）。

    规模：E26_TOTAL 环境变量（默认 3840=大纲原规模；20260929 用户决策缩放
    240）。非默认规模的结果目录为 e26n<total>，与原规模数据隔离。
    """
    exp_dir = "e26" if SCALE_TOTAL == N_TOTAL else f"e26n{SCALE_TOTAL}"
    d = out_dir(stage, exp_dir)
    if mpc_events is None:
        mpc_events = int(os.environ.get("E26_MPC_EVENTS", 20000))
    only = [x for x in os.environ.get("E26_ONLY", "").split(",") if x]
    shard = os.environ.get("E26_SHARD", "main")
    if stage == "smoke":
        cells = [c for c in all_cells()
                 if c[0] == SMOKE_CELL[0] and c[3] == SMOKE_CELL[1] and c[4] == 0]
    else:
        cells = all_cells()
        if only:
            # 条目 "tag"（该格点全部 rep）或 "tag@rep"（单 rep）
            picked = []
            for ent in only:
                if "@" in ent:
                    tg, rp = ent.split("@", 1)
                    picked += [c for c in cells
                               if cell_tag(c[0], c[3]) == tg and c[4] == int(rp)]
                else:
                    picked += [c for c in cells if cell_tag(c[0], c[3]) == ent]
            cells = picked
    part_path = os.path.join(d, f"records_part_{shard}.json")
    policy_set = _POLICY_SETS[os.environ.get("E26_POLICIES", "full")]
    records = json.load(open(part_path, encoding="utf-8")) if os.path.exists(part_path) else []
    done = {(r["cell"], r["policy"], r.get("theta"), r["rep"]) for r in records}
    print_progress(f"E26 {stage}: {len(cells)} cells, mpc_events={mpc_events}, "
                   f"shard={shard}, policies={os.environ.get('E26_POLICIES', 'full')}, "
                   f"已完成 run={len(records)}")
    for cell in cells:
        tag, mode, param, alpha, rep, seed = cell
        specs = None
        for pid, theta in policy_set:
            if (cell_tag(tag, alpha), pid, theta, rep) in done:
                continue    # 逐 run 续跑（增量落盘，中断最多丢当前 run）
            if specs is None:
                specs = build_specs(cell)
            recs = run_one_cell(tag, alpha, rep, d, specs, mpc_events,
                                policy_set=[(pid, theta)])
            records.extend(recs)
            save_json(part_path, records)     # 每个 run 完成立即落盘
            done.update((r["cell"], r["policy"], r.get("theta"), r["rep"])
                        for r in recs)
    print_progress(f"E26 {stage} done -> {part_path} ({len(records)} 记录)")
    return {"n_records": len(records)}


if __name__ == "__main__":
    main([0])
