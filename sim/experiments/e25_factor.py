"""E25：调度因子地图与时间序列面板（设计与测试规格 20260916）。

两种模式：FW 有限工作集（θ∈{M,S,T,R}，M 为真实 makespan）与 OL 在线流
（θ∈{S,T,R} + drain_M 描述性口径）。组织为基线格点 + 六条单因子扫描线
（§3.2），线间不做笛卡尔积。每个格点先跑 cq_fcfs 锚定 run 确定 T_anchor，
格点内所有策略共用 K=200 锚定桶网格；时间序列事后聚合（§5）。
图 cq_fig_e25_{objectives,factor_*,pareto,ts_device,ts_storage,ts_queue}。

MPC/local 沿用 E21/E23 eval 先例取 H=1（简单策略跨 θ 复用同一物理 run）。
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
from fractions import Fraction as F

import numpy as np

from sim.cq.config import ProfileConfig
from sim.cq.metrics import summarize
from sim.cq.policies import SIMPLE_POLICIES
from sim.cq.profile import batch_compute_s, make_T0
from sim.cq.simrun import run_case
from sim.cq.timeseries import (TsConservationError, aggregate_run, anchor_T,
                               b_max_in, load_ts, row_fractions, save_ts)
from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL, trace_hash
from sim.cq.types import HardLimits, RequestSpec
from sim.experiments.e21_batch import quadrant_guard
from sim.experiments.cq_common import (ALL_POLICIES, TRACE_DIR_DEFAULT,
                                       TRACE_WIDE_LIMITS, MooncakeSource,
                                       build_policy, c_ref_of,
                                       default_scenario, out_dir,
                                       print_progress, save_json,
                                       setup_matplotlib)

BASE_WINDOWS = [0, 4, 9, 14, 19]
FW_THETAS = ["M", "S", "T", "R"]
OL_THETAS = ["S", "T", "R"]
TS_POLICIES = ["cq_fcfs", "cq_spt", "cq_edf", "cq_mpc"]
MPC_LIKE = ("cq_mpc", "cq_local")
SIMPLE_PIDS = ["cq_fcfs", "cq_lpm", "cq_spt", "cq_edf", "cq_slack", "cq_pair"]
SMOKE_POLICIES = ["cq_fcfs", "cq_spt", "cq_edf", "cq_mpc"]

PROFILES = {
    "default": ProfileConfig(),
    "no_speedup": ProfileConfig(no_speedup=True),
    "n_sat128": ProfileConfig(N_sat=128),
}

MIN_REQS = 16          # 窗口最小有效请求数（不足则记 skip，不静默缺失）
MPC_BUDGET_S = 3600.0  # 20260917 重跑：MPC 理想上限档（CPU 放开，事件预算 2 万仍为硬界）
TS_WINDOW_DEFAULT = 9  # 时间序列图默认窗口


def windows_for_seeds(seeds: int):
    """--seeds 映射为窗口编号列表（BASE_WINDOWS 前缀，§9）。"""
    return BASE_WINDOWS[:max(1, min(int(seeds), len(BASE_WINDOWS)))]


def cell_id(c: dict) -> str:
    rho = "-" if c.get("rho") is None else f"{c['rho']:g}"
    return (f"{_stem(c['file'])}|{c['mode']}|B{c['B']:g}|rho{rho}"
            f"|a{c['alpha']:g}|{c['cap']}|{c['profile']}|w{c['window']}")


def _stem(fname: str) -> str:
    return fname.replace("_trace.jsonl", "").replace(".jsonl", "")


def _label(fname: str) -> str:
    return TRACE_LABEL.get(fname, _stem(fname))


def cell_limits(cap: str) -> HardLimits:
    if cap == "Cmech":   # 四类机制组能力（8192 token/16 GB，规格 §5.5；§10.3 预案夹具）
        return HardLimits(8, 8192, F(16))
    n_max = 1 if cap == "A" else TRACE_WIDE_LIMITS.n_max
    return HardLimits(n_max, TRACE_WIDE_LIMITS.token_max,
                      TRACE_WIDE_LIMITS.workspace_gb)


def policies_for(cell: dict, stage: str) -> list:
    if stage == "smoke":
        base = SMOKE_POLICIES
    else:
        base = ALL_POLICIES
    if cell["cap"] == "B":
        # 象限约束钩子仅简单策略支持（e21 先例）：B 块只比较简单策略
        return [p for p in base if p in SIMPLE_PIDS]
    return list(base)


def thetas_for(mode: str) -> list:
    return list(FW_THETAS) if mode == "FW" else list(OL_THETAS)


# ---------------------------------------------------------------------------
# 规格构造
# ---------------------------------------------------------------------------

def fw_specs(src: MooncakeSource, window: int, alpha, profile, n_req: int):
    """FW 有限工作集：窗口前 n 条、arrival=0、T0 用本线 profile 生成（规格
    §9.4：各 profile 自身生成 T0）、deadline=α·T0。"""
    blk = src.windows[window]
    rows = [r for r in src.imp.rows
            if blk.start_ms <= r.timestamp_ms < blk.end_ms][:n_req]
    specs = []
    for i, r in enumerate(rows):
        h, u, _full = src.imp.h_u_of(r)
        specs.append(RequestSpec(rid=i, arrival_s=0.0, h_tokens=h, u_tokens=u,
                                 class_id="mooncake", T0_s=1, deadline_s=1,
                                 source_file=src.fname,
                                 source_line=r.source_line,
                                 input_length=r.input_length))
    if not specs:
        return []
    T0 = make_T0(specs, profile, F(80), F(200))
    return [RequestSpec(s.rid, 0.0, s.h_tokens, s.u_tokens, s.class_id,
                        T0[s.rid], float(T0[s.rid]) * float(alpha),
                        source_file=s.source_file, source_line=s.source_line,
                        input_length=s.input_length)
            for s in specs]


def ol_specs(src: MooncakeSource, window: int, rho, alpha, duration_cap):
    lam = src.lam0 * F(str(rho))
    return src.window_specs(window, lam, F(int(alpha)),
                            duration_cap=duration_cap,
                            limits=TRACE_WIDE_LIMITS)


# ---------------------------------------------------------------------------
# 单格点执行（锚定 run + 策略循环 + ts 落盘 + 记录）
# ---------------------------------------------------------------------------

def _batch_stats(eng, scn, specs, med_h=None, med_u=None):
    """批统计：最大批人数、G>1 批数、象限违约数（仅 cap B 传入 med）。"""
    sizes, g_gt1, quad_viol = [], 0, 0
    by = {s.rid: s for s in specs}
    for b in eng.w.batches.values():
        sizes.append(len(b.members))
        sk = sum(float(batch_compute_s([by[r]], scn.profile)) * scn.profile.L
                 for r in b.members)
        bk = sum(float(c) for c in b.c_layers)
        if bk > 0 and sk / bk > 1.0 + 1e-9:
            g_gt1 += 1
        if med_h is not None:
            quads = {(by[r].h_tokens <= med_h, by[r].u_tokens <= med_u)
                     for r in b.members}
            if len(quads) > 1:
                quad_viol += 1
    return {"max_batch_size": max(sizes) if sizes else 0,
            "n_batches_G_gt1": g_gt1, "n_quadrant_viol": quad_viol}


def _record(eng, scn, cell, pid, theta, s, ts_rel, d_sim):
    s.update({"policy": pid, "theta": theta, "mode": cell["mode"],
              "factor_line": cell["factor_line"], "file": cell["file"],
              "window": cell["window"], "B": cell["B"],
              "rho": cell.get("rho"), "alpha": cell["alpha"],
              "cap": cell["cap"], "profile": cell["profile"],
              "cell_id": cell_id(cell), "ts_path": ts_rel,
              "T_anchor_s": cell["_anchor"], "d_sim": d_sim,
              "n_req": len(eng.w.specs)})
    if cell["mode"] == "OL":
        s["drain_M"] = s.get("makespan_M_lower")
    return s


def run_cell_specs(cell: dict, specs, d: str, records: list, progress: dict,
                   d_sim=None, min_reqs: int = 1):
    """核心执行器：specs 直接给出（run_cell 的底层，亦供 FM 夹具使用）。"""
    cid = cell_id(cell)
    if len(specs) < min_reqs:
        progress[cid] = {"skip": f"insufficient_data:{len(specs)}<{min_reqs}"}
        save_progress(d, progress)
        return []
    scn = default_scenario(B_gbps=cell["B"], alpha=F(int(cell["alpha"])),
                           limits=cell_limits(cell["cap"]),
                           profile=PROFILES[cell["profile"]])
    c_ref = c_ref_of(specs, scn)
    med_h = med_u = None
    if cell["cap"] == "B":
        med_h = float(np.median([s.h_tokens for s in specs]))
        med_u = float(np.median([s.u_tokens for s in specs]))
    extra = (quadrant_guard({s.rid: s for s in specs}, med_h, med_u)
             if med_h is not None else None)
    thetas = thetas_for(cell["mode"])
    ts_dir = os.path.join(d, "ts", cid)
    os.makedirs(ts_dir, exist_ok=True)
    thash = trace_hash(specs)[:16]

    def _one(pid, theta):
        # MPC/local 用"理想上限"配置（20260917 重跑）：CPU 预算放开，
        # 仅保留 2 万事件确定性硬界——衡量当前候选集/H=1 结构下的可达
        # 上限；实际开销经 ctrl_p50/p95_ms 与 n_scored/n_overrun 落盘。
        if pid in MPC_LIKE:
            from sim.cq.search import MPCPolicy
            pol = MPCPolicy(pid=pid, local=(pid == "cq_local"), H=1,
                            theta=theta, c_ref=c_ref,
                            budget_s=MPC_BUDGET_S)
        else:
            pol = build_policy(pid, theta, c_ref, H=1)
        if extra is not None and pid in SIMPLE_POLICIES:
            pol.feasibility_extra = extra
        eng = run_case(scn, specs, pol, numeric=float, seed=cell["window"],
                       record_intervals=True)
        s = summarize(eng, scn,
                      arrival_stop=(d_sim if cell["mode"] == "OL" else None))
        ctrl = sorted(d["ctrl_s"] for d in eng.decisions) or [0.0]
        s["ctrl_p50_ms"] = 1000.0 * ctrl[len(ctrl) // 2]
        s["ctrl_p95_ms"] = 1000.0 * ctrl[max(0, int(0.95 * len(ctrl)) - 1)]
        s["n_scored"] = getattr(pol, "n_scored", None)
        s["n_overrun"] = getattr(pol, "n_overrun", None)
        s["mpc_budget_s"] = (MPC_BUDGET_S if pid in MPC_LIKE else None)
        s["code_rev"] = "mpc-fixed-20260917"
        bstat = _batch_stats(eng, scn, specs, med_h, med_u)
        ts_rel, delta, ovf = None, None, None
        if specs:
            suffix = f"_{theta}" if pid in MPC_LIKE else ""
            cell["_anchor"] = cell.get("_anchor") or anchor_T(
                float(eng.w.t) if pid == "cq_fcfs" else 0.0)
            ts_rel = f"ts/{cid}/{pid}{suffix}.json.gz"
            try:
                ts = aggregate_run(eng, cell["_anchor"], cell=cid, policy=pid,
                                   theta=(theta if pid in MPC_LIKE else None),
                                   mode=cell["mode"])
                save_ts(os.path.join(d, ts_rel), ts)
                delta, ovf = ts["meta"]["delta_s"], ts["meta"]["overflow_used"]
            except TsConservationError as e:      # invalid run 单列，不静默
                s["invalid_ts"] = str(e)
                ts_rel = None
        rec = _record(eng, scn, cell, pid, theta, s, ts_rel, d_sim)
        rec.update(bstat)
        rec["delta_s"], rec["overflow_used"] = delta, ovf
        rec["trace_hash"] = thash
        rec["scenario"] = scn.scenario_name
        return rec

    # 锚定 run：FCFS 先跑，确定格点共用 T_anchor（简单策略跨 θ 复用）
    cell["_anchor"] = None
    out = []
    for pid in policies_for(cell, cell.get("_stage", "eval")):
        if pid in MPC_LIKE:
            for th in thetas:
                out.append(_one(pid, th))
        else:
            base = _one(pid, thetas[0])
            for th in thetas:
                r = dict(base) if th != thetas[0] else base
                r["theta"] = th
                out.append(r)
    records.extend(out)
    progress[cid] = {"n_records": len(out), "trace_hash": thash}
    save_progress(d, progress)
    save_json(os.path.join(d, "e25_records.json"), records)
    return out


def run_cell(src: MooncakeSource, cell: dict, d: str, records: list,
             progress: dict, n_req: int, duration_cap):
    """trace 侧入口：按模式构造 specs 后委托 run_cell_specs。"""
    if cell["mode"] == "FW":
        specs = fw_specs(src, cell["window"], cell["alpha"],
                         PROFILES[cell["profile"]], n_req)
        return run_cell_specs(cell, specs, d, records, progress,
                              min_reqs=MIN_REQS)
    specs, d_sim = ol_specs(src, cell["window"], cell["rho"], cell["alpha"],
                            duration_cap)
    return run_cell_specs(cell, specs, d, records, progress, d_sim=d_sim,
                          min_reqs=MIN_REQS)


def save_progress(d: str, progress: dict):
    save_json(os.path.join(d, "progress.json"), progress)


def load_existing(d: str):
    rp = os.path.join(d, "e25_records.json")
    pp = os.path.join(d, "progress.json")
    records = json.load(open(rp, encoding="utf-8")) if os.path.exists(rp) else []
    progress = json.load(open(pp, encoding="utf-8")) if os.path.exists(pp) else {}
    return records, progress


# ---------------------------------------------------------------------------
# 矩阵构造（§3.1 基线 + §3.2 六条线）
# ---------------------------------------------------------------------------

def build_cells(stage: str, windows: list) -> list:
    files = (["conversation_trace.jsonl", "synthetic_trace.jsonl"]
             if stage == "smoke"
             else [f for f, _n, _m, _s in MOONCAKE_FILES])
    base = {"B": 80.0, "alpha": 4, "cap": "C", "profile": "default"}
    cells = []
    for f in files:
        for w in windows:
            cells.append({"factor_line": "baseline", "mode": "FW", "file": f,
                          "rho": None, "window": w, **base})
            cells.append({"factor_line": "baseline", "mode": "OL", "file": f,
                          "rho": 0.6, "window": w, **base})
    if stage == "smoke":
        return cells
    for f in files:
        for w in windows:
            for lvl in (20.0, 320.0):
                for mode in ("FW", "OL"):
                    rho = 0.6 if mode == "OL" else None
                    cells.append({"factor_line": "line_B", "mode": mode,
                                  "file": f, "B": lvl, "rho": rho,
                                  "alpha": 4, "cap": "C",
                                  "profile": "default", "window": w})
            for lvl in (2, 8):
                for mode in ("FW", "OL"):
                    rho = 0.6 if mode == "OL" else None
                    cells.append({"factor_line": "line_alpha", "mode": mode,
                                  "file": f, "B": 80.0, "rho": rho,
                                  "alpha": lvl, "cap": "C",
                                  "profile": "default", "window": w})
            for lvl in (0.3, 0.9, 1.1):
                cells.append({"factor_line": "line_rho", "mode": "OL",
                              "file": f, "B": 80.0, "rho": lvl, "alpha": 4,
                              "cap": "C", "profile": "default", "window": w})
            for lvl in ("A", "B"):
                cells.append({"factor_line": "line_cap", "mode": "FW",
                              "file": f, "B": 80.0, "rho": None, "alpha": 4,
                              "cap": lvl, "profile": "default", "window": w})
            for lvl in ("no_speedup", "n_sat128"):
                cells.append({"factor_line": "line_profile", "mode": "FW",
                              "file": f, "B": 80.0, "rho": None, "alpha": 4,
                              "cap": "C", "profile": lvl, "window": w})
    return cells


# ---------------------------------------------------------------------------
# 图（§6 A–F）；数据源唯一：records 与 ts 文件
# ---------------------------------------------------------------------------

_METRIC_FW = [("M", "makespan_M", "M 全部完成 (s)"),
              ("SLO率", "slo_rate", "SLO 满足率"),
              ("T", "ttft_mean_lower", "平均 TTFT (s)"),
              ("R", "ttft_norm_mean_lower", "平均归一化 TTFT")]


def _slo_rate(r):
    return r["slo_success"] / r["n_cohort"] if r.get("n_cohort") else None


# MPC/local 不同 θ 是不同物理轨迹：按指标取对应 θ 的记录，禁止跨 θ 混池
_THETA_OF_METRIC = {"M": "M", "SLO率": "S", "S": "S", "T": "T", "R": "R",
                    "drainM": "S"}


def _theta_match(r, mkey):
    if r["policy"] in MPC_LIKE:
        return r["theta"] == _THETA_OF_METRIC[mkey]
    return True


def _metric_get(r, key):
    if key == "slo_rate":
        return _slo_rate(r)
    v = r.get(key)
    if key.startswith("makespan") and v is None:
        v = r.get("makespan_M_lower")
    return v


def fig_objectives(records, d: str, plt):
    """图 A：FW 基线四目标绝对值（2×2；x=策略；点=窗口值；线=中位数）。"""
    sub = [r for r in records if r["factor_line"] == "baseline"
           and r["mode"] == "FW"]
    pols = ALL_POLICIES
    files = sorted({r["file"] for r in sub})
    marks = {"conversation_trace.jsonl": "o", "toolagent_trace.jsonl": "s",
             "synthetic_trace.jsonl": "^"}
    cache = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (mkey, key, ylab) in zip(axes.flat, _METRIC_FW):
        vals = {p: [] for p in pols}
        for r in sub:
            v = _metric_get(r, key)
            if r["policy"] in vals and v is not None and _theta_match(r, mkey):
                vals[r["policy"]].append(v)
        for pi, p in enumerate(pols):
            for r in sub:
                if r["policy"] == p and _theta_match(r, mkey):
                    v = _metric_get(r, key)
                    if v is not None:
                        ax.scatter(pi, v, marker=marks.get(r["file"], "o"),
                                   s=14, alpha=.45,
                                   color={"conversation_trace.jsonl": "#4C72B0",
                                          "toolagent_trace.jsonl": "#DD8452",
                                          "synthetic_trace.jsonl": "#55A868"
                                          }.get(r["file"], "gray"))
            med = float(np.median(vals[p])) if vals[p] else None
            if med is not None:
                ax.hlines(med, pi - .3, pi + .3, color="k", lw=2)
            cache.setdefault(p, {})[key] = med
        ax.set_xticks(range(len(pols)))
        ax.set_xticklabels([p.replace("cq_", "") for p in pols], rotation=30)
        ax.set_ylabel(ylab)
        ax.set_title(ylab)
    fig.suptitle("图A｜E25 四目标绝对值对比（FW 有限工作集基线：B=80, α=4, "
                 "能力C, synthetic-v1；点=窗口×文件，粗线=中位数；M 未完成时为下界）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e25_objectives.png"), dpi=130)
    plt.close(fig)
    save_json(os.path.join(d, "fig_e25_objectives_data.json"),
              {"median": cache,
               "note": "粗横线=各策略全部窗口×文件的中位数（FG01 同源缓存）"})


_LINE_LEVELS = {
    "line_B": ([20.0, 80.0, 320.0], "B", lambda c: c["B"], "B (GB/s)"),
    "line_alpha": ([2, 4, 8], "α", lambda c: c["alpha"], "SLO 倍数 α"),
    "line_rho": ([0.3, 0.6, 0.9, 1.1], "ρ", lambda c: c["rho"], "负载 ρ"),
    "line_cap": (["A", "B", "C"], "能力块", lambda c: c["cap"], "批能力块"),
    "line_profile": (["no_speedup", "default", "n_sat128"], "profile",
                     lambda c: c["profile"], "计算画像"),
}


def _line_modes(line):
    return ("FW", "OL") if line in ("line_B", "line_alpha") else (
        ("OL",) if line == "line_rho" else ("FW",))


def fig_factor(line: str, records, d: str, plt):
    """图 B（每线一张）：行=模式×指标，列=三文件；线=策略中位数，带=min–max。"""
    levels, _lk, get_lvl, xlab = _LINE_LEVELS[line]
    modes = _line_modes(line)
    files = [f for f, _n, _m, _s in MOONCAKE_FILES]
    metrics = {"FW": _METRIC_FW,
               "OL": [("SLO率", "slo_rate", "SLO 满足率"),
                      ("T", "ttft_mean_lower", "平均 TTFT (s)"),
                      ("R", "ttft_norm_mean_lower", "平均归一化 TTFT"),
                      ("drainM", "drain_M", "排空 M 下界 (s)")]}
    rows = [(m, *mt) for m in modes for mt in metrics[m]]
    sub_all = [r for r in records if r["factor_line"] in ("baseline", line)
               and r["mode"] in modes]
    pols = sorted({r["policy"] for r in sub_all})
    fig, axes = plt.subplots(len(rows), len(files),
                             figsize=(5.2 * len(files), 2.6 * len(rows)),
                             squeeze=False)
    for ci, f in enumerate(files):
        for ri, (mode, mkey, key, ylab) in enumerate(rows):
            ax = axes[ri][ci]
            sub = [r for r in sub_all if r["file"] == f and r["mode"] == mode]
            for p in pols:
                xs, med, lo, hi = [], [], [], []
                for lv in levels:
                    vs = [_metric_get(r, key) for r in sub
                          if r["policy"] == p and get_lvl(r) == lv
                          and _theta_match(r, mkey)
                          and _metric_get(r, key) is not None]
                    if vs:
                        xs.append(levels.index(lv))
                        med.append(float(np.median(vs)))
                        lo.append(min(vs))
                        hi.append(max(vs))
                if xs:
                    ax.plot(xs, med, marker="o", ms=3, lw=1.2, label=p.replace("cq_", ""))
                    ax.fill_between(xs, lo, hi, alpha=.12)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels([str(l) for l in levels], fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{mode}·{ylab}", fontsize=9)
            if ri == 0:
                ax.set_title(_label(f), fontsize=10)
            if ri == len(rows) - 1:
                ax.set_xlabel(xlab)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(8, len(labels)),
                   fontsize=8)
    fig.suptitle(f"图B｜E25 因子响应：{line}（每线只动一个因子，其余=基线；"
                 "线=窗口中位数，带=窗口 min–max；平线如实保留）")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(d, f"fig_e25_factor_{line.split('_', 1)[1]}.png"),
                dpi=130)
    plt.close(fig)


def fig_pareto(records, d: str, plt):
    """图 C：FW 基线 Pareto 散点（M–SLO、T–R）；取 θ=M 子集保证单轨迹配对。"""
    sub = [r for r in records if r["factor_line"] == "baseline"
           and r["mode"] == "FW" and r["theta"] == "M"]
    files = sorted({r["file"] for r in sub})
    marks = {"conversation_trace.jsonl": "o", "toolagent_trace.jsonl": "s",
             "synthetic_trace.jsonl": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for r in sub:
        m = marks.get(r["file"], "o")
        x1, y1 = _metric_get(r, "makespan_M"), _slo_rate(r)
        x2, y2 = _metric_get(r, "ttft_mean_lower"), _metric_get(r, "ttft_norm_mean_lower")
        c = {"cq_fcfs": "#000000", "cq_spt": "#4C72B0", "cq_edf": "#DD8452",
             "cq_slack": "#55A868", "cq_lpm": "#C44E52", "cq_pair": "#8172B3",
             "cq_local": "#937860", "cq_mpc": "#DA8BC3"}.get(r["policy"], "gray")
        if x1 is not None and y1 is not None:
            axes[0].scatter(x1, y1, marker=m, s=14, alpha=.55, color=c)
        if x2 is not None and y2 is not None:
            axes[1].scatter(x2, y2, marker=m, s=14, alpha=.55, color=c)
    axes[0].set_xlabel("M 全部完成 (s)"); axes[0].set_ylabel("SLO 满足率")
    axes[0].set_title("M–SLO 平面（左上=优）")
    axes[1].set_xlabel("平均 TTFT (s)"); axes[1].set_ylabel("平均归一化 TTFT")
    axes[1].set_title("T–R 平面")
    for ax in axes:
        ax.grid(alpha=.25)
    fh = [plt.Line2D([], [], marker=marks.get(f, "o"), ls="", color="gray",
                     label=_label(f)) for f in files]
    ph = [plt.Line2D([], [], marker="o", ls="",
                     color={"cq_fcfs": "#000000", "cq_spt": "#4C72B0",
                            "cq_edf": "#DD8452", "cq_slack": "#55A868",
                            "cq_lpm": "#C44E52", "cq_pair": "#8172B3",
                            "cq_local": "#937860", "cq_mpc": "#DA8BC3"}[p],
                     label=p.replace("cq_", "")) for p in ALL_POLICIES]
    axes[0].legend(handles=fh, fontsize=8, loc="best")
    axes[1].legend(handles=ph, fontsize=7, loc="best")
    fig.suptitle("图C｜E25 目标冲突散点（FW 基线；点=策略×文件×窗口；"
                 "左图色=策略/形=文件，右图同）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e25_pareto.png"), dpi=130)
    plt.close(fig)


def _pick_ts(records, d, mode, fname, B, policy, theta=None, factor="baseline",
             window=None):
    """按 (mode,file,B,policy[,theta]) 找记录并载入其 ts 文件。"""
    cand = [r for r in records if r["factor_line"] == factor
            and r["mode"] == mode and r["file"] == fname and r["B"] == B
            and r["policy"] == policy and r.get("ts_path")]
    if theta is not None:
        cand = [r for r in cand if r["theta"] == theta]
    else:
        cand = [r for r in cand if r["policy"] not in MPC_LIKE]
    if window is not None:
        cand = [r for r in cand if r["window"] == window] or cand
    if not cand:
        return None
    p = os.path.join(d, cand[0]["ts_path"])
    return load_ts(p) if os.path.exists(p) else None


def fig_timeseries(kind: str, records, d: str, plt):
    """图 D/E/F：行=策略×{B80,B20}，列=三文件（OL 基线窗口 9，回退首个有效）。

    kind ∈ {device, storage, queue}；空尾桶（width<=0）留白不画 0（FG02）。
    """
    files = [f for f, _n, _m, _s in MOONCAKE_FILES]
    rows = [(p, b) for b in (80.0, 20.0) for p in TS_POLICIES]
    fig, axes = plt.subplots(len(rows), len(files),
                             figsize=(5.0 * len(files), 2.1 * len(rows)),
                             squeeze=False)
    for ci, f in enumerate(files):
        win = TS_WINDOW_DEFAULT
        if not any(r["file"] == f and r["mode"] == "OL" and r["window"] == win
                   and r["ts_path"] for r in records):
            cands = sorted({r["window"] for r in records
                            if r["file"] == f and r["mode"] == "OL"
                            and r["ts_path"]})
            win = cands[0] if cands else None
        for ri, (p, b) in enumerate(rows):
            ax = axes[ri][ci]
            factor = "baseline" if b == 80.0 else "line_B"
            theta = "S" if p in MPC_LIKE else None
            ts = _pick_ts(records, d, "OL", f, b, p, theta=theta,
                          factor=factor, window=win)
            if ts is None or not ts["rows"]:
                ax.text(.5, .5, "无数据", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            meta, rows_ts = ts["meta"], ts["rows"]
            xs = [r["t_start_s"] for r in rows_ts]
            if kind == "device":
                fr = [row_fractions(r, meta["m_workers"]) for r in rows_ts]
                xs_f, comp, stall, idle = [], [], [], []
                for x, f2 in zip(xs, fr):
                    if f2 is None:
                        continue
                    xs_f.append(x)
                    comp.append(f2[0]); stall.append(f2[1]); idle.append(f2[2])
                ax.stackplot(xs_f, comp, stall, idle,
                             labels=["Compute", "STALL", "Idle"],
                             colors=["#4C72B0", "#DD8452", "#B0B0B0"],
                             alpha=.9)
                ax.set_ylim(0, 1)
                ax.set_ylabel(f"{p.replace('cq_', '')}·B{b:g}\n占用占比",
                              fontsize=8)
                if ri == 0 and ci == 0:
                    ax.legend(loc="upper right", fontsize=7)
            elif kind == "storage":
                xs_v, util, rate = [], [], []
                for r in rows_ts:
                    if r["width_s"] <= 0 or r["capacity_gb"] <= 0:
                        continue
                    xs_v.append(r["t_start_s"])
                    util.append(r["served_gb"] / r["capacity_gb"])
                    rate.append(r["served_gb"] / r["width_s"])
                ax.plot(xs_v, util, lw=1.1, color="#4C72B0",
                        label="利用率 served/∫B")
                ax.set_ylim(0, 1.02)
                ax.set_ylabel(f"{p.replace('cq_', '')}·B{b:g}\n利用率", fontsize=8)
                ax2 = ax.twinx()
                sched = meta["b_schedule"]
                st = [t for t, _v in sched]
                sv = [v for _t, v in sched]
                step_x = [x for x in st if x <= max(xs_v, default=0)] + \
                         [max(xs_v, default=0)]
                step_y = [sv[min(i, len(sv) - 1)]
                          for i in range(len(step_x))]
                ax2.step(step_x, step_y, where="post", color="k", ls="--",
                         lw=.8, label="B(t)")
                ax2.set_ylabel("GB/s", fontsize=8)
                if ri == 0 and ci == 0:
                    ax.legend(loc="upper left", fontsize=7)
            else:  # queue
                xs_q, qm, qmax = [], [], []
                for r in rows_ts:
                    if r["width_s"] <= 0 or r["queue_mean"] is None:
                        continue
                    xs_q.append(r["t_start_s"])
                    qm.append(r["queue_mean"])
                    qmax.append(r["queue_max"])
                ax.plot(xs_q, qm, lw=1.1, color="#4C72B0", label="均值")
                ax.fill_between(xs_q, qm, qmax, alpha=.18, color="#4C72B0",
                                label="峰值")
                ax.set_ylabel(f"{p.replace('cq_', '')}·B{b:g}\nQ+ACTIVE",
                              fontsize=8)
                if ri == 0 and ci == 0:
                    ax.legend(loc="upper right", fontsize=7)
            if ri == len(rows) - 1:
                ax.set_xlabel("仿真时间 (s)")
            if ci == 0:
                pass
            if ri == 0:
                ax.set_title(f"{_label(f)} w{win}", fontsize=10)
    names = {"device": "ts_device", "storage": "ts_storage",
             "queue": "ts_queue"}
    titles = {"device": "图D｜NPU 三状态时间线",
              "storage": "图E｜存储带宽利用率时间线",
              "queue": "图F｜队列 Q+ACTIVE 时间线"}
    fig.suptitle(titles[kind] +
                 "（OL 在线流基线 B=80 与 line_B B=20；窗口默认 9；"
                 "横轴为缩放后仿真时间；空尾=策略快于锚，留白）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"fig_e25_{names[kind]}.png"), dpi=130)
    plt.close(fig)


def publish_figures(d: str, docs_dir: str):
    """仅新增 cq_fig_e25_*，不触碰任何旧图（FG05）。

    图 A/C（objectives/pareto）的正式版为箱线/条形设计，由
    tools/e25_redraw.py 生成——此处跳过，防止本文件的旧散点版覆盖正式图。
    """
    os.makedirs(docs_dir, exist_ok=True)
    n = 0
    skip = {"fig_e25_objectives.png", "fig_e25_pareto.png"}
    for fn in sorted(os.listdir(d)):
        if fn.startswith("fig_e25_") and fn.endswith(".png") and fn not in skip:
            shutil.copy2(os.path.join(d, fn),
                         os.path.join(docs_dir, f"cq_{fn}"))
            n += 1
    return n


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(seeds, procs=None, duration=150.0, stage="smoke",
         trace_dir=TRACE_DIR_DEFAULT, manifest=None, seed_start=0,
         out_dir_root=None, **kw):
    plt = setup_matplotlib()
    d = out_dir(stage, "e25")
    windows = windows_for_seeds(len(seeds))
    records, progress = load_existing(d)
    cells = build_cells(stage, windows)
    n_req = 32 if stage == "smoke" else 128
    ol_cap = float(duration) if stage == "smoke" else None
    srcs = {}
    for cell in cells:
        if cell_id(cell) in progress:
            continue   # 续跑：已完成格点跳过（progress 含 trace_hash）
        fname = cell["file"]
        if fname not in srcs:
            srcs[fname] = MooncakeSource(fname, trace_dir)
        cell["_stage"] = stage
        out = run_cell(srcs[fname], cell, d, records, progress, n_req, ol_cap)
        print_progress(f"E25 {cell_id(cell)} -> {len(out)} 条记录")
    # 图 A–F（数据源唯一：records 与 ts 文件）
    fig_objectives(records, d, plt)
    fig_pareto(records, d, plt)
    for line in ("line_B", "line_alpha", "line_rho", "line_cap", "line_profile"):
        fig_factor(line, records, d, plt)
    for kind in ("device", "storage", "queue"):
        fig_timeseries(kind, records, d, plt)
    if stage == "eval":
        docs = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "docs", "figures")
        n = publish_figures(d, docs)
        print_progress(f"E25 eval 图已发布 {n} 张 -> {docs}")
    n_runs = sum(v.get("n_records", 0) for v in progress.values()
                 if isinstance(v, dict))
    print_progress(f"E25 done -> {d} ({len(records)} 记录 / {n_runs} run)")
    return {"n_records": len(records), "n_cells": len(progress)}


if __name__ == "__main__":
    main([0])
