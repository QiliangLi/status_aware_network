"""E25.2 MPC 收益的工况地图（20260920，验证"间歇负载才有收益"假设）。

假设（用户提出）：MPC 的收益集中在"存储间歇过载/欠载"的工况——
一直欠载（到达即服务、没队可排）或一直过载（带宽长期打满、队列单调
积压）时，调度没有操作空间，收益应当很小。

三步：
①补跑缺失格点（OL、α=4、5 窗 {0,4,9,14,19}、策略 {fcfs, edf,
  mpc(θ=S,T)}、MPC 理想上限档 budget_s=3600 与主矩阵一致）：
  - B=20 × ρ∈{0.3, 0.9, 1.1}：B×ρ 交互（主线只扫过 B×ρ0.6 与 B80×全 ρ）；
  - m∈{2, 8} × B=80 × ρ=0.6：NPU 数量（λ0 锚按 m 重算 min(80/E[V], m/E[K])，
    同 ρ 下 m 越大到达率越高、存储需求越大——工况随 m 漂移，正是要测的）；
②实测负载状态（不以 ρ 名义值分类）：在 FCFS 运行的 w9 时间序列上算
  comp=NPU 计算占比、sat=带宽利用率≥90% 的桶占比、shallow=排空时段
  (队列≤2)占比、deep=深积压时段(队列≥20)占比、q_med/q_peak；
  分类阈值（在全部格点上一次设定、报告如实披露）：
    持续欠载：shallow ≥ 0.5 且 deep < 0.1（大半时间到达即服务）
    持续过载：deep ≥ 0.5 或 comp ≥ 0.80（大半时间深积压/NPU 满转）
    间歇：其余（既有排空或浅队时段、又有积压时段）
③地图：既有 records（B/ρ 因子线）+ 新跑 records → 每格点 MPC vs EDF
  的逐窗配对收益（θ=T 的 TTFT%、θ=S 的 SLO pp）→ 总表 + 图K + case 图。

用法：
  python tools/e25_regime.py run       # 补跑（progress 落盘，可续跑）
  python tools/e25_regime.py analyze   # 地图总表 -> stdout + map.json
  python tools/e25_regime.py figs      # 图K + case Gantt（重用图 I 绘制代码）
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.cq.config import ProfileConfig
from sim.cq.metrics import summarize
from sim.cq.profile import singleton_K
from sim.cq.simrun import run_case
from sim.cq.timeseries import aggregate_run, anchor_T, save_ts
from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL
from sim.cq.types import RequestSpec
from sim.experiments.cq_common import (TRACE_DIR_DEFAULT, TRACE_WIDE_LIMITS,
                                       MooncakeSource, build_policy,
                                       c_ref_of, default_scenario)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "results", "cq", "eval", "e25_regime")
E25 = os.path.join(REPO, "results", "cq", "eval", "e25")
FIG = os.path.join(REPO, "docs", "figures")
W5 = [0, 4, 9, 14, 19]
MPC_BUDGET_S = 3600.0          # 与主矩阵理想上限档一致
POLICY_THETAS = [("cq_fcfs", ""), ("cq_edf", ""), ("cq_mpc", "S"),
                 ("cq_mpc", "T")]

# 新格点：B×ρ 交互 + NPU 数量（OL、α=4）
NEW_CELLS = []
for _f, _n, _m, _s in MOONCAKE_FILES:
    for _r in (0.3, 0.9, 1.1):
        NEW_CELLS.append(dict(file=_f, B=20.0, rho=_r, m=4))
    for _m in (2, 8):
        NEW_CELLS.append(dict(file=_f, B=80.0, rho=0.6, m=_m))

TCOL = {"conversation_trace.jsonl": "#4C72B0",
        "toolagent_trace.jsonl": "#DD8452",
        "synthetic_trace.jsonl": "#55A868"}
MMARK = {2: "s", 4: "o", 8: "^"}


def cid_new(c, w):
    stem = c["file"].replace("_trace.jsonl", "").replace(".jsonl", "")
    return f"{stem}|OL|B{c['B']:g}|rho{c['rho']:g}|a4|m{c['m']}|w{w}"


def cid_e25(file, B, rho, w=9):
    stem = file.replace("_trace.jsonl", "").replace(".jsonl", "")
    r = "-" if rho is None else f"{rho:g}"
    return f"{stem}|OL|B{B:g}|rho{r}|a4|C|default|w{w}"


def lam0_m(src: MooncakeSource, m: int) -> F:
    """按 NPU 数量重算容量锚：λ0(m) = min(80/E[V_layer], m/E[K_singleton])。

    与 cq_common.MooncakeSource.lam0 同公式（那里 m 固定为 4）。
    """
    prof = ProfileConfig()
    hus = [(r.hit_tokens, r.u_tokens) for r in src.imp.rows]
    e_v = sum(F(prof.kappa_gb_per_token_layer) * h for h, _u in hus) / len(hus)
    ks = []
    for h, u in hus:
        rs = RequestSpec(0, 0, h, u, "x", 1, 1)
        ks.append(F(singleton_K(rs, prof, F(80), F(200))))
    e_k = sum(ks) / len(ks)
    return min(F(80) / e_v, F(m) / e_k)


def load_progress():
    p = os.path.join(ROOT, "progress.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def save_progress(prog):
    os.makedirs(ROOT, exist_ok=True)
    json.dump(prog, open(os.path.join(ROOT, "progress.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)


def run_all():
    from sim.cq.search import MPCPolicy
    prog = load_progress()
    recs_path = os.path.join(ROOT, "e25_regime_records.json")
    recs = (json.load(open(recs_path, encoding="utf-8"))
            if os.path.exists(recs_path) else [])
    srcs = {}
    for c in NEW_CELLS:
        f = c["file"]
        if f not in srcs:
            srcs[f] = MooncakeSource(f, TRACE_DIR_DEFAULT)
        src = srcs[f]
        for w in W5:
            cid = cid_new(c, w)
            if cid in prog:
                continue
            lam = lam0_m(src, c["m"]) * F(str(c["rho"]))
            specs, d_sim = src.window_specs(w, lam, F(4), duration_cap=None,
                                            limits=TRACE_WIDE_LIMITS)
            if len(specs) < 16:
                prog[cid] = {"skip": f"insufficient:{len(specs)}"}
                save_progress(prog)
                continue
            scn = default_scenario(B_gbps=c["B"], m=c["m"], alpha=F(4),
                                   limits=TRACE_WIDE_LIMITS)
            c_ref = c_ref_of(specs, scn)
            for pid, th in POLICY_THETAS:
                if pid == "cq_mpc":
                    pol = MPCPolicy(pid="cq_mpc", H=1, theta=th,
                                    c_ref=c_ref, budget_s=MPC_BUDGET_S)
                else:
                    pol = build_policy(pid, th, c_ref, H=1)
                eng = run_case(scn, specs, pol, numeric=float, seed=w,
                               record_intervals=True)
                s = summarize(eng, scn, arrival_stop=d_sim)
                ctrl = sorted(d["ctrl_s"] for d in eng.decisions) or [0.0]
                s["ctrl_p50_ms"] = 1000.0 * ctrl[len(ctrl) // 2]
                s.update({"policy": pid, "theta": th, "file": f,
                          "B": c["B"], "rho": c["rho"], "m": c["m"],
                          "window": w, "n_req": len(specs), "d_sim": d_sim,
                          "drain_M": s.get("makespan_M_lower"),
                          "code_rev": "e252-regime-20260920"})
                recs.append(s)
                if w == 9:                      # 时间序列只存 w9（图+分类用）
                    try:
                        ts = aggregate_run(eng, anchor_T(float(eng.w.t)),
                                           cell=cid, policy=pid,
                                           theta=(th or None), mode="OL")
                        ts_p = os.path.join(
                            ROOT, "ts", cid,
                            f"{pid}{('_' + th) if pid == 'cq_mpc' else ''}.json.gz")
                        os.makedirs(os.path.dirname(ts_p), exist_ok=True)
                        save_ts(ts_p, ts)
                    except Exception as e:      # TsConservationError 等
                        print(f"  ! ts 保存失败 {cid}/{pid}{th}: {e}")
            prog[cid] = {"n_req": len(specs), "d_sim": d_sim}
            save_progress(prog)
            json.dump(recs, open(recs_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            print(f"done {cid} ({len(specs)} reqs, d_sim={d_sim:.0f}s)")
    print(f"E25.2 run 完成：{len(recs)} 条新记录 -> {recs_path}")


# ---------------------------------------------------------------- 分析
def load_ts(root, cid, pid, suffix=""):
    p = os.path.join(root, "ts", cid, f"{pid}{suffix}.json.gz")
    if not os.path.exists(p):
        return None
    return json.load(gzip.open(p, "rt", encoding="utf-8"))


def regime_metrics(ts):
    """负载状态量（在 FCFS 运行上测）：
    comp=NPU 计算占比；sat=带宽饱和(≥90%)桶占比；shallow=排空时段
    (队列≤2)占比；deep=深积压时段(队列≥20)占比；q_med/q_peak=队列。
    """
    rows = [r for r in ts["rows"] if r["width_s"] > 0]
    b = float(ts["meta"]["b_schedule"][0][1])
    m = int(ts["meta"].get("m_workers", 4))
    utils = [r["served_gb"] / r["width_s"] / b for r in rows]
    q = [r["queue_mean"] for r in rows if r["queue_mean"] is not None]
    comp = (sum(r["compute_s"] for r in rows)
            / sum(m * r["width_s"] for r in rows)) if rows else 0.0
    return dict(comp=float(comp),
                sat=float(np.mean([u >= 0.9 for u in utils])) if utils else 0.0,
                bw_peak=float(max(utils)) if utils else 0.0,
                shallow=float(np.mean([x <= 2 for x in q])) if q else 0.0,
                deep=float(np.mean([x >= 20 for x in q])) if q else 0.0,
                q_med=float(np.median(q)) if q else 0.0,
                q_peak=float(max(q)) if q else 0.0)


def classify(mt):
    """三态分类（阈值在全部格点上一次设定，报告披露）：
    持续欠载=大半时间排空；持续过载=大半时间深积压或 NPU 长期满转；
    间歇=既有排空/浅队时段又有积压时段。"""
    if mt["shallow"] >= 0.5 and mt["deep"] < 0.1:
        return "持续欠载"
    if mt["deep"] >= 0.5 or mt["comp"] >= 0.80:
        return "持续过载"
    return "间歇"


def gains_from_records(recs, file, B, rho, m):
    """逐窗配对：mpc(θ=T) vs edf 的 TTFT%、mpc(θ=S) vs edf 的 SLO pp。

    旧记录（主矩阵）里简单策略也带 theta 字段（同值重复），统一把
    非 MPC 策略的 theta 归一化为空串后再做键。
    """
    sub = [r for r in recs if r["file"] == file and r.get("B") == B
           and r.get("rho") == rho and r.get("m", 4) == m]

    def _key(r):
        if r["policy"] == "cq_mpc":
            return (r["policy"], r.get("theta", ""), r["window"])
        return (r["policy"], "", r["window"])

    by = {_key(r): r for r in sub}
    t_gains, s_gain = [], []
    for w in W5:
        edf = by.get(("cq_edf", "", w))
        mT = by.get(("cq_mpc", "T", w))
        mS = by.get(("cq_mpc", "S", w))
        if edf and mT and edf["ttft_mean_lower"]:
            t_gains.append((edf["ttft_mean_lower"] - mT["ttft_mean_lower"])
                           / edf["ttft_mean_lower"] * 100.0)
        if edf and mS:
            s_gain.append((mS["slo_success"] / mS["n_cohort"]
                           - edf["slo_success"] / edf["n_cohort"]) * 100.0)
    return (float(np.median(t_gains)) if t_gains else None,
            float(np.median(s_gain)) if s_gain else None)


def analyze():
    old = json.load(open(os.path.join(E25, "e25_records.json"),
                         encoding="utf-8"))
    old = [r for r in old if r["mode"] == "OL" and r["alpha"] == 4
           and r["cap"] == "C" and r["profile"] == "default"
           and r["policy"] in ("cq_fcfs", "cq_edf", "cq_mpc")
           and r["theta"] in ("S", "T") and r["window"] in W5]
    new_p = os.path.join(ROOT, "e25_regime_records.json")
    new = (json.load(open(new_p, encoding="utf-8"))
           if os.path.exists(new_p) else [])
    recs = old + new

    # 格点清单：既有（B∈{20,80,320}×ρ0.6、B80×ρ{0.3,0.9,1.1}，m=4）+ 新
    keys = set()
    for f, _n, _m, _s in MOONCAKE_FILES:
        for B in (20.0, 80.0, 320.0):
            keys.add((f, B, 0.6, 4))
        for r in (0.3, 0.9, 1.1):
            keys.add((f, 80.0, r, 4))
        for r in (0.3, 0.9, 1.1):
            keys.add((f, 20.0, r, 4))
        for m in (2, 8):
            keys.add((f, 80.0, 0.6, m))

    rows = []
    for (f, B, rho, m) in sorted(keys):
        mt = None
        if m == 4 and B in (20.0, 80.0, 320.0) and not (B == 20.0 and rho != 0.6):
            ts = load_ts(E25, cid_e25(f, B, rho), "cq_fcfs")
        else:
            ts = load_ts(ROOT, cid_new(dict(file=f, B=B, rho=rho, m=m), 9),
                         "cq_fcfs")
        if ts is not None:
            mt = regime_metrics(ts)
        tg, sg = gains_from_records(recs, f, B, rho, m)
        rows.append(dict(file=f, B=B, rho=rho, m=m, metrics=mt,
                         ttft_gain=tg, slo_gain=sg,
                         regime=(classify(mt) if mt else "?")))
    json.dump(rows, open(os.path.join(ROOT, "map.json"), "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{'trace':12s} {'B':>4s} {'ρ':>4s} {'m':>2s} | {'算':>4s} {'sat':>4s} "
          f"{'排空':>4s} {'深积':>4s} {'q中':>5s} {'q峰':>4s} | {'状态':6s} "
          f"{'TTFT增益':>8s} {'SLO增益':>8s}")
    for r in rows:
        mt = r["metrics"] or {}
        print(f"{r['file'].split('_')[0]:12s} {r['B']:>4.0f} {r['rho']:>4g} "
              f"{r['m']:>2d} | {mt.get('comp', float('nan')):>4.0%} "
              f"{mt.get('sat', float('nan')):>4.0%} "
              f"{mt.get('shallow', float('nan')):>4.0%} "
              f"{mt.get('deep', float('nan')):>4.0%} "
              f"{mt.get('q_med', float('nan')):>5.1f} "
              f"{mt.get('q_peak', float('nan')):>4.0f} | {r['regime']:6s} "
              f"{('%+.1f%%' % r['ttft_gain']) if r['ttft_gain'] is not None else '—':>8s} "
              f"{('%+.1fpp' % r['slo_gain']) if r['slo_gain'] is not None else '—':>8s}")
    print("地图已保存 -> results/cq/eval/e25_regime/map.json")


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# ---------------------------------------------------------------- 图
def figs():
    plt = plt_setup()
    from e25_ts_compare import audit_text_overlap, draw_bandwidth, draw_gantt, row_stats
    rows = json.load(open(os.path.join(ROOT, "map.json"), encoding="utf-8"))
    ok = [r for r in rows if r["metrics"] and r["ttft_gain"] is not None]

    # ---- 图K：收益 vs 实测负载状态（三个状态量各一面板）----
    RMARK = {"持续欠载": "s", "间歇": "o", "持续过载": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=125)
    panels = [("sat", "带宽饱和占比 sat（利用率≥90% 的时间，%）", 100.0),
              ("shallow", "排空时段占比（队列≤2 的时间，%）", 100.0),
              ("deep", "深积压时段占比（队列≥20 的时间，%）", 100.0)]
    for ax, (key, xlab, scale) in zip(axes, panels):
        for r in ok:
            ax.scatter(r["metrics"][key] * scale, r["ttft_gain"],
                       color=TCOL[r["file"]], marker=RMARK[r["regime"]],
                       s=95, edgecolor="k", lw=.6, zorder=3)
        ax.axhline(0, color="k", lw=.8)
        ax.set_xlabel(xlab)
        ax.set_ylabel("MPC vs EDF：平均 TTFT 配对增益 (%)")
        ax.grid(alpha=.25)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, marker="o", ls="", label=l.split("_")[0])
               for l, c in TCOL.items()] + \
              [Line2D([], [], color="k", marker=m, ls="", label=k)
               for k, m in RMARK.items()]
    axes[0].legend(handles=handles, fontsize=9, loc="upper left")
    fig.suptitle("图K｜MPC 收益的工况地图（OL、θ=T 逐窗配对、5 窗中位；状态量在 FCFS 运行 w9 上实测；"
                 "B∈{20,80,320}×ρ∈{0.3..1.1}×m∈{2,4,8}，33 格点）")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(FIG, "cq_fig_e25_regime_map.png"), dpi=125)
    ov = audit_text_overlap(fig)
    print(f"图K -> cq_fig_e25_regime_map.png  [文字重叠审计: "
          f"{'OK' if not ov else ov[:3]}]")
    plt.close(fig)

    # ---- case 图（Gantt，重用图 I 的绘制代码）----
    cases = [
        (dict(file="toolagent_trace.jsonl", B=20.0, rho=1.1, m=4),
         "OL 持续过载（B=20, ρ=1.1, m=4）", "cq_fig_e25_cmp_OLB20r11_toolagent"),
        (dict(file="toolagent_trace.jsonl", B=80.0, rho=0.6, m=8),
         "OL NPU 数量 m=8（B=80, ρ=0.6，λ0 锚随 m 翻倍）", "cq_fig_e25_cmp_OLm8_toolagent"),
        (dict(file="toolagent_trace.jsonl", B=80.0, rho=0.6, m=2),
         "OL NPU 数量 m=2（B=80, ρ=0.6，λ0 锚随 m 减半）", "cq_fig_e25_cmp_OLm2_toolagent"),
        (dict(file="conversation_trace.jsonl", B=20.0, rho=0.9, m=4),
         "OL 间歇偏过载（B=20, ρ=0.9, m=4）", "cq_fig_e25_cmp_OLB20r09_conversation"),
    ]
    from matplotlib.patches import Patch
    patches = [Patch(facecolor="#4C72B0", label="计算"),
               Patch(facecolor="#DD8452", label="等待读（STALL）"),
               Patch(facecolor="#C7C7C7", label="闲置")]
    for c, title, out in cases:
        cid = cid_new(c, 9)
        tss = []
        for pid, th, lbl in [("cq_fcfs", "", "FCFS（基线）"),
                             ("cq_edf", "", "EDF（排序类代表）"),
                             ("cq_mpc", "_S", "MPC θ=S"),
                             ("cq_mpc", "_T", "MPC θ=T")]:
            ts = load_ts(ROOT, cid, pid, th)
            if ts is None:
                break
            tss.append((lbl, ts))
        if len(tss) < 4:
            print(f"skip case {out}: ts 缺失")
            continue
        fig = plt.figure(figsize=(14.88, 13.632), dpi=125)
        gs = fig.add_gridspec(4, 2, width_ratios=[3.1, 1.0], hspace=0.34,
                              wspace=0.12, left=0.085, right=0.985,
                              top=0.885, bottom=0.028)
        for ri, (label, ts) in enumerate(tss):
            first, last = ri == 0, ri == 3
            axg = fig.add_subplot(gs[ri, 0])
            draw_gantt(axg, ts)
            if first:
                n_w = int(ts["meta"].get("m_workers", 4))
                axg.set_title(f"{n_w} NPU 泳道（纵轴=NPU，横轴=仿真时间）",
                              fontsize=11)
            if last:
                axg.set_xlabel("仿真时间 (s)", fontsize=10)
            axb = fig.add_subplot(gs[ri, 1])
            draw_bandwidth(axb, ts, first, last)
            comp, stall, qmax = row_stats(ts)
            axg.set_ylabel(
                f"{label}\n算{comp:.0%} 等{stall:.0%}\n队列峰{qmax:.0f}",
                fontsize=10.5)
        fig.suptitle(f"图I-case｜{TRACE_LABEL[c['file']]} · {title}"
                     f"（窗口9；MPC 行为 θ=S/T 运行）", fontsize=13.5, y=0.985)
        fig.legend(handles=patches, loc="upper center", ncol=3, frameon=False,
                   fontsize=11, bbox_to_anchor=(0.5, 0.958))
        fig.savefig(os.path.join(FIG, out + ".png"), dpi=125)
        ov = audit_text_overlap(fig)
        plt.close(fig)
        print(f"case 图 -> {out}.png  [文字重叠审计: "
              f"{'OK' if not ov else ov[:3]}]")
    print("图K + case 图完成")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("run", "all"):
        run_all()
    if mode in ("analyze", "all"):
        analyze()
    if mode in ("figs", "all"):
        figs()


if __name__ == "__main__":
    main()
