"""E25 结果分析草稿脚本：从 e25_records.json + ts 提取报告 headline 数字。

用法：.venv/bin/python tools/e25_analysis.py [stage]  （默认 eval）
"""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = sys.argv[1] if len(sys.argv) > 1 else "eval"
D = os.path.join(REPO, "results", "cq", STAGE, "e25")
FILES = ["conversation_trace.jsonl", "toolagent_trace.jsonl",
         "synthetic_trace.jsonl"]
FLABEL = {f: f.replace("_trace.jsonl", "") for f in FILES}
POLS = ["cq_fcfs", "cq_lpm", "cq_spt", "cq_edf", "cq_slack", "cq_pair",
        "cq_local", "cq_mpc"]


def slo(r):
    return r["slo_success"] / r["n_cohort"] if r.get("n_cohort") else None


def med(vs):
    vs = [v for v in vs if v is not None]
    return float(np.median(vs)) if vs else None


def paired_gain(recs, base, cmp_, key, better_down=True):
    idx = defaultdict(dict)
    for r in recs:
        idx[(r["file"], r["B"], r.get("rho"), r["alpha"], r["window"],
             r["mode"], r["theta"], r["cap"], r["profile"])][r["policy"]] = r
    out = []
    for k, pols in idx.items():
        if base in pols and cmp_ in pols:
            b, p = pols[base], pols[cmp_]
            vb = key(b)
            vp = key(p)
            if vb is not None and vp is not None and vb != 0:
                g = 100 * (vb - vp) / vb
                if not better_down:
                    g = -g
                out.append(g)
    return out


def main():
    recs = json.load(open(os.path.join(D, "e25_records.json")))
    print(f"记录总数 {len(recs)}；invalid_ts {sum(1 for r in recs if r.get('invalid_ts'))}")

    # ---- 1) FW 基线四目标（图 A 数字） ----
    print("\n=== 1) FW 基线：四目标中位数（窗口×文件合并；M/S/T/R） ===")
    base_fw = [r for r in recs if r["factor_line"] == "baseline" and r["mode"] == "FW"]
    print(f"{'policy':10s} {'M(s)':>8s} {'SLO率':>7s} {'TTFT(s)':>8s} {'归一化':>8s} {'G_total':>7s} {'SLO率(各文件)':>36s}")
    for p in POLS:
        sub = [r for r in base_fw if r["policy"] == p]
        mrow = [r for r in sub if r["theta"] == "M"]
        srow = [r for r in sub if r["theta"] == "S"]
        trow = [r for r in sub if r["theta"] == "T"]
        rrow = [r for r in sub if r["theta"] == "R"]
        per_f = []
        for f in FILES:
            fs = [slo(r) for r in srow if r["file"] == f]
            per_f.append(f"{FLABEL[f][:4]}:{med(fs):.3f}" if med(fs) is not None else f"{FLABEL[f][:4]}:-")
        print(f"{p:10s} {med([r.get('makespan_M') or r.get('makespan_M_lower') for r in mrow]):8.2f} "
              f"{med([slo(r) for r in srow]):7.3f} "
              f"{med([r['ttft_mean_lower'] for r in trow]):8.3f} "
              f"{med([r['ttft_norm_mean_lower'] for r in rrow]):8.3f} "
              f"{med([r.get('batch_G_total') for r in sub]):7.3f} {' '.join(per_f):>36s}")

    # ---- 2) OL 基线配对收益（vs FCFS；θ 对应） ----
    print("\n=== 2) OL 基线：相对 FCFS 配对中位数（θ=S pp / θ=T,R %） ===")
    base_ol = [r for r in recs if r["factor_line"] == "baseline" and r["mode"] == "OL"]
    print(f"{'policy':10s} {'SLO(pp)':>9s} {'TTFT(%)':>9s} {'归一化(%)':>9s} {'drainM(%)':>9s}")
    for p in POLS[1:]:
        gs = paired_gain(base_ol, "cq_fcfs", p, slo)
        gt = paired_gain(base_ol, "cq_fcfs", p, lambda r: r["ttft_mean_lower"])
        gr = paired_gain(base_ol, "cq_fcfs", p, lambda r: r["ttft_norm_mean_lower"])
        gm = paired_gain(base_ol, "cq_fcfs", p, lambda r: r.get("drain_M"))
        print(f"{p:10s} {med(gs) if gs else float('nan'):9.2f} {med(gt) if gt else float('nan'):9.2f} "
              f"{med(gr) if gr else float('nan'):9.2f} {med(gm) if gm else float('nan'):9.2f}")

    # ---- 3) 因子线：关键响应（各线取基线 vs 各水平，θ=T 的平均TTFT、θ=S 的SLO） ----
    print("\n=== 3) 因子响应摘要（OL·θ=S 的 FCFS SLO / θ=T 的 spt TTFT 相对基线变化） ===")
    for line, getter, lvname in [
            ("line_B", lambda r: r["B"], "B"),
            ("line_alpha", lambda r: r["alpha"], "α"),
            ("line_rho", lambda r: r["rho"], "ρ")]:
        print(f"-- {line}（OL）FCFS SLO & FCFS TTFT：")
        for lv in sorted({getter(r) for r in recs if r["factor_line"] in ("baseline", line)
                          and r["mode"] == "OL"}):
            s = [slo(r) for r in recs if r["factor_line"] in ("baseline", line)
                 and r["mode"] == "OL" and r["policy"] == "cq_fcfs"
                 and r["theta"] == "S" and getter(r) == lv]
            t = [r["ttft_mean_lower"] for r in recs if r["factor_line"] in ("baseline", line)
                 and r["mode"] == "OL" and r["policy"] == "cq_fcfs"
                 and r["theta"] == "T" and getter(r) == lv]
            print(f"   {lvname}={lv}: SLO={med(s):.3f}  TTFT={med(t):.3f}s")
    print("-- line_cap / line_profile（FW，θ=S 的 FCFS/SPT SLO、G_total）：")
    for line, getter in [("line_cap", lambda r: r["cap"]),
                         ("line_profile", lambda r: r["profile"])]:
        for lv in sorted({getter(r) for r in recs if r["factor_line"] in ("baseline", line)
                          and r["mode"] == "FW"}, key=str):
            row = {}
            for p in ("cq_fcfs", "cq_spt"):
                s = [slo(r) for r in recs if r["factor_line"] in ("baseline", line)
                     and r["mode"] == "FW" and r["policy"] == p
                     and r["theta"] == "S" and getter(r) == lv]
                g = [r.get("batch_G_total") for r in recs
                     if r["factor_line"] in ("baseline", line) and r["mode"] == "FW"
                     and r["policy"] == p and getter(r) == lv]
                row[p] = (med(s), med(g))
            print(f"   {line}={lv}: fcfs SLO={row['cq_fcfs'][0]:.3f} G={row['cq_fcfs'][1]:.3f} | "
                  f"spt SLO={row['cq_spt'][0]:.3f} G={row['cq_spt'][1]:.3f}")

    # ---- 4) MPC vs EDF 增量（各线汇总） ----
    print("\n=== 4) MPC 相对 EDF 的配对差（θ 对应；0=无增量） ===")
    for mode in ("FW", "OL"):
        sub = [r for r in recs if r["mode"] == mode]
        gs = paired_gain(sub, "cq_edf", "cq_mpc", slo)
        gt = paired_gain(sub, "cq_edf", "cq_mpc", lambda r: r["ttft_mean_lower"])
        gm = paired_gain(sub, "cq_edf", "cq_mpc", lambda r: r.get("drain_M") or r.get("makespan_M") or r.get("makespan_M_lower"))
        print(f"   {mode}: SLO(pp)={med(gs) if gs else float('nan'):+.3f} "
              f"TTFT(%)={med(gt) if gt else float('nan'):+.3f} M(%)={med(gm) if gm else float('nan'):+.3f}")

    # ---- 5) 时间序列读数（基线 OL B80/B20，窗口默认；STALL 占比/利用率） ----
    print("\n=== 5) ts 读数（OL 基线，fcfs/spt/edf/mpc，各文件） ===")
    for B in (80.0, 20.0):
        factor = "baseline" if B == 80.0 else "line_B"
        for f in FILES:
            for p in ("cq_fcfs", "cq_spt", "cq_mpc"):
                th = "S" if p == "cq_mpc" else None
                cand = [r for r in recs if r["factor_line"] == factor
                        and r["mode"] == "OL" and r["file"] == f and r["B"] == B
                        and r["policy"] == p and r.get("ts_path")
                        and (th is None or r["theta"] == th)
                        and (p not in ("cq_mpc",) or True)]
                simple_cand = [r for r in cand if r["policy"] != "cq_mpc"]
                cand = simple_cand or cand
                if not cand:
                    continue
                w9 = [r for r in cand if r["window"] == 9] or cand
                ts = json.load(gzip.open(os.path.join(D, w9[0]["ts_path"]), "rt"))
                m = ts["meta"]["m_workers"]
                rows = ts["rows"]
                cov = [x for x in rows if x["width_s"] > 0]
                comp = sum(x["compute_s"] for x in cov)
                stall = sum(x["stall_s"] for x in cov)
                idle = sum(x["idle_s"] for x in cov)
                tot = comp + stall + idle
                utils = [x["served_gb"] / x["capacity_gb"] for x in cov
                         if x["capacity_gb"] > 0]
                qm = [x["queue_mean"] for x in cov if x["queue_mean"] is not None]
                print(f"   B{B:g} {FLABEL[f][:12]:13s} {p:8s} w{w9[0]['window']} "
                      f"Compute={comp/tot:.1%} STALL={stall/tot:.1%} Idle={idle/tot:.1%} "
                      f"util中位={med(utils):.1%} util峰值={max(utils):.1%} Q中位={med(qm):.0f} "
                      f"({'含溢出桶' if ts['meta']['overflow_used'] else '锚内完成'})")

    # ---- 6) n_sat128 通道确认 ----
    print("\n=== 6) line_profile n_sat128：G>1 批数与 G_total（各策略中位） ===")
    for p in ("cq_fcfs", "cq_spt", "cq_mpc"):
        sub = [r for r in recs if r["factor_line"] == "line_profile"
               and r["profile"] == "n_sat128" and r["policy"] == p]
        if sub:
            print(f"   {p}: G_total={med([r.get('batch_G_total') for r in sub]):.3f} "
                  f"G>1批数中位={med([r['n_batches_G_gt1'] for r in sub])} "
                  f"max_batch={med([r['max_batch_size'] for r in sub])}")


if __name__ == "__main__":
    main()
