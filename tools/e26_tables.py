"""E26 表格与验收检查：合并 records 分片 → 按 (格点×策略) 多 seed 中位数 →
Markdown 汇总表（§6 指标）+ 验收判据检查（§7）。

用法：.venv/bin/python tools/e26_tables.py [--stage eval] [--mode D1|D2|all]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.environ.get("E26_STAGE", "eval")
ROOT = os.path.join(REPO, "results", "cq", STAGE, "e26")

POL_ORDER = [("cq_fcfs", None), ("cq_edf", None), ("cq_local", "S"),
             ("cq_local", "T"), ("cq_mpc", "S"), ("cq_mpc", "T")]
KEYS_MED = ["makespan_s", "ttft_mean_lower", "ttft_norm_mean_lower",
            "ttft_p95_lower", "slo_rate", "A_ttft_mean", "A_slo_rate",
            "B_ttft_mean", "B_slo_rate", "io_wait_share2", "bw_sat_frac",
            "bw_over_frac", "a_conc_p50", "a_conc_p90", "a_conc_max",
            "n_scored", "n_fallback", "wall_s"]


def load_records():
    recs = []
    for p in sorted(glob.glob(os.path.join(ROOT, "records_part*.json"))):
        recs.extend(json.load(open(p, encoding="utf-8")))
    for r in recs:
        if r.get("slo_rate") is None and r.get("n_cohort"):
            r["slo_rate"] = r["slo_success"] / r["n_cohort"]
    return recs


def key_of(r):
    return (r["cell"], r["policy"], r.get("theta"))


def median_table(recs):
    """(cell, policy, theta) -> 中位数指标 dict。"""
    by = defaultdict(list)
    for r in recs:
        by[key_of(r)].append(r)
    out = {}
    for k, rs in by.items():
        m = {"n_reps": len(rs), "alpha": rs[0]["alpha"], "mode": rs[0]["mode"],
             "level": rs[0]["level"], "mpc_events": rs[0].get("mpc_events")}
        for key in KEYS_MED:
            vals = [r[key] for r in rs
                    if r.get(key) is not None and r.get("invalid_ts") is None]
            vals = [float(v) for v in vals]
            m[key] = float(np.median(vals)) if vals else None
        m["n_invalid_ts"] = sum(1 for r in rs if r.get("invalid_ts"))
        m["n_unfinished"] = int(np.median([r["n_unfinished"] for r in rs]))
        m["slo_rate"] = m.get("slo_rate")
        out[k] = m
    return out


def fmt(v, pct=False, s2=False, nd=2):
    if v is None:
        return "—"
    if pct:
        return f"{100*v:.1f}%"
    return f"{v:.{nd}f}"


def cell_table(tab, cell):
    rows = []
    for pid, th in POL_ORDER:
        m = tab.get((cell, pid, th))
        if m is None:
            continue
        label = pid.replace("cq_", "") + (f" θ={th}" if th else "")
        rows.append(
            f"| {label} | {fmt(m['makespan_s'],nd=1)} | "
            f"{fmt(m['ttft_mean_lower'],nd=3)} | {fmt(m['ttft_norm_mean_lower'])} | "
            f"{fmt(m['slo_rate'],pct=True)} | {fmt(m['A_ttft_mean'],nd=3)} | "
            f"{fmt(m['A_slo_rate'],pct=True)} | {fmt(m['B_ttft_mean'],nd=3)} | "
            f"{fmt(m['B_slo_rate'],pct=True)} | {fmt(m['io_wait_share2'],pct=True)} | "
            f"{fmt(m['bw_sat_frac'],pct=True)} | "
            f"{fmt(m['a_conc_p50'],nd=1)}/{fmt(m['a_conc_p90'],nd=1)}/"
            f"{fmt(m['a_conc_max'],nd=0)} | "
            f"{fmt(m['n_scored'],nd=0)} | {m['n_reps']} |")
    return rows


def gap_row(tab, cell, a, b):
    """b 相对 a 的 TTFT 降幅% 与 SLO 差 pp（θ=S 口径调用方保证）。"""
    ra, rb = tab.get((cell, *a)), tab.get((cell, *b))
    if ra is None or rb is None:
        return None
    ttft_gain = (None if ra["ttft_mean_lower"] in (None, 0)
                 else 100 * (ra["ttft_mean_lower"] - rb["ttft_mean_lower"])
                 / ra["ttft_mean_lower"])
    slo_pp = (None if ra["slo_rate"] is None or rb["slo_rate"] is None
              else 100 * (rb["slo_rate"] - ra["slo_rate"]))
    return ttft_gain, slo_pp


def main():
    args = sys.argv[1:]
    mode_filter = "all"
    if "--mode" in args:
        mode_filter = args[args.index("--mode") + 1]
    recs = load_records()
    if not recs:
        print("无记录"); return
    tab = median_table(recs)
    cells = sorted({k[0] for k in tab})
    if mode_filter != "all":
        cells = [c for c in cells if c.startswith(mode_filter)]
    hdr = ("| 策略 | 排空(s) | TTFT均(s) | 归一TTFT | SLO率 | A-TTFT(s) | A-SLO | "
           "B-TTFT(s) | B-SLO | 计算/(算+等) | 带宽饱和 | 并发A p50/p90/max | "
           "n_scored | reps |\n|---|" + "---|" * 13)
    print("# E26 汇总（多 seed 中位数）\n")
    for cell in cells:
        any_key = next(k for k in tab if k[0] == cell)
        m0 = tab[any_key]
        print(f"\n## {cell}（α={m0['alpha']}，mpc_events={m0.get('mpc_events')}）\n")
        print(hdr)
        for r in cell_table(tab, cell):
            print(r)
    # 验收判据（§7）
    print("\n# 验收检查（θ=S 口径）\n")
    print("| 格点 | local−FCFS TTFT | local−EDF TTFT | mpc−FCFS TTFT | "
          "mpc−EDF TTFT | mpc−local TTFT | mpc−local SLO(pp) |")
    print("|---|" + "---|" * 6)
    best = []
    for cell in cells:
        g = {}
        for name, a, b in (("local-fcfs", ("cq_fcfs", None), ("cq_local", "S")),
                           ("local-edf", ("cq_edf", None), ("cq_local", "S")),
                           ("mpc-fcfs", ("cq_fcfs", None), ("cq_mpc", "S")),
                           ("mpc-edf", ("cq_edf", None), ("cq_mpc", "S")),
                           ("mpc-local", ("cq_local", "S"), ("cq_mpc", "S"))):
            g[name] = gap_row(tab, cell, a, b)
        def s(name, idx):
            v = g.get(name)
            return "—" if v is None or v[idx] is None else f"{v[idx]:+.1f}%"
        slo_v = g.get("mpc-local")
        slo_s = "—" if slo_v is None or slo_v[1] is None else f"{slo_v[1]:+.1f}"
        print(f"| {cell} | {s('local-fcfs',0)} | {s('local-edf',0)} | "
              f"{s('mpc-fcfs',0)} | {s('mpc-edf',0)} | {s('mpc-local',0)} | {slo_s} |")
        best.append((cell, g))
    return tab, best


if __name__ == "__main__":
    main()
