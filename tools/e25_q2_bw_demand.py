"""三问深答（20260921）：带宽需求 vs 实际供给 的归档统计。

Q2：B=80 时桶均峰值 ~40+ GB/s、B=20 时曲线不超 20 的原因拆解。
Q3：从 B=80 运行的需求分布推导"间歇性过载"的 B 配置窗口。

数据源：results/cq/eval/e25*/ts/<cell>/cq_fcfs.json.gz 的逐桶
served_gb / requested_gb / capacity_gb（FCFS 锚定 run，需求口径不受策略污染）。
"""
from __future__ import annotations

import gzip
import json
import os
from glob import glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_DIRS = [
    os.path.join(REPO, "results/cq/eval/e25/ts"),
    os.path.join(REPO, "results/cq/eval/e25_regime/ts"),
]
FILES = ["conversation", "toolagent", "synthetic"]
FWIN = [0, 4, 9, 14, 19]
TRACE_CN = {"conversation": "Conv", "toolagent": "Tool", "synthetic": "Syn"}


def load_rows(cell_dir, policy="cq_fcfs"):
    p = os.path.join(cell_dir, policy + ".json.gz")
    if not os.path.exists(p):
        return None
    d = json.load(gzip.open(p))
    return d


def cell_stats(d, b):
    rows = d["rows"]
    dem = []  # 桶均需求速率 GB/s
    srv = []
    wt = []
    for r in rows:
        w = r["width_s"]
        if w <= 0:
            continue
        dem.append(r["requested_gb"] / w)
        srv.append(r["served_gb"] / w)
        wt.append(w)
    tot = sum(wt)
    n = len(dem)

    def q(xs, p):
        ys = sorted(xs)
        return ys[min(n - 1, int(p * n))] if ys else float("nan")

    def frac(cond):
        return sum(w for w, x in zip(wt, dem) if cond(x)) / tot

    # 需求被削顶的桶：requested 与 served 的差 > 容量的 1%
    clipped = sum(w for w, r in zip(wt, rows)
                  if r["requested_gb"] - r["served_gb"] > 0.01 * r["capacity_gb"]) / tot
    # 桶均需求超容量（即该桶内出现过载压力）
    over_cap = frac(lambda x: x > b)
    sat90 = sum(w for w, x in zip(wt, srv) if x >= 0.9 * b) / tot
    return {
        "B": b, "n": n, "w_med": sorted(wt)[n // 2],
        "dem_mean": sum(d * w for d, w in zip(dem, wt)) / tot,
        "dem_p50": q(dem, 0.5), "dem_p90": q(dem, 0.9), "dem_p99": q(dem, 0.99),
        "dem_max": max(dem),
        "srv_max": max(srv),
        "over_cap_frac": over_cap,      # 桶均需求>B 的时间占比
        "clipped_frac": clipped,        # 桶内实际被削顶的时间占比
        "sat90_frac": sat90,            # 实际利用率≥90% 的时间占比
    }


def sat_curve_from_demand(d, bs):
    """用 B=80 运行的桶均需求分布，反推'若带宽为 B'的过载占比。"""
    rows = d["rows"]
    dem = [(r["requested_gb"] / r["width_s"], r["width_s"]) for r in rows
           if r["width_s"] > 0]
    tot = sum(w for _, w in dem)
    out = {}
    for b in bs:
        out[b] = sum(w for x, w in dem if x > b) / tot
    return out


def main():
    print("=" * 100)
    print("A. OL 基线（ρ=0.6, m=4, w9, FCFS）：B=80 与 B=20 的需求/实际/容量")
    print("=" * 100)
    hdr = (f"{'trace':5s} {'B':>4s} {'桶均需求p50/p90/p99/max':>34s} "
           f"{'实际峰':>7s} {'需求>B占比':>9s} {'削顶占比':>8s} {'sat90':>7s}")
    print(hdr)
    results = {}
    for f in FILES:
        for B in (80, 20):
            for td in TS_DIRS:
                cell = os.path.join(td, f"{f}|OL|B{B}|rho0.6|a4|C|default|w9")
                if not os.path.isdir(cell):
                    cell = os.path.join(td, f"{f}|OL|B{B}|rho0.6|a4|m4|w9")
                if not os.path.isdir(cell):
                    continue
                d = load_rows(cell)
                if d is None:
                    continue
                s = cell_stats(d, float(B))
                results[(f, B)] = s
                print(f"{TRACE_CN[f]:5s} {B:4d} "
                      f"{s['dem_p50']:7.1f}{s['dem_p90']:8.1f}{s['dem_p99']:8.1f}"
                      f"{s['dem_max']:9.1f}   {s['srv_max']:7.1f} "
                      f"{s['over_cap_frac']:9.1%} {s['clipped_frac']:8.1%} "
                      f"{s['sat90_frac']:7.1%}")
                break

    print()
    print("=" * 100)
    print("B. 其他 ρ 的 B=80 需求分布（w9, FCFS）——为 Q3 提供需求侧数据")
    print("=" * 100)
    for f in FILES:
        for rho in ("0.3", "0.6", "0.9", "1.1"):
            cell = None
            for td in TS_DIRS:
                for pat in (f"{f}|OL|B80|rho{rho}|a4|C|default|w9",
                            f"{f}|OL|B80|rho{rho}|a4|m4|w9"):
                    c = os.path.join(td, pat)
                    if os.path.isdir(c):
                        cell = c
                        break
                if cell:
                    break
            if not cell:
                print(f"{TRACE_CN[f]:5s} rho{rho}: 无归档")
                continue
            d = load_rows(cell)
            s = cell_stats(d, 80.0)
            sc = sat_curve_from_demand(d, [5, 10, 15, 20, 30, 40, 60, 80])
            print(f"{TRACE_CN[f]:5s} rho{rho}: 桶均需求 mean={s['dem_mean']:5.1f} "
                  f"p50={s['dem_p50']:5.1f} p90={s['dem_p90']:5.1f} "
                  f"p99={s['dem_p99']:5.1f} max={s['dem_max']:5.1f} | "
                  f"若B=5/10/15/20/30/40/60 → 过载占比 "
                  + "/".join(f"{sc[b]:.0%}" for b in [5, 10, 15, 20, 30, 40, 60]))

    print()
    print("=" * 100)
    print("C. m 维度（B=80, ρ=0.6, w9, FCFS）：m=2/8 的需求分布")
    print("=" * 100)
    for f in FILES:
        for m in (2, 8):
            cell = os.path.join(TS_DIRS[1], f"{f}|OL|B80|rho0.6|a4|m{m}|w9")
            if not os.path.isdir(cell):
                print(f"{TRACE_CN[f]:5s} m={m}: 无归档")
                continue
            d = load_rows(cell)
            s = cell_stats(d, 80.0)
            print(f"{TRACE_CN[f]:5s} m={m}: 桶均需求 mean={s['dem_mean']:5.1f} "
                  f"p50={s['dem_p50']:5.1f} p90={s['dem_p90']:5.1f} "
                  f"p99={s['dem_p99']:5.1f} max={s['dem_max']:5.1f}")

    print()
    print("=" * 100)
    print("D. 五窗汇总（OL 基线 ρ=0.6 m=4，FCFS）——验证 w9 代表性")
    print("=" * 100)
    for f in FILES:
        for B in (80, 20):
            agg = []
            for w in FWIN:
                for td in TS_DIRS:
                    for pat in (f"{f}|OL|B{B}|rho0.6|a4|C|default|w{w}",
                                f"{f}|OL|B{B}|rho0.6|a4|m4|w{w}"):
                        c = os.path.join(td, pat)
                        if os.path.isdir(c):
                            d = load_rows(c)
                            if d:
                                agg.append(cell_stats(d, float(B)))
                            break
            if not agg:
                continue
            print(f"{TRACE_CN[f]:5s} B={B}: 5窗中位 mean={sorted(a['dem_mean'] for a in agg)[len(agg)//2]:5.1f} "
                  f"max(各窗峰值)={max(a['dem_max'] for a in agg):5.1f} "
                  f"sat90中位={sorted(a['sat90_frac'] for a in agg)[len(agg)//2]:.1%} "
                  f"over_cap中位={sorted(a['over_cap_frac'] for a in agg)[len(agg)//2]:.1%}")


if __name__ == "__main__":
    main()
