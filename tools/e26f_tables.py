"""E26f 汇总表:分格点 × 分臂,主口径=多种子中位数(1 seed 即该值)。"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "results", "cq", "eval", "e26f")
ARMS = ["fcfs", "edf", "localS_v2.1", "mpcS_v2.1", "mpcT_v2.1"]
CELLS = ["D1fA_k16", "D1fB_k16"]


def med(records, cell, arm, key):
    vals = [r for r in records if r["cell"] == cell and r["policy"] == arm]
    if not vals:
        return None
    return float(np.median([key(r) for r in vals])), [key(r) for r in vals]


def main():
    records = []
    for p in sorted(glob.glob(os.path.join(ROOT, "records_part_*.json"))):
        records.extend(json.load(open(p, encoding="utf-8")))
    records.sort(key=lambda r: (r["cell"], r["rep"]))   # 方括号统一 rep 升序
    print(f"{len(records)} records")
    for cell in CELLS:
        print(f"\n== {cell}")
        print(f"{'臂':14s} {'总SLO':>14s} {'A-SLO':>14s} {'B-SLO':>14s} "
              f"{'C-SLO':>14s} {'makespan':>8s} {'簇发A均/峰':>10s} rep数")
        for arm in ARMS:
            rows = [r for r in records if r["cell"] == cell and r["policy"] == arm]
            if not rows:
                continue

            def m(fn):
                return float(np.median([fn(r) for r in rows]))

            def show(fn, pct=True):
                v = [fn(r) for r in rows]
                s = f"{100*m(fn):.1f}%" if pct else f"{m(fn):.2f}"
                br = "/".join((f"{100*x:.0f}" if pct else f"{x:.2f}")
                              for x in v)
                return f"{s}[{br}]"

            shock = [r["cluster_A_disp_max"] for r in rows]
            print(f"{arm:14s} {show(lambda r: r['slo_rate']):>14s} "
                  f"{show(lambda r: r['A']['slo_rate']):>14s} "
                  f"{show(lambda r: r['B']['slo_rate']):>14s} "
                  f"{show(lambda r: r['C']['slo_rate']):>14s} "
                  f"{show(lambda r: r['makespan_s'], False):>8s} "
                  f"{m(lambda r: r['cluster_A_disp_mean']):.1f}/"
                  f"{int(max(shock))}{'':>{max(0, 6-len(shock))}s} "
                  f"{len(rows)}")
        # MPC−FCFS 判定口径
        for k, fn in (("A-SLO", lambda r: r["A"]["slo_rate"]),
                      ("总SLO", lambda r: r["slo_rate"])):
            d = None
            if (rows_f := [r for r in records if r["cell"] == cell
                           and r["policy"] == "fcfs"]) and \
               (rows_m := [r for r in records if r["cell"] == cell
                           and r["policy"] == "mpcS_v2.1"]):
                # 按 rep 对齐后取中位数差(CRN)
                reps = sorted(set(r["rep"] for r in rows_f)
                              & set(r["rep"] for r in rows_m))
                if reps:
                    diff = [fn(next(r for r in rows_m if r["rep"] == rp))
                            - fn(next(r for r in rows_f if r["rep"] == rp))
                            for rp in reps]
                    d = float(np.median(diff))
                    print(f"  mpcS−FCFS {k}(逐 rep 对齐): median={100*d:+.1f}pp "
                          f"reps={[f'{100*x:+.0f}' for x in diff]}")
        # EDF 排名(垫底检验)
        edf_tot = med(records, cell, "edf", lambda r: r["slo_rate"])
        if edf_tot:
            others = [med(records, cell, a, lambda r: r["slo_rate"])[0]
                      for a in ARMS if a != "edf"]
            print(f"  EDF 总SLO 中位数 {100*edf_tot[0]:.1f}% vs 其余最小 "
                  f"{100*min(others):.1f}% → "
                  f"{'垫底' if edf_tot[0] < min(others) else '不垫底(该 seed)'}"
                  f"  各rep={[f'{100*x:.0f}' for x in edf_tot[1]]}")
        # 逐 NPU 利用率表(纯算口径;与 util 图箱线同数据,给精确数字)
        # 均值列=全机平均=device_compute_s/device_total_mh_s(=32 NPU 纯算
        # 利用率的算术均值,records 未存逐 NPU 数组,由此恒等式恢复)
        print(f"\n  逐 NPU 利用率(纯算=计算/makespan,中位数;方括号=各 rep)")
        print(f"  {'臂':14s} {'min':>12s} {'p50':>12s} {'p90':>12s} "
              f"{'max':>12s} {'均值':>12s} {'CV':>10s} {'含等均值':>12s} makespan")
        for arm in ARMS:
            rows_u = [r for r in records
                      if r["cell"] == cell and r["policy"] == arm]
            if not rows_u:
                continue

            def um(key):
                return med(records, cell, arm, lambda r: r[key])

            def cell_of(key, pct=True):
                m, v = um(key)
                s = f"{100*m:.1f}%" if pct else f"{m:.2f}"
                return f"{s}[{'/'.join(f'{100*x:.0f}' if pct else f'{x:.2f}' for x in v)}]"

            mk = um("makespan_s")
            mean_u = med(records, cell, arm,
                         lambda r: r["device_compute_s"] / r["device_total_mh_s"])
            print(f"  {arm:14s} {cell_of('util_min'):>12s} "
                  f"{cell_of('util_p50'):>12s} {cell_of('util_p90'):>12s} "
                  f"{cell_of('util_max'):>12s} "
                  f"{100*mean_u[0]:.1f}%"
                  f"[{'/'.join(f'{100*x:.0f}' for x in mean_u[1])}]"
                  f" {um('util_cv')[0]:.2f}{'':>4s} "
                  f"{cell_of('util_wait_mean'):>12s} "
                  f"{mk[0]:.2f}s")


if __name__ == "__main__":
    main()
