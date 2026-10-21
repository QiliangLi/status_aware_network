"""E26f 盘维度三方对照分析：母（fcfs 分配）/ 单池 max_min / 盘维度 max_min_disk。

读三组 records（同 (cell, rep, policy) 对齐；cA 取 rep0/1/2 中位数、cB rep0），
产出结果文档用 markdown 表：主表（总/分类 SLO）、H1（排序与 mpcS−fcfs 差距）、
H2（B 类长尾签名）、H3（功守恒与盘间不均）、H4（MPC 墙钟与 fb）、H5（B 层读
公平化代理）。用法：
  .venv/bin/python tools/e26f_disk3.py [--cell=D1fA_k16]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = {"母(fcfs分配)": "e26f", "单池max_min": "e26f_max_min",
         "盘维度": "e26f_max_min_disk"}
ARMS = ["fcfs", "edf", "localS_v2.1", "mpcS_v2.1", "mpcT_v2.1", "tfs"]
ARM_LABEL = {"fcfs": "FCFS", "edf": "EDF", "localS_v2.1": "local v2.1",
             "mpcS_v2.1": "mpc θ=S", "mpcT_v2.1": "mpc θ=T", "tfs": "TFS"}
CELLS = ["D1fA_k16", "D1fB_k16"]


def load(root_name):
    recs = []
    for p in sorted(glob.glob(os.path.join(REPO, "results", "cq", "eval",
                                           root_name, "records_part_*.json"))):
        recs.extend(json.load(open(p, encoding="utf-8")))
    return recs


def _num(v):
    """records 数值兼容：float 或 'a/b' Fraction 字符串（首批 records 的
    max_disk_frac 以字符串入库，后批为 float）。"""
    if v is None:
        return None
    if isinstance(v, str):
        from fractions import Fraction
        return float(Fraction(v))
    return float(v)


def med(recs, cell, arm, key, cls=None):
    vals = []
    for r in recs:
        if r["cell"] == cell and r["policy"] == arm:
            v = r[key] if cls is None else r[cls][key]
            if v is not None:
                vals.append(_num(v))
    return float(np.median(vals)) if vals else None


def fmt(v, pct=True, nd=1):
    if v is None:
        return "—"
    return f"{100*v:.{nd}f}" if pct else f"{v:.{nd}f}"


def main():
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--cell="):
            only = a[len("--cell="):]
    data = {name: load(root) for name, root in ROOTS.items()}
    for name, recs in data.items():
        print(f"# {name}: {len(recs)} runs", file=sys.stderr)
    cells = [c for c in CELLS if only in (None, c)]

    for cell in cells:
        print(f"\n## {cell}\n")
        # 主表
        print("| 臂 | " + " | ".join(
            f"{n} 总SLO | {n} A | {n} B | {n} C" for n in ROOTS) + " |")
        print("|" + "---|" * (1 + 4 * len(ROOTS)))
        for arm in ARMS:
            row = [ARM_LABEL[arm]]
            for n in ROOTS:
                recs = data[n]
                row += [fmt(med(recs, cell, arm, "slo_rate")),
                        fmt(med(recs, cell, arm, "slo_rate", "A")),
                        fmt(med(recs, cell, arm, "slo_rate", "B")),
                        fmt(med(recs, cell, arm, "slo_rate", "C"))]
            print("| " + " | ".join(row) + " |")
        # H1: 排序与差距
        print(f"\n### H1 排序与 mpcS−fcfs 差距（{cell}，分类 pp）\n")
        print("| 分配 | 臂排序(总SLO降序) | mpcS−fcfs | mpcT−fcfs | tfs−fcfs |")
        print("|---|---|---|---|---|")
        for n, recs in data.items():
            scores = {arm: med(recs, cell, arm, "slo_rate") for arm in ARMS}
            order = " > ".join(ARM_LABEL[a] for a in
                               sorted(ARMS, key=lambda x: -(scores[x] or -1)))
            gaps = [fmt((scores["mpcS_v2.1"] - scores["fcfs"]) * 100 if
                        scores["mpcS_v2.1"] is not None and
                        scores["fcfs"] is not None else None, pct=False, nd=1)]
            gaps += [fmt((scores["mpcT_v2.1"] - scores["fcfs"]) * 100
                         if scores["mpcT_v2.1"] is not None and
                         scores["fcfs"] is not None else None, pct=False)]
            gaps += [fmt((scores["tfs"] - scores["fcfs"]) * 100
                         if scores["tfs"] is not None and
                         scores["fcfs"] is not None else None, pct=False)]
            print(f"| {n} | {order} | " + " | ".join(gaps) + " |")
        # H2: 长尾类签名
        print(f"\n### H2 B 类长尾（{cell}）\n")
        print("| 分配 | B ttft_p95(s) | B ttft_mean(s) | B SLO | A ttft_p95(s) | C SLO |")
        print("|---|---|---|---|---|---|")
        for n, recs in data.items():
            print(f"| {n} | " + " | ".join([
                fmt(med(recs, cell, "mpcS_v2.1", "ttft_p95", "B"), pct=False, nd=3),
                fmt(med(recs, cell, "mpcS_v2.1", "ttft_mean", "B"), pct=False, nd=3),
                fmt(med(recs, cell, "mpcS_v2.1", "slo_rate", "B")),
                fmt(med(recs, cell, "mpcS_v2.1", "ttft_p95", "A"), pct=False, nd=3),
                fmt(med(recs, cell, "mpcS_v2.1", "slo_rate", "C"))]) + " |")
        # H3: 功守恒与盘间
        print(f"\n### H3 功守恒与盘间不均（{cell}）\n")
        print("| 分配 | 总带宽利用率 | bw饱和占比 | 盘均利用率(逐盘) | 盘间CV | CV>0.15时间占比 |")
        print("|---|---|---|---|---|---|")
        for n, recs in data.items():
            util = med(recs, cell, "fcfs", "storage_utilization")
            sat = med(recs, cell, "fcfs", "bw_sat_frac")
            if n == "盘维度":
                disks = med(recs, cell, "fcfs", "disks") or 3
                per = [fmt(med(recs, cell, "fcfs", f"disk_util_mean_{d}"))
                       for d in range(int(disks))]
                cv = fmt(med(recs, cell, "fcfs", "disk_util_cv"))
                cv15 = fmt(med(recs, cell, "fcfs", "disk_cv15_frac"))
            else:
                per, cv, cv15 = ["—"] * 3
            print(f"| {n}(fcfs臂) | {fmt(util)} | {fmt(sat)} | "
                  f"{'/'.join(per) if isinstance(per, list) else per} | {cv} | {cv15} |")
        # H4: MPC 成本
        print(f"\n### H4 MPC 墙钟与 fb（{cell}，s）\n")
        print("| 分配 | localS | mpcS | mpcT | mpcS fb | 决策数 |")
        print("|---|---|---|---|---|---|")
        for n, recs in data.items():
            w = [fmt(med(recs, cell, a, "wall_s"), pct=False, nd=0)
                 for a in ("localS_v2.1", "mpcS_v2.1", "mpcT_v2.1")]
            fb = fmt(med(recs, cell, "mpcS_v2.1", "n_fallback"), pct=False, nd=0)
            nd_ = fmt(med(recs, cell, "mpcS_v2.1", "n_decides"), pct=False, nd=0)
            print(f"| {n} | " + " | ".join(w) + f" | {fb} | {nd_} |")
    # 盘维度类常量
    recs = data["盘维度"]
    if recs:
        r0 = next(r for r in recs if r.get("max_disk_frac_A") is not None)
        print("\n## 盘维度类常量（records）\n")
        print(f"disks={_num(r0['disks'])} disk_bps={_num(r0['disk_bps_gbps'])} "
              f"max_disk_frac A={_num(r0['max_disk_frac_A']):.4f} "
              f"B={_num(r0['max_disk_frac_B']):.4f} "
              f"(A 块84/层 B 块19/层；最慢盘字节占比≈0.39/0.44)")


if __name__ == "__main__":
    main()
