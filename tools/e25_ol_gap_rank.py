"""合并全部 ol_gap_search*.json 并排序（OL mpc-local 差距搜索汇总）。"""
from __future__ import annotations

import glob
import json
import os

db = {}
for p in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "results", "cq", "eval", "e25_q2",
                         "ol_gap_search*.json"))):
    for k, v in json.load(open(p)).items():
        db[k] = v

print(f"共 {len(db)} 格点。按 |TTFT中位|+|SLO中位| 排序：")
rank = sorted(db.items(),
              key=lambda kv: -(abs(kv[1]["median_d_ttft_pct"])
                               + abs(kv[1]["median_d_slo_pp"])))
for k, v in rank:
    print(f"{k:42s} TTFT中位{v['median_d_ttft_pct']:+7.2f}% "
          f"SLO中位{v['median_d_slo_pp']:+6.2f}pp | "
          f"单窗最大{v['max_abs_d_ttft_pct']:6.2f}%/{v['max_abs_d_slo_pp']:5.2f}pp")

print("\n按单窗最大 |TTFT| 排序（找极端分化 case）：")
for k, v in sorted(db.items(), key=lambda kv: -kv[1]["max_abs_d_ttft_pct"])[:6]:
    best_w = max(v["per_window"].items(),
                 key=lambda x: abs(x[1]["d_ttft_pct"]))
    print(f"{k:42s} 单窗最大|TTFT|{v['max_abs_d_ttft_pct']:6.2f}% "
          f"(w{best_w[0]}: {best_w[1]['d_ttft_pct']:+.2f}%) "
          f"SLO {best_w[1]['d_slo_pp']:+.2f}pp")

print("\n按单窗最大 |SLO| 排序：")
for k, v in sorted(db.items(), key=lambda kv: -kv[1]["max_abs_d_slo_pp"])[:6]:
    best_w = max(v["per_window"].items(), key=lambda x: abs(x[1]["d_slo_pp"]))
    print(f"{k:42s} 单窗最大|SLO|{v['max_abs_d_slo_pp']:5.2f}pp "
          f"(w{best_w[0]}) TTFT {best_w[1]['d_ttft_pct']:+.2f}%")
