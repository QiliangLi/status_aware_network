"""生成 TFS 收益场景交互页(单文件 HTML,无外部依赖)。

数据:曲线(超订占比 vs B,B 取对数网格 5~200,净口径)+ 逐秒到达读取
速率时序(最大值池化到 ≤600 点)+ 明细指标。全部由 tools/tfs_trace_gain.py
的装载与画像函数重算,不读 results(可独立复现)。

用法:.venv/bin/python tools/tfs_gain_html.py [输出路径]
默认输出 docs/TFS收益场景判定-20261015.html。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
sys.path.insert(0, os.getcwd())

from tfs_trace_gain import (KAPPA, L, M_NPU, _c_s, agentx_global, analyze,
                            load_rows)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "docs", "TFS收益场景判定-20261015.html")

TRACES = [
    ("conversation_trace.jsonl", "Conversation", "conv"),
    ("toolagent_trace.jsonl", "ToolAgent", "tool"),
    ("synthetic_trace.jsonl", "Mooncake-Synthetic", "synth"),
    ("agentx", "AgentX 样本(2 会话)", "ax"),
]
B_GRID = np.geomspace(5.0, 200.0, 60)
SERIES_POINTS = 600


def curves_and_series(rows):
    rows = sorted(rows, key=lambda r: r[0])
    t = np.array([r[0] for r in rows])
    h = np.array([r[1] for r in rows], dtype=np.int64)
    u = np.array([r[2] for r in rows], dtype=np.int64)
    v_total = L * KAPPA * h
    dur = float(t[-1] - t[0]) or 1.0
    n_win = int(np.ceil(dur)) or 1
    idx = np.clip(t.astype(int), 0, n_win - 1)
    v_rate = np.bincount(idx, weights=v_total, minlength=n_win)
    n_arr = np.bincount(idx, minlength=n_win)
    multi = n_arr >= 2
    b_star = float(v_total.sum() / dur)
    cur = {"B": [round(float(b), 2) for b in B_GRID],
           "win": [], "req": [], "rho": []}
    for B in B_GRID:
        hot2 = (v_rate > B) & multi
        cur["win"].append(round(float(hot2.mean()), 5))
        cur["req"].append(round(float(
            sum(n_arr[i] for i in np.flatnonzero(hot2)) / len(rows)), 5))
        cur["rho"].append(round(b_star / float(B), 4))
    # 时序:最大值池化到 ≤SERIES_POINTS
    step = max(1, n_win // SERIES_POINTS)
    pooled = np.maximum.reduceat(v_rate, np.arange(0, n_win, step)) \
        if n_win % step == 0 else None
    if pooled is None:
        pooled = [float(v_rate[i:i + step].max())
                  for i in range(0, n_win, step)]
    series = [[i * step, round(float(v), 3)]
              for i, v in enumerate(pooled)]
    return cur, series, b_star


def main():
    data = {"traces": []}
    for ident, label, key in TRACES:
        kind = "ax" if ident == "agentx" else "mc"
        if kind == "mc":
            rows = load_rows("mc", ident)
            note = "Mooncake · 首现命中"
        else:
            rows = load_rows("ax", "agentx")
            note = "AgentX · 理想前缀命中(口径不同,不可与左三列互推)"
        summ = analyze(rows)
        cur, series, b_star = curves_and_series(rows)
        s3, s2, s4 = summ["S3_mix"], summ["S2_cluster"], summ["S4_compute"]
        data["traces"].append({
            "key": key, "label": label, "note": note,
            "n": summ["n"], "dur_s": round(summ["dur_s"]),
            "arr_rate": round(summ["arr_rate"], 2),
            "b_star": round(b_star, 3),
            "v_peak": round(s4["v_rate_peak_gbps"], 1),
            "speedup80": round(s4["speedup_to_rho1_at_B80"] or 0),
            "cluster_frac": round(s2["req_in_cluster_ge8_frac"], 4),
            "peak1s": s2["peak_1s"], "mean1s": round(s2["mean_1s"], 1),
            "d_cv": round(s3["d_cv"], 2),
            "d_p25": round(s3["d_p25"], 2), "d_p50": round(s3["d_p50"], 2),
            "d_p75": round(s3["d_p75"], 2), "d_p90": round(s3["d_p90"], 2),
            "zero_read": round(s3["zero_read_frac"], 4),
            "near_light": round(s3["near_light_frac"], 4),
            "heavy_v": round(s3["heavy_v_share"], 4),
            "hit": round(s3["token_hit_ratio"], 4),
            "comp_peak": round(s4["comp_peak_win_npu"], 1),
            "curves": cur, "series": series})
    tpl = open(os.path.join(REPO, "tools", "tfs_gain_html_template.html"),
               encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False,
                                              separators=(",", ":")))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML -> {OUT} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
