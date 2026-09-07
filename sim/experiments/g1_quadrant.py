"""G1：存储状态进入 D1/D2 的信息增量与策略收益（设计 §6，四象限压力）。

对照：B-RR / B-LL / B-DynKV / B-Moon-S / B-Hist（+Hist+Joint）/ D1b / D2b / D1a。
判读要点：增强只赢 B-RR/B-LL 不算数（S-G 下主流默认不消费存储状态）；
必须赢 B-Hist 才构成接口增量。
"""
from __future__ import annotations

from .g_common import G1_DEFS, G1_POLICIES, QUADRANTS, run_grid


def main(seeds, procs=None, duration: float = 300.0, lam: float = 2.5):
    run_grid(exp="g1", policies=G1_POLICIES, quadrants=list(QUADRANTS),
             seeds=list(seeds), defs=G1_DEFS, lam=lam, duration=duration,
             out_path="results/g1_quadrant.json",
             timeline_marks={("B-Hist", "io", 0), ("D1a 联合", "io", 0),
                             ("B-LL", "both", 0), ("D2b 动态取算", "both", 0)})
