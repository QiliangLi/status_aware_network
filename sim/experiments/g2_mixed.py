"""G2：混合长短负载（纲领 §7.2 H×U×O 全 8 类）下 D2 排序策略族的收益与代价。

对照：E-FCFS / E-LPM / E-Prio / D2a 就绪 / D2c 预算 / D2d 门控 / D2b / D1a，
路由固定 B-Hist（设计 §6 G2）。必报含 H×U 动作分布与 E-LPM herd 假设检验。
"""
from __future__ import annotations

from .g_common import G2_DEFS, G2_POLICIES, run_grid


def main(seeds, procs=None, duration: float = 300.0, lam: float = 2.2):
    # io（存储瓶颈）/ gpu（计算瓶颈，排序策略的竞争空间）/ loose（无收益负控制）
    run_grid(exp="g2", policies=G2_POLICIES, quadrants=["io", "gpu", "loose"],
             seeds=list(seeds), defs=G2_DEFS, lam=lam, duration=duration,
             out_path="results/g2_mixed.json",
             timeline_marks={("E-FCFS(B-Hist)", "io", 0), ("D2a 就绪排序", "gpu", 0),
                             ("E-LPM", "gpu", 0)})
