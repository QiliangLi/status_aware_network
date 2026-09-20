"""E25 trace 特征图（图 T）：三份 trace 的请求形状与到达结构对比。

四个面板：
- 左上：u_tokens 分布（对数 x 轴直方图）——请求要新算的 token 数
- 右上：h_tokens 分布——命中前缀长度（要读回的 KV 量）
- 左下：T0 分布（对数 x 轴）——独占参考时间
- 右下：相邻到达间隔分布（对数 x 轴）——到达节奏

每面板三条线=三份 trace，颜色一致。
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "docs", "figures")
DPI = 125

TCOL = {"conversation_trace.jsonl": "#4C72B0",
        "toolagent_trace.jsonl": "#DD8452",
        "synthetic_trace.jsonl": "#55A868"}
TLAB = {"conversation_trace.jsonl": "Conversation",
        "toolagent_trace.jsonl": "ToolAgent",
        "synthetic_trace.jsonl": "Synthetic"}


def main():
    from sim.cq.trace import MOONCAKE_FILES
    from sim.experiments.cq_common import MooncakeSource, TRACE_DIR_DEFAULT
    from sim.cq.config import ProfileConfig
    from sim.cq.profile import singleton_K
    from sim.cq.types import RequestSpec
    from fractions import Fraction as F

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from e25_ts_compare import audit_text_overlap

    prof = ProfileConfig()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), dpi=DPI)

    for fname, _n, _s, _h in MOONCAKE_FILES:
        src = MooncakeSource(fname, TRACE_DIR_DEFAULT)
        us = np.array([r.u_tokens for r in src.imp.rows], dtype=float)
        hs = np.array([r.hit_tokens for r in src.imp.rows], dtype=float)
        ks = np.array([float(singleton_K(
            RequestSpec(0, 0, int(h), int(u), "x", 1, 1), prof, F(80), F(200)))
            for h, u in zip(hs, us)])
        gaps = np.diff(np.array(
            sorted(r.timestamp_ms for r in src.imp.rows), dtype=float)) / 1000.0
        c = TCOL[fname]
        lab = TLAB[fname]

        axes[0][0].hist(np.log10(us[us > 0] + 1), bins=60, alpha=.55,
                        color=c, label=lab, density=True)
        axes[0][1].hist(np.log10(hs + 1), bins=60, alpha=.55, color=c,
                        label=lab, density=True)
        axes[1][0].hist(np.log10(ks[ks > 0] + 0.001), bins=60, alpha=.55,
                        color=c, label=lab, density=True)
        axes[1][1].hist(np.log10(gaps[gaps > 0] + 0.001), bins=60, alpha=.55,
                        color=c, label=lab, density=True)

    axes[0][0].set_xlabel("log10(u_tokens + 1)　新算 token 数")
    axes[0][0].set_title("u_tokens 分布（要新算的量）")
    axes[0][0].legend(fontsize=10)

    axes[0][1].set_xlabel("log10(h_tokens + 1)　命中前缀长度")
    axes[0][1].set_title("h_tokens 分布（要读回的 KV 量）")
    axes[0][1].legend(fontsize=10)

    axes[1][0].set_xlabel("log10(T0 + 0.001)　独占参考时间 (s)")
    axes[1][0].set_title("T0 分布（请求耗时）")
    axes[1][0].legend(fontsize=10)

    axes[1][1].set_xlabel("log10(gap + 0.001)　相邻到达间隔 (s)")
    axes[1][1].set_title("相邻到达间隔分布（到达节奏）")
    axes[1][1].legend(fontsize=10)

    for ax in axes.flat:
        ax.set_ylabel("密度")
        ax.grid(alpha=.25)

    fig.suptitle("图T｜三份 Mooncake trace 的请求形状与到达结构对比（全文件）",
                 fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG, "cq_fig_e25_trace_profile.png")
    fig.savefig(out, dpi=DPI)
    ov = audit_text_overlap(fig)
    plt.close(fig)
    print(f"图T -> {os.path.basename(out)}  [文字重叠审计: "
          f"{'OK' if not ov else ov[:3]}]")


if __name__ == "__main__":
    main()
