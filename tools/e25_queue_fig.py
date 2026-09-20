"""E25 队列深度对比图（图 Q）：直接展示 MPC 改善 TTFT/SLO 的机制。

甘特图展示资源结构（谁在算/等/闲），但 TTFT/SLO 是请求级结果——
因果链是：更精的批组合 → 清积压更快（队列更浅）→ 请求等得更短 →
更多请求赶上 deadline。图 Q 把这条链的第二环可视化：
同一场景下 FCFS / EDF / MPC 的队列深度随时间的叠加对比。

另附第二面板：累积完成曲线（cum_done）——MPC 的曲线更早抬升
= 更多请求更早完成 = TTFT 更短、SLO 更高。

数据源：归档时间序列（w9），不重跑仿真。
"""
from __future__ import annotations

import gzip
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_ROOT = os.path.join(REPO, "results", "cq", "eval", "e25", "ts")
FIG = os.path.join(REPO, "docs", "figures")
DPI = 125

POL_STYLE = {
    "cq_fcfs":  ("#555555", "-",  "FCFS"),
    "cq_edf":   ("#DD8452", "--", "EDF"),
    "cq_mpc_S": ("#4C72B0", "-",  "MPC θ=S"),
}

SCENES = [
    ("conversation_trace.jsonl", "OL", 80.0, 0.6,
     "Conversation 基线（B=80, ρ=0.6）", "OLbase_conversation"),
    ("toolagent_trace.jsonl", "OL", 80.0, 0.3,
     "ToolAgent 低负载（B=80, ρ=0.3）★TTFT 最优格", "OLr03_toolagent"),
]


def load(cid, pid, suffix=""):
    p = os.path.join(TS_ROOT, cid, f"{pid}{suffix}.json.gz")
    if not os.path.exists(p):
        return None
    return json.load(gzip.open(p, "rt", encoding="utf-8"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    from e25_ts_compare import audit_text_overlap

    for fname, mode, B, rho, title, tag in SCENES:
        stem = fname.replace("_trace.jsonl", "").replace(".jsonl", "")
        cid = f"{stem}|{mode}|B{B:g}|rho{rho:g}|a4|C|default|w9"
        fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.5), dpi=DPI,
                                 sharex=True)
        for pid, suf in [("cq_fcfs", ""), ("cq_edf", ""), ("cq_mpc", "_S")]:
            ts = load(cid, pid, suf)
            if ts is None:
                continue
            color, ls, label = POL_STYLE[f"{pid}{suf}"]
            rows = [r for r in ts["rows"] if r["width_s"] > 0]
            xs = [r["t_start_s"] for r in rows]
            # 面板 1：队列深度
            q = [r["queue_mean"] if r["queue_mean"] is not None else 0
                 for r in rows]
            axes[0].plot(xs, q, color=color, ls=ls, lw=1.6, label=label)
            axes[0].fill_between(xs, q, alpha=.08, color=color)
            # 面板 2：累积完成数
            done = [r["cum_done"] for r in rows if "cum_done" in r]
            if done:
                axes[1].plot(xs[:len(done)], done, color=color, ls=ls,
                             lw=1.6, label=label)

        axes[0].set_ylabel("队列深度（等待+执行中的请求数）", fontsize=12)
        axes[0].legend(fontsize=11, loc="upper right")
        axes[0].grid(alpha=.25)
        axes[0].set_title(f"图Q｜{title}：队列深度与累积完成对比（窗口9；"
                          f"FCFS/EDF/MPC θ=S）", fontsize=13)

        axes[1].set_ylabel("累积完成请求数", fontsize=12)
        axes[1].set_xlabel("仿真时间 (s)", fontsize=11)
        axes[1].legend(fontsize=11, loc="lower right")
        axes[1].grid(alpha=.25)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = os.path.join(FIG, f"cq_fig_e25_queue_{tag}.png")
        fig.savefig(out, dpi=DPI)
        ov = audit_text_overlap(fig)
        plt.close(fig)
        print(f"图Q -> {os.path.basename(out)}  [文字重叠审计: "
              f"{'OK' if not ov else ov[:3]}]")


if __name__ == "__main__":
    main()
