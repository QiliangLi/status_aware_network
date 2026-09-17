"""E25 方案对比时序图（图 I 系列）：同一场景下 fcfs/edf/mpc 并排，
列 = NPU0..3 每 NPU 三状态 + 存储带宽占用（GB/s，含 B(t) 参考）。

直接读取已归档的 E25.1 时间序列（ts/<cell_id>/<policy>[_θ].json.gz），
不重跑仿真。覆盖：OL 基线/line_B(B20)/line_rho(ρ0.3, ρ1.1) 与 FW 基线，
各 ×3 trace。同时打印每策略的机制统计（三态占比/峰值利用率/队列），
供报告图注引用。
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_ROOT = os.path.join(REPO, "results", "cq", "eval", "e25", "ts")
FIG_DIR = os.path.join(REPO, "docs", "figures")
WINDOW = 9
ROWS = [("cq_fcfs", "", "FCFS"), ("cq_edf", "", "EDF（排序类代表）"),
        ("cq_mpc", "_S", "MPC（θ=S）")]

CELLS = []
for f, _n, _m, _s in MOONCAKE_FILES:
    CELLS += [
        dict(tag="OLbase", mode="OL", file=f, B=80.0, rho=0.6, cap="C",
             profile="default", title_extra="OL 基线（B=80, ρ=0.6）"),
        dict(tag="OLB20", mode="OL", file=f, B=20.0, rho=0.6, cap="C",
             profile="default", title_extra="OL 带宽受限（B=20, ρ=0.6）"),
        dict(tag="OLr03", mode="OL", file=f, B=80.0, rho=0.3, cap="C",
             profile="default", title_extra="OL 低负载（B=80, ρ=0.3）"),
        dict(tag="OLr11", mode="OL", file=f, B=80.0, rho=1.1, cap="C",
             profile="default", title_extra="OL 过载（B=80, ρ=1.1）"),
        dict(tag="FWbase", mode="FW", file=f, B=80.0, rho=None, cap="C",
             profile="default", title_extra="FW 有限工作集（128 条同时到达）"),
    ]


def cell_id(c):
    rho = "-" if c["rho"] is None else f"{c['rho']:g}"
    stem = c["file"].replace("_trace.jsonl", "").replace(".jsonl", "")
    return (f"{stem}|{c['mode']}|B{c['B']:g}|rho{rho}|a4|{c['cap']}|"
            f"{c['profile']}|w{WINDOW}")


def load_ts(c, pid, suffix):
    p = os.path.join(TS_ROOT, cell_id(c), f"{pid}{suffix}.json.gz")
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def stats(ts):
    rows = [r for r in ts["rows"] if r["width_s"] > 0]
    comp = sum(r["compute_s"] for r in rows)
    stall = sum(r["stall_s"] for r in rows)
    idle = sum(r["idle_s"] for r in rows)
    tot = comp + stall + idle
    utils = [(r["served_gb"] / r["width_s"], r["t_start_s"]) for r in rows
             if r["width_s"] > 0]
    pk_rate, pk_t = max(utils) if utils else (0.0, 0.0)
    b = ts["meta"]["b_schedule"][0][1]
    q = [r["queue_mean"] for r in rows if r["queue_mean"] is not None]
    return dict(comp=comp / tot, stall=stall / tot, idle=idle / tot,
                pk=f"{pk_rate / b:.0%}@{pk_t:.0f}s",
                qmax=max(q) if q else 0)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    made = 0
    for c in CELLS:
        tss = []
        for pid, suffix, label in ROWS:
            ts = load_ts(c, pid, suffix)
            if ts is None:
                break
            tss.append((pid, label, ts))
        if len(tss) < len(ROWS):
            print(f"skip {c['tag']} {c['file'][:12]}: ts 缺失")
            continue
        fig, axes = plt.subplots(len(ROWS), 5,
                                 figsize=(17.5, 3.1 * len(ROWS)),
                                 squeeze=False)
        for ri, (pid, label, ts) in enumerate(tss):
            rows = ts["rows"]
            m = ts["meta"]["m_workers"]
            for wi in range(4):
                ax = axes[ri][wi]
                xs, cc, ss, ii = [], [], [], []
                for r in rows:
                    if r["width_s"] <= 0:
                        continue
                    xs.append(r["t_start_s"])
                    cv, sv, iv = r["workers"][wi]
                    cc.append(cv / r["width_s"])
                    ss.append(sv / r["width_s"])
                    ii.append(iv / r["width_s"])
                ax.stackplot(xs, cc, ss, ii, colors=["#4C72B0", "#DD8452", "#B0B0B0"],
                             alpha=.92)
                ax.set_ylim(0, 1)
                ax.set_yticks([0, .5, 1])
                if ri == 0:
                    ax.set_title(f"NPU{wi}", fontsize=10)
                if wi == 0:
                    ax.set_ylabel(label, fontsize=9)
                if ri == len(ROWS) - 1:
                    ax.set_xlabel("仿真时间 (s)", fontsize=8)
            # 第 5 列：存储带宽占用（GB/s）
            ax = axes[ri][4]
            xs, rate = [], []
            for r in rows:
                if r["width_s"] <= 0:
                    continue
                xs.append(r["t_start_s"])
                rate.append(r["served_gb"] / r["width_s"])
            ax.fill_between(xs, rate, color="#8fb8e0", alpha=.5)
            ax.plot(xs, rate, color="#4C72B0", lw=1.0, label="实际占用")
            b = ts["meta"]["b_schedule"][0][1]
            x_end = max(xs, default=0)
            ax.axhline(float(b), color="k", ls="--", lw=.9)
            ax.text(0.99, float(b), f" B={float(b):g}", color="k", fontsize=7.5,
                    va="bottom", ha="right", transform=ax.get_yaxis_transform())
            ax.set_ylim(0, float(b) * 1.12)
            if ri == 0:
                ax.set_title("存储带宽占用 (GB/s)", fontsize=10)
            if ri == len(ROWS) - 1:
                ax.set_xlabel("仿真时间 (s)", fontsize=8)
        fig.suptitle(f"图I｜方案对比时序：{TRACE_LABEL[c['file']]} · {c['title_extra']} "
                     f"（窗口{WINDOW}；蓝=在算 橙=等数据 灰=闲置；EDF 行代表全部排序类）",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(FIG_DIR, f"cq_fig_e25_cmp_{c['tag']}_{c['file'].split('_')[0]}.png")
        fig.savefig(out, dpi=125)
        plt.close(fig)
        made += 1
        st = {lbl.split("（")[0]: stats(ts) for _p, lbl, ts in tss}
        print(f"{os.path.basename(out)}")
        for k, v in st.items():
            print(f"   {k:5s} 算{v['comp']:.0%} 等{v['stall']:.0%} 闲{v['idle']:.0%} "
                  f"带宽峰{v['pk']} 队列峰{v['qmax']:.0f}")
    print(f"共生成 {made} 张 -> docs/figures")


if __name__ == "__main__":
    main()
