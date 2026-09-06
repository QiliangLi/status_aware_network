"""R0/M0 验证报告工具：生成结果文档所需的三张图与误差表。

用法：.venv/bin/python tools/r0_m0_report.py
输出：docs/figures/fig_r0_threshold_sweep.png
      docs/figures/fig_r0_two_request_gantt.png
      docs/figures/fig_r0_analytic_vs_sim.png
并在 stdout 打印可嵌入结果文档的 Markdown 表格。

所有场景参数取纲领 v1.1 §3.1/§3.3 的算例（占位值，非实测）：
R=10s、P=2s、B=4GB、ℓ=0.5s；两请求各 1GB、I/O 1GB/s、重算 0.2s/2s。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.m0 import RecoveryCost, two_request_completion
from sim.m0sim import run_fetch_path, run_recompute_path, run_two_request

# macOS 中文字体；找不到时退回默认字体（图内标签仍可读，仅中文变方框需修字体配置）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

CASE = RecoveryCost(R=10.0, P=2.0, B=4.0, ell=0.5, bw=1.0)
TIMES = (0.2, 2.0)
BYTES_EACH = 1.0
IO_BW = 1.0
FIG_DIR = "docs/figures"

ALLOCATIONS = (
    ("both_fetch", "都取", 2.0),
    ("both_recompute", "都算", 2.2),
    ("fetch_first", "便宜的取、贵的算", 2.0),
    ("fetch_last", "贵的取、便宜的算", 1.0),
)


def fig_threshold_sweep() -> list[tuple[float, float, float]]:
    """图 1：完成时间随取回带宽的变化，解析曲线 + 仿真点，标出交叉带宽 b*。"""
    b_star = CASE.threshold_bandwidth()
    bws = np.linspace(0.25, 1.4, 300)
    fetch_analytic = CASE.ell + CASE.B / bws + CASE.P

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(bws, fetch_analytic, color="#1f77b4",
            label="取回路径 F+P（解析）")
    ax.axhline(CASE.R, color="#d62728", ls="--", label="重算路径 R（解析）")
    ax.axvline(b_star, color="gray", ls=":", label=f"交叉带宽 b*={b_star:.3f} GB/s")

    sim_points = []
    for bw in (0.3, 0.45, 0.6, 0.8, 1.0, 1.3):
        case = RecoveryCost(R=CASE.R, P=CASE.P, B=CASE.B, ell=CASE.ell, bw=bw)
        run = run_fetch_path(case)
        sim_points.append((bw, run.completion, case.fetch_path_time()))
    bws_pts, sim_vals, ana_vals = zip(*sim_points)
    ax.scatter(bws_pts, sim_vals, color="#1f77b4", zorder=5, s=42,
               label="取回路径（仿真）")
    sim_rec = run_recompute_path(CASE).completion
    ax.scatter([bws[-1]], [sim_rec], color="#d62728", zorder=5, s=42, marker="s",
               label="重算路径（仿真）")

    ax.set_xlabel("取回可用带宽 b（GB/s）")
    ax.set_ylabel("完成时间（秒）")
    ax.set_title("单请求取回/重算交叉边界：仿真点落在解析曲线上，交叉于 b*")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_r0_threshold_sweep.png", dpi=150)
    plt.close(fig)
    return sim_points


def fig_two_request_gantt() -> list[tuple[str, float, float, dict[str, float]]]:
    """图 2：两请求四种分配的甘特图（I/O 与 GPU 两资源，串行 FCFS）。"""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.6), sharex=True)
    results = []
    colors = {"io": "#1f77b4", "gpu": "#d62728"}
    y_pos = {"gpu": 0, "io": 1}  # 显式数值映射，避免 barh 类别自动编号与刻度标签错位
    for ax, (alloc, label, expected) in zip(axes.flat, ALLOCATIONS):
        run = run_two_request(TIMES, BYTES_EACH, IO_BW, alloc)
        results.append((alloc, run.completion, expected, run.ledger))
        for s in run.spans:
            ax.barh(y_pos[s.resource], s.duration, left=s.start, height=0.45,
                    color=colors[s.resource], edgecolor="black", linewidth=0.5)
        ax.axvline(run.completion, color="black", ls=":", lw=1)
        ax.axvline(expected, color="green", ls=":", lw=1)
        ax.set_title(f"{label}：仿真 {run.completion:.2f}s（解析 {expected:.1f}s）",
                     fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["GPU", "I/O"])
        ax.set_ylim(-0.55, 1.55)
        ax.set_xlim(0, 2.5)
        ax.grid(alpha=0.3, axis="x")
    for ax in axes[1]:
        ax.set_xlabel("时间（秒）")
    fig.suptitle("两请求分配算例甘特图：黑虚线为仿真完成时间，绿虚线为解析值", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{FIG_DIR}/fig_r0_two_request_gantt.png", dpi=150)
    plt.close(fig)
    return results


def fig_analytic_vs_sim(sim_points, gantt_results) -> float:
    """图 3：解析值 vs 仿真值散点（含重算路径点），返回最大相对误差。"""
    pairs = [(ana, sim) for _, sim, ana in sim_points]
    pairs.append((CASE.R, run_recompute_path(CASE).completion))
    for alloc, sim_completion, expected, _ in gantt_results:
        pairs.append((expected, sim_completion))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    xs, ys = zip(*pairs)
    lim = (0, max(max(xs), max(ys)) * 1.15)
    ax.plot(lim, lim, "k--", lw=1, label="y = x（完全一致）")
    ax.scatter(xs, ys, s=48, color="#1f77b4", zorder=5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("解析值（秒）")
    ax.set_ylabel("仿真值（秒）")
    max_rel = max(abs(y - x) / x for x, y in pairs)
    ax.set_title(f"解析值 vs 仿真值：{len(pairs)} 个场景，最大相对误差 {max_rel:.2e}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_r0_analytic_vs_sim.png", dpi=150)
    plt.close(fig)
    return max_rel


def main() -> None:
    sim_points = fig_threshold_sweep()
    gantt_results = fig_two_request_gantt()
    max_rel = fig_analytic_vs_sim(sim_points, gantt_results)

    print("### 表：阈值扫描仿真点（取回路径）\n")
    print("| 带宽 b（GB/s） | 解析 F+P（秒） | 仿真完成（秒） | 相对误差 |")
    print("|---|---:|---:|---:|")
    for bw, sim_val, ana in sim_points:
        rel = abs(sim_val - ana) / ana
        print(f"| {bw:.2f} | {ana:.4f} | {sim_val:.4f} | {rel:.1e} |")

    print("\n### 表：两请求四种分配\n")
    print("| 分配方式 | 解析完成（秒） | 仿真完成（秒） | I/O 账本（秒） | GPU 账本（秒） |")
    print("|---|---:|---:|---:|---:|")
    io_expected = {"both_fetch": 2.0, "both_recompute": 0.0,
                   "fetch_first": 1.0, "fetch_last": 1.0}
    gpu_expected = {"both_fetch": 0.0, "both_recompute": 2.2,
                    "fetch_first": 2.0, "fetch_last": 0.2}
    for alloc, sim_completion, expected, ledger in gantt_results:
        print(f"| {dict((a, l) for a, l, _ in ALLOCATIONS)[alloc]} | {expected:.1f} "
              f"| {sim_completion:.4f} | {ledger['io']:.2f}（应 {io_expected[alloc]:.1f}） "
              f"| {ledger['gpu']:.2f}（应 {gpu_expected[alloc]:.1f}） |")

    print(f"\n全部场景最大相对误差：{max_rel:.2e}")


if __name__ == "__main__":
    main()
