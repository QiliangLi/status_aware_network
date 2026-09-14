"""cq 报告工具：从结果目录生成汇总表，并可显式发布正式图到 docs/figures。

默认图留 results；--publish-figures 复制已核验的 eval 图到
docs/figures/fig_e19..e24_*.png（不覆盖旧正式图，新文件名加 cq 前缀区分）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PANELS = {
    "e19": ["fig_e19_trace_profile.png", "fig_e19_g_curve.png"],
    "e20": ["fig_e20_four_plan.png", "fig_e20_gap_runtime.png"],
    "e21": ["fig_e21_batch_tradeoff.png"],
    "e22": ["fig_e22_timeline.png", "fig_e22_wait_decomposition.png"],
    "e23": ["fig_e23_gain_overview.png", "fig_e23_slo_load.png"],
    "e24": ["fig_e24_robustness.png"],
}


def summarize_results(results_dir: str, stage: str) -> str:
    lines = [f"# cq {stage} 结果汇总\n"]
    for exp in sorted(PANELS):
        d = os.path.join(results_dir, stage, exp)
        if not os.path.isdir(d):
            continue
        lines.append(f"## {exp}\n")
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                lines.append(f"- 数据：`{f}`")
            elif f.endswith(".png"):
                lines.append(f"- 图：`{f}`")
        # E23 headline 表
        rec = os.path.join(d, "e23_records.json")
        if os.path.exists(rec):
            with open(rec) as fh:
                rows = json.load(fh)
            from collections import defaultdict
            import numpy as np
            agg = defaultdict(list)
            for r in rows:
                agg[r["policy"]].append(r)
            lines.append("\n| 策略 | SLO 满足率(均值) | TTFT 下界均值(s) | runs |")
            lines.append("|---|---:|---:|---:|")
            for p, rs in sorted(agg.items()):
                slo = np.mean([r["slo_success"] / max(1, r["n_cohort"])
                               for r in rs])
                ttft = np.mean([r["ttft_mean_lower"] for r in rs
                                if r.get("ttft_mean_lower") is not None])
                lines.append(f"| {p} | {slo:.3f} | {ttft:.3f} | {len(rs)} |")
        lines.append("")
    return "\n".join(lines)


def publish_figures(results_dir: str, stage: str, out_root: str):
    """复制已核验的图到 docs/figures（新名 cq_ 前缀，不覆盖旧正式图）。"""
    dest_root = os.path.join(out_root, "docs", "figures")
    os.makedirs(dest_root, exist_ok=True)
    copied = []
    for exp, figs in PANELS.items():
        d = os.path.join(results_dir, stage, exp)
        for f in figs:
            src = os.path.join(d, f)
            if os.path.exists(src):
                dst = os.path.join(dest_root, "cq_" + f)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copy(src, dst)
                copied.append("cq_" + f)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--results", default=os.path.join(REPO, "results", "cq"))
    ap.add_argument("--stage", default="smoke")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--publish-figures", action="store_true")
    ap.add_argument("--out-dir", default=REPO)
    args = ap.parse_args()
    if args.check_only:
        rep = summarize_results(args.results, args.stage)
        out = os.path.join(args.results, args.stage, "report_summary.md")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            f.write(rep)
        print(rep)
        print(f"[check-only] 摘要写入 {out}；图未发布")
        return 0
    if args.publish_figures:
        copied = publish_figures(args.results, args.stage, args.out_dir)
        print("[publish] " + "; ".join(copied) if copied else "[publish] 无图可复制")
    return 0


if __name__ == "__main__":
    sys.exit(main())
