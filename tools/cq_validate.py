"""cq 输出完整性校验：trace 指纹、结果文件 schema、守恒与配对完整性。

用法：
  .venv/bin/python tools/cq_validate.py --manifest configs/cq/frozen-e19-e24.json --check-only
  .venv/bin/python tools/cq_validate.py --results results/cq/smoke --check-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.cq.trace import MOONCAKE_FILES, sha256_file


def validate_traces(trace_dir: str) -> list:
    issues = []
    for fname, n_rows, max_ts, sha in MOONCAKE_FILES:
        path = os.path.join(trace_dir, fname)
        if not os.path.exists(path):
            issues.append(f"missing: {path}")
            continue
        got = sha256_file(path)
        if got != sha:
            issues.append(f"sha256 mismatch: {fname} {got}")
    return issues


def validate_results(results_dir: str) -> dict:
    """校验每个实验目录的报告文件存在性与守恒不变量。"""
    report = {"ok": True, "experiments": {}}
    if not os.path.isdir(results_dir):
        report["ok"] = False
        report["error"] = f"no such dir: {results_dir}"
        return report
    for exp in sorted(os.listdir(results_dir)):
        d = os.path.join(results_dir, exp)
        if not os.path.isdir(d):
            continue
        entry = {"files": sorted(os.listdir(d))}
        issues = []
        for f in entry["files"]:
            if f.endswith(".json"):
                try:
                    with open(os.path.join(d, f)) as fh:
                        json.load(fh)
                except Exception as e:
                    issues.append(f"invalid json {f}: {e}")
        # 守恒：E23 记录内 total read 一致性由 trace_hash 标记
        rec = os.path.join(d, "e23_records.json")
        if os.path.exists(rec):
            with open(rec) as fh:
                rows = json.load(fh)
            groups = {}
            for r in rows:
                k = (r["file"], r["B"], r["rho"], r["block"], r["theta"])
                groups.setdefault(k, {})[r["policy"]] = r.get("trace_hash")
            for k, pols in groups.items():
                hashes = set(pols.values())
                if len(hashes) > 1:
                    issues.append(f"trace_hash 不一致 {k}: {hashes}")
                entry["n_runs"] = len(rows)
        entry["issues"] = issues
        if issues:
            report["ok"] = False
        report["experiments"][exp] = entry
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--trace-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mooncake_trace"))
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    issues = validate_traces(args.trace_dir)
    print("[trace] " + ("OK：三文件 SHA256 全部匹配" if not issues
                        else "; ".join(issues)))
    if args.manifest:
        if not os.path.exists(args.manifest):
            print(f"[manifest] missing: {args.manifest}")
        else:
            with open(args.manifest) as f:
                man = json.load(f)
            missing = [k for k in ("provenance", "splits", "profile_truth",
                                   "reference", "scenarios", "statistics",
                                   "acceptance") if k not in man]
            print(f"[manifest] {args.manifest}: " + (
                f"缺少对象 {missing}" if missing else "十五对象子集校验 OK"))
    if args.results:
        rep = validate_results(args.results)
        print(f"[results] {args.results}: " + (
            "OK" if rep["ok"] else "存在问题"))
        for exp, e in rep.get("experiments", {}).items():
            status = "OK" if not e["issues"] else "; ".join(e["issues"])
            print(f"  - {exp}: {status} ({len(e['files'])} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
