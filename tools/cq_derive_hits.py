"""Mooncake trace 命中预计算工具(变更设计 v1.4)。

算法决策(全部定案,单一实现):
- D1 命中为从头连续前缀,首个未命中块即停;
- D2 同一请求内部重复块不可见(整条处理完才入集合);
- D3 尾块(不满 512)参与判定与入集合,hash_id 为内容指纹(已检验自洽);
- D7 同一 timestamp 并发到达的请求互相不可见(组内视图=组前集合)。

输出:
- mooncake_trace/derived/<原名>.hit.jsonl:原行全部字段 +
  hit_tokens / u_tokens / full_hit_adjusted 三个新字段;
- mooncake_trace/derived/derive_report.json:输入 SHA256、行数、输出 SHA256、
  全量统计(作为新口径验收指纹)。

用法:
  .venv/bin/python tools/cq_derive_hits.py                 # 生成三份派生文件
  .venv/bin/python tools/cq_derive_hits.py --check-only    # 校验已生成文件与报告一致
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.cq.trace import MOONCAKE_FILES, load_mooncake

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(REPO, "mooncake_trace")
DERIVED_DIR = os.path.join(TRACE_DIR, "derived")

BLOCK = 512


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_one(in_path: str, fname: str, out_path: str) -> dict:
    """按 D1/D2/D3/D7 扫描单份 trace 并写派生文件,返回统计。"""
    rows = load_mooncake(in_path, fname)
    seen: set = set()
    out_lines: List[str] = []
    n_full_hit = 0
    tot_input = 0
    tot_hit = 0
    n_hit_req = 0
    i = 0
    N = len(rows)
    while i < N:
        # [i, j) 为同一 timestamp 的并发组(D7)
        j = i
        while j < N and rows[j].timestamp_ms == rows[i].timestamp_ms:
            j += 1
        for r in rows[i:j]:
            n_full = r.input_length // BLOCK
            k = 0
            for hh in r.hash_ids:          # 尾块也参与(D3)
                if hh in seen:
                    k += 1
                else:
                    break                  # 连续前缀,首个未命中即停(D1)
            hit = r.input_length - 1 if k > n_full else min(
                BLOCK * k, r.input_length - 1)
            u = r.input_length - hit
            full = hit >= r.input_length - 1
            rec = {"timestamp": r.timestamp_ms,
                   "input_length": r.input_length,
                   "output_length": r.output_length,
                   "hash_ids": list(r.hash_ids),
                   "hit_tokens": hit,
                   "u_tokens": u,
                   "full_hit_adjusted": bool(full)}
            out_lines.append(json.dumps(rec, separators=(",", ":")))
            tot_input += r.input_length
            tot_hit += hit
            n_full_hit += 1 if full else 0
            n_hit_req += 1 if hit > 0 else 0
        for r in rows[i:j]:                 # 组处理完后统一入集合(D2+D7)
            seen.update(r.hash_ids)
        i = j
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    ts_groups = len({r.timestamp_ms for r in rows})
    max_group = max(
        sum(1 for x in rows if x.timestamp_ms == t)
        for t in {r.timestamp_ms for r in rows})
    return {
        "file": fname, "n_rows": N, "timestamps": ts_groups,
        "max_same_ts_group": max_group,
        "token_hit_ratio": tot_hit / tot_input,
        "request_hit_ratio": n_hit_req / N,
        "full_hit_rows": n_full_hit,
        "input_sha256": _sha256_file(in_path),
        "derived_sha256": _sha256_file(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=TRACE_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.trace_dir, "derived")
    report_path = os.path.join(out_dir, "derive_report.json")

    if args.check_only:
        with open(report_path) as f:
            report = json.load(f)
        ok = True
        for entry in report["files"]:
            p = os.path.join(out_dir, entry["file"] + ".hit.jsonl")
            if not os.path.exists(p):
                print(f"[MISS] {p}")
                ok = False
                continue
            got = _sha256_file(p)
            stat = derive_one(
                os.path.join(args.trace_dir, entry["file"]),
                entry["file"], p + ".tmp")
            os.remove(p + ".tmp")
            match = (got == entry["derived_sha256"]
                     and abs(stat["token_hit_ratio"] - entry["token_hit_ratio"]) < 1e-12
                     and stat["n_rows"] == entry["n_rows"])
            print(f"[{'OK' if match else 'MISMATCH'}] {entry['file']}: "
                  f"rows={stat['n_rows']} token_hit={stat['token_hit_ratio']:.6f}")
            ok = ok and match
        print("check-only:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    files = []
    for fname, _n, _m, _s in MOONCAKE_FILES:
        in_path = os.path.join(args.trace_dir, fname)
        out_path = os.path.join(out_dir, fname + ".hit.jsonl")
        stat = derive_one(in_path, fname, out_path)
        files.append(stat)
        print(f"[DERIVED] {fname}: rows={stat['n_rows']} "
              f"token_hit={stat['token_hit_ratio']:.6f} "
              f"ts_groups={stat['timestamps']} max_group={stat['max_same_ts_group']}")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": 1, "algorithm": "first_seen(D1/D2/D3/D7)",
                   "files": files}, f, ensure_ascii=False, indent=1)
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
