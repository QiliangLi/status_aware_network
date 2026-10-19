#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentX (SemiAnalysis cc-traces-weka) trace -> Mooncake-style public trace converter.

Input : Weka trace JSON (one object per file, from the HuggingFace dataset
        semianalysisai/cc-traces-weka-* or a local directory of trace files).
Output: Mooncake-style JSONL, one line per request, compatible with the
        Mooncake FAST'25 public trace format and AIPerf's mooncake_trace loader:

        {"timestamp": <int ms>, "input_length": N, "output_length": N,
         "hash_ids": [<int>, ...], "session_id": "..."}

Behavior:
- Flattens main-agent requests and subagent inner requests into one flat stream.
  Subagent requests get session_id "<trace_id>::sa:<agent_id>".
- hash_ids are per-trace local (hash_id_scope: "local", every trace numbers its
  blocks from 0). This script re-keys them into one global namespace so a single
  merged file never aliases content across traces, while intra-trace prefix
  sharing (main-chain continuation, sibling-subagent shared system prefix) is
  preserved exactly.
- --prefix-hit adds "prefix_hit_tokens": ideal prefix-cache hit in tokens
  (number of leading 64-token blocks whose content was already seen earlier in
  the same trace, i.e. an unlimited content-addressed KV cache).
- --keep-model adds the recorded "model" field per request (extra fields are
  allowed by AIPerf's MooncakeTrace schema, extra="allow").

Replay note:
- The corpus uses block_size = 64 tokens (per-trace field). AIPerf's mooncake
  loader defaults to 512, so pass --isl-block-size 64 when replaying:
    aiperf profile --input-file out.jsonl --custom-dataset-type mooncake_trace \
        --isl-block-size 64 --fixed-schedule ...
- Timestamps are per-trace relative (seconds from conversation start, converted
  to ms), matching the Weka format; sessions in one merged file all start near 0.

Usage:
  python convert_agentx_to_mooncake.py --input-dir <dir> -o agentx_mooncake.jsonl [--prefix-hit] [--keep-model]
  python convert_agentx_to_mooncake.py --input-jsonl <traces.jsonl> -o agentx_mooncake_full.jsonl
  python convert_agentx_to_mooncake.py --hf semianalysisai/cc-traces-weka-062126 -o agentx_mooncake.jsonl

--input-jsonl expects one trace object per line (the format of the dataset's
bundled traces.jsonl); it streams line by line so a full 1.8 GB corpus needs
no more memory than one trace at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def iter_traces_from_dir(directory: str):
    for path in sorted(Path(directory).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            yield json.load(f)


def iter_traces_from_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_traces_from_hf(repo_id: str):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("HF mode requires: pip install datasets")
    ds = load_dataset(repo_id, split="train")
    for row in ds:
        yield dict(row)


def collect_entries(trace: dict):
    """Yield (t, seq, session_id, request) in stable time order."""
    tid = trace["id"]
    entries = []
    seq = 0
    for r in trace.get("requests", []):
        rtype = r.get("type")
        if rtype in ("n", "s"):
            entries.append((r["t"], seq, tid, r))
            seq += 1
        elif rtype == "subagent":
            sid = "{}::sa:{}".format(tid, r.get("agent_id", "unknown"))
            for ir in r.get("requests", []):
                entries.append((ir["t"], seq, sid, ir))
                seq += 1
    entries.sort(key=lambda e: (e[0], e[1]))
    return entries


def convert(traces, prefix_hit: bool = False, keep_model: bool = False, emit=None):
    """Convert traces to Mooncake rows.

    emit(row), when given, receives each row as it is produced (streaming,
    constant memory); otherwise all rows are collected and returned, which
    keeps the old return-based behavior for small inputs.
    """
    rows = []
    stats = {
        "traces": 0,
        "requests": 0,
        "subagent_sessions": 0,
        "inconsistent_isl": 0,
        "global_blocks": 0,
    }
    next_global_id = [0]

    for trace in traces:
        stats["traces"] += 1
        tid = trace["id"]
        block_size = trace.get("block_size", 64) or 64

        mapping = {}

        def rekey(ids):
            out = []
            for h in ids:
                nh = mapping.get(h)
                if nh is None:
                    nh = next_global_id[0]
                    mapping[h] = nh
                    next_global_id[0] += 1
                out.append(nh)
            return out

        seen = set()  # content-addressed ideal cache, local id space
        subagent_sessions = set()

        for t, _seq, sid, r in collect_entries(trace):
            if sid != tid:
                subagent_sessions.add(sid)
            ids = r.get("hash_ids", [])
            in_len = r.get("in", 0)
            out_len = r.get("out", 0)

            # consistency: (len-1)*bs < in <= len*bs
            if ids and not ((len(ids) - 1) * block_size < in_len <= len(ids) * block_size):
                stats["inconsistent_isl"] += 1

            hit_blocks = 0
            if prefix_hit:
                for h in ids:
                    if h in seen:
                        hit_blocks += 1
                    else:
                        break
            seen.update(ids)

            row = {
                "timestamp": int(round(t * 1000)),
                "input_length": in_len,
                "output_length": out_len,
                "hash_ids": rekey(ids),
                "session_id": sid,
            }
            if keep_model and r.get("model"):
                row["model"] = r["model"]
            if prefix_hit:
                row["prefix_hit_tokens"] = hit_blocks * block_size
            if emit is not None:
                emit(row)
            else:
                rows.append(row)
            stats["requests"] += 1

        stats["subagent_sessions"] += len(subagent_sessions)

    stats["global_blocks"] = next_global_id[0]
    return rows, stats

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-dir", help="directory of Weka trace JSON files")
    g.add_argument("--input-jsonl", help="JSONL file with one Weka trace object per line (dataset's bundled traces.jsonl)")
    g.add_argument("--hf", help="HuggingFace dataset repo id, e.g. semianalysisai/cc-traces-weka-062126")
    ap.add_argument("-o", "--output", default="agentx_mooncake.jsonl")
    ap.add_argument("--prefix-hit", action="store_true", help="add prefix_hit_tokens field")
    ap.add_argument("--keep-model", action="store_true", help="keep per-request model field")
    args = ap.parse_args()

    if args.input_dir:
        traces = iter_traces_from_dir(args.input_dir)
    elif args.input_jsonl:
        traces = iter_traces_from_jsonl(args.input_jsonl)
    else:
        traces = iter_traces_from_hf(args.hf)

    with open(args.output, "w", encoding="utf-8") as f:
        def emit(row):
            f.write(json.dumps(row) + "\n")
        rows, stats = convert(traces, prefix_hit=args.prefix_hit,
                              keep_model=args.keep_model, emit=emit)
        n_rows = stats["requests"]

    print(json.dumps(stats, indent=2))
    print("wrote {} lines -> {}".format(n_rows, args.output))


if __name__ == "__main__":
    main()
