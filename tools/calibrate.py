"""标定脚手架（改进方案问题①）：用真实系统测量值反推仿真参数。

三个子命令（均需可达的真实服务；仅输出框架，数值口径见 docstring）：
  lmcache  — 轮询 LMCache /metrics，统计 L1<->L2 端到端吞吐与 inflight，
             反推各 tier 的 b_total / t_base（拟合并发-吞吐曲线，检验 PS 假设；
             若并发升高而单流吞吐不降，需为 SharedKVStorage 启用 max_active 模型）。
  prefill  — 对 vLLM OpenAI 兼容服务发不同长度 prompt，测 TTFT 拟合 prefill_table
             （需独占 GPU；带背景负载模式用于验证 reserved-capacity 假设）。
  rdma     — 节点间带宽/延迟打点，填 TopoConfig.fabric / path_lat。

输出 YAML 片段，人工确认后填入 sim/config.py。用法：
  python tools/calibrate.py lmcache --url http://host:8000/metrics --duration 60
  python tools/calibrate.py prefill --url http://host:8000/v1 --model qwen
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def _get(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def scrape_prom(text: str) -> dict:
    """极简 Prometheus 文本解析：name{...} value -> [(labels, value)]。"""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        try:
            val = float(rest.strip())
        except ValueError:
            continue
        out.setdefault(name, []).append(val)
    return out


def cmd_lmcache(args):
    t0 = time.time()
    series = []
    print(f"polling {args.url} every {args.interval}s for {args.duration}s ...")
    while time.time() - t0 < args.duration:
        try:
            m = scrape_prom(_get(args.url))
        except Exception as e:  # noqa: BLE001
            print("  scrape failed:", e)
        else:
            series.append(m)
            keys = [k for k in m if "l2" in k.lower() or "throughput" in k.lower()]
            print("  " + "  ".join(f"{k}={m[k][-1]:.1f}" for k in keys[:6]))
        time.sleep(args.interval)
    if not series:
        return
    # TODO(标定): 1) 用受控并发负载扫 (并发 N, 单流吞吐) 曲线
    #             2) PS 假设检验：吞吐 ~ min(B/N_total, B/K_active)
    #             3) 最小二乘拟合 b_total / t_base / max_active，输出 YAML
    print(json.dumps({"samples": len(series),
                      "note": "scaffold: fit b_total/t_base/max_active from controlled sweep"}, indent=2))


def cmd_prefill(args):
    for n in args.tokens:
        prompt = "data " * (n // 5)
        payload = json.dumps({"model": args.model, "prompt": prompt, "max_tokens": 1}).encode()
        req = urllib.request.Request(args.url + "/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
        t = time.time()
        try:
            urllib.request.urlopen(req, timeout=120).read()
            print(f"  tokens={n:6d}  ttft={time.time() - t:.3f}s")
        except Exception as e:  # noqa: BLE001
            print(f"  tokens={n:6d}  FAILED: {e}")
    print("note: scaffold -> fill GpuConfig.prefill_table with (tokens, ttft) pairs")


def cmd_rdma(args):
    print("scaffold: use qperf/ib_write_bw between compute and storage nodes;")
    print("fill TopoConfig.fabric.b_total and path_lat matrix; measure under representative load.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("lmcache")
    p1.add_argument("--url", required=True)
    p1.add_argument("--duration", type=float, default=60.0)
    p1.add_argument("--interval", type=float, default=1.0)
    p1.set_defaults(fn=cmd_lmcache)
    p2 = sub.add_parser("prefill")
    p2.add_argument("--url", required=True)
    p2.add_argument("--model", required=True)
    p2.add_argument("--tokens", type=int, nargs="+",
                    default=[4096, 8192, 16384, 32768, 65536])
    p2.set_defaults(fn=cmd_prefill)
    p3 = sub.add_parser("rdma")
    p3.set_defaults(fn=cmd_rdma)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
