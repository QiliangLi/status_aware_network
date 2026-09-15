"""CLI 入口。

用法：
  python -m sim.run --exp all --seeds 10 --procs 7      # v1: E1-E4
  python -m sim.run --exp v2  --seeds 8                 # v2: E5-E9（共享分布式 KV 存储拓扑）
  python -m sim.run --exp v3  --seeds 8                 # v3: 改进方案实验 E9b/E10/E11/E12
  python -m sim.run --exp e5 --seeds 3 --duration 200   # 单实验快速冒烟
"""
from __future__ import annotations

import argparse

_MODULES = {
    "e1a": ("e1_sweep", "main", {"exps": ("e1a",)}),
    "e1b": ("e1_sweep", "main", {"exps": ("e1b",)}),
    "e2": ("e2_routing", "main", {}),
    "e3": ("e3_signal", "main", {}),
    "e4": ("e4_burst", "main", {}),
    "e5": ("e5_routing", "main", {}),
    "e6": ("e6_replica", "main", {}),
    "e7": ("e7_partial", "main", {}),
    "e8": ("e8_replicate", "main", {}),
    "e9": ("e9_coord", "main", {}),
    "e9b": ("e9b_clairvoyant", "main", {}),
    "e10": ("e10_prefetch", "main", {}),
    "e11": ("e11_tiering", "main", {}),
    "e12": ("e12_gpu_signal", "main", {}),
    "e16": ("e16_params", "main", {}),
    "e17": ("e17_session_mech", "main", {}),
    "e18": ("e18_protect_fluid", "main", {}),
    "e15": ("e15_prefetch_pred", "main", {}),
    "e14": ("e14_clair_cap", "main", {}),
    "e13": ("e13_engine_ctrl", "main", {}),
    "g1": ("g1_quadrant", "main", {}),
    "g2": ("g2_mixed", "main", {}),
}


def _run_exp(exp: str, seeds, procs, duration: float):
    import importlib
    mod_name, fn, extra = _MODULES[exp]
    mod = importlib.import_module(f".experiments.{mod_name}", package="sim")
    getattr(mod, fn)(seeds, procs, duration=duration, **extra)


# cq 实验族（E19–E24）：共享 KV 统一队列 Batch 与错峰调度（规格 20260915）
_CQ_MODULES = {
    "e19": ("e19_profile", "main"),
    "e20": ("e20_exact", "main"),
    "e21": ("e21_batch", "main"),
    "e22": ("e22_stagger", "main"),
    "e23": ("e23_trace", "main"),
    "e24": ("e24_robustness", "main"),
    "e25": ("e25_factor", "main"),
}


def _run_cq(exp: str, args, seeds):
    import importlib
    mod_name, fn = _CQ_MODULES[exp]
    mod = importlib.import_module(f".experiments.{mod_name}", package="sim")
    getattr(mod, fn)(seeds, args.procs, duration=args.duration,
                     stage=args.stage, trace_dir=args.trace_dir,
                     manifest=args.manifest, seed_start=args.seed_start,
                     out_dir_root=args.out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all", "v2", "v3", "v4", "all2", *sorted(_MODULES),
                             *sorted(_CQ_MODULES), "cq",
                             "smoke", "smoke2", "smokeg", "smokecq"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--duration", type=float, default=400.0)
    # cq 实验族新增参数（仅 cq 实验消费；旧实验签名不变）
    ap.add_argument("--stage", default="smoke", choices=["smoke", "train", "eval"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    exps = (["e1a", "e1b", "e2", "e3", "e4"] if args.exp in ("all", "all2")
            else ["e5", "e6", "e7", "e8", "e9"] if args.exp == "v2" else
            ["e9b", "e10", "e11", "e12"] if args.exp == "v3"
            else ["e13", "e14", "e15", "e16", "e17", "e18"] if args.exp == "v4"
            else ["e5", "e6", "e7", "e8", "e9", "e9b", "e10", "e11", "e12", "e13", "e14", "e15", "e16"] if args.exp == "all2"
            else [args.exp])
    if args.exp == "smoke":
        exps, seeds, args.duration = ["e1a"], seeds[:2], 120.0
    if args.exp == "smoke2":
        exps, seeds, args.duration = ["e5"], seeds[:2], 120.0
    if args.exp == "smokeg":
        exps, seeds, args.duration = ["g1"], seeds[:2], 150.0
    if args.exp == "smokecq":
        exps, seeds, args.duration = (["e19", "e20"], seeds[:2], 150.0)
    if args.exp == "cq":
        exps = list(_CQ_MODULES)

    if args.exp in _CQ_MODULES or args.exp in ("cq", "smokecq"):
        import os
        if args.trace_dir is None:
            args.trace_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "mooncake_trace")
        if args.stage != "smoke" and not args.manifest:
            ap.error("train/eval 阶段必须提供 --manifest（仅 smoke 可无 manifest）")
        for exp in exps:
            _run_cq(exp, args, seeds)
        return

    for exp in exps:
        _run_exp(exp, seeds, args.procs, args.duration)


if __name__ == "__main__":
    main()
