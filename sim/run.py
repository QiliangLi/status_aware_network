"""CLI 入口。

用法：
  python -m sim.run --exp all --seeds 10 --procs 7      # v1: E1-E4
  python -m sim.run --exp v2  --seeds 8                 # v2: E5-E9（共享分布式 KV 存储拓扑）
  python -m sim.run --exp e5 --seeds 3 --duration 200   # 单实验快速冒烟
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all", "v2", "all2", "e1a", "e1b", "e2", "e3", "e4",
                             "e5", "e6", "e7", "e8", "e9", "smoke", "smoke2"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--duration", type=float, default=400.0)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    exps = (["e1a", "e1b", "e2", "e3", "e4"] if args.exp in ("all", "all2")
            else ["e5", "e6", "e7", "e8", "e9"] if args.exp in ("v2", "all2")
            else [args.exp])
    if args.exp == "smoke":
        exps, seeds, args.duration = ["e1a"], seeds[:2], 120.0
    if args.exp == "smoke2":
        exps, seeds, args.duration = ["e5"], seeds[:2], 120.0

    from .experiments import (e1_sweep, e2_routing, e3_signal, e4_burst,
                              e5_routing, e6_replica, e7_partial, e8_replicate, e9_coord)
    for exp in exps:
        if exp == "e1a":
            e1_sweep.main(seeds, args.procs, duration=args.duration, exps=("e1a",))
        elif exp == "e1b":
            e1_sweep.main(seeds, args.procs, duration=args.duration, exps=("e1b",))
        elif exp == "e2":
            e2_routing.main(seeds, args.procs, duration=args.duration)
        elif exp == "e3":
            e3_signal.main(seeds, args.procs, duration=args.duration)
        elif exp == "e4":
            e4_burst.main(seeds)
        elif exp == "e5":
            e5_routing.main(seeds, args.procs, duration=args.duration)
        elif exp == "e6":
            e6_replica.main(seeds, args.procs, duration=args.duration)
        elif exp == "e7":
            e7_partial.main(seeds, args.procs, duration=args.duration)
        elif exp == "e8":
            e8_replicate.main(seeds, args.procs, duration=args.duration)
        elif exp == "e9":
            e9_coord.main(seeds, args.procs, duration=args.duration)


if __name__ == "__main__":
    main()
