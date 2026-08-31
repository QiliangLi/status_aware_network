"""CLI 入口。

用法：
  python -m sim.run --exp all --seeds 10 --procs 7
  python -m sim.run --exp e1a --seeds 3 --duration 200   # 快速冒烟
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all", "e1a", "e1b", "e2", "e3", "e4", "smoke"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=None)
    ap.add_argument("--duration", type=float, default=400.0)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    exps = ["e1a", "e1b", "e2", "e3", "e4"] if args.exp == "all" else [args.exp]
    if args.exp == "smoke":
        exps, seeds, args.duration = ["e1a"], seeds[:2], 120.0

    from .experiments import e1_sweep, e2_routing, e3_signal, e4_burst
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


if __name__ == "__main__":
    main()
