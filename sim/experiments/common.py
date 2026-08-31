"""实验公共：并行运行、结果落盘、绘图样式。"""
from __future__ import annotations

import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")


def run_pool(jobs, procs=None):
    from ..simrun import _run_job
    import multiprocessing as mp
    procs = procs or max(1, (os.cpu_count() or 4) - 1)
    if procs <= 1 or len(jobs) <= 2:
        return [_run_job(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs) as pool:
        return pool.map(_run_job, jobs)


def save_rows(rows, exp: str) -> str:
    import pandas as pd
    out = os.path.join(RESULTS_DIR, exp, "summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 10,
    })
    return plt


def savefig(plt, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[fig] {path}")
