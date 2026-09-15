"""E19：画像与 trace 导入可信度（§9.2）。

三项必做：本地三文件哈希/单位/目录切分/指纹；批形画像（synthetic-v1 真值与
预测）；容量覆盖率（trace-wide 必须 100%）。图 fig_e19_trace_profile.png。
"""
from __future__ import annotations

import os
from fractions import Fraction as F

import numpy as np

from sim.experiments.cq_common import (MECH_LIMITS, TRACE_DIR_DEFAULT,
                                          TRACE_WIDE_LIMITS, MooncakeSource,
                                          c_ref_of, out_dir, print_progress,
                                          save_json, setup_matplotlib,
                                          default_scenario)
from sim.cq.profile import (ProfileConfig, ProfilePredictor, SplitHasher,
                            batch_compute_s, is_feasible, mem_peak_gb)
from sim.cq.trace import MOONCAKE_FILES, TRACE_LABEL
from sim.cq.types import RequestSpec


def import_report(trace_dir: str) -> dict:
    """三文件导入验收:first_seen 派生统计 + 原始 SHA 核对(变更设计 v1.4)。"""
    import sim.cq.trace as T
    report = {}
    for fname, n_rows, max_ts, sha in MOONCAKE_FILES:
        src = MooncakeSource(fname, trace_dir)
        imp = src.imp
        hs_a = np.array([r.hit_tokens for r in imp.rows])
        us_a = np.array([r.u_tokens for r in imp.rows])
        prof = ProfileConfig()
        specs = [RequestSpec(i, 0, hs_a[i], us_a[i], "mooncake", 1, 1)
                 for i in range(len(hs_a))]
        wide_ok = all(is_feasible([s], TRACE_WIDE_LIMITS, prof) for s in specs)
        mech_ok = sum(1 for s in specs if is_feasible([s], MECH_LIMITS, prof))
        arr_ms = np.array([r.timestamp_ms for r in imp.rows], dtype=float)
        inter = np.diff(np.sort(arr_ms)) / 1000.0
        ts_groups = len(set(arr_ms.tolist()))
        report[fname] = {
            "label": TRACE_LABEL[fname], "n_rows": n_rows, "max_ts_ms": max_ts,
            "input_sha256_ok": T.sha256_file(os.path.join(trace_dir, fname)) == sha,
            "derived_sha256": imp.derived_sha256[:16],
            "token_hit_ratio": imp.token_hit_ratio,
            "request_hit_ratio": imp.request_hit_ratio,
            "full_hit_adjusted": imp.full_hit_adjusted,
            "max_u": int(us_a.max()),
            "h_median": float(np.median(hs_a)),
            "u_p50": float(np.median(us_a)),
            "u_p95": float(np.percentile(us_a, 95)),
            "interarrival_p50_s": float(np.median(inter)) if len(inter) else None,
            "ts_groups": ts_groups,
            "max_same_ts_group": int(max(
                sum(1 for x in imp.rows if x.timestamp_ms == t)
                for t in set(r.timestamp_ms for r in imp.rows))),
            "coverage_wide_100pct": wide_ok,
            "coverage_mech_frac": mech_ok / max(1, len(specs)),
            "lam0_full_trace": float(src.lam0),
        }
    return report


def profile_report() -> dict:
    """synthetic-v1 批形画像 + 留出预测（合成真值直接查询，误差=0）。"""
    prof = ProfileConfig()
    classes = [("HS_US", 512, 128), ("HS_UL", 512, 2048),
               ("HL_US", 8192, 128), ("HL_UL", 8192, 2048)]
    shapes = {}
    # n∈{1,2,4,8} 的全部可行多重集 + 4 个校验点
    from itertools import combinations_with_replacement
    test_keys = []
    for n in (1, 2, 4, 8):
        for combo in combinations_with_replacement(range(4), n):
            hu = tuple(sorted((classes[i][1], classes[i][2]) for i in combo))
            test_keys.append(hu)
    for hu in [(2048, 128)]:
        test_keys.append(((hu),))
    pred = ProfilePredictor(cfg=prof)
    train_shapes = {}
    errs, splits = [], []
    for hu in test_keys:
        specs = [RequestSpec(i, 0, h, u, "x", 1, 1) for i, (h, u) in enumerate(hu)]
        if not is_feasible(specs, MECH_LIMITS, prof):
            continue
        c_true = float(batch_compute_s(specs, prof))
        split = SplitHasher.split_of(hu)
        splits.append(split)
        if split != 0:
            train_shapes[hu] = F(batch_compute_s(specs, prof))
    pred.fit(train_shapes)
    n_hold, errs = 0, []
    for hu in test_keys:
        specs = [RequestSpec(i, 0, h, u, "x", 1, 1) for i, (h, u) in enumerate(hu)]
        if not is_feasible(specs, MECH_LIMITS, prof):
            continue
        if SplitHasher.split_of(hu) != 0:
            continue
        n_hold += 1
        c_true = float(batch_compute_s(specs, prof))
        c_hat, exact, dist = pred.predict(list(hu))
        errs.append(abs(c_hat - c_true) / c_true)
    g_curves = {}
    for n in (1, 2, 4, 8):
        for cls in classes:
            specs = [RequestSpec(i, 0, cls[1], cls[2], "x", 1, 1)
                     for i in range(n)]
            if not is_feasible(specs, MECH_LIMITS, prof):
                continue
            sk = sum(float(batch_compute_s([s], prof)) for s in specs)
            bk = float(batch_compute_s(specs, prof))
            g_curves[f"{cls[0]}x{n}"] = {"G": sk / bk if bk > 0 else None,
                                         "mem_peak_gb": float(mem_peak_gb(specs, prof))}
    return {
        "n_train_shapes": len(train_shapes), "n_holdout": n_hold,
        "holdout_ape_median": float(np.median(errs)) if errs else None,
        "holdout_ape_p95": float(np.percentile(errs, 95)) if errs else None,
        "holdout_err_is_formula_query_zero": bool(
            all(e == 0.0 for e in errs)) if errs else None,
        "g_curves": g_curves,
        "hardware_status": "not_available",
    }


def main(seeds, procs=None, duration=150.0, stage="smoke",
         trace_dir=TRACE_DIR_DEFAULT, **kw):
    plt = setup_matplotlib()
    d = out_dir(stage, "e19")
    rep_imp = import_report(trace_dir)
    rep_prof = profile_report()
    save_json(os.path.join(d, "e19_report.json"),
              {"import": rep_imp, "profile": rep_prof})
    # 图：三文件到达间隔 / h-u 联合 / 命中比 + G 曲线
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    labels, hit_r, hit_t, lam0s = [], [], [], []
    for j, (fname, _n, _m, _s) in enumerate(MOONCAKE_FILES):
        src = MooncakeSource(fname, trace_dir)
        rows = src.imp.rows
        arr = np.array([r.timestamp_ms for r in rows], dtype=float) / 1000.0
        inter = np.diff(np.sort(arr))
        axes[0, j].hist(np.clip(inter, 0, np.percentile(inter, 99)), bins=60,
                        color="#4C72B0", alpha=.8)
        axes[0, j].set_title(f"{TRACE_LABEL[fname]} 到达间隔 (s)")
        axes[0, j].set_yscale("log")
        hus = np.array([src.imp.h_u_of(r)[:2] for r in rows])
        axes[1, j].scatter(np.log10(hus[:, 0] + 1), np.log10(hus[:, 1] + 1),
                           s=2, alpha=.15, color="#DD8452")
        axes[1, j].set_xlabel("log10(h+1)"); axes[1, j].set_ylabel("log10(u+1)")
        labels.append(TRACE_LABEL[fname])
        hit_r.append(rep_imp[fname]["request_hit_ratio"])
        hit_t.append(rep_imp[fname]["token_hit_ratio"])
        lam0s.append(rep_imp[fname]["lam0_full_trace"])
    fig.suptitle("E19：Mooncake 三文件导入（命中比 req/tok：" +
                 " / ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(hit_r, hit_t)) +
                 f"；λ0：" + " / ".join(f"{x:.1f}" for x in lam0s) + " req/s）")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_e19_trace_profile.png"), dpi=130)
    plt.close(fig)
    # G 曲线单独一张（按批人数的计算加速）
    fig2, ax = plt.subplots(figsize=(7, 4.2))
    gs = rep_prof["g_curves"]
    for cls in ("HS_US", "HS_UL", "HL_US", "HL_UL"):
        xs, ys = [], []
        for n in (1, 2, 4, 8):
            k = f"{cls}x{n}"
            if k in gs and gs[k]["G"]:
                xs.append(n); ys.append(gs[k]["G"])
        ax.plot(xs, ys, marker="o", label=cls)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xlabel("批人数 n"); ax.set_ylabel("G_batch（计算节省比）")
    ax.set_title("synthetic-v1 组批计算曲线（B_ref=80）")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(d, "fig_e19_g_curve.png"), dpi=130)
    plt.close(fig2)
    print_progress(f"E19 done -> {d}")
    return {"import": rep_imp, "profile": rep_prof}


if __name__ == "__main__":
    main([0])
