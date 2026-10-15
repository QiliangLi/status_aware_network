"""TFS 收益场景在真实 trace 中的比重分析(docs/TFS规则式调度器-20261014.md §2.1)。

收益场景四条件的逐 trace 实测:
  S1 带宽分诊区(重流聚合需求>供给):逐 1s 窗口到达读取速率 vs B 扫描;
     临界带宽 B*=全程读取量/时长;报窗口超订占比与涉及请求占比;
  S2 簇到达:1s/50ms 窗口到达数分布(峰/均值)与 ≥k 簇的请求占比;
  S3 零读/轻类:h=0 或 d<0.1·d_heavy 的请求占比;重流读取量集中度
     (前 25% d 的请求占总读取量的份额);
  S4 计算饱和:逐窗口计算需求(NPU·s/s)vs m=32(E26 口径)。

口径说明:
- Mooncake 三文件:derived 首现命中(sim/cq/trace.py 口径);
- AgentX 样本:prefix_hit_tokens 理想前缀命中(AGENTS §9,与首现口径不同,
  结论不可互推);u=max(1, input−hit)(与 Mooncake full_hit_adjusted 同规则);
- c/V 用默认 ProfileConfig(L=32,与 E23 trace 实验一致);d=κh/c(层口径);
- AgentX 时间戳为会话内相对时间(两 session 近 0 起点),窗口指标报两种
  叠放口径:overlap(原样,突发上界)与 sequential(顺次拼接,突发下界)。

用法:.venv/bin/python tools/tfs_trace_gain.py [--json OUT]
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from fractions import Fraction as F

import numpy as np

sys.path.insert(0, os.getcwd())

from sim.cq.config import ProfileConfig
from sim.cq.trace import MOONCAKE_FILES, import_mooncake

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(REPO, "mooncake_trace")
AGENTX_PATH = os.path.join(REPO, "AgentX及trace转换分析", "转换结果",
                           "agentx_mooncake_sample.jsonl")
B_SWEEP = (30.0, 60.0, 80.0, 120.0, 200.0)
M_NPU = 32                       # E26 口径
CLUSTER_K = 8                    # E26 簇规模参照(16 的一半,50ms 内)
WIN_S = 1.0
WIN_MS = 0.05

PROF = ProfileConfig()
KAPPA = float(PROF.kappa_gb_per_token_layer)
L = PROF.L


def _c_s(h: int, u: int) -> float:
    """默认画像的 singleton 每层计算(浮点,与 bw_demand 同式)。"""
    eta = 1.0 if u >= PROF.N_sat else max(float(PROF.eta_floor), u / PROF.N_sat)
    A = u * h + u * (u + 1) / 2
    return (float(PROF.t_launch_s) + float(PROF.a_s_per_token) * u / eta
            + float(PROF.b_s_per_pair) * A)


def load_rows(kind: str, ident: str):
    """返回 [(t_s, h, u)],已按时间排序。kind: 'mc' | 'ax'。"""
    if kind == "mc":
        imp = import_mooncake(
            os.path.join(TRACE_DIR, "derived", ident + ".hit.jsonl"), ident)
        return [(r.timestamp_ms / 1000, r.hit_tokens, r.u_tokens)
                for r in imp.rows]
    rows = []
    with open(AGENTX_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            h = int(obj["prefix_hit_tokens"])
            il = int(obj["input_length"])
            u = max(1, il - h)            # 与 Mooncake u≥1 规则对齐
            rows.append((obj["timestamp"] / 1000, h, u,
                         obj.get("session_id", "")))
    return rows


def analyze(rows, ts_scale=None, label=""):
    """rows: [(t_s, h, u)];ts_scale: AgentX 两种叠放的重排(传入则替换 t)。"""
    if ts_scale is not None:
        rows = [(ts_scale[i], h, u) for i, (t, h, u) in enumerate(rows)]
    rows = sorted(rows, key=lambda r: r[0])
    t = np.array([r[0] for r in rows])
    h = np.array([r[1] for r in rows], dtype=np.int64)
    u = np.array([r[2] for r in rows], dtype=np.int64)
    v_layer = KAPPA * h                          # GB/层
    v_total = L * v_layer                        # GB(全程)
    c = np.array([_c_s(hi, ui) for hi, ui in zip(h, u)])
    d = np.where(c > 0, v_layer / c, 0.0)
    dur = float(t[-1] - t[0]) or 1.0
    b_star = float(v_total.sum() / dur)          # 临界带宽:ρ=1 处

    # --- S1:逐 1s 窗口读取速率 vs B ---
    n_win = int(np.ceil(dur / WIN_S)) or 1
    idx = np.clip((t / WIN_S).astype(int), 0, n_win - 1)
    v_rate = np.bincount(idx, weights=v_total, minlength=n_win) / WIN_S
    n_arr = np.bincount(idx, minlength=n_win)
    comp_rate = (np.bincount(idx, weights=L * c, minlength=n_win) / WIN_S
                 / M_NPU)                        # 需求 NPU 数/窗口
    s1 = {}
    multi = n_arr >= 2                     # 净口径:窗口内≥2 条才有并发争抢
    for B in B_SWEEP:
        hot = v_rate > B
        hot2 = hot & multi
        hot_req = sum(n_arr[i] for i in np.flatnonzero(hot))
        hot2_req = sum(n_arr[i] for i in np.flatnonzero(hot2))
        s1[f"B{B:g}"] = {
            "win_frac": float(hot.mean()),
            "req_frac": float(hot_req / len(rows)),
            "win_frac_multi": float(hot2.mean()),
            "req_frac_multi": float(hot2_req / len(rows)),
            "rho": float(v_total.sum() / (dur * B))}
    # --- S2:簇到达 ---
    idx_ms = np.clip((t / WIN_MS).astype(int), 0,
                     int(np.ceil(dur / WIN_MS)))
    cnt_ms = np.bincount(idx_ms, minlength=int(np.ceil(dur / WIN_MS)) + 1)
    in_cluster = sum(int(cnt_ms[i]) for i in np.flatnonzero(cnt_ms >= CLUSTER_K))
    s2 = {"peak_1s": int(n_arr.max()), "mean_1s": float(n_arr.mean()),
          "peak_50ms": int(cnt_ms.max()),
          "req_in_cluster_ge8_frac": in_cluster / len(rows)}
    # --- S3:轻重结构 ---
    d_heavy = float(np.quantile(d, 0.75))
    heavy_mask = d >= d_heavy
    s3 = {
        "d_p25": float(np.quantile(d, 0.25)),
        "d_p50": float(np.quantile(d, 0.50)),
        "d_p75": d_heavy,
        "d_p90": float(np.quantile(d, 0.90)),
        "d_cv": float(np.std(d) / np.mean(d)) if d.mean() > 0 else 0.0,
        "d_p75_over_p25": float(d_heavy / max(np.quantile(d, 0.25), 1e-12)),
        "zero_read_frac": float((h == 0).mean()),
        "near_light_frac": float((d < 0.1 * d_heavy).mean()),
        "heavy_v_share": float(v_total[heavy_mask].sum() / v_total.sum()),
        "token_hit_ratio": float(h.sum() / (h + u).sum()),
        "req_full_hit_frac": float((u <= 1).mean())}
    # --- S4:计算需求 ---
    s4 = {"comp_mean_npu": float((L * c).sum() / dur / M_NPU),
          "comp_peak_win_npu": float(comp_rate.max()),
          "comp_win_over_frac": float((comp_rate > 1.0).mean()),
          "b_star_gbps": b_star,
          "v_rate_peak_gbps": float(v_rate.max()),
          "speedup_to_rho1_at_B80": 80.0 / b_star if b_star > 0 else None}
    return {"S1_bw": s1, "S2_cluster": s2, "S3_mix": s3, "S4_compute": s4,
            "n": len(rows), "dur_s": dur,
            "arr_rate": len(rows) / dur}


def agentx_global(kind: str, rows):
    """AgentX 两种叠放:overlap=原样;sequential=会话顺次拼接。"""
    if kind != "ax":
        return None
    sess = defaultdict(list)
    for t, h, u, sid in rows:
        sess[sid].append((t, h, u))
    order = sorted(sess, key=lambda s: min(x[0] for x in sess[s]))
    seq_t, cur = [], 0.0
    for sid in order:
        rs = sorted(sess[sid])
        off = cur - rs[0][0]
        for t, h, u in rs:
            seq_t.append(t + off)
        cur = max(seq_t) + 0.0
    return seq_t


def main():
    out_json = None
    if "--json" in sys.argv:
        out_json = sys.argv[sys.argv.index("--json") + 1]
    results = {}
    for fname, _n, _ln, _sha in MOONCAKE_FILES:
        rows = load_rows("mc", fname)
        results[fname] = analyze(rows, label=fname)
    ax = load_rows("ax", "agentx")
    # overlap:原样时间戳(两 session 起点≈0);sequential:顺次拼接
    results["agentx_sample(overlap)"] = analyze(ax)
    seq_t = agentx_global("ax", ax)
    ax_seq = [(tt, h, u) for tt, (t, h, u, _s) in zip(seq_t, sorted(ax))]
    results["agentx_sample(sequential)"] = analyze(ax_seq)

    for name, r in results.items():
        print(f"\n===== {name}  n={r['n']} dur={r['dur_s']:.0f}s "
              f"λ={r['arr_rate']:.1f}/s")
        print(f"  S1 临界带宽 B*={r['S4_compute']['b_star_gbps']:.1f} GB/s;"
              f"窗口峰速率 {r['S4_compute']['v_rate_peak_gbps']:.1f} GB/s;"
              f"B=80 推至 ρ=1 需加速 {r['S4_compute']['speedup_to_rho1_at_B80']:.0f}×;"
              f"计算需求均值 {r['S4_compute']['comp_mean_npu']:.1f} NPU"
              f"/峰 {r['S4_compute']['comp_peak_win_npu']:.1f}")
        for k, v in r["S1_bw"].items():
            print(f"  S1 {k}: 窗口超订 {v['win_frac']:.1%}(多请求净口径 "
                  f"{v['win_frac_multi']:.1%}) / 涉及请求 {v['req_frac']:.1%}"
                  f"(净 {v['req_frac_multi']:.1%}) / ρ={v['rho']:.2f}")
        s2, s3 = r["S2_cluster"], r["S3_mix"]
        print(f"  S2 1s 峰/均 {s2['peak_1s']}/{s2['mean_1s']:.1f};50ms 峰 "
              f"{s2['peak_50ms']};≥8 簇请求占比 {s2['req_in_cluster_ge8_frac']:.1%}")
        print(f"  S3 d P25/50/75/90 = {s3['d_p25']:.2f}/{s3['d_p50']:.2f}/"
              f"{s3['d_p75']:.2f}/{s3['d_p90']:.2f} GB/s;CV={s3['d_cv']:.2f};"
              f"P75/P25={s3['d_p75_over_p25']:.1f}")
        print(f"  S3 零读 {s3['zero_read_frac']:.1%};近轻流(<0.1·P75) "
              f"{s3['near_light_frac']:.1%};重流读取量集中度 "
              f"{s3['heavy_v_share']:.1%};token 命中率 "
              f"{s3['token_hit_ratio']:.1%};全命中请求 {s3['req_full_hit_frac']:.1%}")
    if out_json:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"\nJSON -> {out_json}")


if __name__ == "__main__":
    main()
