"""cq MPC/local 搜索健康审计（20260929 常设监测机制）。

背景：20260928 修复的推演引擎批/流 id 错配曾使含在飞读取的决策点推演
100% 卡死、mpc/local 静默退化为 EDF，而旧 records 的 n_fallback 字段是
引擎同名计数（恒 0），完全不可见。本工具从 records 里读取策略级计数器，
对每个 mpc/local run 给出健康判定——任何新实验在写结果文档前应先跑它。

判定口径：
- fallback_ratio > 5%  → 【推演退化】引擎级问题（参照 20260928 bug）；
- n_searchd==0 或缺失 → 【无法评估】旧口径（20260929 之前的 records）；
- deviate_ratio == 0   → 【零偏离】搜索在跑但从未改变动作（结合工况解读，
                         不一定是错，但必须写进结果文档，不得默认"生效"）；
- 其余                 → 【健康】有搜索有调整。

用法：.venv/bin/python tools/cq_mpc_health.py [results/cq/eval/<dir> ...]
不带参数则审计 e26n240、e26、e25 三个已知目录（存在才审）。
"""
from __future__ import annotations

import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = ["results/cq/eval/e26n240", "results/cq/eval/e26",
            "results/cq/eval/e25"]


def load_records(root: str):
    recs = []
    for pat in ("records_part*.json", "records.json", "*_records.json"):
        for p in glob.glob(os.path.join(root, pat)):
            try:
                recs.extend(json.load(open(p, encoding="utf-8")))
            except Exception as e:
                print(f"  [warn] 读取失败 {p}: {e}")
    return recs


def verdict(r):
    ns = r.get("n_searched")
    nf = r.get("n_fallback")
    nd = r.get("n_deviate")
    if ns is None or nf is None:
        # E26 早期记录（e26_ab 落盘，含 cell/mpc_events 字段）：fb 为策略级
        if r.get("mpc_events") is not None or ("cell" in r and "cell_id" not in r):
            return "E26_PARTIAL"
        return "OLD"      # 旧口径：无策略级搜索计数（e25 存档 n_fallback 是引擎计数）
    if ns == 0:
        return "NOSEARCH"
    if nf / ns > 0.05:
        return "DEGRADED"
    if (nd or 0) == 0:
        return "ZERO_DEV"
    return "OK"


_TEXT = {
    "OLD": "【旧口径】无策略级搜索计数（e25 存档 n_fallback 为引擎计数恒 0，"
           "推演健康未知）——mpc/local 数值按退化口径对待",
    "E26_PARTIAL": "【部分新口径】策略级 fallback 已记录且为 0（修复后代码，推演健康），"
                   "但搜索率/偏离率未记录——以补测或重跑为准",
    "NOSEARCH": "【无搜索】n_searched=0（未见任何推演发起）",
    "DEGRADED": "【推演退化】fallback 比例 >5% —— 引擎级问题，结果不可用",
    "ZERO_DEV": "【零偏离】搜索正常但从未改变动作（结合工况解读并写入文档）",
    "OK": "【健康】有搜索有调整",
}


def main():
    roots = sys.argv[1:] or [os.path.join(REPO, d) for d in DEFAULTS]
    exit_code = 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        recs = [r for r in load_records(root)
                if r.get("policy") in ("cq_mpc", "cq_local")
                or r.get("pid") in ("cq_mpc", "cq_local")]
        if not recs:
            print(f"== {root}: 无 mpc/local 记录")
            continue
        vs = {verdict(r) for r in recs}
        counts = {v: sum(1 for r in recs if verdict(r) == v) for v in vs}
        print(f"== {os.path.relpath(root, REPO)}（{len(recs)} 条 mpc/local run）")
        for v in ("DEGRADED", "NOSEARCH", "OLD", "E26_PARTIAL", "ZERO_DEV", "OK"):
            if counts.get(v):
                print(f"   {_TEXT[v]}：{counts[v]} 条")
        if "OLD" in vs:
            print("   （旧口径 run 明细略——一律按退化口径对待；新 run 逐条如下）")
        for r in sorted(recs, key=lambda r: (r.get("cell") or r.get("cell_id",
                       ""), r["policy"], r.get("theta") or "", r.get("rep", 0))):
            v = verdict(r)
            if v in ("OLD", "E26_PARTIAL"):
                continue
            flag = "⚠️" if v in ("DEGRADED", "NOSEARCH") else (
                "·" if v == "ZERO_DEV" else "✓")
            fr = (f"{r.get('n_fallback')}/{r.get('n_searched')}")
            dr = (f"{r.get('n_deviate')}/{r.get('n_searched')}")
            print(f"  {flag} {r.get('cell') or r.get('cell_id'):22s} "
                  f"{r['policy']:9s}θ{r.get('theta') or '-'} rep{r.get('rep', 0)} "
                  f"decides={r.get('n_decides', '—')} fb={fr} dev={dr} "
                  f"scored={r.get('n_scored', '—')}")
            if v == "DEGRADED":
                exit_code = 1
        print()
    print("判定口径：DEGRADED=推演退化（结果不可用）；ZERO_DEV=零偏离（须写入结果文档，"
          "不得默认 MPC 生效）；OLD=旧口径无计数（按退化口径对待）。")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
