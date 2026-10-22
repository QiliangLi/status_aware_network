"""从 results/adv/rounds.jsonl 生成：迭代记录 md 的逐轮表 + 交互页用的 JSON 摘要。"""
import json
import sys
from collections import Counter, defaultdict

SRC = "results/adv/rounds.jsonl"
OUT_JSON = "results/adv/iteration_summary.json"

rows = []
last_round = None
for line in open(SRC, encoding="utf-8"):
    r = json.loads(line)
    if "verify_of" in r:
        if r.get("confirmed") and last_round is not None:
            last_round["verdict"] = "kill_verified"
            last_round["verify_seeds"] = r["seeds"]
        continue
    last_round = r
    rows.append(r)

rounds = rows
waves = defaultdict(list)
for r in rounds:
    waves[r["wave"]].append(r)

summary = {"waves": [], "total_rounds": len(rounds)}
for w in sorted(waves):
    rs = waves[w]
    c = Counter(r["gen"] for r in rs)
    kills = [r for r in rs if r["verdict"] == "kill_verified"]
    summary["waves"].append({
        "wave": w,
        "n_rounds": len(rs),
        "by_gen": dict(c),
        "kills": [{"rnd": k["rnd"], "gen": k["gen"], "kind": k["kind"],
                   "detail": k.get("detail", ""),
                   "metrics": {p: m.get("slo") for p, m in k.get("metrics", {}).items()}}
                  for k in kills],
        "verdicts": dict(Counter(r["verdict"] for r in rs)),
    })
summary["total_kills"] = sum(len(w["kills"]) for w in summary["waves"])
json.dump(summary, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"rounds={len(rounds)} waves={len(waves)} kills={summary['total_kills']}")
for w in summary["waves"]:
    print(f"  wave{w['wave']}: {w['n_rounds']} rounds, verdicts={w['verdicts']}")
