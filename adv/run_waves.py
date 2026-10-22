"""波次运行器：一轮 = 抽生成器 → 抽参数 → 全策略评估 → 判定 → 记录。

用法：
  .venv/bin/python -m adv.run_waves --wave 1 --rounds 40 \
      --flags hope,classfreeze --keymode slack --gens G1,G3,G14,G2,G5,G6,GX
  .venv/bin/python -m adv.run_waves --regress   # 回归：历史击杀 + 母工况锚点
"""
from __future__ import annotations

import argparse
import json
import os
import random

from adv.core import ADV_ROOT, build_scenario, evaluate, base_policies, judge_tfs
from adv.generators import SINGLE_GENS, TWO_GENS
from adv.generators_w3 import W_SINGLE, W_TWO
SINGLE_GENS = {**SINGLE_GENS, **W_SINGLE}
TWO_GENS = {**TWO_GENS, **W_TWO}
from adv.tfs_versions import make_single
from adv.two_layer import (TwoLayerEngine, TwoLayerPolicy, judge_two,
                           policy_immediate, policy_single)

JSONL = os.path.join(ADV_ROOT, "rounds.jsonl")
KILLS = os.path.join(ADV_ROOT, "kill_archive.jsonl")
HOME_SEEDS = [416, 417, 418]


def run_single_round(gen_name, payload, wave, tfs_factory, rnd, seed=0):
    from fractions import Fraction as F
    kw = {}
    if payload.get("b_sched"):
        kw["b_schedule"] = tuple((F(int(t * 1000)), F(int(v)))
                                 for t, v in payload["b_sched"])
    if payload.get("sigma"):
        from sim.cq.config import ObservationConfig
        kw["observation"] = ObservationConfig(sigma=F(payload["sigma"]).limit_denominator(10**6))
    if payload.get("alloc"):
        kw["alloc"] = payload["alloc"]
    scn = build_scenario(b_gbps=payload.get("B", 120.0),
                         b_ref=payload.get("b_ref", None),
                         guard_w=payload.get("guard_w", 0.0), **kw)
    res = evaluate(payload["specs"], scn, base_policies(tfs_factory))
    v = judge_tfs(res, payload["specs"])
    return {
        "wave": wave, "rnd": rnd, "gen": gen_name, "kind": "single",
        "seed": seed, "params": payload.get("params", {}),
        "verdict": "kill_unverified" if v.kill else "repelled",
        "kill_kind": v.kind, "detail": v.detail, "margin_pp": v.margin_pp,
        "metrics": {k: {"slo": round(m.slo, 4),
                        "cls": {c: round(x, 3) for c, x in m.cls_slo.items()},
                        "busy": round(m.busy_frac, 3)}
                    for k, m in res.items()},
    }


def run_two_round(gen_name, payload, wave, two_flags, rnd, seed=0):
    from adv.tfs_versions import make_two_version
    outs = {}
    vkw = payload.get("two_kwargs", {})
    pols = {"immediate": policy_immediate(), "single": policy_single()}
    mk = make_two_version(two_flags)
    pols["tfs2"] = TwoLayerPolicy(name=mk["name"], **mk["policy_kw"])
    bp = vkw.pop("bp_override", None)
    if bp:
        pols["tfs2"].backpressure = int(bp)
    for k, pol in pols.items():
        eng = TwoLayerEngine(payload["trace"], payload["classes"], pol,
                             m_inst=vkw.get("m_inst", 4),
                             pipe=vkw.get("pipe", 120.0),
                             max_seqs=vkw.get("max_seqs", 8))
        outs[k] = eng.run()
    kill, detail = judge_two(outs["tfs2"],
                             {"immediate": outs["immediate"],
                              "single": outs["single"]})
    return {
        "wave": wave, "rnd": rnd, "gen": gen_name, "kind": "two",
        "seed": seed, "params": payload.get("params", {}),
        "verdict": "kill_unverified" if kill else "repelled",
        "detail": detail, "margin_pp": 0.0,
        "metrics": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in v.items() if kk != "cls_slo"}
                    | {"cls": {c: round(x, 3) for c, x in v["cls_slo"].items()}}
                    for k, v in outs.items()},
    }


def make_payload(gen_name, seed):
    rng = random.Random(seed)
    params, payload = SINGLE_GENS[gen_name](rng) if gen_name in SINGLE_GENS \
        else TWO_GENS[gen_name](rng)
    payload["params"] = params
    return payload


def log(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--flags", default="")
    ap.add_argument("--keymode", default="slack")
    ap.add_argument("--two-flags", default="")
    ap.add_argument("--gens", default="")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--regress", action="store_true")
    args = ap.parse_args()

    if args.regress:
        rc = regress(args)
        print(json.dumps(rc, ensure_ascii=False))
        return

    flags = [x for x in args.flags.split(",") if x]
    tfs_factory = make_single(flags, args.keymode)
    two_flags = [x for x in args.two_flags.split(",") if x]
    gens = ([x for x in args.gens.split(",") if x] or
            list(SINGLE_GENS.keys()) + list(TWO_GENS.keys()))
    kills = []
    rng = random.Random(args.wave * 7919 + args.seed_base)
    for i in range(args.rounds):
        g = rng.choice(gens)
        seed = args.seed_base + i
        try:
            payload = make_payload(g, seed)
            if payload["kind"] == "single":
                rec = run_single_round(g, payload, args.wave, tfs_factory,
                                       i + 1, seed)
            else:
                rec = run_two_round(g, payload, args.wave, two_flags,
                                    i + 1, seed)
        except Exception as e:  # noqa: BLE001
            rec = {"wave": args.wave, "rnd": i + 1, "gen": g, "kind": "?",
                   "seed": seed, "verdict": "error", "detail": repr(e)[:300]}
        log(JSONL, rec)
        status = rec["verdict"]
        print(f"wave{args.wave} r{i+1:03d} {g:4s} {status:16s} "
              f"{rec.get('detail','')[:80]}")
        if status == "kill_unverified":
            # 3 seed 复核
            votes = []
            for s in (seed + 5001, seed + 5002, seed + 5003):
                try:
                    p2 = make_payload(g, s)
                    if p2["kind"] == "single":
                        r2 = run_single_round(g, p2, args.wave, tfs_factory,
                                              i + 1, s)
                    else:
                        r2 = run_two_round(g, p2, args.wave, two_flags,
                                           i + 1, s)
                    votes.append(1 if r2["verdict"].startswith("kill") else 0)
                except Exception:  # noqa: BLE001
                    votes.append(0)
            if sum(votes) >= 2:
                rec["verdict"] = "kill_verified"
                rec["verify_seeds"] = votes
                kills.append(rec)
                log(JSONL, rec)   # 确认后回写（含 kill_verified 与复核票）
                log(KILLS, {"wave": args.wave, "gen": g, "seed": seed,
                            "params": rec.get("params", {}),
                            "detail": rec.get("detail", ""),
                            "kind": rec.get("kill_kind", "")})
                log(JSONL, {"verify_of": rec["rnd"], "seeds": votes,
                            "confirmed": True})
                print(f"   -> KILL VERIFIED ({votes})")
            else:
                log(JSONL, {"verify_of": rec["rnd"], "seeds": votes,
                            "confirmed": False})
                print(f"   -> not confirmed ({votes})")
    print(json.dumps({"wave": args.wave, "rounds": args.rounds,
                      "kills": len(kills),
                      "kill_gens": sorted({k['gen'] for k in kills})},
                     ensure_ascii=False))


def regress(args):
    """回归：历史确认击杀全部复现参数 → 当前版本必须全部免疫 + 母工况锚点。"""
    flags = [x for x in args.flags.split(",") if x]
    tfs_factory = make_single(flags, args.keymode)
    two_flags = [x for x in args.two_flags.split(",") if x]
    out = {"kill_regress": [], "home": []}
    if os.path.exists(KILLS):
        with open(KILLS, encoding="utf-8") as f:
            for line in f:
                k = json.loads(line)
                try:
                    payload = make_payload(k["gen"], k["seed"])
                    if payload["kind"] == "single":
                        r = run_single_round(k["gen"], payload, -1,
                                             tfs_factory, 0, k["seed"])
                    else:
                        r = run_two_round(k["gen"], payload, -1, two_flags,
                                          0, k["seed"])
                    out["kill_regress"].append({"gen": k["gen"], "seed": k["seed"],
                                                "still_kill": r["verdict"].startswith("kill"),
                                                "detail": r.get("detail", "")[:120]})
                except Exception as e:  # noqa: BLE001
                    out["kill_regress"].append({"gen": k["gen"], "error": repr(e)[:200]})
    # 母工况锚点：d1f 三 seed，当前版本不得差 base 2pp 以上、零读类必须满
    from sim.experiments import e26f_third as e26f
    scn = build_scenario()
    for s in HOME_SEEDS:
        specs, _ = e26f.d1f_specs(16, s, e26f.C_A_CLASS, n=40)
        res = evaluate(specs, scn, base_policies(tfs_factory))
        out["home"].append({
            "seed": s,
            "tfs": round(res["tfs"].slo, 4),
            "base": round(res["base"].slo, 4),
            "c_slo": round(res["tfs"].cls_slo.get("C", 0.0), 4),
            "ok": res["tfs"].slo >= res["base"].slo - 0.02
                  and res["tfs"].cls_slo.get("C", 0.0) >= 0.999})
    return out


if __name__ == "__main__":
    main()
