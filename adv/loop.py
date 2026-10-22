"""对抗波次循环器：攻击（生成器×参数搜索）→ 评估 → 判定 → 验证 → 记录。

波次协议（md 记录的骨架）：
- 每波绑定一个 TFS 版本（单层 factory + 两层 policy）；
- 每轮 = 一次攻击尝试：从生成器登记表按权重抽一个，抽参数（含随机变异），
  跑全部策略（CRN 同 trace），判定击杀；
- 击杀 → 3 seed 复核（中位数口径再判）→ 确认后交防守方修补；
- 修补后跑回归：全部历史确认击杀 trace + 母工况锚点（d1f 三 seed）。
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from adv import core
from adv.core import (RunMetrics, Verdict, base_policies, build_scenario,
                      evaluate, judge_tfs, log_round)
from adv.two_layer import (TwoLayerEngine, TwoLayerPolicy, judge_two,
                           policy_immediate, policy_single)

JSONL = os.path.join(core.ADV_ROOT, "rounds.jsonl")


@dataclass
class RoundRecord:
    wave: int
    rnd: int                 # 全局轮次号（1 起）
    gen: str                 # 生成器名
    kind: str                # single | two
    params: dict
    seed: int
    verdict: str             # repelled | kill_unverified | kill_verified | error
    kind_detail: str = ""
    detail: str = ""
    margin_pp: float = 0.0
    metrics: dict = field(default_factory=dict)

    def to_json(self):
        return self.__dict__.copy()


class AdvLoop:
    def __init__(self, single_factory: Callable, two_policy_factory: Callable[[], TwoLayerPolicy]):
        self.single_factory = single_factory
        self.two_factory = two_policy_factory
        self.generators: Dict[str, Callable] = {}
        self.kill_traces: List[dict] = []     # 确认击杀的 trace（回归用）
        self.round_no = 0

    def register(self, name, fn):
        self.generators[name] = fn

    # -- 单层一轮 --
    def run_single(self, gen_name, make_specs, seed_params) -> RoundRecord:
        self.round_no += 1
        specs = make_specs()
        scn = build_scenario(b_gbps=seed_params.get("B", 120.0))
        try:
            res = evaluate(specs, scn, base_policies(self.single_factory))
            v: Verdict = judge_tfs(res, specs)
            rec = RoundRecord(
                wave=seed_params.get("wave", 0), rnd=self.round_no, gen=gen_name,
                kind="single", params=seed_params, seed=seed_params.get("seed", 0),
                verdict="kill_unverified" if v.kill else "repelled",
                kind_detail=v.kind, detail=v.detail, margin_pp=v.margin_pp,
                metrics={k: {"slo": m.slo, "cls": m.cls_slo, "ttft": m.ttft_mean,
                             "busy": m.busy_frac} for k, m in res.items()})
        except Exception as e:  # noqa: BLE001
            rec = RoundRecord(wave=seed_params.get("wave", 0), rnd=self.round_no,
                              gen=gen_name, kind="single", params=seed_params,
                              seed=seed_params.get("seed", 0), verdict="error",
                              detail=repr(e)[:300])
        log_round(JSONL, json.dumps(rec.to_json(), ensure_ascii=False, default=str))
        return rec

    # -- 两层一轮 --
    def run_two(self, gen_name, trace, classes, two_kwargs, seed_params) -> RoundRecord:
        self.round_no += 1
        try:
            outs = {}
            pols = {"immediate": policy_immediate(), "single": policy_single(),
                    "tfs2": self.two_factory()}
            for k, pol in pols.items():
                eng = TwoLayerEngine(trace, classes, pol,
                                     m_inst=two_kwargs.get("m_inst", 4),
                                     pipe=two_kwargs.get("pipe", 120.0),
                                     max_seqs=two_kwargs.get("max_seqs", 8))
                outs[k] = eng.run()
            kill, detail = judge_two(outs["tfs2"],
                                     {"immediate": outs["immediate"],
                                      "single": outs["single"]})
            rec = RoundRecord(
                wave=seed_params.get("wave", 0), rnd=self.round_no, gen=gen_name,
                kind="two", params=seed_params, seed=seed_params.get("seed", 0),
                verdict="kill_unverified" if kill else "repelled",
                detail=detail,
                margin_pp=0.0,
                metrics={k: v for k, v in outs.items()})
        except Exception as e:  # noqa: BLE001
            rec = RoundRecord(wave=seed_params.get("wave", 0), rnd=self.round_no,
                              gen=gen_name, kind="two", params=seed_params,
                              seed=seed_params.get("seed", 0), verdict="error",
                              detail=repr(e)[:300])
        log_round(JSONL, json.dumps(rec.to_json(), ensure_ascii=False, default=str))
        return rec

    # -- 击杀复核（3 seed 中位数）--
    def verify_kill(self, rec: RoundRecord, rerun: Callable[[int], RoundRecord]) -> Optional[RoundRecord]:
        verdicts = []
        for s in (1, 2, 3):
            r = rerun(s)
            verdicts.append(r.verdict)
        n_kill = sum(1 for v in verdicts if v.startswith("kill"))
        if n_kill >= 2:
            rec.verdict = "kill_verified"
            log_round(JSONL, json.dumps({"verify_of": rec.rnd, "seeds": verdicts},
                                        ensure_ascii=False))
            return rec
        rec.verdict = "repelled_after_verify"
        log_round(JSONL, json.dumps({"verify_of": rec.rnd, "seeds": verdicts},
                                    ensure_ascii=False))
        return None
