"""cq 运行组装：run_case(spec, trace, policy) → engine（§11.1）。

负责 Observable/CqEngine/GuardedEDF 回退的组装与排空控制；不做指标计算
（metrics.py）也不读结果文件。
"""
from __future__ import annotations

from typing import Optional, Sequence

from .config import CqScenario
from .engine import CqEngine
from .observable import Observable
from .policies import GuardedEDF
from .types import RequestSpec


def run_case(scenario: CqScenario, specs: Sequence[RequestSpec], policy,
             numeric=float, seed: int = 0,
             drain_budget_s=None) -> CqEngine:
    """执行一个 run。有限工作集（全部 arrival=0/有限集合）与在线模式共用；
    排空预算 = max(drain_budget, arrival_stop + drain_budget)。"""
    obs = Observable(scenario, numeric=numeric, seed=seed)
    eng = CqEngine(scenario, specs, policy, numeric=numeric,
                   fallback_policy=GuardedEDF(), observable=obs)
    obs.attach(eng)
    last_arrival = max((numeric(s.arrival_s) for s in specs), default=numeric(0))
    db = numeric(drain_budget_s if drain_budget_s is not None
                 else scenario.cohort.drain_budget_s)
    deadline = last_arrival + db
    eng.run(drain_deadline=deadline)
    return eng


def make_oracle_engine(scenario: CqScenario, specs, policy, numeric=float,
                       seed: int = 0) -> CqEngine:
    """cq_current_oracle 诊断：snapshot 含真值通道（include_oracle）。"""
    obs = _OracleObservable(scenario, numeric, seed)
    eng = CqEngine(scenario, specs, policy, numeric=numeric,
                   fallback_policy=GuardedEDF(), observable=obs)
    obs.attach(eng)
    return eng


class _OracleObservable(Observable):
    """current_oracle：当前资源真值 + 常值外推（仅诊断，不称最优上界）。"""

    def snapshot(self, t, include_oracle=True):
        return super().snapshot(t, include_oracle=True)
