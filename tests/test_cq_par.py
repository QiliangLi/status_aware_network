"""V0：候选评分并行(fork-per-decision)与串行逐位等价性测试。

设计:docs/MPC推演并行加速-候选评分进程池-20261011.md §6-V0。
三断言:①逐决策动作序列逐位一致;②策略计数器(含 n_overrun)全等;
③预算截断工况(max_events 压小使池耗尽)下重放等价——截断/接受/终止点
与串行一致。附④小预算+θ=T 的第二工况(不同打分路径)。
"""
import sys
from fractions import Fraction as F

import numpy as np
import pytest

sys.path.insert(0, ".")

from sim.cq.search import MPCPolicy
from sim.cq.simrun import run_case
from sim.experiments import e26_ab as e26
from sim.experiments.e26e_cluster import d1c_specs


def _run(pol):
    # 小工况只验等价性不验工况意义:24 条+λ=98 浅队列(决策点少、每推演快),
    # 仍是随机到达+全量搜索路径
    specs, _centers = d1c_specs(4, 7, n_a=12, n_b=12, lam=98.0)
    scn = e26.e26_scenario(4)
    c_ref = float(np.median([float(s.T0_s) for s in specs]))
    pol.c_ref = c_ref
    eng = run_case(scn, specs, pol, numeric=float, seed=0)
    return eng


def _mk(par, max_events=20000, theta="S"):
    return MPCPolicy(pid="cq_mpc", H=1, theta=theta, c_ref=0.01,
                     budget_s=3600.0, max_events=max_events,
                     base="bw_edf", composer=True, par_workers=par)


CNT = ["n_decides", "n_searched", "n_scored", "n_fallback",
       "n_deviate", "n_tie", "n_overrun"]


def _assert_equal(p0, e0, p1, e1, tag):
    a0 = [d["action"] for d in e0.decisions]
    a1 = [d["action"] for d in e1.decisions]
    assert len(a0) == len(a1), f"{tag}: 决策数 {len(a0)} != {len(a1)}"
    for i, (x, y) in enumerate(zip(a0, a1)):
        assert x == y, f"{tag}: 第 {i} 个决策动作分歧"
    for rid in e0.w.requests:
        f0 = e0.w.requests[rid].F_s
        f1 = e1.w.requests[rid].F_s
        assert f0 == f1, f"{tag}: rid={rid} F_s {f0} != {f1}"
    for c in CNT:
        v0, v1 = getattr(p0, c), getattr(p1, c)
        assert v0 == v1, f"{tag}: 计数器 {c} {v0} != {v1}"


def test_v0_normal_budget():
    p0, p1 = _mk(0), _mk(4)
    e0, e1 = _run(p0), _run(p1)
    _assert_equal(p0, e0, p1, e1, "常规预算")
    assert p1.n_scored > 0            # 并行分支确实在评分(非静默回退)


def test_v0_budget_exhausted_replay():
    # max_events 压小:基础排空+候选推演共享池必然耗尽 → 触发截断/清零/
    # 终止路径,验证"满预算推演+原序重放"与串行逐位一致
    p0, p1 = _mk(0, max_events=250), _mk(4, max_events=250)
    e0, e1 = _run(p0), _run(p1)
    assert p0.n_overrun > 0           # 工况确认:预算确实耗尽过
    _assert_equal(p0, e0, p1, e1, "截断预算")


def test_v0_theta_T():
    p0, p1 = _mk(0, theta="T"), _mk(4, theta="T")
    e0, e1 = _run(p0), _run(p1)
    _assert_equal(p0, e0, p1, e1, "θ=T")


def test_v0_local_ablation():
    p0 = MPCPolicy(pid="cq_local", local=True, H=1, theta="S", c_ref=0.01,
                   budget_s=3600.0, max_events=20000, base="bw_edf",
                   composer=True, par_workers=0)
    p1 = MPCPolicy(pid="cq_local", local=True, H=1, theta="S", c_ref=0.01,
                   budget_s=3600.0, max_events=20000, base="bw_edf",
                   composer=True, par_workers=4)
    e0, e1 = _run(p0), _run(p1)
    _assert_equal(p0, e0, p1, e1, "local 消融")


def test_v0_env_var_switch(monkeypatch):
    monkeypatch.setenv("MPC_PAR", "4")
    p = MPCPolicy(pid="cq_mpc", H=1, theta="S", base="bw_edf", composer=True)
    assert p.par_workers == 4
    monkeypatch.delenv("MPC_PAR")
    p2 = MPCPolicy(pid="cq_mpc", H=1, theta="S", base="bw_edf", composer=True)
    assert p2.par_workers == 0       # 默认串行,零扰动
