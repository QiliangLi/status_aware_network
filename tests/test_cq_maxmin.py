"""cq 线 max_min 分配模式单测（docs/存储带宽maxmin替换设计-20261016.md §cq 移植）。

覆盖：Fraction 精确 waterfill 向量、StorageSim 在途重配（新流提交改变全部
在途流速率）、FCFS 对照（[30,10] vs max_min [20,20]）、boost_due、clone 穿透、
配置校验、引擎端到端冒烟。
"""
from __future__ import annotations

from fractions import Fraction as F

import pytest

from sim.cq.config import CqScenario, ProfileConfig, StorageConfig
from sim.cq.engine import CqEngine, WorldState
from sim.cq.policies import make_fcfs
from sim.cq.simrun import run_case
from sim.cq.storage import StorageSim, waterfill
from sim.cq.types import HardLimits, RequestSpec


# ---------------------------------------------------------------------------
# waterfill（Fraction 精确）
# ---------------------------------------------------------------------------

def test_waterfill_fraction_exact():
    # 用户确认算例：40 容量，需求 [30,20] -> [20,20]（精确）
    assert waterfill(F(40), [F(30), F(20)], F(0)) == [F(20), F(20)]
    # 源设计 §3.4：[10,28,31,31]（精确）
    assert waterfill(F(100), [F(10), F(28), F(100), F(100)], F(0)) == \
        [F(10), F(28), F(31), F(31)]
    # 供过于求：bonus = 26/3
    assert waterfill(F(40), [F(2), F(4), F(8)], F(0)) == \
        [F(32, 3), F(38, 3), F(50, 3)]
    # 均匀 / 单流 / 边界 / 空
    assert waterfill(F(30), [F(5)] * 3, F(0)) == [F(10)] * 3
    assert waterfill(F(40), [F(30)], F(0)) == [F(40)]
    assert waterfill(F(40), [F(10), F(30)], F(0)) == [F(10), F(30)]
    assert waterfill(F(40), [], F(0)) == []
    # 容量 0：全 0（等带宽断点吃满）
    assert waterfill(F(0), [F(30), F(20)], F(0)) == [F(0), F(0)]


# ---------------------------------------------------------------------------
# StorageSim 在途重配
# ---------------------------------------------------------------------------

def _flow(st, fid, t, V, q, due=None):
    return st.submit(fid, fid * 1000, 0, t, V, q, due)


def test_max_min_dynamic_reallocation():
    st = StorageSim(((F(0), F(40)),), F(200), numeric=F, alloc="max_min")
    f1 = _flow(st, 1, F(0), F(40), F(30))
    st.allocate(F(0))
    assert f1.rate_gbps == F(40)                    # 独占：30 + 10 剩余
    f2 = _flow(st, 2, F(0), F(10), F(20))
    st.allocate(F(0))
    assert f1.rate_gbps == F(20) and f2.rate_gbps == F(20)   # 在途重配
    # 完成时刻：f2 0.5；f2 走后 f1 回 40
    st.advance(F(0), F(1, 2))
    assert st.is_complete(f2) and not st.is_complete(f1)
    assert f1.remaining_gb == F(30)
    st.remove(f2.submit_seq, F(1, 2))
    st.allocate(F(1, 2))
    assert f1.rate_gbps == F(40)
    comp = dict(st.completions(F(1, 2)))
    assert comp[f1.submit_seq] == F(1, 2) + F(30) / F(40)


def test_fcfs_contrast():
    """同流同带宽：FCFS [30,10]（先到全得）vs max_min [20,20]（公平重配）。"""
    stf = StorageSim(((F(0), F(40)),), F(200), numeric=F, alloc="fcfs")
    a, b = _flow(stf, 1, F(0), F(40), F(30)), _flow(stf, 2, F(0), F(10), F(20))
    stf.allocate(F(0))
    assert (a.rate_gbps, b.rate_gbps) == (F(30), F(10))
    stm = StorageSim(((F(0), F(40)),), F(200), numeric=F, alloc="max_min")
    a2, b2 = _flow(stm, 1, F(0), F(40), F(30)), _flow(stm, 2, F(0), F(10), F(20))
    stm.allocate(F(0))
    assert (a2.rate_gbps, b2.rate_gbps) == (F(20), F(20))


def test_new_flow_retimes_all_inflight():
    """新流提交 -> 全部在途流速率/完成时刻重算（用户强调的语义）。"""
    st = StorageSim(((F(0), F(30)),), F(200), numeric=F, alloc="max_min")
    fs = [_flow(st, i, F(0), F(10), F(15)) for i in (1, 2, 3)]
    st.allocate(F(0))
    assert all(f.rate_gbps == F(10) for f in fs)
    t_done_3 = dict(st.completions(F(0)))
    assert all(v == F(1) for v in t_done_3.values())       # 均为 t=1 完成
    f4 = _flow(st, 4, F(0), F(10), F(15))
    st.allocate(F(0))
    assert all(f.rate_gbps == F(30) / 4 for f in (*fs, f4))  # 全部重配 7.5
    t_done_4 = dict(st.completions(F(0)))
    assert all(v == F(4, 3) for v in t_done_4.values())      # 全部推迟到 4/3


def test_boost_due_max_min():
    st = StorageSim(((F(0), F(40)),), F(200), numeric=F, alloc="max_min")
    f1 = _flow(st, 1, F(0), F(100), F(30), due=F(1))
    f2 = _flow(st, 2, F(0), F(10), F(20))
    st.allocate(F(0))
    assert (f1.rate_gbps, f2.rate_gbps) == (F(20), F(20))
    st.advance(F(0), F(2))
    assert st.boost_due(F(2))                      # 到期提升 q -> q_max
    assert f1.q_gbps == F(200) and f1.boosted
    st.allocate(F(2))
    # 升速后：Σq=220 > 40，均分 20；f2 需求 20 定格，余 20 给 f1
    assert (f1.rate_gbps, f2.rate_gbps) == (F(20), F(20))


def test_clone_preserves_alloc():
    scn = CqScenario(scenario_name="t", profile=ProfileConfig(),
                     storage=StorageConfig(alloc="max_min"),
                     limits=HardLimits(), m_workers=2)
    specs = [RequestSpec(0, F(0), 100, 100, "A", F(1), F(10))]
    w = WorldState(scn, specs, numeric=float)
    assert w.storage.alloc == "max_min"
    assert w.clone().storage.alloc == "max_min"    # 推演世界同规则


def test_config_alloc_validation_and_json():
    with pytest.raises(ValueError):
        StorageConfig(alloc="bogus")
    assert StorageConfig(alloc="max_min").to_json()["alloc"] == "max_min"
    assert StorageConfig().alloc == "fcfs"          # 默认旧口径


def _specs2():
    return [RequestSpec(0, F(0), 4096, 512, "A", F(1), F(10)),
            RequestSpec(1, F(0), 4096, 512, "A", F(1), F(10))]


def test_engine_end_to_end_both_allocs():
    from dataclasses import replace
    base = CqScenario(scenario_name="t", profile=ProfileConfig(),
                      storage=StorageConfig(b_schedule=((F(0), F(40)),)),
                      limits=HardLimits(n_max=1), m_workers=2)   # 单成员批 -> 双流并发
    outs = {}
    for alloc in ("fcfs", "max_min"):
        scn = replace(base, storage=replace(base.storage, alloc=alloc))
        eng = run_case(scn, _specs2(), make_fcfs(), numeric=float, seed=0)
        assert eng.status == "done"
        assert eng.w.storage.alloc == alloc
        outs[alloc] = [float(eng.w.requests[i].F_s) for i in (0, 1)]
    # 两种分配都确定性完成；速率规则不同 -> 完成时刻不同
    assert outs["fcfs"] != outs["max_min"]
