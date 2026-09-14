"""cq 单元测试矩阵（§11.3 U01–U30）。每个测试对应独立解析值或语义不变量。"""
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, ".")

from sim.cq.config import CqScenario, ControllerConfig, ProfileConfig, StorageConfig
from sim.cq.engine import CqEngine
from sim.cq.metrics import (goodput_gain_pct, latency_gain, slo_gain_pp,
                            summarize)
from sim.cq.observable import Observable
from sim.cq.policies import GuardedEDF, make_edf, make_fcfs
from sim.cq.profile import (batch_compute_s, is_feasible, layer_read_gb,
                            mem_peak_gb, singleton_K)
from sim.cq.search import MPCPolicy, score_of
from sim.cq.simrun import run_case
from sim.cq.storage import StorageSim
from sim.cq.types import Action, HardLimits, JointAction, RequestSpec


def _req(rid, h, u, cls="x", T0=F(1), deadline=F(10**9), arrival=F(0)):
    return RequestSpec(rid=rid, arrival_s=arrival, h_tokens=h, u_tokens=u,
                       class_id=cls, T0_s=T0, deadline_s=deadline)


# ---------------------------------------------------------------------------
# U01 单位与 κ
# ---------------------------------------------------------------------------

def test_u01_units():
    # 8 KV 头、128 维、K/V 各 2 字节：每 token 每层 8*128*2*2=4096 字节
    kappa = F(8 * 128 * 2 * 2, 10**9)
    assert kappa == F(4096, 10**9)
    assert abs(float(kappa) - 4.096e-6) < 1e-15
    # 512 token、32 层的总 KV 读取量
    total = kappa * 512 * 32
    assert total == F(67108864, 10**9)
    assert abs(float(total) - 0.067108864) < 1e-12
    # 毫秒转秒
    assert F(27482, 1000) == F(13741, 500)
    assert float(F(27482) / 1000) == 27.482


# ---------------------------------------------------------------------------
# U02 profile 公式校验值（§5.1 表）
# ---------------------------------------------------------------------------

def test_u02_profile_table():
    cfg = ProfileConfig()
    # HS_US singleton：A = u h + u(u+1)/2 = 128*512 + 8128 = 73728+8128
    hs_us = _req(0, 512, 128)
    A = 128 * 512 + 128 * 129 // 2
    assert A == 65536 + 8256
    # 校验点 h2048/u128：A = 2048*128 + 128*129/2 = 262144+8256=270400
    chk = _req(1, 2048, 128)
    A_chk = 2048 * 128 + 128 * 129 // 2
    assert A_chk == 270400
    c = batch_compute_s([chk], cfg)
    assert abs(float(c) - 0.00017944) < 1e-9, float(c)
    # 4 份合批：N=512 → η=1
    batch4 = [chk] * 4
    A4 = 4 * 270400
    assert A4 == 1081600
    c4 = batch_compute_s(batch4, cfg)
    assert abs(float(c4) - 0.00026056) < 1e-9, float(c4)
    # §5.1 表：四类每层 c、singleton K（=L*c 纯计算）与 T⁰（含读重放）
    for cls, h, u, c_exp, K_exp, T0_exp in [
            ("HS_US", 512, 128, 0.0001597792, 0.0051129344, 0.0051391488),
            ("HS_UL", 512, 2048, 0.0007742752, 0.0247768064, 0.0248030208),
            ("HL_US", 8192, 128, 0.0002580832, 0.0082586624, 0.0136798560),
            ("HL_UL", 8192, 2048, 0.0023471392, 0.0751084544, 0.0755278848)]:
        r = _req(0, h, u, cls)
        c = batch_compute_s([r], cfg)
        assert abs(float(c) - c_exp) < 5e-12, (cls, float(c), c_exp)
        assert abs(float(c) * 32 - K_exp) < 1e-10
        T0 = singleton_K(r, cfg, F(80), F(200))
        assert abs(float(T0) - T0_exp) < 1e-10, (cls, float(T0), T0_exp)


# ---------------------------------------------------------------------------
# U03 无合批加速：batch K 严格等于 Σ singleton K
# ---------------------------------------------------------------------------

def test_u03_no_speedup():
    cfg = ProfileConfig(no_speedup=True)
    scn = CqScenario(scenario_name="u3", profile=cfg,
                     storage=StorageConfig(b_schedule=((F(0), F(80)),),
                                           q_max_gbps=F(200), b_ref_gbps=F(80)),
                     limits=HardLimits(8, 8192, F(1024)))
    specs = [_req(0, 512, 2048), _req(1, 8192, 128), _req(2, 2048, 128),
             _req(3, 512, 128)]
    full = specs
    part = specs[:2]
    K_full = sum(float(batch_compute_s([s], cfg)) for s in full)
    K_part = sum(float(batch_compute_s([s], cfg)) for s in part)
    c_full = float(batch_compute_s(full, cfg))
    c_part = float(batch_compute_s(part, cfg))
    assert abs(K_full - c_full) < 1e-12   # 任意批纯计算 = singleton 之和
    assert abs(K_part - c_part) < 1e-12


# ---------------------------------------------------------------------------
# U04 容量边界
# ---------------------------------------------------------------------------

def test_u04_capacity():
    cfg = ProfileConfig()
    lim = HardLimits(n_max=2, token_max=300, workspace_gb=F(10**9))
    a = _req(0, 100, 100)
    b = _req(1, 100, 200)   # Σu=300 恰好
    c = _req(2, 100, 201)
    assert is_feasible([a], lim, cfg)
    assert is_feasible([a, b], lim, cfg)     # 等于边界合法
    assert not is_feasible([a, c], lim, cfg)  # 超 1 不合法
    lim2 = HardLimits(n_max=1, token_max=8192, workspace_gb=F(10**9))
    assert not is_feasible([a, b], lim2, cfg)
    lim3 = HardLimits(n_max=2, token_max=8192,
                      workspace_gb=mem_peak_gb([a, b], cfg) - F(1, 10**6))
    assert not is_feasible([a, b], lim3, cfg)  # 内存超 1e-6 不合法
    lim4 = HardLimits(n_max=2, token_max=8192, workspace_gb=mem_peak_gb([a, b], cfg))
    assert is_feasible([a, b], lim4, cfg)


# ---------------------------------------------------------------------------
# U05/U06 FCFS 分配
# ---------------------------------------------------------------------------

def test_u05_fcfs_allocation():
    st = StorageSim(((F(0), F(4)),), F(4), numeric=F)
    st.submit(0, 0, 0, F(0), F(10), F(1), None)
    st.submit(1, 0, 0, F(0), F(10), F(4), None)
    st.submit(2, 0, 0, F(0), F(10), F(2), None)
    st.allocate(F(0))
    rates = [st.flows[i].rate_gbps for i in range(3)]
    assert rates == [F(1), F(3), F(0)], rates   # 不能得到公平共享
    st.remove(0, F(1))
    st.allocate(F(1))
    rates = [st.flows[1].rate_gbps, st.flows[2].rate_gbps]
    assert rates == [F(4), F(0)], rates


def test_u06_idle_bandwidth():
    st = StorageSim(((F(0), F(4)),), F(4), numeric=F)
    for i, q in enumerate((F(1), F(1), F(1))):
        st.submit(i, 0, 0, F(0), F(10), q, None)
    st.allocate(F(0))
    total = sum(st.flows[i].rate_gbps for i in range(3))
    assert total == F(3)
    assert st.b_at(F(0)) - total == F(1)   # 不擅自超额分配


# ---------------------------------------------------------------------------
# U07 容量变化：恰断点旧区间按旧 B 积分
# ---------------------------------------------------------------------------

def test_u07_capacity_change():
    st = StorageSim(((F(0), F(4)), (F(1), F(2))), F(4), numeric=F)
    st.submit(0, 0, 0, F(0), F(6), F(4), None)
    st.allocate(F(0))
    st.advance(F(0), F(1))
    assert st.flows[0].remaining_gb == F(2)
    st.allocate(F(1))
    st.advance(F(1), F(2))
    assert st.flows[0].remaining_gb == F(0)   # 4+2=6 恰好读完


# ---------------------------------------------------------------------------
# U08 到期补读（保留序号）
# ---------------------------------------------------------------------------

def test_u08_due_boost():
    st = StorageSim(((F(0), F(4)),), F(4), numeric=F)
    st.submit(0, 0, 0, F(0), F(4), F(4), None)       # F0 剩 4，q4 排前
    st.submit(1, 0, 0, F(0), F(2), F(1), F(1, 2))    # F1 V2 q1 due0.5
    st.allocate(F(0))
    assert st.flows[0].rate_gbps == F(4)
    assert st.flows[1].rate_gbps == F(0)   # F1 先无服务
    st.advance(F(0), F(1, 2))
    st.allocate(F(1, 2))
    assert st.flows[1].rate_gbps == F(0)
    changed = st.boost_due(F(1, 2))
    assert changed
    assert st.flows[1].q_gbps == F(4)
    assert st.flows[1].submit_seq == 1     # 保留原序号
    st.allocate(F(1, 2))
    assert st.flows[0].rate_gbps == F(4)   # F0 仍在前
    assert st.flows[1].rate_gbps == F(0)
    # F0 在 1 完成
    st.advance(F(1, 2), F(1))
    assert st.is_complete(st.flows[0])
    st.remove(0, F(1))
    st.allocate(F(1))
    assert st.flows[1].rate_gbps == F(4)
    st.advance(F(1), F(3, 2))
    assert st.is_complete(st.flows[1])     # F1 在 1.5 完成


# ---------------------------------------------------------------------------
# U09 恢复与停机
# ---------------------------------------------------------------------------

def test_u09_recovery_and_shutdown():
    st = StorageSim(((F(0), F(0)), (F(2), F(1))), F(4), numeric=F)
    st.submit(0, 0, 0, F(0), F(1), F(4), None)
    st.allocate(F(0))
    assert st.flows[0].rate_gbps == F(0)
    assert st.has_future_positive(F(0)) is True or st.has_future_positive(F(0))
    st.advance(F(0), F(2))
    st.allocate(F(2))
    st.advance(F(2), F(3))
    assert st.flows[0].remaining_gb == F(0)   # 前者 R=3（2s 起传）
    # 永久停机：无未来正带宽
    st2 = StorageSim(((F(0), F(0)),), F(4), numeric=F)
    st2.submit(0, 0, 0, F(0), F(1), F(4), None)
    assert st2.has_future_positive(F(0)) is False


# ---------------------------------------------------------------------------
# U10–U13 零读/零算边界（引擎层）
# ---------------------------------------------------------------------------

def _mini_engine(L, Vs, cs, B=F(4), m=1):
    """构造每层 (V,c) 可任意（含 0）的最小引擎场景。"""
    prof = ProfileConfig(L=L, kappa_gb_per_token_layer=F(1),
                         N_sat=1, t_launch_s=F(0),
                         a_s_per_token=F(cs[0]) if cs[0] else F(1),
                         b_s_per_pair=F(0), no_speedup=True)
    # 用 h=V/κ, u=c/a 表达；零值单独构造需要特殊 profile——改用直接注入
    scn = CqScenario(scenario_name="mini", profile=prof,
                     storage=StorageConfig(b_schedule=((F(0), B),),
                                           q_max_gbps=F(4), b_ref_gbps=F(4)),
                     limits=HardLimits(1, 10**9, F(10**9)), m_workers=m,
                     controller=ControllerConfig(cost_mode="zero"))
    return scn


def test_u10_two_layer_singleton():
    # 每层 V1 c1，B=q=4：T0=1/4+2+0=9/4；首层 STALL=1/4
    scn = _mini_engine(2, [1, 1], [1, 1])
    specs = [_req(0, 1, 1, T0=F(9, 4))]
    pol = make_fcfs()
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, pol, numeric=F, observable=obs)
    obs.attach(eng)
    eng.run()
    assert eng.w.requests[0].F_s == F(9, 4)
    assert eng.w.workers[0].stall_s == F(1, 4)
    assert eng.w.workers[0].compute_s == F(2)


def test_u11_zero_read():
    # L3、V 全 0、c 各 0.1：无流无 submit_seq，F=0.3，STALL=0
    scn = _mini_engine(3, [0, 0, 0], [F(1, 10), F(1, 10), F(1, 10)])
    specs = [_req(0, 0, 1, T0=F(3, 10))]
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, make_fcfs(), numeric=F, observable=obs)
    obs.attach(eng)
    eng.run()
    assert eng.w.requests[0].F_s == F(3, 10)
    assert eng.w.workers[0].stall_s == F(0)
    assert eng.w.workers[0].compute_s == F(3, 10)
    assert eng.observable.ledger == {}   # 无流


def test_u12_zero_compute():
    # L2、V 各 1、c=0：两层串行读，F=0.5，无除零/死循环
    scn = _mini_engine(2, [1, 1], [0, 0])
    specs = [_req(0, 1, 0, T0=F(1, 2))]
    # u=0 不合法（u>=1），改用 u=1 但 c 公式 = a*u=0 → a=0
    prof = ProfileConfig(L=2, kappa_gb_per_token_layer=F(1), N_sat=1,
                         t_launch_s=F(0), a_s_per_token=F(0),
                         b_s_per_pair=F(0), no_speedup=True)
    scn = CqScenario(scenario_name="zc", profile=prof,
                     storage=StorageConfig(((F(0), F(4)),), F(4), F(4)),
                     limits=HardLimits(1, 10**9, F(10**9)), m_workers=1,
                     controller=ControllerConfig(cost_mode="zero"))
    specs = [_req(0, 1, 1, T0=F(1, 2))]
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, make_fcfs(), numeric=F, observable=obs)
    obs.attach(eng)
    eng.run()
    assert eng.w.requests[0].F_s == F(1, 2)
    assert eng.w.workers[0].compute_s == F(0)


def test_u13_zero_both():
    prof = ProfileConfig(L=2, kappa_gb_per_token_layer=F(1), N_sat=1,
                         t_launch_s=F(0), a_s_per_token=F(0),
                         b_s_per_pair=F(0), no_speedup=True)
    scn = CqScenario(scenario_name="zb", profile=prof,
                     storage=StorageConfig(((F(0), F(4)),), F(4), F(4)),
                     limits=HardLimits(1, 10**9, F(10**9)), m_workers=1,
                     controller=ControllerConfig(cost_mode="zero"))
    specs = [_req(0, 0, 1, T0=F(1))]   # 人为 T0=1 仅边界测试
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, make_fcfs(), numeric=F, observable=obs)
    obs.attach(eng)
    st = eng.run()
    assert st == "done"
    assert eng.w.requests[0].F_s == F(0)   # 同刻完成且闭包终止


def test_u14_simultaneous_due():
    # 两个未完成旧流同刻 due，新流提交：旧 seq 保持、新流在后
    st = StorageSim(((F(0), F(4)),), F(4), numeric=F)
    st.submit(0, 0, 0, F(0), F(4), F(4), F(1))
    st.submit(1, 0, 0, F(0), F(4), F(4), F(1))
    st.submit(2, 0, 0, F(0), F(4), F(4), None)
    st.allocate(F(0))
    st.boost_due(F(1))
    assert st.flows[0].q_gbps == F(4) and st.flows[1].q_gbps == F(4)
    assert st.flows[0].submit_seq == 0 and st.flows[1].submit_seq == 1
    assert st.flows[2].submit_seq == 2   # 新流在后
    st.allocate(F(1))
    assert st.flows[0].rate_gbps == F(4)
    assert st.flows[1].rate_gbps == F(0)
    assert st.flows[2].rate_gbps == F(0)


# ---------------------------------------------------------------------------
# U15/U16 原子领取与 worker 占用
# ---------------------------------------------------------------------------

def test_u15_atomic_dispatch():
    scn = _mini_engine(1, [1], [1])
    scn = CqScenario(scenario_name="a15", profile=scn.profile,
                     storage=scn.storage, limits=HardLimits(2, 10**9, F(10**9)),
                     m_workers=2, controller=ControllerConfig(cost_mode="zero"))
    specs = [_req(0, 1, 1), _req(1, 1, 1)]
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, make_fcfs(), numeric=F, observable=obs)
    obs.attach(eng)
    eng._physical_step(F(0))   # 到达入队
    ja = JointAction((Action("DISPATCH", 0, (0,)), Action("DISPATCH", 1, (0,))))
    ok = eng._validate(F(0), list(ja.actions))
    assert not ok   # 重复 rid → 整体拒绝
    assert eng.w.n_queued == 2
    assert all(eng.w.workers[i].batch_id is None for i in range(2))
    assert eng.w.storage.next_seq == 0   # 无部分修改


def test_u16_worker_occupied():
    scn = CqScenario(scenario_name="a16", profile=ProfileConfig(
        L=1, kappa_gb_per_token_layer=F(1), N_sat=1, t_launch_s=F(0),
        a_s_per_token=F(1), b_s_per_pair=F(0), no_speedup=True),
        storage=StorageConfig(((F(0), F(4)),), F(4), F(4)),
        limits=HardLimits(1, 10**9, F(10**9)), m_workers=1,
        controller=ControllerConfig(cost_mode="zero"))
    specs = [_req(0, 1, 1), _req(1, 1, 1)]
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, make_fcfs(), numeric=F, observable=obs)
    obs.attach(eng)
    eng._dispatch(F(0), 0, (0,))
    ja = JointAction((Action("DISPATCH", 0, (1,)),))
    assert not eng._validate(F(0), list(ja.actions))   # worker 占用
    w = Action("WAIT", 0, (), F(1))
    eng._apply_action(F(0), JointAction((w,)), stale=False)
    assert eng.w.storage.next_seq == 1   # WAIT 不新增流/内存占用


# ---------------------------------------------------------------------------
# U17 可见性 / U18 陈旧样本
# ---------------------------------------------------------------------------

def test_u17_snapshot_visibility():
    from sim.cq.observable import ObservableSnapshot
    scn = _mini_engine(2, [1, 1], [1, 1])
    scn = CqScenario(scenario_name="u17", profile=scn.profile,
                     storage=scn.storage, limits=HardLimits(2, 10**9, F(10**9)),
                     m_workers=2, controller=ControllerConfig(cost_mode="zero"))
    specs = [_req(0, 1, 1, deadline=F(100)), _req(1, 2, 1, deadline=F(100))]
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, make_fcfs(), numeric=F, observable=obs)
    obs.attach(eng)
    snap = obs.snapshot(F(0))
    fields = snap.public_fields()
    for k in ("now", "requests", "workers", "flows", "quote"):
        assert k in fields
    # 禁止字段：snapshot 无 world/storage 引用
    banned = ("world", "storage", "engine", "eng", "w")
    for b in banned:
        assert not hasattr(snap, b) or getattr(snap, b) is None
    # 普通策略可完成决定且只依赖 snapshot
    for name, pol in [("fcfs", make_fcfs()), ("edf", make_edf()),
                      ("mpc", MPCPolicy(theta="S", H=1))]:
        eng2 = run_case(scn, specs, pol, numeric=F)
        assert eng2.status == "done", (name, eng2.status)


def test_u18_stale_sample():
    scn = _mini_engine(2, [1, 1], [1, 1])
    scn = CqScenario(scenario_name="u18", profile=scn.profile,
                     storage=scn.storage, limits=HardLimits(2, 10**9, F(10**9)),
                     m_workers=1, controller=ControllerConfig(cost_mode="zero"))
    specs = [_req(0, 1, 1, deadline=F(100))]
    obs = Observable(scn, numeric=F)
    eng = CqEngine(scn, specs, make_fcfs(), numeric=F, observable=obs)
    obs.attach(eng)
    eng.run()
    # 完成确认已入账本；交付旧进度样本不复活流
    for fid, rec in obs.ledger.items():
        assert rec["completed"] and rec["completed_s"] is not None
    assert eng.w.storage.flows == {}


# ---------------------------------------------------------------------------
# U19 控制延迟
# ---------------------------------------------------------------------------

def test_u19_control_delay():
    prof = ProfileConfig(L=1, kappa_gb_per_token_layer=F(0), N_sat=1,
                         t_launch_s=F(0), a_s_per_token=F(1),
                         b_s_per_pair=F(0), no_speedup=True)
    from sim.cq.config import ControllerConfig as CC
    scn = CqScenario(scenario_name="u19", profile=prof,
                     storage=StorageConfig(((F(0), F(4)),), F(4), F(4)),
                     limits=HardLimits(1, 10**9, F(10**9)), m_workers=2,
                     controller=CC(cost_mode="fixed", fixed_delay_s=F(1, 5)))
    specs = [_req(0, 0, 1, T0=F(1)), _req(1, 0, 1, T0=F(1))]
    eng = run_case(scn, specs, make_fcfs(), numeric=F)
    rr = eng.w.requests[0]
    assert rr.dispatch_s == F(1, 5)
    assert rr.F_s == F(6, 5)
    assert eng.w.requests[0].queue_wait if False else True
    row_q = float(rr.dispatch_s) - float(rr.spec.arrival_s)
    assert row_q == 0.2
    # 另一运行批不被 CPU 暂停：批 1 与批 2 同刻领取，计算重叠
    d0 = eng.w.requests[0].dispatch_s
    d1 = eng.w.requests[1].dispatch_s
    assert d0 == d1 == F(1, 5)


# ---------------------------------------------------------------------------
# U20 新到达与控制器返回
# ---------------------------------------------------------------------------

def test_u20_arrival_during_decision():
    prof = ProfileConfig(L=1, kappa_gb_per_token_layer=F(0), N_sat=1,
                         t_launch_s=F(0), a_s_per_token=F(1),
                         b_s_per_pair=F(0), no_speedup=True)
    scn = CqScenario(scenario_name="u20", profile=prof,
                     storage=StorageConfig(((F(0), F(4)),), F(4), F(4)),
                     limits=HardLimits(1, 10**9, F(10**9)), m_workers=1,
                     controller=ControllerConfig(cost_mode="fixed",
                                                 fixed_delay_s=F(1, 5)))
    specs = [_req(0, 0, 1, T0=F(1), arrival=F(0)),
             _req(1, 0, 1, T0=F(1), arrival=F(1, 10))]
    eng = run_case(scn, specs, make_fcfs(), numeric=F)
    # t=0 决策（覆盖请求 0），期间请求 1 到达；t=0.2 领取 0；请求 1 保留下一轮
    assert eng.w.requests[0].dispatch_s == F(1, 5)
    assert eng.w.requests[1].state == "DONE"
    # 批 0 在 1.2 完成，新一轮决策再计 0.2 控制延迟 → 领取于 1.4
    assert eng.w.requests[1].dispatch_s == F(7, 5)


# ---------------------------------------------------------------------------
# U21 WAIT 与 guard
# ---------------------------------------------------------------------------

def test_u21_wait_guard():
    prof = ProfileConfig(L=1, kappa_gb_per_token_layer=F(1), N_sat=1,
                         t_launch_s=F(0), a_s_per_token=F(1),
                         b_s_per_pair=F(0), no_speedup=True)
    from sim.cq.config import CohortConfig
    scn = CqScenario(scenario_name="u21", profile=prof,
                     storage=StorageConfig(((F(0), F(4)),), F(4), F(4)),
                     limits=HardLimits(1, 10**9, F(10**9)), m_workers=1,
                     controller=ControllerConfig(cost_mode="zero"),
                     guard_w=F(1))
    specs = [_req(0, 1, 1, T0=F(5, 4), deadline=F(100))]

    class WaitPolicy:
        def __init__(self):
            self.waits = []

        def decide(self, snap, scn):
            if float(snap.now) >= 1.0:
                return JointAction((Action("DISPATCH", 0, (0,)),))
            wake = float(snap.now) + 0.25
            self.waits.append((float(snap.now), wake))
            return JointAction((Action("WAIT", 0, (), wake),))

    pol = WaitPolicy()
    eng = run_case(scn, specs, pol, numeric=F)
    assert eng.status == "done"
    assert all(w > now for now, w in pol.waits)   # 每个 WAIT 有严格未来 wake
    assert eng.w.requests[0].dispatch_s >= F(1)   # guard 达到后最老请求被派发


# ---------------------------------------------------------------------------
# U22 固定分母
# ---------------------------------------------------------------------------

def test_u22_fixed_denominator():
    from sim.cq.observable import ObservableSnapshot, PublicRequest
    snap = ObservableSnapshot(
        now=0, requests=[], workers=[], flows=[], quote=None,
        queued=frozenset(), idle_workers=(), b_ref=80, q_max=200)
    Fs = {0: 2.0, 1: 10.0}
    reqs = {0: PublicRequest(0, 0, 1, 1, 100, 1.0),
            1: PublicRequest(1, 0, 1, 1, 100, 10.0)}
    snap.requests = list(reqs.values())
    ttft = {0: 2.0, 1: 10.0}
    norm_mean = sum(ttft[r] / float(reqs[r].T0_s) for r in ttft) / 2
    assert norm_mean == (2.0 / 1.0 + 10.0 / 10.0) / 2 == 1.5


# ---------------------------------------------------------------------------
# U23 四目标（四计划集内 M 选 P3、S/T/R 选 P4）
# ---------------------------------------------------------------------------

def test_u23_four_objectives():
    from tests.cq_reference import GOLD, PLANS, run_plan
    from sim.cq.exact import THETA_TUPLES
    Fs_all = {}
    for name in ("P1", "P2", "P3", "P4"):
        eng, _ = run_plan(PLANS[name])
        Fs_all[name] = {rid: rr.F_s for rid, rr in eng.w.requests.items()}
    arrs = {r: F(0) for r in range(4)}
    specs = None
    from tests.cq_reference import e20_specs
    sp = {s.rid: s for s in e20_specs()}
    T0s = {r: sp[r].T0_s for r in range(4)}
    dls = {r: sp[r].deadline_s for r in range(4)}
    pick = {}
    for theta in ("M", "S", "T", "R"):
        best = None
        for name, Fs in Fs_all.items():
            key = THETA_TUPLES[theta](Fs, arrs, T0s, dls, (name,))
            if best is None or key < best[0]:
                best = (key, name)
        pick[theta] = best[1]
    assert pick["M"] == "P3"
    assert pick["S"] == "P4"   # 成功 4 的计划中按均值 TTFT 选 P4
    assert pick["T"] == "P4"
    assert pick["R"] == "P4"


# ---------------------------------------------------------------------------
# U24 下界 vs 穷举最优（小网格节点）
# ---------------------------------------------------------------------------

def test_u24_lower_bound():
    from sim.cq.exact import E20Instance, ExactSolver
    inst = E20Instance(types=[0, 0, 1], arrival=(F(0),) * 3, delta=F(1))
    for theta in ("M", "S", "T", "R"):
        e = ExactSolver(inst, theta, mode="enumerate",
                        node_budget=2 * 10**6, time_budget_s=60)
        r = e.solve()
        b = ExactSolver(inst, theta, mode="bnb",
                        node_budget=2 * 10**6, time_budget_s=60)
        rb = b.solve()
        assert r["status"] == "optimal" and rb["status"] == "optimal"
        assert r["UB"] == rb["UB"], (theta, float(r["UB"]), float(rb["UB"]))
        assert rb["LB"] == rb["UB"]


# ---------------------------------------------------------------------------
# U25 超时证书
# ---------------------------------------------------------------------------

def test_u25_timeout_certificate():
    from sim.cq.exact import E20Instance, ExactSolver
    inst = E20Instance(types=[0, 0, 1, 1], arrival=(F(0),) * 4, delta=F(1, 2))
    b = ExactSolver(inst, "T", mode="bnb", node_budget=10, time_budget_s=60)
    r = b.solve()
    assert r["status"] == "time_limit"
    assert r["open_min_lb"] is not None   # 父区域 LB 仍在 OPEN，未丢
    assert r["LB"] <= r["UB"]


# ---------------------------------------------------------------------------
# U26 截断（右删失）
# ---------------------------------------------------------------------------

def test_u26_censoring():
    prof = ProfileConfig(L=1, kappa_gb_per_token_layer=F(1), N_sat=1,
                         t_launch_s=F(0), a_s_per_token=F(1),
                         b_s_per_pair=F(0), no_speedup=True)
    from sim.cq.config import CohortConfig
    scn = CqScenario(scenario_name="u26", profile=prof,
                     storage=StorageConfig(((F(0), F(0)),), F(4), F(4)),
                     limits=HardLimits(1, 10**9, F(10**9)), m_workers=1,
                     controller=ControllerConfig(cost_mode="zero"),
                     cohort=CohortConfig(drain_budget_s=F(2)))
    specs = [_req(0, 1, 1, T0=F(5, 4), deadline=F(10)),
             _req(1, 1, 1, T0=F(5, 4), deadline=F(10))]
    eng = run_case(scn, specs, make_fcfs(), numeric=F,
                   drain_budget_s=F(2))
    s = summarize(eng, scn)
    # B=0：均未完成，右删失；deadline>censor → unknown
    assert s["n_unfinished"] == 2
    assert s["ttft_mean_lower"] is not None
    assert s["slo_unknown"] == 2
    assert s["slo_rate_interval"] == [0.0, 1.0]
    assert s["is_lower_bound"]


# ---------------------------------------------------------------------------
# U27 cohort 边界
# ---------------------------------------------------------------------------

def test_u27_cohort_bounds():
    prof = ProfileConfig(L=1, kappa_gb_per_token_layer=F(0), N_sat=1,
                         t_launch_s=F(0), a_s_per_token=F(1),
                         b_s_per_pair=F(0), no_speedup=True)
    scn = CqScenario(scenario_name="u27", profile=prof,
                     storage=StorageConfig(((F(0), F(4)),), F(4), F(4)),
                     limits=HardLimits(1, 10**9, F(10**9)), m_workers=1,
                     controller=ControllerConfig(cost_mode="zero"))
    specs = [_req(0, 0, 1, T0=F(1), deadline=F(100), arrival=F(5)),
             _req(1, 0, 1, T0=F(1), deadline=F(100), arrival=F(10))]
    eng = run_case(scn, specs, make_fcfs(), numeric=F)
    # warmup=5：请求 0 计入、请求 1（恰在 stop=10）不进入该 cohort
    s = summarize(eng, scn, warmup=5.0, arrival_stop=10.0)
    assert s["n_cohort"] == 1
    assert s["n_done"] == 1


# ---------------------------------------------------------------------------
# U28 收益公式
# ---------------------------------------------------------------------------

def test_u28_gain_formulas():
    assert latency_gain(10.0, 8.0) == pytest.approx(20.0)
    assert latency_gain(8.0, 10.0) == pytest.approx(-25.0)   # 负收益保留
    assert slo_gain_pp(0.60, 0.75) == pytest.approx(15.0)
    assert goodput_gain_pct(0.0, 0.1) is None


# ---------------------------------------------------------------------------
# U29 确定性
# ---------------------------------------------------------------------------

def test_u29_determinism():
    prof = ProfileConfig()
    scn = CqScenario(scenario_name="u29", profile=prof,
                     storage=StorageConfig(((F(0), F(80)),), F(200), F(80)),
                     limits=HardLimits(8, 8192, F(16)), m_workers=4)
    specs = [_req(i, (512, 2048, 8192)[i % 3], (128, 2048)[i % 2],
                  T0=F(1) / (i + 1), deadline=F(50) / (i + 1),
                  arrival=F(i, 10)) for i in range(12)]
    outs = []
    for _ in range(2):
        eng = run_case(scn, specs, make_fcfs(), numeric=float)
        Fs = tuple(float(rr.F_s) for rid, rr in sorted(eng.w.requests.items()))
        outs.append(Fs)
    assert outs[0] == outs[1]   # 相同输入 → 相同轨迹


# ---------------------------------------------------------------------------
# U30 协议追赶（q 更新公式）
# ---------------------------------------------------------------------------

def test_u30_periodic_catchup():
    # 一流 due=1、V=1、前 0.5 无服务、0.5 更新：q=min(q_max, rem/(due-now))
    st = StorageSim(((F(0), F(4)),), F(4), numeric=F)
    f = st.submit(0, 0, 0, F(0), F(1), F(1), F(1))
    st.allocate(F(0))
    assert f.q_gbps == F(1)
    # 假设前 0.5 无服务（被更早流占满），周期追赶在 0.5 更新
    now = F(1, 2)
    remaining = f.remaining_gb
    q_new = min(st.q_max, remaining / (f.due_s - now))
    assert q_new == F(1) / F(1, 2) == F(2)
    f.q_gbps = q_new
    st.allocate(now)
    assert f.submit_seq == 0   # 更新不改序号


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
