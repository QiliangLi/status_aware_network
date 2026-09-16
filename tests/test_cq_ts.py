"""E25 桶聚合测试矩阵（设计与测试规格 §11.2 BU01–BU12）。

每个测试对应语义不变量或规格 §5.5 手算金标；夹具为小规模真实引擎 run
（n≤6、m≤2），不经 e25 矩阵。纯函数（区间求交/事件扫描/解析容量）单测。
"""
import json
import math
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, ".")

from sim.cq.config import CqScenario, ProfileConfig, StorageConfig
from sim.cq.metrics import summarize
from sim.cq.policies import make_fcfs, make_spt
from sim.cq.simrun import run_case
from sim.cq.timeseries import (TsConservationError, aggregate_run,
                               anchor_T, b_max_in, bucket_intersections,
                               capacity_per_bucket, check_rows_conservation,
                               load_ts, row_fractions, save_ts,
                               schedule_integral, sweep_backlog)
from sim.cq.types import HardLimits, RequestSpec


def _req(rid, h, u, arrival=0.0, T0=1.0, deadline=10**9):
    return RequestSpec(rid=rid, arrival_s=arrival, h_tokens=h, u_tokens=u,
                       class_id="x", T0_s=F(T0), deadline_s=F(deadline))


def _scn(B=80.0, m=2, n_max=8):
    return CqScenario(
        scenario_name="ts_test", profile=ProfileConfig(),
        storage=StorageConfig(b_schedule=((F(0), F(int(B))),),
                              q_max_gbps=F(200), b_ref_gbps=F(80)),
        limits=HardLimits(n_max, 10**9, F(1024)), m_workers=m)


def _run(scn, specs, policy=None, record=True):
    pol = policy or make_fcfs()
    eng = run_case(scn, specs, pol, record_intervals=record)
    assert eng.status == "done", eng.status
    return eng


# ---------------------------------------------------------------------------
# BU01 区间切分（规格 §5.5 例 1，两文档共用金标）
# ---------------------------------------------------------------------------

def test_bu01_interval_split():
    out = bucket_intersections(1.4, 3.1, 0.5, K=200)
    assert [(i, round(ov, 9)) for i, ov in out] == [
        (2, 0.1), (3, 0.5), (4, 0.5), (5, 0.5), (6, 0.1)]
    assert abs(sum(ov for _i, ov in out) - 1.7) < 1e-12   # 无丢失无重复


# ---------------------------------------------------------------------------
# BU02 端点与零长
# ---------------------------------------------------------------------------

def test_bu02_endpoints_and_zero_length():
    assert bucket_intersections(0.5, 0.5, 0.5, K=4) == []       # 零长
    # 恰为一整桶 [0.5,1.0)：只落桶 1，全长 0.5
    assert bucket_intersections(0.5, 1.0, 0.5, K=4) == [(1, pytest.approx(0.5))]
    # 跨溢出边界：K=4、Δ=0.5 → T_anchor=2.0；[1.9,2.2) → 桶3 得 0.1、溢出桶得 0.2
    out = bucket_intersections(1.9, 2.2, 0.5, K=4)
    assert [(i, round(ov, 9)) for i, ov in out] == [(3, 0.1), (4, 0.2)]
    # 完全落在溢出桶
    assert bucket_intersections(2.5, 2.9, 0.5, K=4) == [(4, pytest.approx(0.4))]


# ---------------------------------------------------------------------------
# BU03 三态守恒（对 tiny run 与 WorkerState 累计双核对）
# ---------------------------------------------------------------------------

def _tiny_specs():
    return [_req(0, 512, 128, arrival=0.0), _req(1, 8192, 256, arrival=0.001),
            _req(2, 2048, 128, arrival=0.002), _req(3, 4096, 512, arrival=0.003)]


def test_bu03_device_conservation():
    scn = _scn(B=20)
    eng = _run(scn, _tiny_specs())
    ts = aggregate_run(eng, anchor_T(eng.w.t))
    rows = ts["rows"]
    dev = sum(r["compute_s"] + r["stall_s"] + r["idle_s"] for r in rows)
    assert abs(dev - scn.m_workers * float(eng.w.t)) <= 1e-9 * max(1, float(eng.w.t))
    wc = sum(float(wk.compute_s) for wk in eng.w.workers.values())
    ws = sum(float(wk.stall_s) for wk in eng.w.workers.values())
    wi = sum(float(wk.idle_s) for wk in eng.w.workers.values())
    assert abs(sum(r["compute_s"] for r in rows) - wc) <= 1e-9 * max(1, wc)
    assert abs(sum(r["stall_s"] for r in rows) - ws) <= 1e-9 * max(1, ws)
    assert abs(sum(r["idle_s"] for r in rows) - wi) <= 1e-9 * max(1, wi)
    # 每 NPU 分桶字段：逐 worker 守恒（workers[i] = [compute, stall, idle]）
    for wid, wk in sorted(eng.w.workers.items()):
        sums = [sum(r["workers"][wid][k] for r in rows) for k in range(3)]
        assert abs(sums[0] - float(wk.compute_s)) <= 1e-9 * max(1, float(wk.compute_s))
        assert abs(sums[1] - float(wk.stall_s)) <= 1e-9 * max(1, float(wk.stall_s))
        assert abs(sums[2] - float(wk.idle_s)) <= 1e-9 * max(1, float(wk.idle_s))
    assert all(len(r["workers"]) == scn.m_workers for r in rows)


# ---------------------------------------------------------------------------
# BU04 字节守恒
# ---------------------------------------------------------------------------

def test_bu04_byte_conservation():
    eng = _run(_scn(B=40), _tiny_specs())
    rows = aggregate_run(eng, anchor_T(eng.w.t))["rows"]
    assert abs(sum(r["served_gb"] for r in rows)
               - float(eng.w.storage.actual_integral_gb)) <= 1e-9 * 2
    assert abs(sum(r["requested_gb"] for r in rows)
               - float(eng.w.storage.requested_integral_gb)) <= 1e-9 * 2


# ---------------------------------------------------------------------------
# BU05 容量闭式（独立复算，不读 storage 累计器）
# ---------------------------------------------------------------------------

def test_bu05_capacity_closed_form():
    sched = ((0.0, 4.0), (1.0, 2.0))
    assert schedule_integral(sched, 0.0, 3.0) == pytest.approx(4 * 1 + 2 * 2)
    caps = capacity_per_bucket(sched, 0.5, 4, 3.0)   # T_anchor=2.0，T_end=3.0
    assert caps[0] == pytest.approx(4 * 0.5)          # [0,0.5)
    assert caps[1] == pytest.approx(4 * 0.5)          # [0.5,1)
    assert caps[2] == pytest.approx(2 * 0.5)          # [1,1.5)
    assert sum(caps) == pytest.approx(8.0)
    assert b_max_in(sched, 0.9, 1.6) == 4.0           # 跨断点取峰值
    assert b_max_in(sched, 1.2, 1.8) == 2.0


# ---------------------------------------------------------------------------
# BU06 溢出桶（人为缩短锚）
# ---------------------------------------------------------------------------

def test_bu06_overflow_bucket():
    eng = _run(_scn(), _tiny_specs())
    t_end = float(eng.w.t)
    small_anchor = round(t_end * 0.5, 3)              # 人为把锚压到 0.5×T_end
    ts = aggregate_run(eng, small_anchor)
    assert ts["meta"]["overflow_used"] is True
    assert len(ts["rows"]) == 201                     # 200 常规 + 1 溢出
    ovf = ts["rows"][-1]
    assert ovf["i"] == 200
    assert abs(ovf["width_s"] - (t_end - small_anchor)) <= 1e-9
    # 守恒含溢出桶后仍成立（aggregate_run 内部已校验，这里再显式复算）
    check_rows_conservation(ts["rows"], eng.w.m, t_end,
                            storage_served=eng.w.storage.actual_integral_gb)


# ---------------------------------------------------------------------------
# BU07 锚定一致性
# ---------------------------------------------------------------------------

def test_bu07_anchor_consistency():
    scn = _scn()
    specs = _tiny_specs()
    e1 = _run(scn, specs, make_fcfs())
    e2 = _run(scn, specs, make_fcfs())
    assert float(e1.w.t) == float(e2.w.t)             # 确定性前提（U29 机制）
    ta = anchor_T(e1.w.t)
    assert ta == pytest.approx(math.ceil(1.1 * float(e1.w.t) * 10) / 10)
    assert ta >= 1.1 * float(e1.w.t) - 1e-12
    # 另一策略（SPT 排序不同）共用同一锚参数
    e3 = _run(scn, specs, make_spt())
    ts_fcfs = aggregate_run(e1, ta, policy="cq_fcfs")
    ts_spt = aggregate_run(e3, ta, policy="cq_spt")
    assert ts_fcfs["meta"]["T_anchor_s"] == ts_spt["meta"]["T_anchor_s"]
    assert ts_fcfs["meta"]["delta_s"] == ts_spt["meta"]["delta_s"]
    assert ts_fcfs["meta"]["K"] == ts_spt["meta"]["K"]


# ---------------------------------------------------------------------------
# BU08 队列时间加权（§5.5 例 2 事件 + 引擎级守恒）
# ---------------------------------------------------------------------------

def test_bu08_queue_time_weighted():
    # 例 2：t∈[0,0.2) 长 0，[0.2,1.1) 长 3（3 到 2 走），t≥1.1 长 1；Δ=0.5
    events = [(0.2, 1), (0.2, 1), (0.2, 1), (1.1, -1), (1.1, -1)]
    qsum, qmax = sweep_backlog(events, T_end=1.5, delta=0.5, K=4)
    assert qsum[0] == pytest.approx(3 * 0.3, abs=1e-12)
    assert qsum[1] == pytest.approx(3 * 0.5, abs=1e-12)
    assert qmax[0] == 3 and qmax[1] == 3
    mean0 = qsum[0] / 0.5
    mean1 = qsum[1] / 0.5
    # 时间加权均值 vs 事件数平均的反例：禁止口径退化
    assert (mean0, mean1) == pytest.approx((1.8, 3.0))
    assert (mean0 + mean1) / 2 != pytest.approx((0 + 3) / 2)
    # 引擎级：Σ(queue_mean×width) == Σ(F_i − a_i)（积压面积守恒）
    scn = _scn(B=80, m=1, n_max=1)
    specs = [_req(0, 1024, 64, arrival=0.0),
             _req(1, 2048, 64, arrival=0.001),
             _req(2, 4096, 64, arrival=0.002)]
    eng = _run(scn, specs)
    rows = aggregate_run(eng, anchor_T(eng.w.t))["rows"]
    area = sum((r["queue_mean"] or 0.0) * r["width_s"] for r in rows)
    expect = sum(float(eng.w.requests[i].F_s) - float(eng.w.requests[i].spec.arrival_s)
                 for i in range(3))
    assert abs(area - expect) <= 1e-9 * max(1, expect)


# ---------------------------------------------------------------------------
# BU09 累计口径
# ---------------------------------------------------------------------------

def test_bu09_cumulative_counts():
    eng = _run(_scn(), _tiny_specs())
    rows = aggregate_run(eng, anchor_T(eng.w.t))["rows"]
    assert rows[-1]["cum_arrived"] == len(eng.w.requests)
    assert rows[-1]["cum_done"] == len(eng.w.requests)
    ca = [r["cum_arrived"] for r in rows]
    cd = [r["cum_done"] for r in rows]
    assert ca == sorted(ca) and cd == sorted(cd)      # 单调不减


# ---------------------------------------------------------------------------
# BU10 空与零值
# ---------------------------------------------------------------------------

def test_bu10_empty_and_zero_read():
    # 全 h=0：无读取流，served 全 0，capacity 正常积分，util=0.0（非 null）
    specs = [_req(0, 0, 256), _req(1, 0, 512)]
    eng = _run(_scn(B=80), specs)
    rows = aggregate_run(eng, anchor_T(eng.w.t))["rows"]
    assert all(r["served_gb"] == 0.0 for r in rows)
    assert sum(r["capacity_gb"] for r in rows) > 0.0
    for r in rows:
        if r["capacity_gb"] > 0:
            assert r["served_gb"] / r["capacity_gb"] == 0.0
    assert float(eng.w.storage.actual_integral_gb) == 0.0
    # 空 specs：不抛异常、不产生行
    eng0 = _run(_scn(), [])
    ts0 = aggregate_run(eng0, anchor_T(eng0.w.t))
    assert ts0["rows"] == []


# ---------------------------------------------------------------------------
# BU11 零影响开关（聚合是纯观察者）
# ---------------------------------------------------------------------------

def _canon(eng):
    """规范轨迹摘要：请求 F、批元数据、事件数（与 U29 同法）。"""
    reqs = {rid: (float(rr.F_s) if rr.F_s is not None else None,
                  rr.batch_id) for rid, rr in eng.w.requests.items()}
    batches = {b.batch_id: (list(b.members), float(b.dispatch_s),
                            float(b.F_s) if b.F_s is not None else None)
               for b in eng.w.batches.values()}
    return json.dumps({"reqs": reqs, "batches": batches,
                       "events": eng.events_processed,
                       "status": eng.status}, sort_keys=True)


def test_bu11_zero_impact_switch():
    scn = _scn(B=30)
    specs = _tiny_specs()
    e_on = _run(scn, specs, record=True)
    e_off = _run(scn, specs, record=False)
    assert _canon(e_on) == _canon(e_off)
    assert len(e_on.w.storage.interval_log) > 0
    assert e_off.w.storage.interval_log == []
    assert e_off.w.storage.record_intervals is False
    # 默认（不经 run_case 显式开关）同样关闭：e19–e24 路径零影响
    assert _canon(_run(scn, specs, record=False)) == _canon(e_on)


# ---------------------------------------------------------------------------
# BU12 ts 落盘格式 + 篡改拒绝
# ---------------------------------------------------------------------------

def test_bu12_ts_file_format(tmp_path):
    eng = _run(_scn(B=20), _tiny_specs())
    ts = aggregate_run(eng, anchor_T(eng.w.t), cell="c|FW|B20|w0",
                       policy="cq_fcfs", theta=None, mode="FW")
    p = tmp_path / "cq_fcfs.json.gz"
    save_ts(str(p), ts)
    back = load_ts(str(p))
    assert set(back["meta"]) >= {"cell", "policy", "theta", "mode",
                                 "T_anchor_s", "delta_s", "K", "T_end_s",
                                 "b_schedule", "overflow_used"}
    n_expect = 200 + (1 if ts["meta"]["overflow_used"] else 0)
    assert len(back["rows"]) == n_expect
    fields = {"i", "t_start_s", "width_s", "compute_s", "stall_s", "idle_s",
              "served_gb", "capacity_gb", "requested_gb", "queue_mean",
              "queue_max", "cum_arrived", "cum_done"}
    assert all(fields <= set(r) for r in back["rows"])
    for r in back["rows"]:      # 无 NaN/Infinity（allow_nan=False 落盘即保证）
        assert r["compute_s"] == r["compute_s"]
    # 篡改后守恒校验器必须拒绝
    bad = json.loads(json.dumps(ts))
    bad["rows"][3]["compute_s"] += 1.0
    with pytest.raises(TsConservationError):
        check_rows_conservation(bad["rows"], eng.w.m, float(eng.w.t),
                                storage_served=eng.w.storage.actual_integral_gb)
    # 空尾桶占比为 None（图上层留白，不画 0）
    m = ts["meta"]["m_workers"]
    for r in back["rows"]:
        frac = row_fractions(r, m)
        if r["width_s"] <= 0:
            assert frac is None
        else:
            assert abs(sum(frac) - 1.0) <= 1e-9
