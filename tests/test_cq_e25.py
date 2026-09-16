"""E25 因子地图测试（设计与测试规格 §11.3 FM01–FM08、§11.4 FG01–FG05）。

FM 用小规模真实 run（手造 specs 或 synthetic 窗口短截断）断言记录/配置口径；
FG 断言图与落盘数据同源、发布不覆盖旧图。聚合器本身由 BU 组覆盖。
"""
import json
import os
import sys
from fractions import Fraction as F

import numpy as np
import pytest

sys.path.insert(0, ".")

from sim.cq.config import ProfileConfig
from sim.cq.metrics import summarize
from sim.cq.policies import SIMPLE_POLICIES, make_fcfs
from sim.cq.simrun import run_case
from sim.cq.timeseries import b_max_in, load_ts, row_fractions
from sim.cq.types import RequestSpec
from sim.experiments import e25_factor as e5
from sim.experiments.cq_common import MooncakeSource, default_scenario
from sim.experiments.e25_factor import (ALL_POLICIES, MPC_LIKE, PROFILES,
                                        cell_id, cell_limits, fw_specs,
                                        policies_for, run_cell,
                                        run_cell_specs, windows_for_seeds)


def _req(rid, h, u, arrival=0.0):
    return RequestSpec(rid=rid, arrival_s=arrival, h_tokens=h, u_tokens=u,
                       class_id="x", T0_s=F(1), deadline_s=F(10**9))


def _cell(**kw):
    c = {"factor_line": "baseline", "mode": "FW",
         "file": "synthetic_trace.jsonl", "B": 80.0, "rho": None,
         "alpha": 4, "cap": "C", "profile": "default", "window": 0,
         "_stage": "eval"}
    c.update(kw)
    return c


def _run_cell_tmp(tmp_path, cell, specs, min_reqs=1, d_sim=None):
    d = str(tmp_path)
    recs, prog = [], {}
    out = run_cell_specs(dict(cell), specs, d, recs, prog, d_sim=d_sim,
                         min_reqs=min_reqs)
    return out, recs, prog, d


# ---------------------------------------------------------------------------
# FM01 四目标齐全 + M 语义独立重算
# ---------------------------------------------------------------------------

def test_fm01_four_objectives(tmp_path):
    specs = [_req(i, 512 * (i % 3 + 1), 64 * (i % 4 + 1)) for i in range(8)]
    out, _r, _p, _d = _run_cell_tmp(tmp_path, _cell(), specs)
    assert len(out) == 32                      # 8 策略 × θ{M,S,T,R}
    for pid in ALL_POLICIES:
        ths = {r["theta"] for r in out if r["policy"] == pid}
        assert ths == {"M", "S", "T", "R"}, (pid, ths)
    # M 语义：与独立重放的引擎逐值一致（确定性）
    scn = default_scenario(B_gbps=80.0, alpha=F(4), limits=cell_limits("C"),
                           profile=PROFILES["default"])
    eng = run_case(scn, specs, make_fcfs())
    s = summarize(eng, scn)
    rec = next(r for r in out if r["policy"] == "cq_fcfs" and r["theta"] == "M")
    if s["makespan_M"] is not None:
        assert rec["makespan_M"] == pytest.approx(s["makespan_M"])
        assert rec["makespan_M"] == pytest.approx(
            max(float(rr.F_s) for rr in eng.w.requests.values()))
    else:
        assert rec["makespan_M"] is None
        assert rec["makespan_M_lower"] is not None


# ---------------------------------------------------------------------------
# FM02 OL drain 口径与主窗/排空拆分
# ---------------------------------------------------------------------------

def test_fm02_ol_drain_split(tmp_path):
    src = MooncakeSource("synthetic_trace.jsonl")
    cell = _cell(mode="OL", rho=0.6)
    d = str(tmp_path)
    recs, prog = [], {}
    out = run_cell(src, dict(cell), d, recs, prog, n_req=128,
                   duration_cap=2.0)
    if not out:                                 # 窗口太短被跳过则放大截断
        out = run_cell(src, dict(cell), d, recs, prog, n_req=128,
                       duration_cap=4.0)
    assert out, "OL 小窗口应产生记录"
    for r in out:
        assert r["drain_M"] is not None
        assert r["drain_M"] == r["makespan_M_lower"]
        if r["makespan_M"] is not None:         # 全部完成时两者一致
            assert r["makespan_M"] == pytest.approx(r["drain_M"])
        for k in ("compute", "stall", "idle"):
            m, dr = r[f"device_{k}_main_s"], r[f"device_{k}_drain_s"]
            full = r[f"device_{k}_s"]
            assert abs(m + dr - full) <= 1e-9 * max(1.0, full)


# ---------------------------------------------------------------------------
# FM03 CRN 配对
# ---------------------------------------------------------------------------

def test_fm03_crn_pairing(tmp_path):
    specs = [_req(i, 512 * (i % 3 + 1), 128 * (i % 4 + 1)) for i in range(8)]
    out, _r, _p, _d = _run_cell_tmp(tmp_path, _cell(), specs)
    assert len({r["cell_id"] for r in out}) == 1
    assert len({r["trace_hash"] for r in out}) == 1
    served = {r["storage_served_gb"] for r in out}
    assert len(served) == 1 or max(served) - min(served) <= 1e-9 * max(served)


# ---------------------------------------------------------------------------
# FM04 单因子隔离（因子元数据逐字段）
# ---------------------------------------------------------------------------

def test_fm04_single_factor_isolation(tmp_path):
    specs = [_req(i, 512, 64) for i in range(8)]
    base_out, *_ = _run_cell_tmp(tmp_path, _cell(), specs)
    b20_out, *_ = _run_cell_tmp(tmp_path, _cell(factor_line="line_B", B=20.0),
                                specs)
    rb = next(r for r in base_out if r["policy"] == "cq_fcfs" and r["theta"] == "M")
    r2 = next(r for r in b20_out if r["policy"] == "cq_fcfs" and r["theta"] == "M")
    for k in ("alpha", "cap", "profile", "rho", "mode", "file", "window"):
        assert rb[k] == r2[k], k
    assert rb["B"] == 80.0 and r2["B"] == 20.0
    assert r2["factor_line"] == "line_B" and rb["factor_line"] == "baseline"
    assert r2["scenario"] != rb["scenario"]


# ---------------------------------------------------------------------------
# FM05 跨 θ 复用
# ---------------------------------------------------------------------------

def test_fm05_theta_reuse(tmp_path):
    specs = [_req(i, 512, 64 * (i + 1)) for i in range(8)]
    out, _r, _p, _d = _run_cell_tmp(tmp_path, _cell(), specs)
    spt = [r for r in out if r["policy"] == "cq_spt"]
    assert len({r["ts_path"] for r in spt}) == 1     # 简单策略共享一份 ts
    for k in ("makespan_M", "n_batches", "device_compute_s"):
        vals = {round(float(r[k]), 12) for r in spt if r[k] is not None}
        assert len(vals) <= 1, k
    mpc = [r for r in out if r["policy"] == "cq_mpc"]
    assert len({r["ts_path"] for r in mpc}) == 4     # MPC 每目标一份


# ---------------------------------------------------------------------------
# FM06 profile 线（组批通道打开/关闭）
# ---------------------------------------------------------------------------

def test_fm06_profile_line(tmp_path):
    specs = [_req(i, 512, 64) for i in range(8)]
    sat, *_ = _run_cell_tmp(tmp_path, _cell(profile="n_sat128"), specs)
    assert any(r["n_batches_G_gt1"] >= 1 for r in sat)
    assert any((r["batch_G_total"] or 1.0) > 1.0 + 1e-9 for r in sat)
    assert "n_sat128" in sat[0]["cell_id"]
    nos, *_ = _run_cell_tmp(tmp_path, _cell(profile="no_speedup"), specs)
    for r in nos:
        assert abs((r["batch_G_total"] or 1.0) - 1.0) <= 1e-9
        assert r["n_batches_G_gt1"] == 0


# ---------------------------------------------------------------------------
# FM07 能力块
# ---------------------------------------------------------------------------

def test_fm07_capability_blocks(tmp_path):
    # 混合形状：4 小 (512,64) + 4 大 (8192,4096)，中位数切分四象限
    specs = ([_req(i, 512, 64) for i in range(4)]
             + [_req(i + 4, 8192, 4096) for i in range(4)])
    a_out, *_ = _run_cell_tmp(tmp_path, _cell(cap="A"), specs)
    assert all(r["max_batch_size"] == 1 for r in a_out)       # 禁止组批
    b_out, *_ = _run_cell_tmp(tmp_path, _cell(factor_line="line_cap", cap="B"),
                              specs)
    assert {r["policy"] for r in b_out} == set(SIMPLE_POLICIES)  # 仅简单策略
    assert all(r["n_quadrant_viol"] == 0 for r in b_out)
    assert all(r["max_batch_size"] <= 4 for r in b_out)       # 同象限限批
    c_out, *_ = _run_cell_tmp(tmp_path, _cell(), specs)
    assert any(r["max_batch_size"] >= 2 for r in c_out)       # 自由混批


# ---------------------------------------------------------------------------
# FM08 窗口映射与跳过
# ---------------------------------------------------------------------------

def test_fm08_window_mapping_and_skip(tmp_path):
    assert windows_for_seeds(2) == [0, 4]
    assert windows_for_seeds(5) == [0, 4, 9, 14, 19]
    specs = [_req(i, 512, 64) for i in range(5)]
    out, _r, prog, _d = _run_cell_tmp(tmp_path, _cell(), specs, min_reqs=16)
    assert out == []
    note = prog[cell_id(_cell())]
    assert "insufficient_data" in note["skip"]


# ---------------------------------------------------------------------------
# FG01 图 A 同源（数据缓存 == records 重算中位数）
# ---------------------------------------------------------------------------

def test_fg01_fig_a_same_source(tmp_path):
    specs = [_req(i, 512 * (i % 3 + 1), 64 * (i % 4 + 1)) for i in range(8)]
    out, recs, _p, d = _run_cell_tmp(tmp_path, _cell(), specs)
    e5.fig_objectives(recs, d, e5.setup_matplotlib())
    cache = json.load(open(os.path.join(d, "fig_e25_objectives_data.json")))
    assert os.path.exists(os.path.join(d, "fig_e25_objectives.png"))
    for mkey, key, _yl in e5._METRIC_FW:
        for pid in ALL_POLICIES:
            vs = [e5._metric_get(r, key) for r in recs
                  if r["policy"] == pid and e5._theta_match(r, mkey)
                  and e5._metric_get(r, key) is not None]
            if vs:
                assert cache["median"][pid][key] == pytest.approx(
                    float(np.median(vs))), (pid, key)


# ---------------------------------------------------------------------------
# FG02/FG03/FG04 时间序列图数据口径
# ---------------------------------------------------------------------------

def _one_ts(records, d):
    r = next(r for r in records if r.get("ts_path"))
    return load_ts(os.path.join(d, r["ts_path"])), r


def test_fg02_frac_normalization(tmp_path):
    specs = [_req(i, 512 * (i + 1), 64 * (i + 1)) for i in range(6)]
    out, _r, _p, d = _run_cell_tmp(tmp_path, _cell(), specs)
    ts, _rec = _one_ts(out, d)
    m = ts["meta"]["m_workers"]
    for r in ts["rows"]:
        frac = row_fractions(r, m)
        if frac is None:
            assert r["width_s"] <= 0.0        # 空尾桶 → 留白（不画 0）
        else:
            assert sum(frac) == pytest.approx(1.0, abs=1e-9)


def test_fg03_rate_within_B(tmp_path):
    specs = [_req(i, 4096 * (i + 1), 128) for i in range(6)]
    out, _r, _p, d = _run_cell_tmp(tmp_path, _cell(B=20.0), specs)
    ts, _rec = _one_ts(out, d)
    sched = ts["meta"]["b_schedule"]
    for r in ts["rows"]:
        if r["width_s"] > 0:
            rate = r["served_gb"] / r["width_s"]
            assert rate <= b_max_in(sched, r["t_start_s"],
                                    r["t_start_s"] + r["width_s"]) + 1e-9


def test_fg04_queue_peak_matches_summary(tmp_path):
    specs = [_req(i, 512, 64 * (i + 1)) for i in range(8)]
    out, _r, _p, d = _run_cell_tmp(tmp_path, _cell(), specs)
    for rec in out:
        if not rec.get("ts_path"):
            continue
        ts = load_ts(os.path.join(d, rec["ts_path"]))
        peak = max(r["queue_max"] for r in ts["rows"])
        assert peak == rec["max_backlog"], rec["policy"]


# ---------------------------------------------------------------------------
# FG05 发布隔离（只新增 cq_fig_e25_*，不触碰旧图）
# ---------------------------------------------------------------------------

def test_fg05_publish_isolation(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    docs = tmp_path / "docs"
    docs.mkdir()
    old = docs / "cq_fig_e19_trace_profile.png"
    old.write_bytes(b"OLD")
    old_mtime = old.stat().st_mtime_ns
    (res / "fig_e25_objectives.png").write_bytes(b"NEW_A")
    (res / "fig_e25_ts_queue.png").write_bytes(b"NEW_F")
    n = e5.publish_figures(str(res), str(docs))
    assert n == 2
    assert (docs / "cq_fig_e25_objectives.png").read_bytes() == b"NEW_A"
    assert (docs / "cq_fig_e25_ts_queue.png").read_bytes() == b"NEW_F"
    assert old.read_bytes() == b"OLD"
    assert old.stat().st_mtime_ns == old_mtime
    assert not any(f.startswith("cq_fig_e2") and "e25" not in f
                   for f in os.listdir(docs) for f in [f]
                   if f != "cq_fig_e19_trace_profile.png") or True
    # 精确断言：docs 下只有 1 个旧文件 + 2 个新文件
    assert sorted(os.listdir(docs)) == [
        "cq_fig_e19_trace_profile.png", "cq_fig_e25_objectives.png",
        "cq_fig_e25_ts_queue.png"]


# ---------------------------------------------------------------------------
# MPC 修复回归（20260917 审计）：WAIT 保持 / 决策点分支 / 预算根因
# ---------------------------------------------------------------------------

def test_mpc_wait_hold_forecast():
    """WAIT 保持：单实例单请求，WAIT δ 的推演 F = δ + 独占服务时间。

    修复前 forecast_drain 丢弃 WAIT（闭环 π 同刻立即派发），F 恒等于立即
    派发值 2.25；修复后 _HoldPi 在唤醒前禁止派发。
    """
    from sim.cq.engine import CqEngine
    from sim.cq.observable import Observable
    from sim.cq.policies import GuardedEDF
    from sim.cq.search import build_forecast_engine, forecast_drain
    from sim.cq.types import Action, JointAction
    from tests.cq_reference import e20_scenario, e20_specs
    scn = e20_scenario(m=1)
    specs = [e20_specs(n_long=0, n_short=1)[0]]
    obs = Observable(scn, numeric=float)
    eng = CqEngine(scn, specs, GuardedEDF(), numeric=float,
                   fallback_policy=GuardedEDF(), observable=obs)
    obs.attach(eng)
    eng._physical_step(0.0)
    snap = obs.snapshot(0.0)
    est = build_forecast_engine(snap, scn, GuardedEDF())
    Fs = forecast_drain(est, JointAction((Action("WAIT", 0, (), 0.725),)),
                        [20000])
    assert abs(Fs[0] - 2.975) < 1e-9        # 0.725 + 2.25（独占服务时间）
    est2 = build_forecast_engine(snap, scn, GuardedEDF())
    Fs2 = forecast_drain(est2, None, [20000])
    assert abs(Fs2[0] - 2.25) < 1e-9         # 无首动作 = 立即派发


def test_mpc_big_budget_deviates():
    """决策点分支修复：大预算下 MPC 偏离 π 且严格改善（四计划金标小世界）。

    修复前候选从已排空的基础副本克隆，全体得分==基础动作，永远回落 π。
    """
    from sim.cq.search import MPCPolicy
    from tests.cq_reference import e20_scenario, e20_specs
    for theta, ok in (("T", lambda Fs: sum(Fs) < 31.5 - 1e-9),
                      ("M", lambda Fs: max(Fs) < 10.75 - 1e-9)):
        pol = MPCPolicy(pid="m", H=1, theta=theta, c_ref=3.625, budget_s=10.0)
        eng = run_case(e20_scenario(), e20_specs(), pol)
        Fs = [float(rr.F_s) for rr in eng.w.requests.values()]
        assert pol.n_scored > 0, theta
        assert ok(Fs), (theta, sorted(Fs))
    # θ=S：最优候选主目标与保底打平（0 超时 = 0 超时）→ gate 拒绝 → 保底轨迹
    pol = MPCPolicy(pid="m", H=1, theta="S", c_ref=3.625, budget_s=10.0)
    eng = run_case(e20_scenario(), e20_specs(), pol)
    Fs = sorted(float(rr.F_s) for rr in eng.w.requests.values())
    assert Fs == [5.0, 5.0, 10.75, 10.75]


def test_mpc_zero_budget_starves_and_h_clamped():
    """预算根因与 H 截断的确定性回归。

    真实 trace 上 2ms 软预算低于单次基础排空成本（审计实测 decide
    P50≈23ms），候选零评分——本测试用近零预算构造同一现象；H>1 因逐
    决策点分支未实现而被按 H=1 执行并计数。
    """
    from sim.cq.search import MPCPolicy
    from tests.cq_reference import e20_scenario, e20_specs
    pol = MPCPolicy(pid="m", H=1, theta="T", c_ref=3.625, budget_s=1e-9)
    eng = run_case(e20_scenario(), e20_specs(), pol)
    assert pol.n_scored == 0                 # 预算耗尽：候选零评分
    assert pol.n_overrun >= 1
    pol2 = MPCPolicy(pid="m", H=2, theta="T", c_ref=3.625, budget_s=10.0)
    eng2 = run_case(e20_scenario(), e20_specs(), pol2)
    assert pol2.n_depth_clamped >= 1         # H=2 被 clamp 为 H=1 并记录
