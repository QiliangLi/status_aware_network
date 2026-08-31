"""仿真器不变量测试：解析解吻合、字节守恒、确定性、策略单调性。"""
from __future__ import annotations

import numpy as np
import pytest
import simpy

from sim.config import GpuConfig, ObsConfig, RunSpec, StorageConfig, stable
from sim.gpu import GpuPool, PrefillCurve
from sim.storage import SharedKVStorage


def drive(env, proc):
    env.process(proc(env))
    env.run()


class TestStorage:
    def test_single_fetch_analytic(self):
        env = simpy.Environment()
        s = SharedKVStorage(env, 0, StorageConfig(b_total=100.0, t_base=0.005, bg_schedule=stable(0.0)))

        def proc(env):
            ev = s.submit(1, 10.0)
            yield ev
            assert env.now == pytest.approx(0.105, abs=1e-6)

        drive(env, proc)

    def test_equal_sharing(self):
        env = simpy.Environment()
        s = SharedKVStorage(env, 0, StorageConfig(b_total=100.0, t_base=0.005))

        def proc(env):
            e1 = s.submit(1, 10.0)
            e2 = s.submit(2, 10.0)
            done = []
            for ev in (e1, e2):
                r = yield ev
                done.append(env.now)
            # 两笔各分 50GB/s：10GB/50 = 0.2s + t_base
            assert done[0] == pytest.approx(0.205, abs=1e-6)
            assert done[1] == pytest.approx(0.205, abs=1e-6)

        drive(env, proc)

    def test_starvation_release(self):
        env = simpy.Environment()
        s = SharedKVStorage(env, 0, StorageConfig(
            b_total=100.0, t_base=0.005, bg_schedule=((0.0, 100.0), (1.0, 0.0))))

        def proc(env):
            ev = s.submit(1, 10.0)
            yield ev
            # t_base 门控(0.005s)在饿死期内已打开，t=1.0 释放后传 10GB/100GB/s
            assert env.now == pytest.approx(1.1, abs=1e-6)

        drive(env, proc)

    def test_byte_conservation(self):
        env = simpy.Environment()
        s = SharedKVStorage(env, 0, StorageConfig(b_total=100.0, t_base=0.0, bg_schedule=stable(40.0)))
        sizes = [3.0, 7.0, 12.0]

        def proc(env):
            evs = [s.submit(i, b) for i, b in enumerate(sizes)]
            for ev in evs:
                yield ev
            assert s.bytes_served == pytest.approx(sum(sizes), rel=1e-9)
            # 全程可用容量 60GB/s，完成时间不可能早于总字节/60
            assert env.now >= sum(sizes) / 60.0 - 1e-9

        drive(env, proc)

    def test_hypothetical_matches_actual(self):
        # oracle 自顾预测：提交前预测的完成时间与真实完成时间应接近
        env = simpy.Environment()
        s = SharedKVStorage(env, 0, StorageConfig(b_total=100.0, t_base=0.005))

        def proc(env):
            for i in range(10):
                s.submit(100 + i, 10.0)
            pred = s.hypothetical_fetch_time(5.0)
            ev = s.submit(999, 5.0)
            yield ev
            assert pred == pytest.approx(env.now, rel=0.15)

        drive(env, proc)


class TestGpu:
    def test_single_job_bg(self):
        env = simpy.Environment()
        g = GpuPool(env, 0, PrefillCurve(GpuConfig().prefill_table),
                    GpuConfig(bg_schedule=stable(0.5)))

        def proc(env):
            ev = g.submit(1, 32768)
            yield ev
            assert env.now == pytest.approx(0.205 / 0.5, abs=1e-6)

        drive(env, proc)

    def test_fcfs_queue(self):
        env = simpy.Environment()
        g = GpuPool(env, 0, PrefillCurve(GpuConfig().prefill_table), GpuConfig())

        def proc(env):
            e1 = g.submit(1, 32768)
            e2 = g.submit(2, 32768)
            t1 = (yield e1) or env.now
            assert env.now == pytest.approx(0.205, abs=1e-6)
            yield e2
            assert env.now == pytest.approx(0.410, abs=1e-6)

        drive(env, proc)


def _spec(policy, bg=80.0, gpu=0.4, seed=0, duration=120.0):
    return RunSpec(
        exp="test", policy=policy, seed=seed, duration=duration, warmup=15.0, margin=60.0,
        storages=(StorageConfig(bg_schedule=stable(bg)),),
        gpu=GpuConfig(bg_schedule=stable(gpu)),
        obs=ObsConfig(interval=0.05),
    )


class TestRun:
    def test_determinism(self):
        from sim.simrun import run_once
        a = run_once(_spec("p2", seed=3))
        b = run_once(_spec("p2", seed=3))
        assert a == b

    def test_low_pressure_all_fetch(self):
        # 存储空闲时应全 fetch，goodput 接近满
        from sim.simrun import run_once
        r = run_once(_spec("p1", bg=0.0, gpu=0.2))
        assert r["fetch_count"] > 0
        assert r["recompute_count"] == pytest.approx(r["n_arr"] * 0.3, abs=r["n_arr"] * 0.05 + 5)
        assert r["slo_rate"] > 0.95

    def test_monotonicity_under_pressure(self):
        from sim.simrun import run_once
        meds = {}
        for pol in ["p0", "p1", "p2", "p4"]:
            vals = [run_once(_spec(pol, seed=s, duration=150.0))["goodput"] for s in range(3)]
            meds[pol] = float(np.median(vals))
        assert meds["p4"] >= meds["p2"] - 1e-9
        assert meds["p2"] >= meds["p1"] - 1e-9
        assert meds["p1"] >= meds["p0"] - 1e-9
        assert meds["p2"] > meds["p1"]  # 高压格子必须有分离
