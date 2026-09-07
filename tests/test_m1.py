"""G0 不变量测试（设计文档 §6 G0 + 纲领 v1.1 §11.1 的 M1 相关项）。

覆盖：解析一致（单请求 fetch/recompute 完成时间手算吻合）、资源/工作守恒、
依赖与完成顺序、容量安全、极限退化（quote→nominal 复现静态决策）、
S-G 无本地动作、排序键单调、确定性。
"""
from __future__ import annotations

import json

import pytest

from sim.config import (GpuConfig, ModelConfig, NodeConfig, ObsConfig,
                        StorageConfig, TopoConfig)
from sim.experiments.g_common import QUADRANTS, build_topo, make_classes
from sim.m1 import M1Class, M1Config, M1Req, M1Spec, M1System
from sim.policies_g import order_key

M1 = M1Config(decode_t0=0.004, decode_t1=0.0008, token_budget=4096, hbm_gb=64.0)


def _topo1(classes, n_workers=1, mem_bg=0.0, fabric_bg=0.0, gpu_bg=0.0):
    nodes = tuple(
        NodeConfig(f"n{i}", mem=StorageConfig(b_total=10.0, t_base=0.05,
                                              bg_schedule=((0.0, mem_bg),)),
                   ssd=StorageConfig(b_total=5.0))
        for i in range(2))
    fabric = StorageConfig(b_total=20.0, t_base=0.05, bg_schedule=((0.0, fabric_bg),))
    replicas = tuple((c.name, ((0, "mem"), (1, "mem"))) for c in classes if c.H > 0)
    return TopoConfig(n_workers=n_workers, nodes=nodes, fabric=fabric,
                      path_lat=((0.01, 0.01),) * n_workers,
                      local_cache_gb=0.0, replicas=replicas,
                      gpu_bgs=tuple(((0.0, gpu_bg),) for _ in range(n_workers)))


def _spec(router="ll", order="fcfs", classes=None, gpu=None, seed=0,
          duration=0.5, lam=1000.0, obs=None, m1=M1, n_workers=1,
          mem_bg=0.0, fabric_bg=0.0, gpu_bg=0.0, fixed_trace=()):
    classes = classes or (M1Class("X", 16384, 512, 3, 1.0),)
    return M1Spec(exp="t", router=router, order=order, seed=seed,
                  duration=duration, warmup=0.0, lam=lam, classes=classes,
                  topo=_topo1(classes, n_workers, mem_bg, fabric_bg, gpu_bg),
                  obs=obs or ObsConfig(interval=0.0, noise_sigma=0.0),
                  m1=m1, model=ModelConfig(), gpu=gpu or GpuConfig(bg_schedule=((0.0, 0.0),)),
                  fixed_trace=fixed_trace)


CURVE = lambda tok: M1System  # placeholder 防误用


class TestAnalyticSingleRequest:
    """解析一致：单请求、无竞争、无背景，完成时间与手算逐项吻合（R0 扩展）。"""

    def test_fetch_path_analytic(self):
        # kvgb = 2*80*8*128*2/1e9；H=16384 → gb≈5.3687 GB
        sysm = M1System(_spec(router="ll", fixed_trace=((0.001, "X"),), duration=0.01))
        out = sysm.run()
        reqs = sysm.cohort
        assert len(reqs) == 1
        r = reqs[0]
        assert r.action == "fetch"
        gb = r.kv_gb
        # 取回链路 = 路径 0.01 + (t_base + gb/10) + (t_base + gb/20)
        t_fetch = 0.01 + 0.05 + gb / 10.0 + 0.05 + gb / 20.0
        curve = sysm.curve
        t_iter1 = (curve(512) + M1.decode_t0) / 1.0        # 首迭代：suffix 512 + d0
        assert r.fetch_t1 - r.fetch_t0 == pytest.approx(t_fetch, abs=1e-9)
        assert r.t_first - r.arrival == pytest.approx(t_fetch + t_iter1, abs=1e-9)
        # 剩余 O-1=2 个 decode 迭代，每迭代 d0 + d1×1
        assert r.t_done - r.t_first == pytest.approx(2 * (M1.decode_t0 + M1.decode_t1), abs=1e-9)

    def test_recompute_path_analytic(self):
        # 放大 prefill 曲线使重算路径被 dynar 选中，并核对手算
        gpu = GpuConfig(prefill_table=((4096, 1.0), (16384, 4.0)),
                        bg_schedule=((0.0, 0.0),))
        sysm = M1System(_spec(router="ll", gpu=gpu,
                              fixed_trace=((0.001, "X"),), duration=0.01))
        out = sysm.run()
        r = sysm.cohort[0]
        assert r.action == "fetch"          # 曲线放大后取回更划算（4.1s vs 0.92s）
        # 反向：恢复默认曲线时 ll 恒取回；用 dynar 验证动态选择重算（大预算避免切块）
        m1_big = M1Config(decode_t0=0.004, decode_t1=0.0008,
                          token_budget=32768, hbm_gb=64.0)
        sysm2 = M1System(_spec(router="dynar", fixed_trace=((0.001, "X"),),
                               duration=0.01, m1=m1_big))
        sysm2.run()
        r2 = sysm2.cohort[0]
        assert r2.action == "recompute"     # 默认曲线下 16896 tokens ≈ 0.108s < 0.92s
        ttft = (sysm2.curve(16896) + M1.decode_t0) / 1.0
        assert r2.t_first - r2.arrival == pytest.approx(ttft, abs=1e-9)

    def test_chunked_prefill_budget(self):
        # U 超过 token 预算时跨迭代切块：完成时间 = ⌈U/B⌉ 段预算迭代 + d0 每迭代
        b = 256
        m1 = M1Config(decode_t0=0.004, decode_t1=0.0008, token_budget=b, hbm_gb=64.0)
        sysm = M1System(_spec(router="ll", classes=(M1Class("X", 0, 700, 2, 1.0),),
                              m1=m1, fixed_trace=((0.001, "X"),), duration=0.01))
        sysm.run()
        r = sysm.cohort[0]
        c = sysm.curve
        expect = (c(256) + M1.decode_t0) + (c(256) + M1.decode_t0) + (c(188) + M1.decode_t0)
        assert r.t_first - r.arrival == pytest.approx(expect, abs=1e-9)


class TestConservation:
    """资源/工作守恒：存储各段服务字节 == 取回字节；全部请求完成且 token 数正确。"""

    def test_bytes_and_completion(self):
        classes = (M1Class("A", 16384, 512, 8, 0.5), M1Class("B", 512, 2048, 16, 0.5))
        spec = _spec(router="ll", classes=classes, duration=4.0, lam=8.0,
                     m1=M1Config(decode_t0=0.004, decode_t1=0.0008,
                                 token_budget=4096, hbm_gb=64.0, margin_s=120.0))
        sysm = M1System(spec)
        out = sysm.run()
        fetch = [r for r in sysm.cohort if r.action == "fetch"]
        gb = sum(r.kv_gb for r in fetch)
        tier_served = sum(sysm.world.res(n, "mem").bytes_served for n in range(2))
        assert tier_served == pytest.approx(gb, abs=1e-9)
        assert sysm.world.fabric.bytes_served == pytest.approx(gb, abs=1e-9)
        # 全部完成、token 守恒、依赖顺序（首 token 在取回后）
        for r in sysm.cohort:
            assert r.t_done > 0 and r.tok == r.O and r.t_first <= r.t_done
            if r.action == "fetch":
                assert r.fetch_t1 <= r.t_first + 1e-12


class TestCapacityAndDiscipline:
    def test_hbm_capacity_safe(self):
        classes = (M1Class("A", 16384, 8192, 2048, 1.0),)   # 预留 ≈ 8.74 GB
        spec = _spec(router="ll", classes=classes, duration=6.0, lam=12.0,
                     m1=M1Config(decode_t0=0.004, decode_t1=0.0008,
                                 token_budget=4096, hbm_gb=20.0, margin_s=300.0))
        sysm = M1System(spec)
        sysm.run()
        for w in sysm.workers:
            assert w.hbm_peak_gb <= 20.0 + 1e-9
        assert sysm.workers[0].hbm_peak_gb > 0    # 确实触发过准入

    def test_no_local_action_in_sg(self):
        sysm = M1System(_spec(router="ll", duration=2.0, lam=6.0))
        sysm.run()
        acts = {r.action for r in sysm.cohort}
        assert acts <= {"fetch", "recompute", "prefill"}
        assert all(not sysm.world.locals[w].holds(r.cls)
                   for w in range(sysm.world.n_workers) for r in sysm.cohort)


class TestDegeneration:
    """极限退化：live 无噪声时 quote 估计 == nominal 静态成本（设计 §4.1 不变量）。"""

    def test_quote_equals_static(self):
        sysm = M1System(_spec(router="ll"))
        r = sysm.cohort[0] if sysm.cohort else None
        gb, w, n, t = 5.0, 0, 0, "mem"
        q = sysm.quote.estimate(gb, w, n, t, noisy=False)["time"]
        from sim.policies_g import GCtx
        ctx = GCtx(world=sysm.world, quote=sysm.quote, hist=sysm.hist,
                   curve=sysm.curve, workers=sysm.workers, m1=M1,
                   rng=None, guardband=1.2)
        s = ctx and sysm.router.static_fetch_t(gb, w, n, t)
        assert q == pytest.approx(s, abs=1e-9)


class TestOrderKeys:
    def test_key_monotonicity(self):
        sysm = M1System(_spec(router="ll", duration=0.1, lam=1.0))
        ctx = sysm.ctx
        a = M1Req(rid=1, cls="X", arrival=1.0, H=512, U=512, N=1024, O=8, O_max=8,
                  kv_gb=0.1, prio=1, slo_ttft=1.0, slo_tpot=0.05, hit=True)
        b = M1Req(rid=2, cls="X", arrival=2.0, H=4096, U=512, N=4608, O=8, O_max=8,
                  kv_gb=0.9, prio=0, slo_ttft=1.0, slo_tpot=0.05, hit=True)
        b.action, b.worker, b.node, b.tier = "fetch", 0, 0, "mem"
        for r in (a, b):
            r.okey = None
        assert order_key("fcfs", a, ctx) < order_key("fcfs", b, ctx)
        assert order_key("lpm", b, ctx) < order_key("lpm", a, ctx)      # H 大者先
        assert order_key("prio", b, ctx) < order_key("prio", a, ctx)    # prio 小者先
        # ready：无取回请求 r=0，fetch 请求 r≥0
        ka, kb = order_key("ready", a, ctx), order_key("ready", b, ctx)
        assert ka[0] == 0.0 and kb[0] >= 0.0


class TestDeterminism:
    def test_same_seed_same_summary(self):
        classes = (M1Class("A", 16384, 512, 8, 0.5), M1Class("B", 512, 2048, 16, 0.5))
        kwargs = dict(router="d1a", classes=classes, duration=4.0, lam=6.0, seed=3)

        def run():
            s = M1System(_spec(**kwargs))
            out = s.run()
            out.pop("timeline_iters", None), out.pop("timeline_res", None)
            return json.dumps(out, sort_keys=True, default=str)
        assert run() == run()


class TestSGScenario:
    """S-G 场景装配：双副本目录全局可见、象限背景生效。"""

    def test_quadrant_bg_and_replicas(self):
        classes = make_classes([("A", 16384, 512, 128, 1.0, 0)])
        topo = build_topo(classes, QUADRANTS["io"])
        spec = M1Spec(exp="t", router="ll", order="fcfs", seed=0, duration=0.1,
                      lam=1.0, classes=classes, topo=topo, obs=ObsConfig(),
                      m1=M1, model=ModelConfig(), gpu=GpuConfig(bg_schedule=((0.0, 0.0),)))
        sysm = M1System(spec)
        assert sysm.world.dir.holders("A") == {(0, "mem"), (1, "mem")}
        assert sysm.world.res(0, "mem").bg_at(0.0) == 42.0
        assert sysm.world.fabric.bg_at(0.0) == 78.0
        assert sysm.world.gpus[0].bg_at(0.0) == 0.10
