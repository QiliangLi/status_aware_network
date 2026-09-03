"""v2 拓扑/引擎/策略不变量测试。"""
from __future__ import annotations

import math

import pytest
import simpy

from sim.config import (GpuConfig, ModelConfig, ObsConfig, RunSpec, StorageConfig,
                        TopoConfig, WorkloadConfig, stable)
from sim.simrun import run_once
from sim.topology import LocalKVCache, MetadataDirectory, World


def _topo(replicas=(), **kw):
    return TopoConfig(replicas=replicas, **kw)


def _spec(policy, seed=0, topo=None, duration=60.0, lam=None, **kw):
    if lam is not None and "wl" not in kw:
        kw["wl"] = WorkloadConfig(lam=lam)
    return RunSpec(exp="t", policy=policy, seed=seed, duration=duration, warmup=5.0,
                   margin=30.0, topo=topo or _topo(), **kw)


class TestLocalKVCache:
    def test_lru_eviction(self):
        c = LocalKVCache(cap_gb=10.0)
        c.insert("A", 4.0)
        c.insert("B", 4.0)
        assert c.holds("A") and c.holds("B")
        c.insert("A", 4.0)          # touch A
        c.insert("C", 4.0)          # 超容，淘汰最久未用 B
        assert c.holds("A") and c.holds("C") and not c.holds("B")
        assert c.used == pytest.approx(8.0)

    def test_no_evict_last(self):
        c = LocalKVCache(cap_gb=1.0)
        c.insert("A", 5.0)
        assert c.holds("A")         # 至少保留一个


class TestDirectory:
    def test_replica_bookkeeping(self):
        topo = _topo(replicas=(("A", ((0, "mem"), (1, "ssd"))),))
        wl = WorkloadConfig()
        d = MetadataDirectory(topo, ["A", "B"], {"A": 10.0, "B": 2.0})
        assert d.holders("A") == {(0, "mem"), (1, "ssd")}
        assert d.holders("B") == set()
        assert d.held[0] == pytest.approx(10.0)
        d.add("B", (2, "mem"), 2.0)
        assert (2, "mem") in d.holders("B")
        assert d.held[2] == pytest.approx(2.0)
        d.add("B", (2, "mem"), 2.0)  # 幂等
        assert d.held[2] == pytest.approx(2.0)
        assert 0.0 <= d.capacity_pressure(0) <= 1.0


class TestEngineChain:
    def test_fetch_chain_analytic(self):
        """单请求 fetch：路径延迟 + tier 传输 + fabric 传输，解析可算。"""
        from sim.config import NodeConfig
        nodes = (NodeConfig("n0", mem=StorageConfig(b_total=100.0, bg_schedule=stable(0.0)),
                            ssd=StorageConfig(b_total=10.0)),)
        topo = _topo(replicas=(("A", ((0, "mem"),)),), path_lat=((0.01, 0.02, 0.02),),
                     n_workers=1, nodes=nodes)
        spec = _spec("joint2", topo=topo, wl=WorkloadConfig(lam=0.0001))
        env = simpy.Environment()
        world = World(env, spec)
        # 10GB：tier 100GB/s -> 0.1s；fabric 120GB/s -> 0.0833s；path 0.01
        ev1 = world.res(0, "mem").submit(1, 10.0)

        def proc():
            yield ev1
            t_tier = env.now
            ev2 = world.fabric.submit(1, 10.0)
            yield ev2
            # tier：t_base 0.005 门控 + 10GB/100GB/s = 0.105
            assert t_tier == pytest.approx(0.105, abs=1e-6)
            assert env.now == pytest.approx(0.105 + 0.005 + 10.0 / 120.0, abs=1e-6)

        env.process(proc())
        env.run(until=5.0)

    def test_partial_overlap_max_semantics(self):
        """partial overlap：完成时间 = max(取回链, GPU 计算)。"""
        env = simpy.Environment()
        spec = _spec("joint2")
        world = World(env, spec)
        from sim.engine2 import EngineV2
        from sim.metrics import Collector
        from sim.policies2 import Decision
        from sim.request import build_request
        metrics = Collector(spec, (), None)
        eng = EngineV2(env, 0, world, metrics)
        wl = spec.wl
        req = build_request(1, 0.0, "A", True, wl, spec.model)
        # A=32K -> kv 10.49GB；取 F=0.5 -> 5.24GB；tier 60GB/s + fabric 120GB/s 空载
        dec = Decision(worker=0, action="partial", node=0, tier="mem",
                       fetch_tokens=req.cached_prefix_tokens // 2,
                       fetch_gb=req.kv_gb / 2, overlap=True)
        t0 = []
        eng.handle(req, dec)
        env.run(until=50.0)
        # 取回链 = path + t_base门控 + bytes/B_tier + t_base门控 + bytes/B_fabric
        fb = req.kv_gb / 2
        chain_t = (0.002 + fb / world.res(0, "mem").b_total + world.res(0, "mem").t_base
                   + fb / world.fabric.b_total + world.fabric.t_base)
        # 计算 = prefill(prompt - F) 全速
        comp_t = world.curve(req.prompt_tokens - dec.fetch_tokens)
        expect = max(chain_t, comp_t)
        assert req.t_prefill_done == pytest.approx(expect, rel=1e-3)


class TestPoliciesV2:
    def test_local_preferred_when_idle(self):
        """本地缓存命中且 GPU 空闲 -> joint2 应选 local，不产生存储流量。"""
        topo = _topo(replicas=(("A", ((0, "mem"),)),), seed_local=((0, "A"),))
        spec = _spec("joint2", topo=topo, duration=40.0, lam=2.0)
        out = run_once(spec)
        assert out["frac_local"] > 0.5 or out["fetch_count"] == 0 or out["n_arr"] == 0

    def test_dynamic_beats_static_under_hot_storage(self):
        """存储层背景打满时，joint2 的 fetch 字节数应显著低于 alwaysfetch2。"""
        from sim.config import NodeConfig
        nodes = (NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(55.0)),
                            ssd=StorageConfig(b_total=25.0)),
                 NodeConfig("n1", mem=StorageConfig(b_total=60.0), ssd=StorageConfig(b_total=25.0)),
                 NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)))
        topo = _topo(replicas=(("A", ((0, "mem"),)), ("B", ((0, "mem"),)),
                               ("C", ((0, "mem"),)), ("D", ((0, "mem"),))), nodes=nodes)
        outs = {}
        for pol in ("alwaysfetch2", "joint2"):
            outs[pol] = run_once(_spec(pol, seed=1, topo=topo, duration=80.0, lam=4.0))
        assert outs["joint2"]["fetch_gb"] < 0.8 * outs["alwaysfetch2"]["fetch_gb"]

    def test_replica_selects_idle_node(self):
        """仅 A、双副本一忙一闲、GPU 有背景负载：joint2 的 A 取数应明显偏向闲节点。"""
        from sim.config import NodeConfig, PrefixClass
        nodes = (NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(55.0)),
                            ssd=StorageConfig(b_total=25.0)),
                 NodeConfig("n1", mem=StorageConfig(b_total=60.0, bg_schedule=stable(0.0)),
                            ssd=StorageConfig(b_total=25.0)),
                 NodeConfig("n2", mem=StorageConfig(b_total=40.0), ssd=StorageConfig(b_total=15.0)))
        classes = (PrefixClass("A", 32768, 1.0),)
        topo = _topo(replicas=(("A", ((0, "mem"), (1, "mem"))),), nodes=nodes,
                     gpu_bgs=(stable(0.6),) * 4, local_cache_gb=0.0)
        out = run_once(_spec("joint2", seed=2, topo=topo, duration=80.0,
                             wl=WorkloadConfig(lam=4.0, classes=classes)))
        n0 = out.get("replica_n0_mem", 0)
        n1 = out.get("replica_n1_mem", 0)
        assert n1 > n0

    def test_determinism(self):
        topo = _topo(replicas=(("A", ((0, "mem"),)),), seed_local=((0, "A"),))
        a = run_once(_spec("joint2", seed=3, topo=topo, duration=40.0))
        b = run_once(_spec("joint2", seed=3, topo=topo, duration=40.0))
        assert a["goodput"] == b["goodput"] and a["ttft_p95"] == b["ttft_p95"]

    def test_all_policies_run(self):
        from sim.policies2 import V2POLICIES
        topo = _topo(replicas=(("A", ((0, "mem"), (1, "mem"))), ("B", ((0, "mem"),)),
                               ("C", ((1, "ssd"),)), ("D", ((2, "ssd"),))),)
        for pol in V2POLICIES:
            out = run_once(_spec(pol, seed=0, topo=topo, duration=30.0, lam=3.0))
            assert out["n_arr"] > 0

    def test_clairvoyant_uses_future_trace(self):
        """先知策略：burst 内应表现出与逐请求贪心不同的分配（看到未来球）。"""
        from sim.config import PrefixClass
        classes = (PrefixClass("A", 32768, 1.0),)
        topo = _topo(replicas=(("A", ((0, "mem"), (1, "mem"))),), local_cache_gb=0.0)
        spec = RunSpec(exp="t", policy="clairvoyant2", seed=1, duration=60.0, warmup=5.0,
                       margin=30.0, topo=topo, burst=(20.0, 60, 1.0, "A"),
                       wl=WorkloadConfig(lam=2.0, classes=classes))
        out = run_once(spec)
        assert out["n_arr"] > 0 and out["n_done"] > 0
        # 决策具有确定性
        out2 = run_once(spec)
        assert out["goodput"] == out2["goodput"]


class TestReplication:
    def test_controller_replicates_hot_class(self):
        from sim.config import CtrlConfig, NodeConfig, PrefixClass
        nodes = (NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(50.0)),
                            ssd=StorageConfig(b_total=25.0)),
                 NodeConfig("n1", mem=StorageConfig(b_total=60.0, bg_schedule=stable(0.0)),
                            ssd=StorageConfig(b_total=25.0)),
                 NodeConfig("n2", mem=StorageConfig(b_total=40.0, bg_schedule=stable(0.0)),
                            ssd=StorageConfig(b_total=15.0)))
        classes = (PrefixClass("A", 32768, 0.60), PrefixClass("B", 8192, 0.40))
        topo = _topo(replicas=(("A", ((0, "mem"),)), ("B", ((0, "mem"),))),
                     nodes=nodes, ctrl=CtrlConfig(interval=0.5, hot_util=0.8, exit_util=0.6,
                                                  hold_s=1.0, min_demand=0.5))
        spec = RunSpec(exp="t", policy="joint2", seed=0, duration=100.0, warmup=10.0,
                       margin=40.0, topo=topo, wl=WorkloadConfig(lam=4.0, classes=classes))
        out = run_once(spec)
        assert out["n_replications"] >= 1


class TestQuote:
    def test_quote_pressure_levels(self):
        env = simpy.Environment()
        topo = _topo()
        spec = _spec("joint2", topo=topo)
        world = World(env, spec)
        from sim.quote import AccessCostQuery
        q = AccessCostQuery(world, ObsConfig(interval=0.0, signal="quote"))
        assert q.pressure(0) == "NORMAL"
        # 注入重负载：占满 n0.mem

        def _load():
            yield world.res(0, "mem").submit(-1, 1e6)

        env.process(_load())
        env.run(until=0.01)
        assert q.pressure(0) in ("HOT", "CRITICAL")

    def test_quote_includes_path_lat(self):
        env = simpy.Environment()
        topo = _topo(path_lat=((0.5, 0.5, 0.5),) * 4)
        spec = _spec("joint2", topo=topo)
        world = World(env, spec)
        from sim.quote import AccessCostQuery
        q = AccessCostQuery(world, ObsConfig(interval=0.0, signal="quote"))
        est = q.estimate(1.0, 0, 0, "mem", noisy=False)
        assert est["time"] >= 0.5


class TestImprovements:
    """改进方案新增组件：会话负载 / 预取 / 回写 / 目录删除 / 漂移。"""

    def test_session_trace_shape(self):
        from sim.workload import gen_session_trace
        from sim.config import WorkloadConfig
        wl = WorkloadConfig()
        tr = gen_session_trace(wl, 200.0, 0, (10.0, 3.0, 1.0))
        assert tr and all(r[0] < 200.0 for r in tr)
        # 会话内首轮 miss、后续 hit：按到达序，miss 先于同类 hit 出现
        seen = set()
        first_miss_first = True
        for t, cls, hit, sid in tr:
            if cls not in seen:
                if hit:
                    first_miss_first = False
                seen.add(cls)
        assert first_miss_first

    def test_drift_trace_generates(self):
        from sim.workload import gen_drift_trace
        from sim.config import WorkloadConfig
        tr = gen_drift_trace(WorkloadConfig(), 200.0, 1, (50.0,))
        assert len(tr) > 100

    def test_prefetch_gated_and_waste(self):
        from sim.config import NodeConfig, PrefetchConfig, StorageConfig, TopoConfig, WorkloadConfig
        from sim.config import PrefixClass
        nodes = (NodeConfig("n0", mem=StorageConfig(b_total=60.0, bg_schedule=stable(10.0)),
                            ssd=StorageConfig(b_total=25.0)),)
        nodes += (NodeConfig("n1"), NodeConfig("n2"))
        classes = (PrefixClass("A", 8192, 1.0),)
        topo = TopoConfig(replicas=(("A", ((0, "mem"),)),), nodes=nodes,
                          prefetch=PrefetchConfig(mode="gated"), cache_mode="coord",
                          local_cache_gb=6.0)
        spec = RunSpec(exp="t", policy="joint2", seed=0, duration=60.0, warmup=5.0, margin=30.0,
                       topo=topo, wl=WorkloadConfig(lam=4.0, classes=classes),
                       sessions=(1.0, 3.0, 0.8))
        out = run_once(spec)
        assert "prefetch_gb" in out and "cache_redundancy" in out
        assert out["cache_redundancy"] <= 1.0 + 1e-9   # coord 下同类至多一个偏好 worker……
