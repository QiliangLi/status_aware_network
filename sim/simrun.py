"""单次仿真运行的装配与执行。"""
from __future__ import annotations

import os

import numpy as np
import simpy

from .config import RunSpec
from .gpu import GpuPool, PrefillCurve
from .metrics import Collector
from .policies import PolicyCtx, WorkerView, make_policy, NEEDS_OBS
from .request import build_request
from .scheduler import Scheduler
from .storage import SharedKVStorage, StorageObservable
from .worker import Engine
from . import workload

_TRACE_CACHE: dict = {}


def get_trace(spec: RunSpec) -> list:
    key = (spec.seed, spec.duration, spec.wl, spec.burst, spec.sessions, spec.drift)
    if key not in _TRACE_CACHE:
        if spec.drift:
            _TRACE_CACHE[key] = workload.gen_drift_trace(spec.wl, spec.duration, spec.seed, spec.drift)
        elif spec.sessions:
            _TRACE_CACHE[key] = workload.gen_full_trace_session(
                spec.wl, spec.duration, spec.seed, spec.burst, spec.sessions)
        else:
            _TRACE_CACHE[key] = workload.gen_full_trace(spec.wl, spec.duration, spec.seed, spec.burst)
    return _TRACE_CACHE[key]


def run_once(spec: RunSpec) -> dict:
    if spec.topo is not None:
        return run_once_v2(spec)
    trace = get_trace(spec)
    env = simpy.Environment()
    curve = PrefillCurve(spec.gpu.prefill_table)
    cbm = spec.class_backend_map()
    n_backends = len(spec.storages)

    metrics = Collector(spec, spec.worker_backend, cbm)
    storages = [SharedKVStorage(env, i, sc) for i, sc in enumerate(spec.storages)]
    gpus = [GpuPool(env, w, curve, spec.gpu) for w in range(len(spec.worker_backend))]

    mean_hit_gb = spec.wl.mean_hit_tokens * spec.model.kv_gb_per_token
    obs = [None] * n_backends
    if spec.policy in NEEDS_OBS:
        for b in range(n_backends):
            rng = np.random.default_rng([spec.seed, b, 4])
            obs[b] = StorageObservable(env, storages[b], spec.obs, mean_hit_gb, rng)

    views = [
        WorkerView(worker_id=w, gpu=gpus[w], storage=storages[spec.worker_backend[w]],
                   obs=obs[spec.worker_backend[w]])
        for w in range(len(spec.worker_backend))
    ]
    ctx = PolicyCtx(b_total=spec.storages[0].b_total, t_base=spec.storages[0].t_base,
                    curve=curve, margin=spec.pol.margin, signal=spec.obs.signal)
    policy = make_policy(spec.policy, ctx)
    scheduler = Scheduler(policy, spec.worker_backend, cbm)
    engines = [Engine(env, w, gpus[w], storages[spec.worker_backend[w]], metrics)
               for w in range(len(spec.worker_backend))]

    # ---------- 供 ticker 读取的世界快照 ----------
    last_bytes = [0.0] * n_backends
    last_busy = [0.0] * len(gpus)
    last_tick = [env.now]

    def world(t):
        nonlocal_last = max(last_tick[0], 0.0)
        dt = max(t - nonlocal_last, 1e-9)
        stw = []
        for b, s in enumerate(storages):
            fg = (s.bytes_served - last_bytes[b]) / dt
            stw.append(dict(qdepth=len(s.active), inflight=sum(tr.remaining for tr in s.active),
                            fg_gbps=fg, util=min(1.0, (s.bg_at(t) + fg) / s.b_total)))
            last_bytes[b] = s.bytes_served
        gpw = []
        for w, g in enumerate(gpus):
            gpw.append(dict(qjobs=len(g.queue), util_win=(g.busy_time - last_busy[w]) / dt))
            last_busy[w] = g.busy_time
        last_tick[0] = t
        return dict(storages=stw, gpus=gpw)

    def _tickers():
        def coarse():
            while True:
                yield env.timeout(0.1)
                for s in storages:
                    s.advance_to(env.now)
                for g in gpus:
                    g.advance_to(env.now)
                metrics.coarse_tick(env.now, world(env.now))

        env.process(coarse())
        if spec.collect_ts:
            metrics.enable_ts()

            def fine():
                while True:
                    yield env.timeout(0.01)
                    for s in storages:
                        s.advance_to(env.now)
                    for g in gpus:
                        g.advance_to(env.now)
                    metrics.ts_tick(env.now, world(env.now))

            env.process(fine())

    _tickers()

    def _arrivals():
        for rid, row in enumerate(trace):
            t, cls, hit = row[0], row[1], row[2]
            sid = row[3] if len(row) > 3 else -1
            yield env.timeout(max(0.0, t - env.now))
            req = build_request(rid, t, cls, hit, spec.wl, spec.model, sid=sid)
            metrics.on_arrival(req)
            wid, action = scheduler.dispatch(req, views)
            req.worker = wid
            req.action = action
            metrics.on_decision(req, action)
            engines[wid].handle(req, action)

    env.process(_arrivals())
    env.run(until=spec.duration + spec.margin)
    summary = metrics.finalize()

    if spec.out_dir:
        os.makedirs(spec.out_dir, exist_ok=True)
        if spec.save_requests:
            import pandas as pd
            rows = [{
                "rid": r.rid, "t_arr": r.arrival, "cls": r.cls, "hit": r.hit,
                "worker": r.worker, "action": r.action, "t_fetch_done": r.t_fetch_done,
                "t_prefill_done": r.t_prefill_done, "ttft": r.ttft if r.ttft else np.nan,
                "slo_met": (r.ttft is not None and r.ttft <= r.ttft_slo),
            } for r in metrics.reqs.values()]
            pd.DataFrame(rows).to_csv(
                os.path.join(spec.out_dir, f"requests_{spec.policy}_s{spec.seed}.csv"), index=False)
        if spec.save_ts and metrics.ts:
            import pandas as pd
            pd.DataFrame(metrics.ts).to_csv(
                os.path.join(spec.out_dir, f"ts_{spec.policy}_s{spec.seed}.csv"), index=False)
    return summary


def _run_job(job):
    meta, spec = job
    return {**meta, **run_once(spec)}


# ==================== v2：共享分布式 KV 存储拓扑 ====================

def _ctrl_loop(env, world, quote, ctrl, metrics, spec):
    """存储侧控制器：复制（跨层降级）/ 迁移（冷类回收 mem）/ 容量淘汰（防孤儿）。

    问题④扩展后的行为：
    - 源副本持续高压 + 类需求高 -> 复制到 (util, 容量压力) 最小的目标；
      cross_tier=True 时目标层含 ssd（mem 拥堵而 ssd 空闲时降层）。
    - 类需求低于 cold_demand 且源在 mem -> 迁移（复制完成后淘汰源，回收快层容量）。
    - 节点容量压力 > cap_evict -> 淘汰该节点上需求率最低的类至 evict_target，
      仅当该类在别处仍有副本（Directory.remove 内建孤儿检查）。
    """
    hot_since = {}
    op_id = [0]
    last_op = {}   # cls -> 上次复制/迁移时刻（冷却）

    def _do_transfer(cls, src, dst, nbytes, evict_src):
        op_id[0] += 1
        rid = -(10 ** 6 + op_id[0])
        t0 = env.now
        e1 = world.res(*src).submit(rid, nbytes)
        e2 = world.res(*dst).submit(rid, nbytes)
        yield e1 & e2
        world.dir.add(cls, dst, nbytes)
        metrics.on_replication(cls, src, dst, nbytes, env.now - t0)
        if evict_src:
            world.dir.remove(cls, src, nbytes)

    while True:
        yield env.timeout(ctrl.interval)
        if env.now < spec.warmup:
            continue
        demand = metrics.demand_rates(env.now)
        enter = ctrl.hot_util - (0.10 if ctrl.predictive else 0.0)
        hold = ctrl.hold_s / (2.0 if ctrl.predictive else 1.0)
        for ri in range(len(world.resources) - 1):   # fabric 不承载副本
            u = quote.util_of(ri)
            if u >= enter:
                hot_since.setdefault(ri, env.now)
            elif u < ctrl.exit_util:
                hot_since.pop(ri, None)
        demand = metrics.demand_rates(env.now)

        # ---- 容量淘汰（防孤儿，内建检查）----
        for n in range(world.n_nodes):
            while world.dir.capacity_pressure(n) > ctrl.cap_evict:
                victims = sorted(world.dir.replicas.items(),
                                 key=lambda kv: demand.get(kv[0], 0.0))
                evicted = False
                for cls, hs in victims:
                    if demand.get(cls, 0.0) >= ctrl.evict_demand:
                        continue   # 只淘汰真正冷的类
                    for (nn, t_) in list(hs):
                        if nn == n and world.dir.remove(cls, (nn, t_), world.cls_bytes.get(cls, 0.0)):
                            evicted = True
                            break
                    if evicted:
                        break
                if not evicted:
                    break

        # ---- 复制 / 迁移 ----
        for cls, rate in sorted(demand.items()):
            if len(world.dir.holders(cls)) >= ctrl.max_replicas:
                continue
            if env.now - last_op.get(cls, -1e9) < ctrl.cooldown_s:
                continue
            holders = world.dir.holders(cls)
            src = next(((n, t) for (n, t) in holders
                        if world.res_idx(n, t) in hot_since
                        and env.now - hot_since[world.res_idx(n, t)] >= hold), None)
            if src is None:
                continue
            migrate = (rate < ctrl.cold_demand and src[1] == "mem"
                       and len(holders) >= 1)
            tiers = ("mem", "ssd") if ctrl.cross_tier else (src[1],)
            best_dst, best_key = None, None
            for n in range(world.n_nodes):
                for t_ in tiers:
                    dst = (n, t_)
                    if dst in holders:
                        continue
                    ri = world.res_idx(n, t_)
                    key = (quote.util_of(ri), world.dir.capacity_pressure(n))
                    if best_dst is None or key < best_key:
                        best_dst, best_key = dst, key
            nbytes = world.cls_bytes.get(cls, 0.0)
            if best_dst is not None and nbytes > 0:
                last_op[cls] = env.now
                env.process(_do_transfer(cls, src, best_dst, nbytes, evict_src=migrate))


def run_once_v2(spec: RunSpec) -> dict:
    import numpy as np

    trace = get_trace(spec)
    env = simpy.Environment()
    from .topology import World
    from .quote import AccessCostQuery
    from .engine2 import EngineV2
    from .policies2 import V2Ctx, make_v2_policy

    world = World(env, spec)
    quote = AccessCostQuery(world, spec.obs)
    metrics = Collector(spec, (), None)
    rng = np.random.default_rng([spec.seed, 9, 7])
    gpu_obs = None
    if spec.policy not in ("oracle2", "clairvoyant2"):
        from .gpu_obs import GpuObservable
        gpu_obs = [
            GpuObservable(env, g, spec.obs.gpu_interval, spec.obs.gpu_ema,
                          spec.obs.gpu_noise, np.random.default_rng([spec.seed, 200 + w, 5]))
            for w, g in enumerate(world.gpus)
        ]
    ctx = V2Ctx(world=world, quote=quote, margin=spec.pol.margin,
                kv_gb_per_token=spec.model.kv_gb_per_token, rng=rng,
                gpu_obs=gpu_obs, guardband=spec.pol.guardband)
    policy = make_v2_policy(spec.policy, ctx)
    if spec.policy == "clairvoyant2":
        import bisect
        _future = {}
        for t_, cls, hit in trace:
            if hit:
                _future.setdefault(cls, []).append(t_)
        for cls in _future:
            _future[cls].sort()

        def future_count(cls, t, horizon):
            arr = _future.get(cls)
            if not arr:
                return 0
            return bisect.bisect_right(arr, t + horizon) - bisect.bisect_right(arr, t)

        ctx.future = future_count

    for (w, cls) in spec.topo.seed_local:
        world.locals[w].insert(cls, world.cls_bytes.get(cls, 0.0))

    engines = [EngineV2(env, w, world, metrics) for w in range(world.n_workers)]
    prefetcher = None
    if spec.topo.prefetch is not None:
        from .prefetch import Prefetcher
        prefetcher = Prefetcher(env, world, quote, metrics, spec.topo.prefetch)
        for e in engines:
            e.on_complete_hook = prefetcher.on_complete

    # ---------- 世界快照（ticker 用） ----------
    n_res = len(world.resources)
    last_bytes = [0.0] * n_res
    last_busy = [0.0] * world.n_workers
    last_tick = [env.now]

    def world_now(t):
        dt = max(t - last_tick[0], 1e-9)
        stw = []
        for i, r in enumerate(world.resources):
            fg = (r.bytes_served - last_bytes[i]) / dt
            last_bytes[i] = r.bytes_served
            stw.append(dict(qdepth=len(r.active), inflight=sum(tr.remaining for tr in r.active),
                            fg_gbps=fg, util=min(1.0, (r.bg_at(t) + fg) / r.b_total)))
        gpw = []
        for w, g in enumerate(world.gpus):
            gpw.append(dict(qjobs=len(g.queue), util_win=(g.busy_time - last_busy[w]) / dt))
            last_busy[w] = g.busy_time
        last_tick[0] = t
        return dict(storages=stw, gpus=gpw)

    def _tickers():
        def coarse():
            while True:
                yield env.timeout(0.1)
                for r in world.resources:
                    r.advance_to(env.now)
                for g in world.gpus:
                    g.advance_to(env.now)
                metrics.coarse_tick(env.now, world_now(env.now))

        env.process(coarse())
        if spec.collect_ts:
            metrics.enable_ts(n_workers=world.n_workers, n_res=n_res)

            def fine():
                while True:
                    yield env.timeout(0.01)
                    for r in world.resources:
                        r.advance_to(env.now)
                    for g in world.gpus:
                        g.advance_to(env.now)
                    metrics.ts_tick(env.now, world_now(env.now))

            env.process(fine())

    _tickers()
    if spec.topo.ctrl is not None:
        env.process(_ctrl_loop(env, world, quote, spec.topo.ctrl, metrics, spec))

    def _arrivals():
        for rid, row in enumerate(trace):
            t, cls, hit = row[0], row[1], row[2]
            sid = row[3] if len(row) > 3 else -1
            yield env.timeout(max(0.0, t - env.now))
            req = build_request(rid, t, cls, hit, spec.wl, spec.model, sid=sid)
            req.local_avail = bool(hit) and any(l.holds(cls) for l in world.locals)
            metrics.on_arrival(req)
            dec = policy.decide(req)
            req.worker = dec.worker
            req.action = dec.action
            req.node, req.tier = dec.node, dec.tier
            req.fetch_tokens = dec.fetch_tokens
            metrics.on_decision_v2(req, dec)
            if prefetcher is not None:
                prefetcher.on_dispatch(req, dec)
            engines[dec.worker].handle(req, dec)

    env.process(_arrivals())
    env.run(until=spec.duration + spec.margin)
    summary = metrics.finalize()
    summary["res_names"] = ",".join(world.res_names)
    # 缓存与预取统计
    held = [c.classes_held() for c in world.locals]
    all_cls = {c.name for c in spec.wl.classes}
    counts = {c: sum(1 for h in held if c in h) for c in all_cls}
    distinct = sum(1 for v in counts.values() if v > 0)
    summary["cache_redundancy"] = sum(counts.values()) / max(1, distinct)
    summary["cache_coverage"] = distinct / max(1, len(all_cls))
    if prefetcher is not None:
        summary.update(prefetcher.stats())
        summary["prefetch_waste_frac"] = (
            summary["prefetch_wasted_gb"] / max(1e-9, summary["prefetch_gb"]))
    d = world.dir
    for n in range(world.n_nodes):
        summary[f"node{n}_occup"] = d.held[n] / max(1e-9, d.cap[n])
    summary["orphan_events"] = d.orphan_events
    summary["n_evictions"] = d.n_evictions
    summary["cross_tier_replicas"] = sum(1 for cls, hs in d.replicas.items() for (_, t) in hs if t == "ssd")

    if spec.out_dir:
        os.makedirs(spec.out_dir, exist_ok=True)
        if spec.save_requests:
            import pandas as pd
            rows = [{
                "rid": r.rid, "t_arr": r.arrival, "cls": r.cls, "hit": r.hit,
                "worker": r.worker, "action": r.action, "node": r.node, "tier": r.tier,
                "fetch_tokens": r.fetch_tokens, "t_fetch_done": r.t_fetch_done,
                "t_prefill_done": r.t_prefill_done, "ttft": r.ttft if r.ttft else np.nan,
                "slo_met": (r.ttft is not None and r.ttft <= r.ttft_slo),
            } for r in metrics.reqs.values()]
            pd.DataFrame(rows).to_csv(
                os.path.join(spec.out_dir, f"requests_{spec.policy}_s{spec.seed}.csv"), index=False)
        if spec.save_ts and metrics.ts:
            import pandas as pd
            pd.DataFrame(metrics.ts).to_csv(
                os.path.join(spec.out_dir, f"ts_{spec.policy}_s{spec.seed}.csv"), index=False)
    return summary
