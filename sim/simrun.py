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
    key = (spec.seed, spec.duration, spec.wl, spec.burst)
    if key not in _TRACE_CACHE:
        _TRACE_CACHE[key] = workload.gen_full_trace(spec.wl, spec.duration, spec.seed, spec.burst)
    return _TRACE_CACHE[key]


def run_once(spec: RunSpec) -> dict:
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
        for rid, (t, cls, hit) in enumerate(trace):
            yield env.timeout(max(0.0, t - env.now))
            req = build_request(rid, t, cls, hit, spec.wl, spec.model)
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
