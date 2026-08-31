"""指标采集：请求级事件、粗粒度统计采样、可选时间序列、汇总。"""
from __future__ import annotations

from collections import deque

import numpy as np


class Collector:
    def __init__(self, spec, worker_backend, class_backend_map):
        self.spec = spec
        self.worker_backend = worker_backend
        self.cbm = class_backend_map
        self.reqs = {}
        # 统计（arrival 在 [warmup, duration) 内的请求）
        self.ttfts = []            # 完成者
        self.n_arr = 0
        self.n_done = 0
        self.n_slo = 0
        self.action_counts = {"fetch": 0, "recompute": 0, "prefill": 0}
        self.fetch_gb = 0.0
        self.recompute_tokens = 0
        self.cls_tot = {}
        self.cls_slo = {}
        self.misroute = 0
        # 窗口指标（可选）
        self.win = spec.window
        self.win_ttfts = []
        self.win_arr = 0
        self.win_slo = 0
        # 粗粒度资源采样
        self.q_samples = []        # (t, backend -> qdepth)
        self.gpu_busy_samples = []
        # 时间序列
        self.ts = None
        # 滚动量
        self._act_hist = deque()   # (t, action)
        self._ttft_hist = deque()  # (t_done, ttft)

    # ---------- 请求级 ----------
    def on_arrival(self, req):
        self.reqs[req.rid] = req
        if self.spec.warmup <= req.arrival < self.spec.duration:
            self.n_arr += 1
            self.cls_tot[req.cls] = self.cls_tot.get(req.cls, 0) + 1
            if self.win and self.win[0] <= req.arrival < self.win[1]:
                self.win_arr += 1

    def on_decision(self, req, action):
        self._act_hist.append((req.arrival, action))
        if self.spec.warmup <= req.arrival < self.spec.duration:
            if action in self.action_counts:
                self.action_counts[action] += 1
            if action == "fetch":
                self.fetch_gb += req.kv_gb
            else:
                self.recompute_tokens += req.prompt_tokens
            holdings = self.cbm.get(req.cls)
            if req.hit and holdings is not None and self.worker_backend[req.worker] not in holdings:
                self.misroute += 1

    def on_fetch_done(self, rid, t):
        pass

    def on_complete(self, req, action):
        ttft = req.ttft
        self._ttft_hist.append((req.t_prefill_done, ttft))
        if self.spec.warmup <= req.arrival < self.spec.duration:
            self.n_done += 1
            self.ttfts.append(ttft)
            if ttft <= req.ttft_slo:
                self.n_slo += 1
                self.cls_slo[req.cls] = self.cls_slo.get(req.cls, 0) + 1
            if self.win and self.win[0] <= req.arrival < self.win[1]:
                self.win_ttfts.append(ttft)
                if ttft <= req.ttft_slo:
                    self.win_slo += 1

    # ---------- 采样 ----------
    def coarse_tick(self, t, world):
        self.q_samples.append([t] + [w["qdepth"] for w in world["storages"]])
        self.gpu_busy_samples.append([t] + [g["util_win"] for g in world["gpus"]])

    def enable_ts(self):
        self.ts = {"t": []}
        for i in range(len(self.worker_backend)):
            self.ts[f"w{i}_qjobs"] = []
            self.ts[f"w{i}_util"] = []
        for b in sorted(set(self.worker_backend)):
            self.ts[f"s{b}_q"] = []
            self.ts[f"s{b}_inflight"] = []
            self.ts[f"s{b}_fg"] = []
            self.ts[f"s{b}_util"] = []
        self.ts["fetch_frac"] = []
        self.ts["recompute_frac"] = []
        self.ts["roll_p95_ttft"] = []

    def ts_tick(self, t, world):
        if self.ts is None:
            return
        self.ts["t"].append(t)
        for i, g in enumerate(world["gpus"]):
            self.ts[f"w{i}_qjobs"].append(g["qjobs"])
            self.ts[f"w{i}_util"].append(g["util_win"])
        for b in sorted(set(self.worker_backend)):
            self.ts[f"s{b}_q"].append(world["storages"][b]["qdepth"])
            self.ts[f"s{b}_inflight"].append(world["storages"][b]["inflight"])
            self.ts[f"s{b}_fg"].append(world["storages"][b]["fg_gbps"])
            self.ts[f"s{b}_util"].append(world["storages"][b]["util"])
        now = t
        while self._act_hist and self._act_hist[0][0] < now - 2.0:
            self._act_hist.popleft()
        n_act = max(1, len(self._act_hist))
        self.ts["fetch_frac"].append(
            sum(1 for _, a in self._act_hist if a == "fetch") / n_act)
        self.ts["recompute_frac"].append(
            sum(1 for _, a in self._act_hist if a != "fetch") / n_act)
        while self._ttft_hist and self._ttft_hist[0][0] < now - 5.0:
            self._ttft_hist.popleft()
        vals = [x[1] for x in self._ttft_hist]
        self.ts["roll_p95_ttft"].append(float(np.percentile(vals, 95)) if vals else np.nan)

    # ---------- 汇总 ----------
    def finalize(self) -> dict:
        dur = self.spec.duration - self.spec.warmup
        arr = np.asarray(self.ttfts) if self.ttfts else np.array([np.nan])
        out = {
            "n_arr": self.n_arr,
            "n_done": self.n_done,
            "n_incomplete": self.n_arr - self.n_done,
            "incomplete_frac": (self.n_arr - self.n_done) / max(1, self.n_arr),
            "n_slo": self.n_slo,
            "goodput": self.n_slo / dur,
            "throughput": self.n_done / dur,
            "slo_rate": self.n_slo / max(1, self.n_arr),
            "ttft_p50": float(np.percentile(arr, 50)),
            "ttft_p95": float(np.percentile(arr, 95)),
            "ttft_p99": float(np.percentile(arr, 99)),
            "fetch_count": self.action_counts["fetch"],
            "recompute_count": self.action_counts["recompute"] + self.action_counts["prefill"],
            "fetch_gb": self.fetch_gb,
            "recompute_tokens": self.recompute_tokens,
            "misroute_rate": self.misroute / max(1, self.n_arr),
        }
        for cls, tot in sorted(self.cls_tot.items()):
            out[f"slo_rate_{cls}"] = self.cls_slo.get(cls, 0) / tot
        if self.q_samples:
            qa = np.asarray(self.q_samples, dtype=float)
            for b in range(1, qa.shape[1]):
                out[f"s{b-1}_q_max"] = float(qa[:, b].max())
                out[f"s{b-1}_q_mean"] = float(qa[:, b].mean())
        if self.gpu_busy_samples:
            ga = np.asarray(self.gpu_busy_samples, dtype=float)
            for w in range(1, ga.shape[1]):
                out[f"w{w-1}_util_mean"] = float(ga[:, w].mean())
        if self.win:
            warr = np.asarray(self.win_ttfts) if self.win_ttfts else np.array([np.nan])
            wdur = self.win[1] - self.win[0]
            out.update({
                "win_arr": self.win_arr,
                "win_goodput": self.win_slo / wdur,
                "win_slo_rate": self.win_slo / max(1, self.win_arr),
                "win_ttft_p95": float(np.percentile(warr, 95)),
                "win_ttft_p99": float(np.percentile(warr, 99)),
            })
        return out
