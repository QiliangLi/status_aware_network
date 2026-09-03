"""Q4 预取（问题②）：会话续写预取（压力门控）+ 重算后异步回写共享存储。

- 预取：命中请求完成后，若该 worker 本地缓存未持有该类，则从最优副本把 KV 预取
  进本地缓存（占用 tier+fabric 带宽）；gated 模式仅在目标副本压力 <= WARM 时发起。
- 回写：重算/prefill 完成后，完整 KV 已在 worker 本地——异步回写到低压 mem 节点
  （内部传输占用带宽），完成后登记进元数据目录，等价于"重算逃逸的免费副本修复"。
"""
from __future__ import annotations


class Prefetcher:
    def __init__(self, env, world, quote, metrics, cfg):
        self.env = env
        self.world = world
        self.quote = quote
        self.metrics = metrics
        self.cfg = cfg
        self.n_prefetch = 0
        self.prefetch_gb = 0.0
        self.n_writeback = 0
        self.writeback_gb = 0.0
        self._op = 0
        self.last_disp = {}     # cls -> 上次命中派发时刻（类级）
        self.gap_ema = {}       # cls -> 轮间隔 EMA（秒）
        self.s_last = {}        # sid -> 上次命中派发时刻（会话级）
        self.s_gap = {}         # sid -> 轮间隔（秒，最近值即可：会话内间隔近似平稳）
        self.horizon = 5.0

    def _update_gap(self, cls: str, t: float, sid: int = -1) -> None:
        last = self.last_disp.get(cls)
        self.last_disp[cls] = t
        if last is not None:
            gap = t - last
            old = self.gap_ema.get(cls)
            a = 0.4   # 简单平滑（约 2 轮收敛）
            self.gap_ema[cls] = gap if old is None else old + a * (gap - old)
        if sid >= 0:
            sl = self.s_last.get(sid)
            if sl is not None:
                self.s_gap[sid] = t - sl
            self.s_last[sid] = t

    # ---------- 派发钩子（预取） ----------
    def on_dispatch(self, req, dec) -> None:
        """请求派发时触发：目标 worker（coord 下为偏好 worker）未持有该类则后台预取。

        引擎在命中完成后总会写完成侧 worker 的缓存，因此预取只对
        (a) 目标 != 服务 worker（coord 协同放置），或 (b) 本次动作为重算
        （不会走取回链路）的场景有增量价值；fetch/partial 到自身则跳过避免重复 IO。
        """
        if not req.hit or self.cfg.mode == "none" or not self.world.dir.holders(req.cls):
            return
        self._update_gap(req.cls, self.env.now, req.sid)
        if self.cfg.mode in ("predictive", "session"):
            if self.cfg.mode == "session" and req.sid >= 0:
                g = self.s_gap.get(req.sid)
            else:
                g = self.gap_ema.get(req.cls)
            if g is None or g > self.horizon:
                return   # 该会话/类轮间隔长，预取大概率被淘汰
        target = dec.worker
        if self.world.locals[target]._coord_target(req.cls) is not None:
            target = self.world.locals[target]._coord_target(req.cls)
        lc = self.world.locals[target]
        if lc.holds(req.cls):
            return
        if target == dec.worker and dec.action in ("fetch", "partial"):
            return
        self._maybe_prefetch(target, req.cls, req.kv_gb)

    # ---------- 引擎完成钩子（回写） ----------
    def on_complete(self, req) -> None:
        if self.cfg.writeback and req.hit and req.action in ("recompute", "prefill"):
            self._maybe_writeback(req)

    def _maybe_prefetch(self, worker: int, cls: str, nbytes: float) -> None:
        best = None
        for (n, t) in sorted(self.world.dir.holders(cls)):
            q = self.quote.estimate(nbytes, worker, n, t)
            if self.cfg.mode == "gated" and q["pressure"] not in ("NORMAL", "WARM"):
                continue
            if best is None or q["time"] < best[0]:
                best = (q["time"], n, t)
        if best is None:
            return
        _, n, t = best
        self._op += 1
        self.env.process(self._prefetch_chain(worker, n, t, cls, nbytes,
                                              -(10 ** 7 + self._op)))

    def _prefetch_chain(self, worker, node, tier, cls, nbytes, rid):
        yield self.env.timeout(self.world.path_lat[worker][node])
        yield self.world.res(node, tier).submit(rid, nbytes)
        yield self.world.fabric.submit(rid, nbytes)
        self.world.locals[worker].insert(cls, nbytes, source="prefetch")

    def _maybe_writeback(self, req) -> None:
        holders = self.world.dir.holders(req.cls)
        best, best_key = None, None
        for n in range(self.world.n_nodes):
            dst = (n, "mem")
            if dst in holders:
                continue
            ri = self.world.res_idx(n, "mem")
            key = (self.quote.util_of(ri), self.world.dir.capacity_pressure(n))
            if best is None or key < best_key:
                best, best_key = dst, key
        if best is None:
            return
        self._op += 1
        self.n_writeback += 1
        nbytes = req.kv_gb
        self.writeback_gb += nbytes
        rid = -(10 ** 7 + self._op)

        def _wb():
            ev = self.world.res(best[0], "mem").submit(rid, nbytes)
            yield ev
            self.world.dir.add(req.cls, best, nbytes)

        self.env.process(_wb())

    def stats(self) -> dict:
        caches = self.world.locals
        return dict(
            n_prefetch=sum(c.n_prefetch for c in caches),
            prefetch_gb=sum(c.prefetch_gb for c in caches),
            prefetch_wasted_gb=sum(c.prefetch_wasted_gb for c in caches),
            n_writeback=self.n_writeback,
            writeback_gb=self.writeback_gb,
        )
