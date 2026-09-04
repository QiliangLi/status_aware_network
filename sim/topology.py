"""v2 拓扑组件：计算节点本地 KV 缓存（LRU）、元数据目录（副本放置）、世界装配。

对应架构图：
  - 计算节点(带小型本地显存缓存)  -> LocalKVCache
  - 元数据目录(KV位置与副本可用性) -> MetadataDirectory
  - 存储节点(内存+固态盘)          -> 每节点两个 SharedKVStorage 实例（mem/ssd tier）
  - 高速网络                       -> 一个共享 SharedKVStorage 实例（fabric）
"""
from __future__ import annotations

from collections import OrderedDict

from .config import TopoConfig
from .gpu import GpuPool, PrefillCurve
from .storage import SharedKVStorage, StorageObservable


class LocalKVCache:
    """单 worker 的本地 KV 缓存：class -> [bytes, source, used]，LRU 淘汰，总容量有限。

    source: "serve"（服务后写入，已使用）| "prefetch"（预取写入，未使用直到被 local 命中）。
    coord 模式：仅偏好 worker（一致性哈希）或已有条目可写入，降低跨 worker 冗余。
    """

    def __init__(self, cap_gb: float, worker_id: int = 0, n_workers: int = 1,
                 coord: bool = False):
        import zlib
        self.cap = cap_gb
        self.worker_id = worker_id
        self.coord = coord
        self._pref = lambda cls: zlib.crc32(cls.encode()) % max(1, n_workers)
        self.used = 0.0
        self._items: OrderedDict[str, list] = OrderedDict()
        self._protected: set = set()
        self.prefetch_wasted_gb = 0.0
        self.prefetch_gb = 0.0
        self.n_prefetch = 0

    def holds(self, cls: str) -> bool:
        return cls in self._items

    def _allowed(self, cls: str) -> bool:
        return (not self.coord) or self._pref(cls) == self.worker_id or cls in self._items

    def _coord_target(self, cls: str):
        """coord 模式下该类的偏好 worker；非 coord 返回 None。"""
        return self._pref(cls) if self.coord else None

    def insert(self, cls: str, nbytes: float, source: str = "serve") -> None:
        if self.cap <= 1e-9:      # 容量 0 = 不启用本地缓存
            return
        if cls in self._items:
            self._items.move_to_end(cls)
            if source == "serve":
                self._items[cls][2] = True
            return
        if not self._allowed(cls):
            return
        self._items[cls] = [nbytes, source, source == "serve"]
        self.used += nbytes
        if source == "prefetch":
            self.prefetch_gb += nbytes
            self.n_prefetch += 1
        while self.used > self.cap + 1e-9 and len(self._items) > 1:
            victim, rec = self._pop_victim()
            if victim is None:
                break
            self.used -= rec[0]
            if rec[1] == "prefetch" and not rec[2]:
                self.prefetch_wasted_gb += rec[0]

    def _pop_victim(self):
        """驱逐：优先非保护的 LRU；保护仅覆盖"预取且未使用"的活跃条目；
        serve 条目始终可驱逐；无候选则回退整体 LRU（防死锁）。"""
        for cls in self._items:
            rec = self._items[cls]
            guarded = (cls in self._protected and rec[1] == "prefetch" and not rec[2])
            if not guarded:
                return cls, self._items.pop(cls)
        if self._items:
            return self._items.popitem(last=False)
        return None, None

    def set_protected(self, classes) -> None:
        self._protected = set(classes)

    def free_gb(self) -> float:
        return self.cap - self.used

    def evictable_unprotected_gb(self) -> float:
        return sum(rec[0] for c, rec in self._items.items()
                   if not (c in self._protected and rec[1] == "prefetch" and not rec[2]))

    def evict(self, cls: str) -> None:
        rec = self._items.pop(cls, None)
        if rec is not None:
            self.used -= rec[0]
            if rec[1] == "prefetch" and not rec[2]:
                self.prefetch_wasted_gb += rec[0]

    def mark_used(self, cls: str) -> None:
        rec = self._items.get(cls)
        if rec is not None:
            rec[2] = True

    def size(self, cls: str) -> float:
        rec = self._items.get(cls)
        return rec[0] if rec else 0.0

    def classes_held(self) -> set:
        return set(self._items.keys())


class MetadataDirectory:
    """元数据目录：class -> {(node_idx, tier)} 副本集合；同时维护节点占用（容量压力状态）。"""

    def __init__(self, topo: TopoConfig, cls_names, cls_bytes: dict):
        self.n_nodes = len(topo.nodes)
        self.cap = [n.cap_gb for n in topo.nodes]
        self.held = [0.0] * self.n_nodes
        self.orphan_events = 0
        self.n_evictions = 0
        self.replicas: dict[str, set] = {c: set() for c in cls_names}
        for cls, placements in topo.replicas:
            if cls not in self.replicas:
                self.replicas[cls] = set()
            for (ni, tier) in placements:
                self.replicas[cls].add((ni, tier))
                self.held[ni] += cls_bytes.get(cls, 0.0)

    def holders(self, cls: str) -> set:
        return self.replicas.get(cls, set())

    def add(self, cls: str, placement: tuple, nbytes: float) -> None:
        if placement in self.replicas.get(cls, set()):
            return
        self.replicas.setdefault(cls, set()).add(placement)
        self.held[placement[0]] += nbytes

    def remove(self, cls: str, placement: tuple, nbytes: float) -> bool:
        """删除副本；若使该类无任何副本则拒绝（防孤儿）并返回 False。"""
        hs = self.replicas.get(cls, set())
        if placement not in hs or len(hs) <= 1:
            return False
        hs.discard(placement)
        self.held[placement[0]] = max(0.0, self.held[placement[0]] - nbytes)
        self.n_evictions += 1
        return True

    def capacity_pressure(self, node_idx: int) -> float:
        return min(1.0, self.held[node_idx] / max(1e-9, self.cap[node_idx]))


class World:
    """v2 世界：worker 侧（GPU+本地缓存）、存储节点侧（mem/ssd tier）、fabric、目录、可观测视图。

    resources 列表（供指标按序采样）：[n0.mem, n0.ssd, n1.mem, n1.ssd, ..., fabric]
    """

    def __init__(self, env, spec):
        import numpy as np

        topo = spec.topo
        self.spec = spec
        self.topo = topo
        self.env = env
        self.curve = PrefillCurve(spec.gpu.prefill_table)
        self.n_workers = topo.n_workers
        self.n_nodes = len(topo.nodes)

        # 计算侧（每 worker 可有独立背景负载）
        from dataclasses import replace as _replace
        self.gpus = []
        for w in range(self.n_workers):
            gcfg = spec.gpu
            if topo.gpu_bgs:
                gcfg = _replace(spec.gpu, bg_schedule=topo.gpu_bgs[w])
            self.gpus.append(GpuPool(env, w, self.curve, gcfg))
        self.locals = [
            LocalKVCache(topo.local_cache_gb, worker_id=w, n_workers=self.n_workers,
                         coord=(topo.cache_mode == "coord"))
            for w in range(self.n_workers)
        ]

        # 存储侧：每节点 mem/ssd 两个流体资源 + 共享 fabric
        self.nodes = []           # [(mem_res, ssd_res, NodeConfig)]
        for ni, ncfg in enumerate(topo.nodes):
            self.nodes.append((SharedKVStorage(env, f"n{ni}.mem", ncfg.mem),
                               SharedKVStorage(env, f"n{ni}.ssd", ncfg.ssd), ncfg))
        self.fabric = SharedKVStorage(env, "fabric", topo.fabric)
        self.res_names = []
        self.resources = []
        for ni, (mem, ssd, _) in enumerate(self.nodes):
            self.resources += [mem, ssd]
            self.res_names += [f"n{ni}.mem", f"n{ni}.ssd"]
        self.resources.append(self.fabric)
        self.res_names.append("fabric")

        cls_bytes = {c.name: c.tokens * spec.model.kv_gb_per_token for c in spec.wl.classes}
        self.cls_bytes = cls_bytes
        self.dir = MetadataDirectory(topo, [c.name for c in spec.wl.classes], cls_bytes)
        self.path_lat = topo.path_lat

        # 可观测视图（策略可见；Oracle 直接读 ground truth 资源对象）
        mean_hit_gb = spec.wl.mean_hit_tokens * spec.model.kv_gb_per_token
        self.obs = []
        for ri, res in enumerate(self.resources):
            rng = np.random.default_rng([spec.seed, 100 + ri, 4])
            self.obs.append(StorageObservable(env, res, spec.obs, mean_hit_gb, rng))

    # ---- 便捷访问 ----
    def res(self, node_idx: int, tier: str) -> SharedKVStorage:
        mem, ssd, _ = self.nodes[node_idx]
        return mem if tier == "mem" else ssd

    def res_idx(self, node_idx: int, tier: str) -> int:
        return node_idx * 2 + (0 if tier == "mem" else 1)

    def obs_for(self, node_idx: int, tier: str) -> StorageObservable:
        return self.obs[self.res_idx(node_idx, tier)]

    def stats_now(self) -> list:
        return [dict(qdepth=len(r.active), inflight=sum(tr.remaining for tr in r.active),
                     bytes_served=r.bytes_served, b_total=r.b_total, bg=r.bg_at(self.env.now))
                for r in self.resources]
