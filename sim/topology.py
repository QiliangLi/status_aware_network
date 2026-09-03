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
    """单 worker 的本地 KV 缓存：class -> bytes，LRU 淘汰，总容量有限。"""

    def __init__(self, cap_gb: float):
        self.cap = cap_gb
        self.used = 0.0
        self._items: OrderedDict[str, float] = OrderedDict()

    def holds(self, cls: str) -> bool:
        return cls in self._items

    def insert(self, cls: str, nbytes: float) -> None:
        if self.cap <= 1e-9:      # 容量 0 = 不启用本地缓存
            return
        if cls in self._items:
            self._items.move_to_end(cls)
            return
        self._items[cls] = nbytes
        self.used += nbytes
        while self.used > self.cap + 1e-9 and len(self._items) > 1:
            victim, b = self._items.popitem(last=False)
            self.used -= b

    def evict(self, cls: str) -> None:
        b = self._items.pop(cls, None)
        if b is not None:
            self.used -= b

    def size(self, cls: str) -> float:
        return self._items.get(cls, 0.0)


class MetadataDirectory:
    """元数据目录：class -> {(node_idx, tier)} 副本集合；同时维护节点占用（容量压力状态）。"""

    def __init__(self, topo: TopoConfig, cls_names, cls_bytes: dict):
        self.n_nodes = len(topo.nodes)
        self.cap = [n.cap_gb for n in topo.nodes]
        self.held = [0.0] * self.n_nodes
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
        self.locals = [LocalKVCache(topo.local_cache_gb) for _ in range(self.n_workers)]

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
