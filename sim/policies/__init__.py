from .base import BasePolicy, PolicyCtx, WorkerView
from .always_fetch import AlwaysFetch
from .static_cost import StaticCost
from .dynamic_cost import DynamicCost
from .storage_aware_routing import StorageAwareRoute
from .oracle import Oracle
from .routing_baselines import RoundRobin, LoadAware, KvAware

POLICIES = {
    "p0": AlwaysFetch,
    "p1": StaticCost,
    "p2": DynamicCost,
    "p3": StorageAwareRoute,
    "p4": Oracle,
    "rr": RoundRobin,
    "load": LoadAware,
    "kv": KvAware,
}

NEEDS_OBS = {name for name, cls in POLICIES.items() if cls.needs_obs}


def make_policy(name: str, ctx: PolicyCtx):
    return POLICIES[name](ctx)
