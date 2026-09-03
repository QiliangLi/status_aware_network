"""访问成本查询接口（架构图右侧）：输入 KV 大小 + 目标节点 -> 预计完成时间 / 压力等级 / 置信度。

仅使用计算侧可见信息（StorageObservable 的陈旧/EMA/带噪视图 + 静态路径延迟与 nominal 带宽），
不触碰 ground truth（Oracle 走资源对象的 hypothetical_*）。
"""
from __future__ import annotations

LEVELS = ("NORMAL", "WARM", "HOT", "CRITICAL")
# (进入阈值, 退出阈值)，util 单调升档需超过 enter，降档需低于 exit（滞回防抖）
_THRESH = ((0.70, 0.60), (0.85, 0.75), (0.97, 0.90))


class AccessCostQuery:
    def __init__(self, world, obs_cfg):
        self.world = world
        self.cfg = obs_cfg
        self._level = [0] * len(world.resources)   # 每资源当前压力档位 idx

    # ---------- 压力等级（升档直达、降档单级滞回） ----------
    def _update_level(self, ri: int, util: float) -> int:
        lv = self._level[ri]
        tgt = 0
        for i in range(3):
            if util >= _THRESH[i][0]:
                tgt = i + 1
        tgt = max(lv, tgt)                      # 升档直达
        if tgt < lv and util < _THRESH[lv - 1][1]:
            tgt = lv - 1                        # 降档一次一级（滞回）
        self._level[ri] = tgt
        return tgt

    def _util_of(self, ri: int) -> float:
        o = self.world.obs[ri]
        s = o.s
        t = self.world.env.now
        if o.interval <= 1e-12 or o.util_ema is None:
            # live 模式：背景负载 + （有排队即在满速率服务）
            base = s.bg_at(t) / max(1e-9, s.b_total)
            if len(s.active) > 0:
                base += max(0.0, s.cap_at(t)) / s.b_total
            return min(1.0, base)
        return o.util_ema

    def util_of(self, ri: int) -> float:
        return self._util_of(ri)

    def pressure(self, ri: int) -> str:
        return LEVELS[self._update_level(ri, self._util_of(ri))]

    # ---------- 完成时间估计：路径延迟 + tier 服务 + fabric 传输 ----------
    def _seg_est(self, ri: int, nbytes: float, signal: str = None) -> float:
        return self.world.obs[ri].estimate(nbytes, signal=signal or self.cfg.signal)

    def estimate(self, nbytes: float, worker: int, node: int, tier: str, noisy: bool = True,
                 signal: str = None) -> dict:
        ti = self.world.res_idx(node, tier)
        fi = len(self.world.resources) - 1          # fabric 恒为最后一个资源
        lat = self.world.path_lat[worker][node]
        tier_t = self._seg_est(ti, nbytes, signal)
        fab_t = self._seg_est(fi, nbytes, signal)
        t = lat + tier_t + fab_t
        if noisy:
            t *= 1.0  # 噪声已在 Observable.estimate 内按段叠加
        util = max(self._util_of(ti), self._util_of(fi))
        self._update_level(ti, self._util_of(ti))
        conf = 1.0 / (1.0 + self.cfg.interval / 0.2) * (1.0 / (1.0 + self.cfg.noise_sigma))
        return dict(time=t, pressure=LEVELS[self._level[ti]], confidence=conf,
                    util=util, tier_time=tier_t, fabric_time=fab_t, path_lat=lat)
