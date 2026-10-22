"""版本化 TFS：防守方修补库的参数化实现（v0 = 原版 cq_tfs）。

开关（对应防守方 Agent 20261022 修补库 D1~D9 + 两项升级项）：
- hope(D8)      死单降级：slack ≤ −T0 的请求排到全部可救请求之后（尾序 FCFS）
- classfreeze(D3) 分类双门槛（P75 ∨ b_ref/m）+ 到达时冻结
- coldabs(D5)   冷启动（hist<12）只用绝对阈
- wl(D1)        加权账本：Σd(在跑重流)+d(候选) ≤ cap；cap 用 est_bw 迟滞钳制
- active(D2)    配额只数仍在读的重批；唤醒对齐带宽释放（末层开算）
- biasfix(D6)   d×(1/(1+e)) 偏差逆修正 + C 保守取整
- lightgate(D7) 观测饱和（连续 2 份报价 util≥0.9）+ 紧重流在场时暂缓宽 slack 轻读流
- dsync(D9)     WAIT 相位去同步（+1.5ms×(wid%4)）
- hardage(D4')  升级项：wait≥2×guard_w 的锚点无条件豁免（不要求无更小 slack 者）
- keymode       'slack'（原版）| 'dslack'（升级项：deadline 为主、slack tie-break）
- captol(升级)  cap 容差 ×1.10（Wave1 G5 归因：2×64.94=129.9 > 120 把毒丸
                串行化、毒窗×2；10% 容差换 8% 超订、读拉伸被期限余量吸收）
- dnmed(升级)   下修用近 5 份报价中位数、阈值 0.85、底 0.75×b_ref（Wave3 W5：
                σ=15 时 0.9 阈值落在噪声带内、双份迟滞不够，低点把 cap 拖到
                ~105 → 4 并发降 3 → 近容量负载 0.018 vs 基线 1.000）
- ovswitch(升级) 过载切换：排队深度 >2m 持续 3 次决策 → 排序切 deadline 主序
                （Wave3 W6：非稳态过载突发期 slack 序的滚动活流窗口饿死受害者，
                稳态不可能定理不覆盖此形态；深度 <m 时切回）
- wl×active 联动：账本只计仍在读的重流（D1×D2 正交叠加，Wave1 G2/G4
  波量化损失的修复——波内早读者的末层计算期放行下一条）

信息边界：只用 ObservableSnapshot 公开字段 + 策略自持状态；零推演；
O(n log n)。母工况（E26f：b 恒 120、σ=0、e=0、类内 d 同质、guard_w=0）
下 hope/classfreeze/coldabs/biasfix/dsync/wl 近似逐动作等价，active/lightgate
非中性（冒烟核对）。
"""
from __future__ import annotations

from sim.cq.policies import TFS, bw_demand
from sim.cq.types import Action, JointAction


class TFSAdv(TFS):
    KEYFLAGS = {"hope", "classfreeze", "coldabs", "wl", "active",
                "biasfix", "lightgate", "dsync", "hardage"}
    N_MIN = 12          # D5 冷启动样本下限
    SAT, SAT_K = 0.90, 2
    TIGHT, WIDE = 0.5, 1.0
    DN_RATIO, DN_K = 0.90, 2
    UP_RATIO, UP_K = 1.25, 3
    DN_MED_N, DN_MED_RATIO, DN_FLOOR_X = 9, 0.80, 0.75
    OV_ON, OV_K = 1.0, 3        # 深度>1×m 持续 3 决策 → 切；<0.5×m → 切回
    CAP_TOL = 1.10

    def __init__(self, flags=None, keymode="slack", hist=None):
        super().__init__(hist)
        self.flags = set(flags or [])
        self.keymode = keymode
        self._heavy = {}          # D3 冻结表
        self._prev_q = None       # D7 报价差分
        self._sat_run = 0
        self._dn = 0; self._up = 0; self._cap_up = None
        self._bw_ring = []        # dnmed：近 DN_MED_N 份 bw_hat
        self._ov_run = 0; self._ov_on = False   # ovswitch 状态

    # -- D6：偏差修正版 d 计算 --
    def _d_map(self, snap):
        d, d_heavy, C, heavy_run, wake = bw_demand(snap, self._hist)
        if "biasfix" in self.flags:
            e = float(getattr(snap, "compute_bias", 0) or 0)
            if e != 0.0:
                d = {k: v * (1.0 + e) for k, v in d.items()}
                vals = sorted(d.values())
                d_heavy = vals[int(0.75 * (len(vals) - 1))] if vals else 0.0
                C = max(1, int(float(snap.b_ref) / d_heavy - 1e-9)) \
                    if d_heavy > 0 else 1 << 30
        return d, d_heavy, C, heavy_run, wake

    # -- D3/D5：分类 --
    def _is_heavy(self, snap, rid, d, d_heavy) -> bool:
        if "classfreeze" not in self.flags and "coldabs" not in self.flags:
            return d[rid] >= d_heavy
        if rid not in self._heavy:
            d_abs = float(snap.b_ref) / max(1, len(snap.workers))
            if "coldabs" in self.flags and len(self._hist) < self.N_MIN:
                self._heavy[rid] = d[rid] >= d_abs
            else:
                self._heavy[rid] = (d[rid] >= d_heavy) or (d[rid] >= d_abs)
        return self._heavy[rid]

    # -- D1：观测钳制 cap（v3：中位数下修 + 上修保留）--
    def _cap(self, snap, d_heavy: float) -> float:
        b_ref = float(snap.b_ref)
        cap = b_ref
        if snap.quote is not None:
            bw = float(snap.quote.bw_hat_gbps)
            if "dnmed" in self.flags:
                self._bw_ring.append(bw)
                if len(self._bw_ring) > self.DN_MED_N:
                    self._bw_ring.pop(0)
                if len(self._bw_ring) >= self.DN_MED_N:
                    med = sorted(self._bw_ring)[len(self._bw_ring) // 2]
                    if med < self.DN_MED_RATIO * b_ref:
                        cap = min(cap, max(med, self.DN_FLOOR_X * b_ref))
            else:
                if bw < self.DN_RATIO * b_ref:
                    self._dn += 1
                else:
                    self._dn = 0
                if self._dn >= self.DN_K:
                    cap = min(cap, bw)
            # 上修路径（G6 参考带宽过期的修复）：与下修正交
            if bw >= self.UP_RATIO * b_ref:
                self._up += 1
                if self._up >= self.UP_K:
                    self._cap_up = bw if self._cap_up is None else min(self._cap_up, bw)
            else:
                self._up = 0
                self._cap_up = None
            if self._cap_up is not None:
                cap = max(cap, min(self._cap_up, 2.0 * b_ref))
        return max(cap, d_heavy)

    # -- D2：active 计数与释放时刻 --
    def _active_and_release(self, snap, d, d_heavy):
        prof = snap._profile_cfg
        L, now = prof.L, float(snap.now)
        req = {r.rid: r for r in snap.requests}
        active, release = 0, None
        a_s, b_s = float(prof.a_s_per_token), float(prof.b_s_per_pair)
        t_l, n_sat = float(prof.t_launch_s), prof.N_sat
        for pw in snap.workers:
            if not pw.members:
                continue
            if not any(d.get(x, 0.0) >= d_heavy for x in pw.members):
                continue
            if pw.layers_done < L:
                active += 1
                rem = L - 1 - pw.layers_done
                if rem > 0:
                    r0 = req[pw.members[0]]
                    u, h = r0.u_tokens, r0.h_tokens
                    eta = 1.0 if u >= n_sat else max(float(prof.eta_floor), u / n_sat)
                    c = t_l + a_s * u / eta + b_s * (u * h + u * (u + 1) / 2)
                    t = now + rem * c
                    release = t if release is None else min(release, t)
        return active, release

    # -- D7：观测饱和 --
    def _saturated(self, snap) -> bool:
        q = snap.quote
        if q is None:
            self._prev_q = None; self._sat_run = 0
            return False
        cur = (float(q.sample_s),
               {fid: float(v) for fid, v in q.reported_bytes.items()},
               float(q.bw_hat_gbps))
        if self._prev_q is not None:
            dt = cur[0] - self._prev_q[0]
            if dt > 0:
                ds = sum(max(0.0, cur[1].get(f, 0.0) - s)
                         for f, s in self._prev_q[1].items())
                util = (ds / dt) / max(1e-6, cur[2])
                self._sat_run = self._sat_run + 1 if util >= self.SAT else 0
        self._prev_q = cur
        return self._sat_run >= self.SAT_K

    def decide(self, snap, scn) -> JointAction:
        f = self.flags
        d, d_heavy, C, heavy_run, wake = self._d_map(snap)
        if "wl" in f:
            cap = self._cap(snap, d_heavy)
            if "captol" in f:
                cap *= self.CAP_TOL
        now = float(snap.now)
        acc, release = heavy_run, None
        if "active" in f:
            acc, release = self._active_and_release(snap, d, d_heavy)
            if release is not None:
                wake = now + min(0.2, max(0.005, release - now))
        sat = self._saturated(snap) if "lightgate" in f else False
        # 在跑重流账本（D1 的 Σd 初值）
        led = 0.0
        if "wl" in f:
            L = snap._profile_cfg.L
            for pw in snap.workers:
                if pw.members and d.get(pw.members[0], 0.0) >= d_heavy:
                    if "active" in f and pw.layers_done >= L:
                        continue          # 末层计算期：读取已完成，不计账本
                    led += d.get(pw.members[0], 0.0)
        hv = {r.rid: self._is_heavy(snap, r.rid, d, d_heavy)
              for r in snap.requests if r.rid in snap.queued}
        queued = [r for r in snap.requests if r.rid in snap.queued]
        slack = {r.rid: float(r.deadline_s) - now - float(r.T0_s) for r in queued}

        # ovswitch：过载突发期切 deadline 主序（slack 序的稳态防饿死不覆盖
        # 非稳态；深度迟滞防抖动）
        if "ovswitch" in f:
            depth = len(queued)
            m = max(1, len(snap.workers))
            if depth > self.OV_ON * m:
                self._ov_run += 1
            elif depth < 0.5 * m:
                self._ov_run = 0
                self._ov_on = False
            if self._ov_run >= self.OV_K:
                self._ov_on = True

        def key(r):
            sl = slack[r.rid]
            if self._ov_on:
                if "hope" in f and sl <= -1.0 * float(r.T0_s):
                    return (1, float(r.arrival_s), r.rid)
                return (0, float(r.deadline_s), r.rid)
            if self.keymode == "dslack":
                return (float(r.deadline_s), sl, r.rid)
            if "hope" in f and sl <= -1.0 * float(r.T0_s):
                return (1, float(r.arrival_s), r.rid)
            return (0, sl, float(r.deadline_s), r.rid)

        order = sorted(queued, key=key)
        anchors = self._aged_anchors(snap, scn)
        gw = float(getattr(scn, "guard_w", 0) or 0)
        acts, used = [], set()
        for wid in sorted(snap.idle_workers):
            pick = None
            tight_heavy = any(hv[r.rid] and r.rid not in used
                              and slack[r.rid] < self.TIGHT * float(r.T0_s)
                              for r in order)
            for r in order:
                if r.rid in used:
                    continue
                heavy = hv[r.rid]
                if heavy:
                    if "wl" in f:
                        if led + d[r.rid] > cap:
                            continue
                    elif acc >= C:
                        continue
                if (sat and tight_heavy and r.h_tokens > 0 and not heavy
                        and slack[r.rid] > self.WIDE * float(r.T0_s)):
                    continue          # D7 软闸
                pick = r.rid
                break
            if pick is None:
                rest = [r for r in order if r.rid not in used]
                if not any(not hv[r.rid] for r in rest):
                    for a in anchors:
                        if a in snap.queued and a not in used:
                            pick = a
                            break
                elif "hardage" in f and gw > 0:
                    req = {r.rid: r for r in snap.requests}
                    for a in anchors:
                        if a not in snap.queued or a in used:
                            continue
                        if now - float(req[a].arrival_s) >= 2.0 * gw:
                            pick = a          # D4'：无条件饥饿保底豁免
                            break
            if pick is None:
                w = wake
                if "dsync" in f:
                    w = now + min(0.2, max(0.005, wake - now)
                                  + 0.0015 * (wid % 4))
                acts.append(Action("WAIT", wid, (), w))
                continue
            acts.append(Action("DISPATCH", wid, (pick,)))
            used.add(pick)
            if hv[pick]:
                if "wl" in f:
                    led += d[pick]
                else:
                    acc += 1
        return JointAction(tuple(acts))


def make_single(flags=None, keymode="slack"):
    def factory():
        return TFSAdv(flags=set(flags or []), keymode=keymode)
    factory.version = ("+".join(sorted(flags)) if flags else "v0") + \
                      ("/" + keymode if keymode != "slack" else "")
    return factory


# ---------------------------------------------------------------------------
# 两层版本（攻防迭代的对象：TwoLayerPolicy 参数组合）
# ---------------------------------------------------------------------------

def make_two_version(flags=None) -> dict:
    """两层版本参数。v0 = 设计原案。防御开关：
    e_ready_aging(D10)  就绪重流老化提升（饿死有界）
    bp_hyst(D11)        反压迟滞（高关低开）
    e_byte_share(D12)   E2 字节份额含轻流
    bp_loadbased(D13)   反压判据改加载位（Wave1 发现）
    """
    fl = set(flags or [])
    return {
        "name": "v0+" + "+".join(sorted(fl)) if fl else "v0",
        "policy_kw": {
            "ready_aging": "e_ready_aging" in fl,
            "light_gate_e": "e_byte_share" in fl,
            "bp_hyst": "bp_hyst" in fl,
            "bp_loadbased": "bp_loadbased" in fl,
        },
    }
