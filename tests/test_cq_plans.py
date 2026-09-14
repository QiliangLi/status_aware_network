"""§11.2 四计划金标：逐数复现 P1–P4（Fraction 严格等号）+ 关键事件断言。"""
import sys
from fractions import Fraction as F

sys.path.insert(0, ".")

from tests.cq_reference import GOLD, PLANS, e20_scenario, e20_specs, run_plan


def _collect(eng):
    w = eng.w
    Fs = {rid: rr.F_s for rid, rr in w.requests.items()}
    return Fs, w


def check_plan(name):
    eng, status = run_plan(PLANS[name])
    g = GOLD[name]
    Fs, w = _collect(eng)
    assert status == "done", f"{name} status={status}"
    assert (Fs[0], Fs[1]) == g["S"], f"{name} S F={Fs[0]},{Fs[1]}"
    assert (Fs[2], Fs[3]) == g["L"], f"{name} L F={Fs[2]},{Fs[3]}"
    M = max(Fs.values())
    assert M == g["M"], f"{name} M={M}"
    ttft = sum(Fs[r] for r in range(4)) / 4
    assert ttft == g["TTFT"], f"{name} TTFT={ttft}"
    norm = sum(Fs[r] / w.specs[r].T0_s for r in range(4)) / 4
    assert norm == g["norm"], f"{name} norm={norm} expect {g['norm']}"
    specs = w.specs
    slo = sum(1 for r in range(4) if Fs[r] <= specs[r].deadline_s)
    assert slo == g["slo"], f"{name} slo={slo}"
    compute = sum(wk.compute_s for wk in w.workers.values())
    assert compute == g["compute"], f"{name} compute={compute}"
    stall = sum(wk.stall_s for wk in w.workers.values())
    assert stall == g["stall"], f"{name} stall={stall} expect {g['stall']}"
    idle = sum(wk.idle_s for wk in w.workers.values())
    assert idle == g["idle"], f"{name} idle={idle} expect {g['idle']}"
    read = w.storage.actual_integral_gb
    assert read == g["read"], f"{name} read={read}"
    return eng


def test_p1():
    check_plan("P1")


def test_p2():
    check_plan("P2")


def test_p3():
    eng = check_plan("P3")
    # 关键事件：worker0 层1 开始计算 229/48，worker1 层1 为 3293/576
    b0 = next(b for b in eng.w.batches.values() if b.worker_id == 0)
    b1 = next(b for b in eng.w.batches.values() if b.worker_id == 1)
    assert b0.C[1] == F(229, 48), b0.C[1]
    assert b1.C[1] == F(3293, 576), b1.C[1]


def test_p4():
    eng = check_plan("P4")
    # 关键事件：SS 层1 流提交序先于 LL 首层流（旧批 lookahead 先于同刻新批派发）
    seqs = {f.flow_id: f.submit_seq for f in
            [x for fl in eng.w.storage.flows.values() for x in []]}
    # 流已全部完成并删除；改从 rate_log/账本核对提交顺序
    led = eng.observable.ledger
    ss_l1 = [v for v in led.values() if v["batch_id"] == 0 and v["layer"] == 1][0]
    ll_l0 = [v for v in led.values() if v["batch_id"] == 1 and v["layer"] == 0][0]
    assert ss_l1["submit_seq"] < ll_l0["submit_seq"], "P4 SS 层1 应先于 LL 首层提交"
    assert ss_l1["submit_s"] == F(1, 2) and ll_l0["submit_s"] == F(1, 2)


def test_p4_closure_order_mutation():
    """反转旧批 lookahead 与新批派发顺序时 P4 不再通过金标（突变测试）。"""
    import sim.cq.engine as E
    from sim.cq.observable import Observable
    from tests.cq_reference import ScriptedPolicy

    orig_step = E.CqEngine._step

    def reversed_step(self, t):
        # 错误顺序：先控制器后旧批闭包
        st = self.w.storage
        done = [s for s, f in st.flows.items() if st.is_complete(f)]
        for s in sorted(done):
            f = st.remove(s, t)
            b = self.w.batches.get(f.batch_id)
            if b is not None:
                b.read_ready[f.layer] = t
        if done:
            st.allocate(t)
        while self.w.arrival_idx < len(self.w.arrival_order):
            rid = self.w.arrival_order[self.w.arrival_idx]
            if self.w.arrival_times[rid] <= t:
                rr = self.w.requests[rid]
                rr.state = "QUEUED"
                self.w.n_queued += 1
                self.w.arrival_idx += 1
            else:
                break
        st.allocate(t)
        if self.observable is not None:
            self.observable.on_instant(t)
        self._controller_tick(t)   # 先派发
        self._closure(t)           # 后旧批
        if st.boost_due(t):
            st.allocate(t)

    E.CqEngine._step = reversed_step
    try:
        scn = e20_scenario()
        specs = e20_specs()
        pol = ScriptedPolicy(PLANS["P4"])
        obs = Observable(scn, numeric=F)
        eng = E.CqEngine(scn, specs, pol, numeric=F, observable=obs)
        obs.attach(eng)
        eng.run()
        Fs = {rid: rr.F_s for rid, rr in eng.w.requests.items()}
        assert (Fs[0], Fs[1]) != GOLD["P4"]["S"] or (Fs[2], Fs[3]) != GOLD["P4"]["L"], \
            "反转顺序后不应通过金标"
    finally:
        E.CqEngine._step = orig_step


if __name__ == "__main__":
    for n in ("P1", "P2", "P3", "P4"):
        check_plan(n)
        print(n, "OK")
