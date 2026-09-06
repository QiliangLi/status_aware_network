"""M0 解析模型不变量测试（纲领 v1.1 R0"解析一致"与 §2.3/§3.1/§3.3 算例）。"""
from __future__ import annotations

import pytest

from sim.config import ModelConfig
from sim.m0 import RecoveryCost, recovery_case_from_tokens, two_request_completion
from sim.m0sim import run_fetch_path, run_recompute_path, run_two_request


def _case(**over) -> RecoveryCost:
    params = dict(R=10.0, P=2.0, B=4.0, ell=0.5, bw=1.0)
    params.update(over)
    return RecoveryCost(**params)


class TestKVBytes:
    """纲领 §2.3：KV 字节按张量维度计数，单位十进制 GB。"""

    def test_bytes_per_token_32l_8h_128d_2b(self):
        model = ModelConfig(layers=32, kv_heads=8, head_dim=128, bytes_per_elem=2)
        # 2(K,V) × 32 × 8 × 128 × 2 = 131072 B/token
        assert model.kv_gb_per_token * 1e9 == pytest.approx(131072.0)

    def test_4096_tokens_about_0p537_gb(self):
        model = ModelConfig(layers=32, kv_heads=8, head_dim=128, bytes_per_elem=2)
        assert model.kv_gb_per_token * 4096 == pytest.approx(4096 * 131072 / 1e9)

    def test_kv_bytes_scale_with_layers_and_heads(self):
        base = ModelConfig(layers=32, kv_heads=8, head_dim=128, bytes_per_elem=2)
        assert ModelConfig(layers=64, kv_heads=8, head_dim=128, bytes_per_elem=2).kv_gb_per_token \
            == pytest.approx(2 * base.kv_gb_per_token)
        assert ModelConfig(layers=32, kv_heads=16, head_dim=128, bytes_per_elem=2).kv_gb_per_token \
            == pytest.approx(2 * base.kv_gb_per_token)


class TestFetchThreshold:
    """纲领 §3.1：b* = B/(ΔC-ℓ)；取回划算 ⇔ F < ΔC（不是 F < R）。"""

    def test_threshold_formula(self):
        c = _case()
        assert c.threshold_bandwidth() == pytest.approx(4.0 / (8.0 - 0.5))

    def test_fetch_wins_above_threshold(self):
        c = _case()  # b* ≈ 0.533 GB/s，bw=1 在阈值之上
        assert c.fetch_wins()

    def test_fetch_loses_below_threshold(self):
        c = _case(bw=0.4)
        assert not c.fetch_wins()

    def test_boundary_equality(self):
        c = _case(bw=4.0 / (8.0 - 0.5))
        assert c.fetch_path_time() == pytest.approx(c.R)

    def test_compare_against_delta_c_not_full_R(self):
        # F 远小于完整 R，但漏掉 P 会高估取回：ΔC 很小时取回不划算
        c = _case(R=10.0, P=9.99, B=0.1, ell=0.3, bw=1.0)
        assert c.fetch_time() < c.R
        assert not c.fetch_wins()

    def test_no_bandwidth_helps_when_startup_exceeds_saving(self):
        # ΔC ≤ ℓ：带宽再大取回也赢不了
        c = _case(R=1.0, P=0.8, ell=0.5, B=0.001, bw=1e9)
        assert c.threshold_bandwidth() is None
        assert not c.fetch_wins()

    def test_linear_scaling_leaves_threshold_unchanged(self):
        # §3.1 反例：ℓ=0 且成本线性时，等比放大不改变取回偏好——
        # 长短差异必须来自非线性来源，不能靠放大同一模板制造
        base = _case(ell=0.0)
        assert base.threshold_bandwidth() is not None
        for k in (0.5, 2.0, 10.0):
            scaled = _case(ell=0.0, R=base.R * k, P=base.P * k, B=base.B * k)
            assert scaled.threshold_bandwidth() == pytest.approx(base.threshold_bandwidth())
            assert scaled.fetch_wins() == base.fetch_wins()


class TestTwoRequestAllocation:
    """纲领 §3.3：共享 I/O 1 GB/s，重算 0.2s/2s，各需 1 GB KV。"""

    TIMES = (0.2, 2.0)
    BYTES = 1.0
    BW = 1.0

    def test_both_fetch(self):
        assert two_request_completion(self.TIMES, self.BYTES, self.BW, "both_fetch") \
            == pytest.approx(2.0)

    def test_both_recompute(self):
        assert two_request_completion(self.TIMES, self.BYTES, self.BW, "both_recompute") \
            == pytest.approx(2.2)

    def test_expensive_fetch_cheap_recompute_is_best(self):
        assert two_request_completion(self.TIMES, self.BYTES, self.BW, "fetch_last") \
            == pytest.approx(1.0)

    def test_cheap_fetch_expensive_recompute(self):
        assert two_request_completion(self.TIMES, self.BYTES, self.BW, "fetch_first") \
            == pytest.approx(2.0)

    def test_allocation_beats_both_single_action(self):
        best = min(
            two_request_completion(self.TIMES, self.BYTES, self.BW, a)
            for a in ("both_fetch", "both_recompute", "fetch_first", "fetch_last")
        )
        assert best == pytest.approx(1.0)
        assert best < two_request_completion(self.TIMES, self.BYTES, self.BW, "both_fetch")
        assert best < two_request_completion(self.TIMES, self.BYTES, self.BW, "both_recompute")

    def test_matches_resource_lower_bound(self):
        # 串行无竞争模型中每种分配恰好达到资源下界，无凭空开销：
        # 完成时间 = max(取回字节的 I/O 串行时间, 被重算请求的 GPU 串行时间和)
        for alloc, (fetched, recomputed) in (
            ("both_fetch", (2, ())),
            ("both_recompute", (0, (0, 1))),
            ("fetch_first", (1, (1,))),
            ("fetch_last", (1, (0,))),
        ):
            io = fetched * self.BYTES / self.BW
            gpu = sum(self.TIMES[i] for i in recomputed)
            assert two_request_completion(self.TIMES, self.BYTES, self.BW, alloc) \
                == pytest.approx(max(io, gpu)), alloc

    def test_unknown_allocation_raises(self):
        with pytest.raises(ValueError):
            two_request_completion(self.TIMES, self.BYTES, self.BW, "both_sleep")


class TestRecoveryCaseFromTokens:
    """token 数到字节/成本的装配一致性。"""

    def test_bytes_follow_model_tensor_count(self):
        model = ModelConfig(layers=32, kv_heads=8, head_dim=128, bytes_per_elem=2)
        c = recovery_case_from_tokens(
            model, miss_tokens=1024, recompute_full_s=1.0, recompute_suffix_s=0.5,
            ell=0.0, bw=1.0,
        )
        assert c.B == pytest.approx(model.kv_gb_per_token * 1024)

    def test_delta_c_uses_given_costs(self):
        model = ModelConfig(layers=32, kv_heads=8, head_dim=128, bytes_per_elem=2)
        c = recovery_case_from_tokens(
            model, miss_tokens=1024, recompute_full_s=3.0, recompute_suffix_s=1.0,
            ell=0.0, bw=1.0,
        )
        assert c.delta_c == pytest.approx(2.0)


class TestM0Simulation:
    """事件级微仿真与解析模型一致（R0：解析值与仿真值误差、守恒、串行不重叠）。"""

    TIMES = (0.2, 2.0)
    BYTES = 1.0
    BW = 1.0

    @pytest.mark.parametrize(
        "allocation,expected",
        [("both_fetch", 2.0), ("both_recompute", 2.2),
         ("fetch_first", 2.0), ("fetch_last", 1.0)],
    )
    def test_completion_matches_analytic(self, allocation, expected):
        run = run_two_request(self.TIMES, self.BYTES, self.BW, allocation)
        assert run.completion == pytest.approx(expected, abs=1e-9)
        assert run.completion == pytest.approx(
            two_request_completion(self.TIMES, self.BYTES, self.BW, allocation), abs=1e-9,
        )

    def test_ledger_conserves_submitted_work(self):
        # 逐资源账本 = 提交的工作量：I/O 只服务取回，GPU 只服务重算
        run = run_two_request(self.TIMES, self.BYTES, self.BW, "fetch_last")
        assert run.ledger["io"] == pytest.approx(self.BYTES / self.BW, abs=1e-9)
        assert run.ledger["gpu"] == pytest.approx(self.TIMES[0], abs=1e-9)

        run = run_two_request(self.TIMES, self.BYTES, self.BW, "both_recompute")
        assert run.ledger["io"] == pytest.approx(0.0, abs=1e-12)
        assert run.ledger["gpu"] == pytest.approx(sum(self.TIMES), abs=1e-9)

    def test_spans_do_not_overlap_per_resource(self):
        # 串行资源不变量：同一资源上的作业区间两两不重叠
        for alloc in ("both_fetch", "both_recompute", "fetch_first", "fetch_last"):
            run = run_two_request(self.TIMES, self.BYTES, self.BW, alloc)
            by_resource: dict[str, list] = {}
            for s in run.spans:
                by_resource.setdefault(s.resource, []).append(s)
            for spans in by_resource.values():
                spans.sort(key=lambda s: s.start)
                for prev, nxt in zip(spans, spans[1:]):
                    assert nxt.start >= prev.end - 1e-12, (alloc, prev, nxt)

    def test_fetch_path_matches_analytic(self):
        # 单请求取回路径：完成时间 = ℓ + B/bw + P；账本各归其位
        case = _case()
        run = run_fetch_path(case)
        assert run.completion == pytest.approx(case.fetch_path_time(), abs=1e-9)
        assert run.ledger["io"] == pytest.approx(case.B / case.bw, abs=1e-9)
        assert run.ledger["gpu"] == pytest.approx(case.P, abs=1e-9)

    def test_recompute_path_matches_analytic(self):
        case = _case()
        run = run_recompute_path(case)
        assert run.completion == pytest.approx(case.R, abs=1e-9)
        assert run.ledger["gpu"] == pytest.approx(case.R, abs=1e-9)
        assert run.ledger.get("io", 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_threshold_crossing_agrees_with_simulation(self):
        # 阈值两侧的仿真判决与解析 fetch_wins 一致；b* 处两者相等
        b_star = _case().threshold_bandwidth()
        for factor, should_win in ((0.75, False), (1.25, True)):
            case = _case(bw=b_star * factor)
            run = run_fetch_path(case)
            sim_wins = run.completion < case.R
            assert sim_wins == should_win == case.fetch_wins()
        at_boundary = run_fetch_path(_case(bw=b_star))
        assert at_boundary.completion == pytest.approx(_case().R)
