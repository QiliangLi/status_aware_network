"""derive 工具单元测试(变更设计 v1.4,T14–T22)。

算法:D1 连续前缀 + D2 请求内不可见 + D3 尾块参与 + D7 同刻不可见。
"""
import json
import os
import sys
from fractions import Fraction as F

import pytest

sys.path.insert(0, ".")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from cq_derive_hits import BLOCK, derive_one

TRACE_DIR = os.path.join(REPO, "mooncake_trace")
requires_trace = pytest.mark.skipif(
    not os.path.isdir(TRACE_DIR), reason="mooncake_trace 不存在")


def _write(tmp_path, rows):
    p = tmp_path / "synthetic_trace.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def _derive(tmp_path, rows):
    p = _write(tmp_path, rows)
    out = str(tmp_path / "derived" / "synthetic_trace.jsonl.hit.jsonl")
    derive_one(p, "synthetic_trace.jsonl", out)
    with open(out) as f:
        return [json.loads(x) for x in f]


def _row(ts, input_len, hashes):
    return {"timestamp": ts, "input_length": input_len, "output_length": 1,
            "hash_ids": hashes}


# ---------------------------------------------------------------------------
# T14 首现未命中
# ---------------------------------------------------------------------------

def test_t14_first_appearance_miss(tmp_path):
    rows = [_row(0, 1024, [1, 2])]
    out = _derive(tmp_path, rows)
    assert out[0]["hit_tokens"] == 0
    assert out[0]["u_tokens"] == 1024
    assert not out[0]["full_hit_adjusted"]


# ---------------------------------------------------------------------------
# T15 二次出现命中(不同 timestamp,先到先可见)
# ---------------------------------------------------------------------------

def test_t15_second_appearance_hit(tmp_path):
    rows = [_row(0, 1024, [1, 2]), _row(5, 1024, [1, 3])]
    out = _derive(tmp_path, rows)
    assert out[0]["hit_tokens"] == 0
    assert out[1]["hit_tokens"] == 512      # 命中连续块 1
    assert out[1]["u_tokens"] == 512


# ---------------------------------------------------------------------------
# T16 非连续不算(块 1 在更早请求第 2 位出现过,但本请求首块是 4)
# ---------------------------------------------------------------------------

def test_t16_non_contiguous_not_counted(tmp_path):
    rows = [_row(0, 1024, [1, 2]), _row(5, 1024, [4, 1])]
    out = _derive(tmp_path, rows)
    assert out[1]["hit_tokens"] == 0        # 首块 4 未见过即停


# ---------------------------------------------------------------------------
# T17 请求内重复块不可见
# ---------------------------------------------------------------------------

def test_t17_within_request_invisible(tmp_path):
    rows = [_row(0, 1024, [7, 7])]
    out = _derive(tmp_path, rows)
    assert out[0]["hit_tokens"] == 0        # 第二个 7 不能命中第一个


# ---------------------------------------------------------------------------
# T18/T19 尾块参与:同形请求(含尾块)二次全命中,留 1 token
# ---------------------------------------------------------------------------

def test_t18_t19_tail_block_participation(tmp_path):
    # 请求 1: input=700 → 完整块 [1] + 尾块(188) [2];首现全未命中,
    # 但完整块与尾块都进入已见集合(D3)。
    rows = [_row(0, 700, [1, 2]), _row(9, 700, [1, 2])]
    out = _derive(tmp_path, rows)
    assert out[0]["hit_tokens"] == 0
    # 请求 2:完整块 1 命中、尾块 2 也命中 → 全命中,留 1 token
    assert out[1]["hit_tokens"] == 699
    assert out[1]["u_tokens"] == 1
    assert out[1]["full_hit_adjusted"] is True


# ---------------------------------------------------------------------------
# T22 同 timestamp 并发组互相不可见;下一时刻可见
# ---------------------------------------------------------------------------

def test_t22_same_timestamp_invisible(tmp_path):
    rows = [_row(0, 1024, [1, 2]), _row(0, 1024, [1, 3]),
            _row(0, 1024, [1, 4]), _row(1, 1024, [1, 5])]
    out = _derive(tmp_path, rows)
    assert all(r["hit_tokens"] == 0 for r in out[:3])   # 同刻三条全部未命中
    assert out[3]["hit_tokens"] == 512                  # 下一时刻命中块 1


# ---------------------------------------------------------------------------
# T20 真实三份 trace:派生统计与设计文档 v1.3 指纹逐位一致
# ---------------------------------------------------------------------------

@requires_trace
def test_t20_real_trace_fingerprint(tmp_path):
    expect = {
        "conversation_trace.jsonl": (12031, 0.374),
        "toolagent_trace.jsonl": (23608, 0.570),
        "synthetic_trace.jsonl": (3993, 0.651),
    }
    for fname, (n_rows, tok) in expect.items():
        out = str(tmp_path / (fname + ".hit.jsonl"))
        stat = derive_one(os.path.join(TRACE_DIR, fname), fname, out)
        assert stat["n_rows"] == n_rows, fname
        assert abs(stat["token_hit_ratio"] - tok) < 1.5e-3, (fname, stat["token_hit_ratio"])
        # 时刻组结构(并发披露)
        if fname == "synthetic_trace.jsonl":
            assert stat["max_same_ts_group"] == 2
        else:
            assert stat["max_same_ts_group"] in (28, 47)


# ---------------------------------------------------------------------------
# T21 字段一致性:全量行 h+u=input、u>=1、行数与原文件一致
# ---------------------------------------------------------------------------

@requires_trace
def test_t21_field_consistency(tmp_path):
    for fname, _n, _m, _s in [
            ("conversation_trace.jsonl", 12031, 0, ""),
            ("toolagent_trace.jsonl", 23608, 0, ""),
            ("synthetic_trace.jsonl", 3993, 0, "")]:
        out = str(tmp_path / (fname + ".hit.jsonl"))
        derive_one(os.path.join(TRACE_DIR, fname), fname, out)
        with open(out) as f:
            rows = [json.loads(x) for x in f]
        assert len(rows) == _n
        for r in rows:
            assert r["hit_tokens"] + r["u_tokens"] == r["input_length"]
            assert r["u_tokens"] >= 1
            assert 0 <= r["hit_tokens"] <= r["input_length"] - 1


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
