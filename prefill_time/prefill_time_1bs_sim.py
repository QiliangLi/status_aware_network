"""Per-request per-layer compute time (us) and KV load (GB) for a trace.

Standalone: GLM5 + D950PR, storage_sim defaults (bs=1, tp=1, dp=32, mla_type=2,
sparse_mode=DSA, dual_pipe=False). All coefficients come from the MODEL/NPU dicts
and operator formulas below (mirroring sim_code_v2.py / models.py); no imports
from the rest of the repo.

Usage:  python compute_per_request.py [trace.jsonl]
Input  : jsonl with fields input_length, u_tokens (e.g. derived/*.hit.jsonl)
Output : per line  per_layer_us, per_layer_kv_gb
"""
import sys, os, json, math

# ---- Precision constants (base.py) ----
FP8 = 1
FP16 = 2

# ---- Model: ModelArgsGLM5 (models.py:1586) ----
MODEL = {
    "model_name": "GLM5",
    "n_layers": 78, "n_dense_layers": 3,
    "dim": 6144, "inter_dim": 12288,
    "n_routed_experts": 256, "n_shared_experts": 1, "n_activated_experts": 8,
    "moe_inter_dim": 2048, "moe_compress_ratio": 1.0,
    "n_heads": 64,
    "q_lora_rank": 2048, "kv_lora_rank": 512,
    "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "v_head_dim": 256,
    "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
    "index_share_interval": 1,
    "kv_precision": FP16, "weight_precision": FP16, "moe_precision": FP16,
    "max_batch_size": 8, "max_seq_len": 4096 * 32, "vocab_size": 154880,
}

# ---- NPU: D950PR (npus.py:63) ----
NPU = {
    "fp16_flops": 512, "fp8_flops": 1024,
    "mem": 128, "mem_bw": 1600,
    "nvlink_bw": 2000, "storage_bw": 350, "pcie_bw": 100,
    "prefill_discount_cal": 0.8, "prefill_discount_mem": 0.8,
    "decode_discount_cal": 0.8, "decode_discount_mem": 0.8,
    "base_delay": 30, "kernel_launch": 7, "iops_limit": 1e12,
    "npu_core_num": 32,
    "pipeline_floor_mla": 0.35, "pipeline_floor_moe": 0.55,
}

# ---- Storage_sim fixed conf (storage_scheduling_sim.py:64-108) ----
NUM_NPU = 32
CONF = {
    "instance_type": "prefill", "mla_type": 2, "sparse_mode": "DSA",
    "dual_pipe": False,
    "dp": NUM_NPU, "ep": NUM_NPU, "tp": 1, "routed_rep_num": 1, "shared_rep_num": 1,
    "total_bs": NUM_NPU, "bs": 1,            # bs = total_bs / dp
    "output_len": 1, "moe_compress_ratio": 1.0,
    "sparse_HBM_hit_ratio": 0.8, "per_layer_miss_rates": None,
}

# Effective NPU throughput/bandwidth with prefill discount (GPU_perf getters)
F_FP16 = NPU["fp16_flops"] * NPU["prefill_discount_cal"] * 1e12   # FLOPS/s
F_FP8 = NPU["fp8_flops"] * NPU["prefill_discount_cal"] * 1e12
B_MEM = NPU["mem_bw"] * NPU["prefill_discount_mem"]               # GB/s


def _roof(cal, mem_bytes, fp=FP16, fp16_cal=False):
    """Roofline: t(us) = max(cal/FLOPS, mem_bytes*fp/1024^3/BW). (get_time/get_time_fp16)"""
    F = F_FP16 if fp16_cal else F_FP8
    return max(cal / F * 1e6, mem_bytes * fp / (1024 ** 3) / B_MEM * 1e6)


def _t_indexer_qb_linear(m, c, g):
    # sim_code_v2.py:551 (FP8 weight/act via get_time)
    bs = c["bs"]; tp = c["tp"]
    cal = 2 * bs * m["q_lora_rank"] * m["index_n_heads"] * m["index_head_dim"] / tp
    mem_r = bs * m["q_lora_rank"] + bs * m["index_n_heads"] * m["index_head_dim"] / tp
    mem_w = m["q_lora_rank"] * m["index_n_heads"] * m["index_head_dim"] / tp
    return _roof(cal, mem_r + mem_w, fp=FP8)


def _t_indexer_qk_score(m, c, g):
    # sim_code_v2.py:598 ¡ª scans full_seq_len, three-way roofline with pipeline_floor
    bs = c["bs"]; tp = c["tp"]
    score_len = c["full_seq_len"]                # DSA: full sequence
    qk_cal = 2 * bs * m["index_n_heads"] * m["index_head_dim"] / tp * score_len
    seq_read = bs * m["index_n_heads"] * m["index_head_dim"] / tp
    s2 = 128
    n_iter = max(1, math.ceil(score_len / s2))
    cal_per_iter = qk_cal / F_FP8 * 1e6 / n_iter
    k_read = bs * s2 * m["index_head_dim"] * FP8 / (1024 ** 3) / B_MEM * 1e6
    q_read = seq_read * FP8 / (1024 ** 3) / B_MEM * 1e6 / n_iter
    score_w = bs * m["index_n_heads"] * s2 * FP16 / (1024 ** 3) / B_MEM * 1e6
    mem_per_iter = k_read + q_read + score_w
    pipe = g["pipeline_floor_mla"]
    t_per_iter = max(cal_per_iter, mem_per_iter, pipe)
    return n_iter * t_per_iter + g["kernel_launch"]


def _t_gather_kv(m, c, g):
    # sim_code_v2.py:1215 ¡ª DSA: seq_len = index_topk (constant)
    bs = c["bs"]; tp = c["tp"]
    seq_len = m["index_topk"]
    io_count = bs * seq_len
    kv_dim = m["kv_lora_rank"] + m["qk_rope_head_dim"]
    data_gb = io_count * kv_dim * FP8 / (1024 ** 3)
    iops_time = io_count / g["iops_limit"] * 1e6
    bw_time = data_gb / B_MEM * 1e6
    s2 = 128
    n_iter = max(1, math.ceil(seq_len / s2))
    pipe = g.get("pipeline_floor_gather", g["pipeline_floor_mla"])
    t_per_iter = max(iops_time / n_iter, bw_time / n_iter, pipe)
    return n_iter * t_per_iter + g["kernel_launch"]


def _t_fused_mla_attention(m, c, g):
    # sim_code_v2.py:1262 ¡ª DSA branch: SFA on index_topk (constant seq_len)
    bs = c["bs"]; tp = c["tp"]
    seq_len = m["index_topk"]
    sfa_cal = (2 * bs * m["n_heads"] * m["qk_nope_head_dim"] * seq_len / tp
               + 2 * bs * m["n_heads"] * seq_len * m["v_head_dim"] / tp)
    mem_r = (bs * m["n_heads"] * m["qk_nope_head_dim"] / tp * FP16
             + bs * seq_len * (m["kv_lora_rank"] + m["qk_rope_head_dim"]) / tp * FP8)
    mem_w = bs * m["n_heads"] * m["v_head_dim"] / tp * FP16
    pipe = g["pipeline_floor_moe"]
    s2 = 128
    n_iter = max(1, math.ceil(seq_len / s2))
    cal_per_iter = sfa_cal / F_FP16 * 1e6 / n_iter
    mem_per_iter = (mem_r + mem_w) / (1024 ** 3) / n_iter / B_MEM * 1e6
    t_per_iter = max(cal_per_iter, mem_per_iter, pipe)
    return n_iter * t_per_iter + g["kernel_launch"]


def _t_mlp(m, c, g):
    # mlp_time ¡ú mlp_1of3/2of3/3of3 (FP8 cal for GLM5)
    bs = c["bs"]; tp = c["tp"]; nql = c["new_query_len"]
    # 1of3: dim¡úinter_dim
    cal = 2 * bs * nql * m["dim"] * m["inter_dim"] / tp
    mem_r = bs * nql * m["dim"] + m["dim"] * m["inter_dim"] / tp
    mem_w = bs * nql * m["inter_dim"] / tp
    t = _roof(cal, mem_r + mem_w, fp=FP8)
    # 2of3: inter_dim¡úinter_dim
    cal = 2 * bs * nql * m["dim"] * m["inter_dim"] / tp
    mem_r = bs * nql * m["inter_dim"] + m["dim"] * m["inter_dim"] / tp
    mem_w = bs * nql * m["dim"] / tp
    t += _roof(cal, mem_r + mem_w, fp=FP8)
    # 3of3: inter_dim¡údim
    cal = 2 * bs * nql * m["inter_dim"] * m["dim"] / tp
    mem_r = bs * nql * m["inter_dim"] / tp + m["dim"] * m["inter_dim"] / tp
    mem_w = bs * nql * m["dim"] / tp
    t += _roof(cal, mem_r + mem_w, fp=FP8)
    return t * c["moe_compress_ratio"]


def _routed_moe_bs(m, c):
    # sim_code_v2.py:28
    return max(c["bs"] * c["dp"] * c["new_query_len"] * m["n_activated_experts"]
               / c["routed_rep_num"] / m["n_routed_experts"], 1)


def _shared_moe_bs(m, c):
    # sim_code_v2.py:17
    return c["bs"] * c["dp"] * c["new_query_len"] * m["n_shared_experts"] / c["ep"] / c["shared_rep_num"]


def _t_moe(m, c, g):
    # moe_time: shared 1of3/2of3/3of3 (op_num=1) + routed 1of3/2of3/3of3 (op_num=routed_experts_num)
    bs_r = _routed_moe_bs(m, c); bs_s = _shared_moe_bs(m, c)
    dim = m["dim"]; mi = m["moe_inter_dim"]
    n_route = math.ceil(min(m["n_routed_experts"], c["bs"] * c["dp"] * m["n_activated_experts"]) / c["ep"])
    # shared 1of3
    cal = 2 * bs_s * dim * mi; mem_r = bs_s * dim + dim * mi * c["moe_compress_ratio"]; mem_w = bs_s * mi
    t = _roof(cal, mem_r + mem_w, fp=FP8)
    # shared 2of3
    cal = 2 * bs_s * dim * mi; mem_r = bs_s * mi + dim * mi * c["moe_compress_ratio"]; mem_w = bs_s * dim
    t += _roof(cal, mem_r + mem_w, fp=FP8)
    # shared 3of3
    cal = 2 * bs_s * dim * mi; mem_r = bs_s * dim + dim * mi * c["moe_compress_ratio"]; mem_w = bs_s * mi
    t += _roof(cal, mem_r + mem_w, fp=FP8)
    # routed 1of3/2of3/3of3 (op_num = n_route)
    for (mr, mw) in [(bs_r * dim + dim * mi, bs_r * mi),
                     (bs_r * mi + dim * mi, bs_r * dim),
                     (bs_r * dim + dim * mi, bs_r * mi)]:
        cal = 2 * bs_r * dim * mi
        t += _roof(cal, mr + mw, fp=FP8) * n_route
    return t


def _t_moe_dispatch(m, c, g):
    # sim_code_v2.py:1964
    size_gb = FP8 * _routed_moe_bs(m, c) * m["dim"] / (1024 ** 3)
    n_route = math.ceil(min(m["n_routed_experts"], c["bs"] * c["dp"] * m["n_activated_experts"]) / c["ep"])
    return g["base_delay"] + n_route * size_gb / g["nvlink_bw"] * 1e6


def _t_moe_combine(m, c, g):
    # sim_code_v2.py:1981
    size_gb = FP16 * _routed_moe_bs(m, c) * m["dim"] / (1024 ** 3)
    n_route = math.ceil(min(m["n_routed_experts"], c["bs"] * c["dp"] * m["n_activated_experts"]) / c["ep"])
    return g["base_delay"] + n_route * size_gb / g["nvlink_bw"] * 1e6


def _t_sparse_kv_load(m, c, g):
    # sim_code_v2.py:1997 + getSparseFetchKVCacheSizeOneLayer (DSA) + miss_rate
    fetch = (m["kv_lora_rank"] + m["qk_rope_head_dim"]) * c["bs"] * m["index_topk"] * m["kv_precision"]
    miss_rate = 1 - c["sparse_HBM_hit_ratio"]
    load_gb = fetch / (1024 ** 3) * miss_rate
    return load_gb / g["storage_bw"] * 1e6

def _t_dense_layer(m, c, g):
    """Dense layer DSA (DSAFusedTime, non-v2): indexer(2) + gather_kv + fused_mla_attention + mlp."""
    cal = (_t_indexer_qb_linear(m, c, g) + _t_indexer_qk_score(m, c, g)
           + _t_gather_kv(m, c, g) + _t_fused_mla_attention(m, c, g) + _t_mlp(m, c, g))
    comm = max(0, _t_moe_dispatch(m, c, g)) + max(0, _t_moe_combine(m, c, g)) + max(0, _t_sparse_kv_load(m, c, g))
    return cal + comm


def _t_moe_layer(m, c, g):
    """MoE layer DSA (DSAFusedTime, non-v2): indexer(2) + gather_kv + fused_mla_attention + moe(shared+routed)."""
    cal = (_t_indexer_qb_linear(m, c, g) + _t_indexer_qk_score(m, c, g)
           + _t_gather_kv(m, c, g) + _t_fused_mla_attention(m, c, g) + _t_moe(m, c, g))
    comm = max(0, _t_moe_dispatch(m, c, g)) + max(0, _t_moe_combine(m, c, g)) + max(0, _t_sparse_kv_load(m, c, g))
    return cal + comm


def per_request(input_length, u_tokens):
    """Return (per_layer_us, per_layer_kv_gb) for one request. Config-traceable: every
    coefficient reads from MODEL/NPU/CONF dicts above; the two trace fields map to
    full_seq_len and new_query_len (storage_scheduling_sim.py:101-102)."""
    c = dict(CONF)
    c["full_seq_len"] = input_length
    c["new_query_len"] = u_tokens
    n_layers = MODEL["n_layers"]; n_dense = MODEL["n_dense_layers"]
    t_total = n_dense * _t_dense_layer(MODEL, c, NPU) + (n_layers - n_dense) * _t_moe_layer(MODEL, c, NPU)
    per_layer_us = t_total / n_layers
    # ---- KV load: models.py:315 getPrefillKVLoadGB (non-V4 branch) ----
    cached = max(0, input_length - u_tokens)
    kv_mla = (MODEL["kv_lora_rank"] + MODEL["qk_rope_head_dim"]) * cached * c["bs"] * MODEL["kv_precision"]
    kv_idx = MODEL["index_head_dim"] * cached * c["bs"] * MODEL["kv_precision"]
    total_kv_gb = (kv_idx * math.ceil(n_layers / MODEL["index_share_interval"]) + kv_mla * n_layers) / (1024 ** 3)
    per_layer_kv_gb = total_kv_gb / n_layers
    return per_layer_us, per_layer_kv_gb


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "derived", "conversation_trace.jsonl.hit.jsonl")
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            us, kv = per_request(r['input_length'], r['u_tokens'])
            print(f"input_length={r['input_length']}\tu_tokens={r['u_tokens']}\t"
                  f"per_layer_us={us:.3f}\tper_layer_kv_gb={kv:.6e}")