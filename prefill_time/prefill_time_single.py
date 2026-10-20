# -*- coding: utf-8 -*-
"""GLM5 + D950PR 逐层 prefill 计算时间（us）与 KV 加载量（GB）估算——单请求版。

独立脚本：不依赖本仓库其他模块。系数抄录自外部仿真器（sim_code_v2.py / models.py /
npus.py / storage_scheduling_sim.py），函数内行号注释指向该外部代码。
与 prefill_time_batch.py 同一套算子模型，本文件把每卡 batch 固定为 1，用于衡量**单个请求**
prefill 时每层的计算时间与需加载的逐层 KV 量。

语义约定（继承外部 TestConfig，bs=1 / dp=ep=32 / tp=1）
-------------------------------------------------------
- 每张 NPU 处理 1 个请求，32 卡各自独立 prefill，MoE 专家按 EP=32 分摊。因此"单请求时间"
  是系统满载（32 并发）时该请求穿过一层的时延，不是空集群单请求时延。
- input_length → full_seq_len（上下文全长，含本次新 token）；
  u_tokens → new_query_len（本次 prefill 的新 token 数，即缓存未命中的部分）。

对抄录版（prefill_time_1bs_sim.py，GBK 编码，无法运行）的修正
----------------------------------------------------------
F1  注意力路径加入 new_query_len。原版 indexer_qb_linear / indexer_qk_score / gather_kv /
    fused_mla_attention 只按"单个 query token"建模，是 decode 语义；prefill 时每个新 token
    都要经 lightning indexer 对上下文打分、各选 top-2048、再做稀疏 MLA，计算量与新 token 数
    成正比。来源：DeepSeek-V3.2 技术报告（arXiv:2512.02556）、vLLM 博文
    （vllm.ai/blog/2025-09-29-deepseek-v3-2）、FlashMLA 仓库（indices 形状 [s_q, h_kv, topk]，
    逐 query token）。ut=1 时退化为原版 decode 情形。
F2  注意力 KV 的 HBM 读取按"每层每请求的上下文至多读一遍"建模：
    entries = min(ut*topk, full_seq_len)。依据：各 query 的 top-k 并集不会超过上下文；
    FlashMLA 稀疏 kernel prefill 实测 640 TFLOPS（H800）/410 TFLOPS（昇腾950，95% 峰值），
    说明 prefill 注意力是算力瓶颈而非逐 query 重复读 KV 的带宽瓶颈。kv_hbm_reuse=False
    给出无复用上界 ut*topk。kernel 内精确复用因子待核实。
F3  fused MLA 的 QK 计算量与 Q 读取补上 qk_rope_head_dim（原版只算 nope 的 192）。
F4  dense 层（前 3 层）去掉 MoE dispatch/combine：无专家路由就没有 all-to-all 通信。
F5  字节单位统一十进制 GB（1e9），与 FLOPS 的 1e12、NPU 带宽 GB/s 及本仓库 sim/config.py
    约定一致。原版按 1024^3 折算字节却按十进制 GB/s 除，访存项低估约 7%（新版在访存受限项
    上高 7.4%，计算受限项不变），故数值与外部 sim 原输出不再逐位可比。
F6  打分扫描长度取因果中点 full_seq_len - new_query_len/2：第 i 个新 token 只能看到
    il-ut+i 个前缀 token（vLLM 博文："Due to causality, every query token only attends to
    the tokens before it"，ke = range(n-q, n)）。原版一律用全长，对首轮长 prefill（ut≈il）
    会把 indexer 二次项高估到 2 倍。
F8  修正用法文件名；默认 trace 路径指向本仓库 mooncake_trace/derived/，缺失时报错提示。
F9  indexer 打分写回按每 (query, key) 对一个标量计：head 归约在 kernel 内完成。依据
    DeepGEMM README（github.com/deepseek-ai/DeepGEMM，mqa_logits）：
    out_ij = relu(q[i,:,:] @ kv_j) * weights[i,:] 后 .sum() 得标量，输出形状
    [seq_len, seq_len_kv]。原版按逐 head FP16 物化（index_n_heads 倍），decode 下可忽略，
    prefill 下会把长上下文的主导项高估 16 倍。标量 dtype 按 FP32（DeepGEMM 惯例，待核实；
    若为 FP16 该项再减半）。

保留的外部 kernel 模型假设（未改，已核实或待核实）
- 索引 K 缓存每 token 128 维 FP8（与 vLLM "key cache of 128 per token" 一致），每请求
  每层从 HBM 读一遍；打分 Q 读一遍；top-k 在物化的 (q, n) logits 上做（vLLM 博文：
  "the logits tensor is materialized before running a row-wise topk"）。
- gather_kv 与 fused_mla 各读一遍 KV（原版 decode 情形亦如此双重计），按 128 条一 tile、
  每 tile 受 pipeline_floor 下限约束，加一次 kernel_launch。
- 稀疏 KV 存储拉取：HBM 命中 0.8，未命中部分按 kv_precision(FP16) 从 storage_bw 拉取，
  与逐层 KV 加载量（getPrefillKVLoadGB，全上下文口径）并存：前者是注意力逐层按需取，
  后者是前缀命中时的整段加载量，二者语义不同。
- MoE/MLP 公式、nvlink dispatch/combine、pipeline_floor 与 kernel_launch 数值均照抄外部仿真器。

单位：时间 us；字节 GB（十进制 1e9）；带宽 GB/s；算力 TFLOPS（1e12）。

用法
----
  python prefill_time_single.py [trace.jsonl]
    输入 jsonl 需含 input_length、u_tokens 字段
    （默认 ../mooncake_trace/derived/conversation_trace.jsonl.hit.jsonl）
    输出每行 per_layer_us、per_layer_kv_gb
  代码调用：
    from prefill_time_single import per_request
    us, kv = per_request(input_length, u_tokens)
"""
import sys, os, json, math

# ---- 精度常量（外部 base.py）----
FP8 = 1
FP16 = 2
FP32 = 4
GB = 1e9   # 十进制 GB（F5）

# ---- 模型：ModelArgsGLM5（外部 models.py:1586）----
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

# ---- NPU：D950PR（外部 npus.py:63）----
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

# ---- storage_sim 固定配置（外部 storage_scheduling_sim.py:64-108），bs=1 ----
NUM_NPU = 32
CONF = {
    "instance_type": "prefill", "mla_type": 2, "sparse_mode": "DSA",
    "dual_pipe": False,
    "dp": NUM_NPU, "ep": NUM_NPU, "tp": 1, "routed_rep_num": 1, "shared_rep_num": 1,
    "total_bs": NUM_NPU, "bs": 1,            # bs = total_bs / dp = 1
    "output_len": 1, "moe_compress_ratio": 1.0,
    "sparse_HBM_hit_ratio": 0.8, "per_layer_miss_rates": None,
    "kv_hbm_reuse": True,                    # F2：注意力 KV 按上下文并集读一遍
}

# prefill 折扣后的有效算力/带宽（GPU_perf getters）
F_FP16 = NPU["fp16_flops"] * NPU["prefill_discount_cal"] * 1e12   # FLOPS/s
F_FP8 = NPU["fp8_flops"] * NPU["prefill_discount_cal"] * 1e12
B_MEM = NPU["mem_bw"] * NPU["prefill_discount_mem"]               # GB/s


def _roof(cal, mem_bytes, fp=FP16, fp16_cal=False):
    """Roofline：t(us) = max(cal/FLOPS, mem_bytes*fp/GB/BW)。（外部 get_time/get_time_fp16）"""
    F = F_FP16 if fp16_cal else F_FP8
    return max(cal / F * 1e6, mem_bytes * fp / GB / B_MEM * 1e6)


def _nq(c):
    """本卡新 query token 总数（F1：prefill 的注意力/索引工作量都按它缩放）。"""
    return c["bs"] * c["new_query_len"]


def _score_len(c):
    """indexer 打分扫描长度，因果中点（F6）。"""
    return max(1.0, c["full_seq_len"] - c["new_query_len"] / 2.0)


def _kv_entries(m, c):
    """本层本卡注意力需从 HBM 读的 KV 条目数（F2）。"""
    per_req = c["new_query_len"] * m["index_topk"]
    if c.get("kv_hbm_reuse", True):
        per_req = min(per_req, c["full_seq_len"])
    return c["bs"] * per_req


def _t_indexer_qb_linear(m, c, g):
    # 外部 sim_code_v2.py:551（FP8 权重/激活）；F1：cal 与激活读写随新 token 数缩放
    nq = _nq(c); tp = c["tp"]
    cal = 2 * nq * m["q_lora_rank"] * m["index_n_heads"] * m["index_head_dim"] / tp
    mem_r = nq * (m["q_lora_rank"] + m["index_n_heads"] * m["index_head_dim"]) / tp
    mem_w = m["q_lora_rank"] * m["index_n_heads"] * m["index_head_dim"] / tp
    return _roof(cal, mem_r + mem_w, fp=FP8)


def _t_indexer_qk_score(m, c, g):
    # 外部 sim_code_v2.py:598；F1/F6/F9：打分计算、Q 读、打分写回均随新 token 数缩放，
    # 扫描长度取因果中点；索引 K 每请求读一遍（不随 token 数重复读）；
    # 打分写回每 (q,k) 对一个标量（head 归约在 kernel 内完成）
    nq = _nq(c); tp = c["tp"]
    sl = _score_len(c)
    qk_cal = 2 * nq * m["index_n_heads"] * m["index_head_dim"] / tp * sl
    q_read = nq * m["index_n_heads"] * m["index_head_dim"] / tp          # FP8，只读一遍
    k_read = c["bs"] * sl * m["index_head_dim"]                          # FP8，索引 K 每请求一遍
    score_w = nq * sl                                                    # FP32，每对一个 logit
    s2 = 128
    n_iter = max(1, math.ceil(sl / s2))
    cal_per_iter = qk_cal / F_FP8 * 1e6 / n_iter
    mem_per_iter = (q_read * FP8 + k_read * FP8 + score_w * FP32) / GB / B_MEM * 1e6 / n_iter
    t_per_iter = max(cal_per_iter, mem_per_iter, g["pipeline_floor_mla"])
    return n_iter * t_per_iter + g["kernel_launch"]


def _t_gather_kv(m, c, g):
    # 外部 sim_code_v2.py:1215；F2：按并集条目数，而非逐请求固定 topk
    io_count = _kv_entries(m, c)
    kv_dim = m["kv_lora_rank"] + m["qk_rope_head_dim"]
    data_gb = io_count * kv_dim * FP8 / GB
    iops_time = io_count / g["iops_limit"] * 1e6
    bw_time = data_gb / B_MEM * 1e6
    s2 = 128
    n_iter = max(1, math.ceil(m["index_topk"] / s2))
    pipe = g.get("pipeline_floor_gather", g["pipeline_floor_mla"])
    t_per_iter = max(iops_time / n_iter, bw_time / n_iter, pipe)
    return n_iter * t_per_iter + g["kernel_launch"]


def _t_fused_mla_attention(m, c, g):
    # 外部 sim_code_v2.py:1262；F1/F2/F3：计算与 Q 读写随新 token 数缩放、KV 按并集
    # 条目读、QK 计算补 rope 维度
    nq = _nq(c); tp = c["tp"]
    seq_len = m["index_topk"]
    qk_d = m["qk_nope_head_dim"] + m["qk_rope_head_dim"]
    sfa_cal = 2 * nq * m["n_heads"] * seq_len * (qk_d + m["v_head_dim"]) / tp
    mem_q = nq * m["n_heads"] * qk_d / tp * FP16
    mem_kv = _kv_entries(m, c) * (m["kv_lora_rank"] + m["qk_rope_head_dim"]) * FP8
    mem_w = nq * m["n_heads"] * m["v_head_dim"] / tp * FP16
    pipe = g["pipeline_floor_moe"]
    s2 = 128
    n_iter = max(1, math.ceil(seq_len / s2))
    cal_per_iter = sfa_cal / F_FP16 * 1e6 / n_iter
    mem_per_iter = (mem_q + mem_kv + mem_w) / GB / B_MEM * 1e6 / n_iter
    t_per_iter = max(cal_per_iter, mem_per_iter, pipe)
    return n_iter * t_per_iter + g["kernel_launch"]


def _t_mlp(m, c, g):
    # 外部 mlp_time → mlp_1of3/2of3/3of3（GLM5 走 FP8 计算）；bs*nql 已含新 token 数
    bs = c["bs"]; tp = c["tp"]; nql = c["new_query_len"]
    # 1of3: dim→inter_dim
    cal = 2 * bs * nql * m["dim"] * m["inter_dim"] / tp
    mem_r = bs * nql * m["dim"] + m["dim"] * m["inter_dim"] / tp
    mem_w = bs * nql * m["inter_dim"] / tp
    t = _roof(cal, mem_r + mem_w, fp=FP8)
    # 2of3: inter_dim→inter_dim
    cal = 2 * bs * nql * m["dim"] * m["inter_dim"] / tp
    mem_r = bs * nql * m["inter_dim"] + m["dim"] * m["inter_dim"] / tp
    mem_w = bs * nql * m["dim"] / tp
    t += _roof(cal, mem_r + mem_w, fp=FP8)
    # 3of3: inter_dim→dim
    cal = 2 * bs * nql * m["inter_dim"] * m["dim"] / tp
    mem_r = bs * nql * m["inter_dim"] / tp + m["dim"] * m["inter_dim"] / tp
    mem_w = bs * nql * m["dim"] / tp
    t += _roof(cal, mem_r + mem_w, fp=FP8)
    return t * c["moe_compress_ratio"]


def _routed_moe_bs(m, c):
    # 外部 sim_code_v2.py:28
    return max(c["bs"] * c["dp"] * c["new_query_len"] * m["n_activated_experts"]
               / c["routed_rep_num"] / m["n_routed_experts"], 1)


def _shared_moe_bs(m, c):
    # 外部 sim_code_v2.py:17
    return c["bs"] * c["dp"] * c["new_query_len"] * m["n_shared_experts"] / c["ep"] / c["shared_rep_num"]


def _t_moe(m, c, g):
    # 外部 moe_time：shared 1of3/2of3/3of3（op_num=1）+ routed 1of3/2of3/3of3（op_num=n_route）
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
    # routed 1of3/2of3/3of3（op_num = n_route）
    for (mr, mw) in [(bs_r * dim + dim * mi, bs_r * mi),
                     (bs_r * mi + dim * mi, bs_r * dim),
                     (bs_r * dim + dim * mi, bs_r * mi)]:
        cal = 2 * bs_r * dim * mi
        t += _roof(cal, mr + mw, fp=FP8) * n_route
    return t


def _t_moe_dispatch(m, c, g):
    # 外部 sim_code_v2.py:1964
    size_gb = FP8 * _routed_moe_bs(m, c) * m["dim"] / GB
    n_route = math.ceil(min(m["n_routed_experts"], c["bs"] * c["dp"] * m["n_activated_experts"]) / c["ep"])
    return g["base_delay"] + n_route * size_gb / g["nvlink_bw"] * 1e6


def _t_moe_combine(m, c, g):
    # 外部 sim_code_v2.py:1981
    size_gb = FP16 * _routed_moe_bs(m, c) * m["dim"] / GB
    n_route = math.ceil(min(m["n_routed_experts"], c["bs"] * c["dp"] * m["n_activated_experts"]) / c["ep"])
    return g["base_delay"] + n_route * size_gb / g["nvlink_bw"] * 1e6


def _t_sparse_kv_load(m, c, g):
    # 外部 sim_code_v2.py:1997 + getSparseFetchKVCacheSizeOneLayer（DSA）+ miss_rate；
    # F2：拉取条目数随并集口径缩放
    fetch = (m["kv_lora_rank"] + m["qk_rope_head_dim"]) * _kv_entries(m, c) * m["kv_precision"]
    miss_rate = 1 - c["sparse_HBM_hit_ratio"]
    load_gb = fetch / GB * miss_rate
    return load_gb / g["storage_bw"] * 1e6


def _t_dense_layer(m, c, g):
    """dense 层 DSA：indexer(2) + gather_kv + fused_mla_attention + mlp。

    F4：dense 层无专家路由，不含 MoE dispatch/combine 通信，只有稀疏 KV 拉取。
    """
    cal = (_t_indexer_qb_linear(m, c, g) + _t_indexer_qk_score(m, c, g)
           + _t_gather_kv(m, c, g) + _t_fused_mla_attention(m, c, g) + _t_mlp(m, c, g))
    comm = max(0, _t_sparse_kv_load(m, c, g))
    return cal + comm


def _t_moe_layer(m, c, g):
    """MoE 层 DSA：indexer(2) + gather_kv + fused_mla_attention + moe(shared+routed) + 通信。"""
    cal = (_t_indexer_qb_linear(m, c, g) + _t_indexer_qk_score(m, c, g)
           + _t_gather_kv(m, c, g) + _t_fused_mla_attention(m, c, g) + _t_moe(m, c, g))
    comm = max(0, _t_moe_dispatch(m, c, g)) + max(0, _t_moe_combine(m, c, g)) + max(0, _t_sparse_kv_load(m, c, g))
    return cal + comm


def per_request(input_length, u_tokens):
    """返回单个请求的（每层计算时间 us，每层 KV 加载量 GB）。

    系数全部可溯源：MODEL/NPU/CONF 字典 + 上方算子公式；两个 trace 字段映射到
    full_seq_len / new_query_len（外部 storage_scheduling_sim.py:101-102）。
    """
    if u_tokens < 0 or u_tokens > input_length:
        raise ValueError(f"u_tokens={u_tokens} 需满足 0 <= u_tokens <= input_length={input_length}")
    c = dict(CONF)
    c["full_seq_len"] = input_length
    c["new_query_len"] = u_tokens
    n_layers = MODEL["n_layers"]; n_dense = MODEL["n_dense_layers"]
    # ---- KV 加载量：外部 models.py:315 getPrefillKVLoadGB（非 V4 分支），F5 改十进制 GB ----
    cached = max(0, input_length - u_tokens)
    kv_mla = (MODEL["kv_lora_rank"] + MODEL["qk_rope_head_dim"]) * cached * c["bs"] * MODEL["kv_precision"]
    kv_idx = MODEL["index_head_dim"] * cached * c["bs"] * MODEL["kv_precision"]
    total_kv_gb = (kv_idx * math.ceil(n_layers / MODEL["index_share_interval"]) + kv_mla * n_layers) / GB
    per_layer_kv_gb = total_kv_gb / n_layers
    if u_tokens == 0:                 # 全命中、无新 token：没有 prefill 计算
        return 0.0, per_layer_kv_gb
    t_total = n_dense * _t_dense_layer(MODEL, c, NPU) + (n_layers - n_dense) * _t_moe_layer(MODEL, c, NPU)
    return t_total / n_layers, per_layer_kv_gb


_DEFAULT_TRACE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "mooncake_trace", "derived",
                              "conversation_trace.jsonl.hit.jsonl")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_TRACE
    if not os.path.exists(path):
        print(f"ERROR: trace 文件不存在：{path}\n"
              f"  本仓库可用输入：mooncake_trace/derived/*.hit.jsonl（512 token 块）、"
              f"AgentX及trace转换分析/转换结果/agentx_mooncake_full.jsonl（64 token 块）",
              file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                us, kv = per_request(r['input_length'], r['u_tokens'])
                print(f"input_length={r['input_length']}\tu_tokens={r['u_tokens']}\t"
                      f"per_layer_us={us:.3f}\tper_layer_kv_gb={kv:.6e}")
    except BrokenPipeError:
        # 下游（如 head）提前关闭管道属正常情形；重定向 stdout 以免解释器退出时再报一次
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
