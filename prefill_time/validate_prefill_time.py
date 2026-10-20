# -*- coding: utf-8 -*-
"""prefill_time 计算时间仿真的合理性验证（单次运行，输出表格与图）。

与 prefill_time_batch.py 配套：对其仿真逻辑做五个维度的合理性论证，全部可重跑溯源。

§0 回归不变量：single≡batch(bs=1)、ut=1 与旧版（git bd8422d）逐算子等价、
   il/ut/bs 单调、KV 随 cached 线性、dense<MoE、两文件算子段逐字一致。
§A 第一性原理复算：按教科书公式独立数每层每 token 的 FLOPs（absorb MLA + DSA + MoE）
   与每层每 NPU 的权重字节，对照脚本的算力受限时间（比值应≈1，超出部分是
   pipeline_floor/kernel_launch 固定开销）与访存受限底线。
§B 极限行为：每 token 每层时间随 ut 呈"权重读底线 → ridge → 算力平台"三段式；
   线性层 ridge 的理论值 F/(2B)=320 token 与曲线拐点对照（对一切 GEMM 形状成立）。
§C bs≠1 自洽：
   C1 N 个请求×ut ≈ 1 个请求×N·ut（计算受限区比值应≈1，差异只来自 indexer 上下文）；
   C2 摊薄收益随 ut 衰减（访存受限区大、算力受限区→0），并对照"固定 dp=32"的另一种
      bs 语义（该语义下专家权重才真正被摊薄）；
   C3 bs↔dp 耦合的守恒：每 NPU 的 MoE 计算量只随 nq=bs·ut 走，dp 只改每卡本地专家数
      （权重字节数），计算与带宽两条守恒式逐 bs 核对；
   C4 per_batch 同构等于 per_request(bs)、异构介于 [max(单请求), bs×max]。
§D 外部锚点：模型隐含吞吐/效率 vs FlashMLA 昇腾950 实测（410 TFLOPS prefill）与
   DeepSeek-V3 生产 H800 prefill 吞吐（73.7k tok/s/节点，含 56.3% 缓存命中）。
§E Mooncake 三条 derived trace 的每请求 78 层 prefill 时延分布（新旧对比箱线图）。

用法：python validate_prefill_time.py
图写入 ../docs/figures/fig_prefill_time_{a,b,c,d}.png
"""
import os, sys, json, math, subprocess, types, warnings
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
FIGDIR = os.path.join(REPO, "docs", "figures")
sys.path.insert(0, HERE)
import prefill_time_batch as M
import prefill_time_single as S


# ---------- 画图 ----------
def get_plt():
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 10,
        "font.sans-serif": ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    return plt


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def load_old():
    """从 git 取原始 GBK 版（bd8422d）用于新旧对比；不可得则返回 None。"""
    r = subprocess.run(["git", "show", "bd8422d:prefill_time/prefill_time_anybs_sim.py"],
                       capture_output=True, cwd=REPO)
    if r.returncode != 0:
        return None
    mod = types.ModuleType("old_gbk"); mod.__file__ = "<git:bd8422d>"
    exec(compile(r.stdout.decode("gbk"), "old_gbk", "exec"), mod.__dict__)
    return mod


# ---------- 第一性原理（独立于脚本公式重写） ----------
def textbook_flops_per_token(m, il, ut):
    """每 token 每层的 FLOPs（absorb MLA + DSA indexer + MoE/MLP），教科书口径。"""
    h = m["n_heads"]; ih = m["index_n_heads"]; idd = m["index_head_dim"]
    d_qk = m["kv_lora_rank"] + m["qk_rope_head_dim"]; d_v = m["kv_lora_rank"]
    sl = max(1.0, il - ut / 2.0)
    proj_params = (m["dim"] * m["q_lora_rank"]
                   + m["q_lora_rank"] * h * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"])
                   + h * m["qk_nope_head_dim"] * m["kv_lora_rank"]
                   + m["dim"] * d_qk
                   + h * m["kv_lora_rank"] * m["v_head_dim"]
                   + h * m["v_head_dim"] * m["dim"])
    return {
        "indexer_q": 2 * m["q_lora_rank"] * ih * idd,
        "indexer_k": 2 * m["dim"] * idd,
        "indexer_score": 2 * ih * idd * sl,
        "proj": 2 * proj_params,
        "attn_core": 2 * h * m["index_topk"] * (d_qk + d_v),
        "moe": (m["n_activated_experts"] + m["n_shared_experts"]) * 3 * 2 * m["dim"] * m["moe_inter_dim"],
        "mlp": 3 * 2 * m["dim"] * m["inter_dim"],
    }


def textbook_weight_bytes(m, dp):
    """每层每 NPU 权重字节（FP8，tp=1）。路由专家按每卡本地专家数（256/dp）计。"""
    h = m["n_heads"]; ih = m["index_n_heads"]; idd = m["index_head_dim"]
    d_qk = m["kv_lora_rank"] + m["qk_rope_head_dim"]
    proj_params = (m["dim"] * m["q_lora_rank"]
                   + m["q_lora_rank"] * h * (m["qk_nope_head_dim"] + m["qk_rope_head_dim"])
                   + h * m["qk_nope_head_dim"] * m["kv_lora_rank"]
                   + m["dim"] * d_qk
                   + h * m["kv_lora_rank"] * m["v_head_dim"]
                   + h * m["v_head_dim"] * m["dim"])
    n_local = math.ceil(min(m["n_routed_experts"], 256) / dp)
    return {
        "indexer": m["q_lora_rank"] * ih * idd + m["dim"] * idd,
        "proj": proj_params,
        "moe_experts": (n_local + m["n_shared_experts"]) * 3 * m["dim"] * m["moe_inter_dim"],
        "mlp": 3 * m["dim"] * m["inter_dim"],
    }


def textbook_moe_layer_cal_us(m, il, ut, bs=1):
    """MoE 层计算时间的第一性原理下界：线性/索引 GEMM 走 FP8，注意力核心走 FP16。"""
    f = textbook_flops_per_token(m, il, ut)
    nq = bs * ut
    lin = f["indexer_q"] + f["indexer_k"] + f["proj"] + f["moe"]
    return (nq * lin / M.F_FP8 * 1e6
            + nq * f["attn_core"] / M.F_FP16 * 1e6
            + nq * f["indexer_score"] / M.F_FP8 * 1e6)


# ---------- 脚本侧 ----------
def conf(il, ut, bs=1, dp=None):
    c = dict(M.CONF)
    c["bs"] = bs
    c["dp"] = dp if dp else max(1, 32 // bs)
    c["ep"] = c["dp"] * c["tp"]; c["total_bs"] = bs * c["dp"]
    c["full_seq_len"] = il; c["new_query_len"] = ut
    return c


def script_moe_ops(il, ut, bs=1, dp=None):
    c = conf(il, ut, bs, dp)
    m, g = M.MODEL, M.NPU
    return {
        "indexer_q": M._t_indexer_qb_linear(m, c, g),
        "indexer_score": M._t_indexer_qk_score(m, c, g),
        "gather": M._t_gather_kv(m, c, g),
        "attn_core": M._t_fused_mla_attention(m, c, g),
        "proj": M._t_mla_proj(m, c, g),
        "moe": M._t_moe(m, c, g),
        "comm": M._t_moe_dispatch(m, c, g) + M._t_moe_combine(m, c, g) + M._t_sparse_kv_load(m, c, g),
    }


def main():
    m = M.MODEL
    print("# prefill_time 合理性验证（validate_prefill_time.py）\n")
    old = load_old()

    # ---------- §0 回归不变量 ----------
    print("## §0 回归不变量")
    src_b = open(os.path.join(HERE, "prefill_time_batch.py"), encoding="utf-8").read()
    src_s = open(os.path.join(HERE, "prefill_time_single.py"), encoding="utf-8").read()
    seg = lambda s: s[s.index("def _roof("):s.index("def per_request(")]
    checks = {"两文件算子段逐字一致": seg(src_b) == seg(src_s)}
    pts = [(1024, 1024), (8192, 512), (8192, 1), (65536, 65536), (300, 300), (4096, 0), (2047, 1)]
    checks["single ≡ batch(bs=1)"] = all(S.per_request(i, u) == M.per_request(i, u, 1) for i, u in pts)
    mono = all(M.per_request(i2, u, 1)[0] >= M.per_request(i1, u, 1)[0]
               for u in (64, 512, 4096) for i1, i2 in zip((4096, 8192, 16384, 32768), (8192, 16384, 32768, 65536)))
    mono &= all(M.per_request(i, u2, 1)[0] >= M.per_request(i, u1, 1)[0]
                for i in (8192, 65536) for u1, u2 in zip((1, 64, 512, 4096), (64, 512, 4096, i)))
    mono &= all(M.per_request(8192, 512, b2)[0] >= M.per_request(8192, 512, b1)[0]
                for b1, b2 in zip((1, 2, 4), (2, 4, 8)))
    checks["时间随 il/ut/bs 单调"] = mono
    kvs = [M.per_request(8192 + c_, 512, 1)[1] for c_ in (0, 1024, 2048, 3072, 4096)]
    d = [kvs[i + 1] - kvs[i] for i in range(4)]
    checks["KV 随 cached 线性（1408B/token/层）"] = all(abs(x - d[0]) < 1e-12 for x in d) and abs(d[0] - 1408 * 1024 / 1e9) < 1e-12
    checks["dense 层 < MoE 层"] = all(
        M._t_dense_layer(m, conf(i, u), M.NPU) < M._t_moe_layer(m, conf(i, u), M.NPU)
        for i, u in ((8192, 512), (65536, 8192)))
    if old is not None:
        eq = True
        for il in (1024, 8192, 65536):
            co = dict(old.CONF); co["full_seq_len"] = il; co["new_query_len"] = 1
            cn = conf(il, 1)
            # 未改动且 floor 主导的算子应逐点相等
            for o in ("_t_indexer_qk_score", "_t_gather_kv"):
                eq &= abs(getattr(old, o)(old.MODEL, co, old.NPU) - getattr(M, o)(m, cn, M.NPU)) < 1e-6
            # 仅单位修正（1024^3→1e9）且访存主导的算子比值应为 1024^3/1e9
            r = M._t_moe(m, cn, M.NPU) / old._t_moe(old.MODEL, co, old.NPU)
            eq &= 1.073 < r < 1.075
            # dispatch/combine 由 base_delay 主导，比值落在 [1, 1.074]
            for o in ("_t_moe_dispatch", "_t_moe_combine"):
                r = getattr(M, o)(m, cn, M.NPU) / getattr(old, o)(old.MODEL, co, old.NPU)
                eq &= 0.999 < r < 1.075
        checks["ut=1 与旧版：floor 主导算子相等、纯单位项 moe 比值=1.074"] = eq
    for k, v in checks.items():
        print(f"- {'PASS' if v else 'FAIL'}  {k}")
    assert all(checks.values()), "§0 有失败项"

    # ---------- §A 第一性原理复算 ----------
    print("\n## §A 第一性原理复算")
    f = textbook_flops_per_token(m, 8192, 512)
    gf = lambda x: x / 1e9
    rows = [
        ["indexer_q（投影 2048→4096）", f["indexer_q"], "FP8"],
        ["indexer_k（投影 6144→128）", f["indexer_k"], "FP8"],
        ["indexer_score（打分 ut×il 二次项）", f["indexer_score"], "FP8"],
        ["proj（MLA 六个投影 + W_UK/W_UV 吸收）", f["proj"], "FP8"],
        ["attn_core（absorb 576/512 × topk2048）", f["attn_core"], "FP16"],
        ["moe（8 路由 + 1 共享，9×3×6144×2048）", f["moe"], "FP8"],
        ["mlp（dense 层 3×6144×12288）", f["mlp"], "FP8"],
    ]
    print(md_table(["组件（每 token 每层）", "FLOPs", "MFLOP"], [[r[0], f"{r[1]:.3e}", f"{r[1]/1e6:.1f}"] for r in rows]))
    tot_moe = f["indexer_q"] + f["indexer_k"] + f["indexer_score"] + f["proj"] + f["attn_core"] + f["moe"]
    act_b = (75 * (f["proj"] + f["indexer_q"] + f["indexer_k"] + f["moe"]) / 2
             + 3 * (f["proj"] + f["indexer_q"] + f["indexer_k"] + f["mlp"]) / 2) / 1e9
    print(f"MoE 层合计：{tot_moe/1e6:.0f} MFLOP/token（il=8192 口径，indexer_score 随上下文增长）；"
          f"激活参数等价 ≈ {act_b:.1f} B（与 DeepSeek-V3 的 37B 同量级）\n")

    print("算力受限区核对（脚本算子时间 / 第一性原理 FLOPs÷F，应≈1，超出部分为 floor/launch 固定开销）：")
    rows = []
    for il, ut in ((8192, 8192), (65536, 65536), (65536, 512)):
        ops = script_moe_ops(il, ut)
        f = textbook_flops_per_token(m, il, ut); nq = ut
        cb = {
            "indexer_q": nq * f["indexer_q"] / M.F_FP8 * 1e6,
            "indexer_score": nq * f["indexer_score"] / M.F_FP8 * 1e6,
            "proj": nq * (f["proj"] + f["indexer_k"]) / M.F_FP8 * 1e6,
            "attn_core": nq * f["attn_core"] / M.F_FP16 * 1e6,
            "moe": nq * f["moe"] / M.F_FP8 * 1e6,
        }
        rows.append([f"il={il}, ut={ut}"] + [f"{ops[k]/cb[k]:.3f}" for k in ("indexer_q", "indexer_score", "proj", "attn_core", "moe")]
                    + [f"{(sum(ops[k] for k in ops if k != 'comm'))/sum(cb.values()):.3f}"])
    print(md_table(["工况", "idx_q", "idx_score", "proj", "attn", "moe", "整层(cal)"], rows) + "\n")

    w = textbook_weight_bytes(m, 32)
    us_of = lambda nbytes: nbytes / M.GB / M.B_MEM * 1e6
    moe_w = w["indexer"] + w["proj"] + w["moe_experts"]
    print(f"每层每 NPU 权重（bs=1，dp=32，FP8）：indexer {w['indexer']/1e6:.1f} MB + proj {w['proj']/1e6:.1f} MB "
          f"+ 专家 {w['moe_experts']/1e6:.1f} MB（dense 层以 mlp {w['mlp']/1e6:.1f} MB 代替专家）"
          f"→ MoE 层访存底线 {us_of(moe_w):.0f} us/层")
    t_ut1 = sum(v for k, v in script_moe_ops(65536, 1).items() if k != "comm")
    print(f"ut=1（il=65536）脚本 MoE 层 cal {t_ut1:.0f} us vs 权重读底线 {us_of(moe_w):.0f} us"
          f" → 固定开销占 {(t_ut1 - us_of(moe_w))/t_ut1*100:.0f}%（pipeline_floor/kernel_launch/索引 K 扫描）")

    # ---------- §B 极限行为 ----------
    print("\n## §B 极限行为（bs=1，MoE 层，不含 comm）")
    ridge = M.F_FP8 / (2 * M.B_MEM * M.GB)
    print(f"线性 GEMM 理论 ridge：F/(2B) = {ridge:.0f} token（FP8、对任意 in×out 形状成立；MoE 按每专家 token 数）")
    uts = [2 ** k for k in range(0, 17)]
    rows = []
    for il in (8192, 65536):
        for ut in (1, 8, 64, 320, 1024, 8192, 65536):
            if ut > il:
                continue
            cal = sum(v for k, v in script_moe_ops(il, ut).items() if k != "comm")
            rows.append([il, ut, f"{cal/ut:.2f}", f"{textbook_moe_layer_cal_us(m, il, ut)/ut:.2f}",
                         f"{cal/textbook_moe_layer_cal_us(m, il, ut):.3f}"])
    print(md_table(["il", "ut", "脚本 us/token/层", "第一性原理下界", "比值"], rows))

    # ---------- §C bs≠1 自洽 ----------
    print("\n## §C bs≠1 自洽")
    print("C1 等价性：每层时间(bs=N, ut) ÷ 每层时间(bs=1, N·ut)——同一张卡上 token 总数相同")
    layer_tot = lambda il_, ut_, bs_, dp=None: sum(script_moe_ops(il_, ut_, bs_, dp).values())
    rows = []
    for il in (8192, 65536):
        for ut in (64, 512, 1024):
            for N in (2, 4, 8):
                if N * ut > il:
                    continue
                base = layer_tot(il, N * ut, 1)
                rows.append([il, ut, N, f"{layer_tot(il, ut, N)/base:.3f}", f"{layer_tot(il, ut, N, dp=32)/base:.3f}"])
    print(md_table(["il", "ut", "N", "耦合 dp=32/N", "固定 dp=32"], rows))
    print("（固定 dp 下比值≈1：bs 只是把同样多的 token 放到同一张卡，GEMM/注意力 FLOPs 完全相同；"
          "耦合语义下算力受限区（ut≥512）≈1，访存受限区（ut=64）偏大是因为 dp 收缩让每卡本地专家数×N、"
          "专家权重读 ×N——这是该语义的定义，不是计算错误。两列共有的残余（il=8k 1–5%，il=64k 最多 33%）"
          "来自随'请求数×上下文'走的项：N 个请求各读一遍索引 K 与各自上下文的 KV 并集、各拉一遍稀疏 KV，"
          "而 1 个长请求只读一遍——N 个不同请求确实有 N 份不同上下文，物理上应当如此）\n")

    print("C2 每请求摊薄比：每层时间(bs=N)/N ÷ 每层时间(bs=1)（il=65536，MoE 层含通信，越小收益越大）")
    rows = []
    for ut in (1, 8, 64, 320, 1024, 8192):
        base = layer_tot(65536, ut, 1)
        row = [ut]
        for N in (2, 4, 8):
            row.append(f"{layer_tot(65536, ut, N)/N/base:.3f}")
            row.append(f"{layer_tot(65536, ut, N, dp=32)/N/base:.3f}")
        rows.append(row)
    print(md_table(["ut", "N=2 耦合", "N=2 固定dp", "N=4 耦合", "N=4 固定dp", "N=8 耦合", "N=8 固定dp"], rows))
    print("（耦合=脚本实现：dp=32/N，全局 32 请求重打包，每卡本地专家数×N，专家权重不被摊薄；"
          "固定dp=假设 dp 恒 32、全局并发=32N，专家权重被摊薄——两种语义各自自洽）\n")

    print("C3 bs↔dp 耦合守恒（MoE 层，il=65536, ut=512）：计算量 ∝ nq=bs·ut，与 dp 无关；权重字节 ∝ 本地专家数")
    rows = []
    for bs in (1, 2, 4, 8):
        dp = 32 // bs
        c = conf(65536, 512, bs)
        tok_per_expert = M._routed_moe_bs(m, c)
        n_local = math.ceil(min(256, bs * dp * 8) / c["ep"])
        moe_cal = M._t_moe(m, c, M.NPU)
        wexp = (n_local + 1) * 3 * m["dim"] * m["moe_inter_dim"]
        rows.append([bs, dp, n_local, f"{tok_per_expert:.0f}", f"{moe_cal:.0f}", f"{moe_cal/bs:.0f}", f"{wexp/1e6:.0f}",
                     f"{wexp/bs/1e6:.0f}"])
    print(md_table(["bs", "dp", "本地专家数", "token/专家", "MoE 时间 us", "÷bs", "专家权重 MB", "÷bs"], rows)
          + "\n（token/专家 = bs·dp·ut·8/256 = ut·(32/32)=512 恒定；MoE 时间÷bs 恒定 ⇒ 每卡计算量只随 nq 走）")

    a = M.per_batch([(8192, 512)] * 4, bs=4)
    b = M.per_request(8192, 512, 4)
    reqs = [(8192, 8192), (1024, 64)]
    pb = M.per_batch(reqs)
    hi = max(M.per_request(*r, 1)[0] for r in reqs)
    print(f"C4 per_batch：同构 4×(8192,512) 与 per_request(bs=4) 相等：{abs(a[0]-b[0])<1e-9}；"
          f"异构 {reqs}：padding {pb[0]:.0f} us ∈ [{hi:.0f}, {4*hi:.0f}]：{hi <= pb[0] <= 4*hi}")

    # ---------- §D 外部锚点 ----------
    print("\n## §D 外部锚点")
    il = ut = 8192
    t_tok = M.per_request(il, ut, 1)[0] * 78 / ut / 1e6      # s/token（78 层）
    tpn = 1 / t_tok
    fd = textbook_flops_per_token(m, il, ut)                  # 算力受限区口径
    moe_layer = fd["indexer_q"] + fd["indexer_k"] + fd["indexer_score"] + fd["proj"] + fd["attn_core"] + fd["moe"]
    dense_layer = fd["indexer_q"] + fd["indexer_k"] + fd["indexer_score"] + fd["proj"] + fd["attn_core"] + fd["mlp"]
    flops_tok = 75 * moe_layer + 3 * dense_layer
    print(f"模型隐含（il=ut=8192，算力受限区）：{tpn:.0f} token/s/NPU（32 卡集群 {tpn*32/1e3:.0f} k tok/s），"
          f"每 token ≈ {flops_tok/1e9:.0f} GFLOP，"
          f"等效算力 {tpn*flops_tok/1e12:.0f} TFLOPS/NPU")
    print("锚点 1（kernel 级）：FlashMLA 稀疏 prefill 昇腾950 实测 410 TFLOPS = 峰值 95%；"
          "本模型 F_FP16 = 512×0.8 = 409.6 TFLOPS 与之同量级（D950PR 身份待核实）")
    ds_tok = 73700 * (1 - 0.563) / 8
    print(f"锚点 2（系统级）：DeepSeek-V3 生产 H800 节点 prefill 73.7k tok/s（含 56.3% 命中）"
          f"→ 计算 token {ds_tok:.0f} tok/s/GPU；DSV3 37B 激活 ⇒ 线性 74 GFLOP/token，加 MLA 注意力"
          f"（128 头×1088 维×上下文，2k–4k 上下文约 +35–70 GFLOP）⇒ "
          f"~{ds_tok*109/1e3:.0f}–{ds_tok*144/1e3:.0f} TFLOPS/GPU ≈ H800 FP8 密集峰值 1979 的 "
          f"{ds_tok*109/1e3/1979*100:.0f}–{ds_tok*144/1e3/1979*100:.0f}%（估算，取决于平均上下文）")
    print("结论：本模型是 kernel 级 roofline（0.8 效率、通信只含 dispatch/combine），"
          "对生产系统级时延是偏乐观的下界（生产 MFU 约 20–30% vs 模型等效 63%）")

    # ---------- §E Mooncake trace ----------
    print("\n## §E Mooncake derived trace（每请求 78 层 prefill 时延，ms）")
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels, new_l, old_l, rows = [], [], [], []
    for name in ("conversation", "toolagent", "synthetic"):
        path = os.path.join(REPO, "mooncake_trace", "derived", f"{name}_trace.jsonl.hit.jsonl")
        nl, ol = [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                nl.append(M.per_request(r["input_length"], r["u_tokens"], 1)[0] * 78 / 1e3)
                if old is not None:
                    ol.append(old.per_request(r["input_length"], r["u_tokens"], 1)[0] * 78 / 1e3)
        new_l.append(nl); old_l.append(ol)
        p = lambda xs, q: sorted(xs)[min(int((len(xs) - 1) * q / 100) + 1, len(xs) - 1)]
        rows.append([name, len(nl), f"{p(nl,50):.0f}", f"{p(nl,90):.0f}", f"{p(nl,99):.0f}", f"{max(nl):.0f}",
                     f"{sum(nl)/sum(ol):.2f}" if ol else "-"])
        labels += [f"{name}\n新版", f"{name}\n旧版"]
    data = [x for pair in zip(new_l, old_l) for x in pair]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_yscale("log"); ax.set_ylabel("每请求 prefill 时延（78 层，ms，log）")
    ax.set_title("Mooncake 三条 trace：修正版 vs 原始版")
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "fig_prefill_time_d.png"), bbox_inches="tight")
    plt.close(fig)
    print(md_table(["trace", "请求数", "p50", "p90", "p99", "max", "总量比 新/旧"], rows))
    print("[fig] fig_prefill_time_d.png")

    # ---------- 图 a：极限行为 ----------
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for il, style in ((65536, "-"), (8192, "--")):
        xs, ys, yb = [], [], []
        for ut in uts:
            if ut > il:
                continue
            cal = sum(v for k, v in script_moe_ops(il, ut).items() if k != "comm")
            xs.append(ut); ys.append(cal / ut); yb.append(textbook_moe_layer_cal_us(m, il, ut) / ut)
        ax.plot(xs, ys, style, marker=".", ms=3, label=f"脚本 il={il}")
        ax.plot(xs, yb, style, color="tab:red", alpha=0.55, lw=1, label=f"第一性原理下界 il={il}")
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.annotate("ridge=320\n(F/2B)", (ridge, 2e-1), fontsize=7, color="gray")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("每请求新 token 数 ut（bs=1）"); ax.set_ylabel("每 token 每层时间（us）")
    ax.set_title("访存底线 → ridge → 算力平台（MoE 层，不含通信）"); ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIGDIR, "fig_prefill_time_a.png"), bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_prefill_time_a.png")

    # ---------- 图 b：算子拆解 ----------
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    cases = [(8192, 64), (8192, 512), (8192, 8192), (65536, 512), (65536, 8192), (65536, 65536)]
    keys = ["indexer_q", "indexer_score", "gather", "attn_core", "proj", "moe", "comm"]
    cols = ["#c7e9c0", "#238b45", "#9ecae1", "#3182bd", "#fdae6b", "#e6550d", "#969696"]
    import numpy as np
    xs = np.arange(len(cases))
    bottom = np.zeros(len(cases))
    for k, c_ in zip(keys, cols):
        vals = np.array([script_moe_ops(il, ut)[k] for il, ut in cases])
        ax.bar(xs, vals, bottom=bottom, label=k, color=c_, width=0.62)
        bottom += vals
    for x, tot in zip(xs, bottom):
        ax.text(x, tot * 1.03, f"{tot/1e3:.1f}k", ha="center", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([f"il={i}\nut={u}" for i, u in cases], fontsize=7)
    ax.set_yscale("log"); ax.set_ylabel("MoE 层时间（us，log）")
    ax.set_title("算子拆解（bs=1）"); ax.legend(fontsize=7, ncol=4)
    fig.savefig(os.path.join(FIGDIR, "fig_prefill_time_b.png"), bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_prefill_time_b.png")

    # ---------- 图 c：bs 自洽 ----------
    plt = get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    layer_tot = lambda il_, ut_, bs_, dp=None: sum(script_moe_ops(il_, ut_, bs_, dp).values())
    for ut, mk in ((64, "o"), (512, "s"), (1024, "^")):
        Ns, rc, rf = [], [], []
        for N in (2, 4, 8):
            if N * ut <= 8192:
                base = layer_tot(8192, N * ut, 1)
                Ns.append(N); rc.append(layer_tot(8192, ut, N) / base); rf.append(layer_tot(8192, ut, N, dp=32) / base)
        ax1.plot(Ns, rc, marker=mk, label=f"ut={ut} 耦合dp")
        ax1.plot(Ns, rf, marker=mk, ls="--", color="tab:red", alpha=0.5, label=f"ut={ut} 固定dp")
    ax1.axhline(1.0, color="gray", ls=":")
    ax1.set_xlabel("N（bs=N, ut）vs（bs=1, N·ut）"); ax1.set_ylabel("每层时间比值")
    ax1.set_title("C1：同卡 token 总数相同时的等价性"); ax1.legend(fontsize=6, ncol=2)
    uts2 = [1, 8, 64, 320, 1024, 8192]
    for N, mk in ((2, "o"), (4, "s"), (8, "^")):
        coup, fixdp = [], []
        for ut in uts2:
            base = layer_tot(65536, ut, 1)
            coup.append(layer_tot(65536, ut, N) / N / base)
            fixdp.append(layer_tot(65536, ut, N, dp=32) / N / base)
        ax2.plot(uts2, coup, marker=mk, label=f"bs={N}（耦合dp）")
        ax2.plot(uts2, fixdp, marker=mk, ls="--", color="tab:red", alpha=0.5, label=f"bs={N}（固定dp）")
    ax2.axhline(1.0, color="gray", ls=":")
    ax2.set_xscale("log", base=2); ax2.set_xlabel("ut"); ax2.set_ylabel("每请求时间比（/bs=1）")
    ax2.set_title("C2：摊薄只在访存受限区（il=65536）"); ax2.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_prefill_time_c.png"), bbox_inches="tight")
    plt.close(fig)
    print("[fig] fig_prefill_time_c.png")


if __name__ == "__main__":
    main()
