# prefill_time — GLM5 + D950PR 逐层 prefill 计算时间估算

独立工具目录：不依赖 `sim/`，也不被 `sim/`、`tests/` 引用。系数抄录自外部仿真器
（sim_code_v2.py / models.py / npus.py / storage_scheduling_sim.py，不在本仓库），
函数内行号注释指向该外部代码。

## 文件

| 文件 | 用途 |
|---|---|
| `prefill_time_single.py` | 单请求：每层计算时间（us）与每层 KV 加载量（GB） |
| `prefill_time_batch.py` | 任意每卡 bs 的一层时间（`per_request(il, ut, bs=N)`）；异构批用 `per_batch([...], bs)`（padding 上界） |
| `validate_prefill_time.py` | 合理性验证：第一性原理对账 / 极限行为 / bs 自洽 / 外部锚点 / trace 分布，产出表与图 |

## 用法

```bash
python prefill_time/prefill_time_single.py [trace.jsonl]           # 每行输出 per_layer_us / per_layer_kv_gb
python prefill_time/prefill_time_batch.py [trace.jsonl] [bs]       # bs 不整除 32 或 >8 时告警
python prefill_time/validate_prefill_time.py                       # 重跑全部验证（图写入 docs/figures/）
```

输入 jsonl 需含 `input_length`、`u_tokens`（如 `mooncake_trace/derived/*.hit.jsonl`）。

## 语义要点（易错）

- **bs 是每卡 batch**：dp = 32//bs、ep = dp，全局并发恒 32 请求重新分组；bs 增大 =
  用更少的卡、每卡本地专家数 ×bs，不是更大的全局 batch。访存受限区专家权重不被摊薄。
- 「单请求时间」是系统满载（32 并发）下穿过一层的时延，不是空集群单请求时延。
- KV 加载量按命中前缀全上下文口径（1408 B/token/层），与 E23 系列存储实验一致。
- **单位：GB 十进制（1e9）**，与外部仿真器（1024^3）输出不可逐位比较。

## 修正与验证

- 原始 GBK 抄录版入库于 `bd8422d`；修正（F1–F11，含 prefill 语义、MLA 投影、
  absorb 注意力口径、单位统一）记录在两个脚本文件头，逐条附外部来源。
- 验证结论：算力受限区与第一性原理比值 1.000–1.07；三段式换挡点落在解析值
  F/(2B)=320 token；bs 守恒与等价成立；FP16 有效算力 409.6 TFLOPS 与 FlashMLA
  昇腾 950 实测 410 同值。
- **定位：kernel 级 roofline 下界**（不含排队、EP 不均、调度空洞）。对照 DeepSeek-V3
  生产集群，真实系统级时延约为此估计的 2–3 倍。方案相对比较可信；绝对值须按实测
  校准 `prefill_discount_cal/mem`。

详细材料：`docs/prefill_time计算时间仿真合理性验证-20261021.md`（验证文档），
`docs/prefill_time仿真解释-20261021.html`（自解释讲解页，含交互计算器）。
