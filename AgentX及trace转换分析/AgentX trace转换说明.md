# AgentX trace 转 Mooncake trace 转换说明

## 0. 本目录内容

| 文件 | 说明 |
|---|---|
| convert_agentx_to_mooncake.py | 转换脚本，含使用说明注释 |
| 原始trace样本/ | 2 个真实的 AgentX trace JSON 文件，小样本转换输入 |
| 原始trace全量/traces.jsonl | HF 数据集全量打包文件，393 行（每行一个会话），1.85 GB，gitignored，重新下载命令见 §4 |
| 转换结果/agentx_mooncake_sample.jsonl | 2 会话样本转换输出，231 行，约 3.5 MB |
| 转换结果/agentx_mooncake_full.jsonl | 全量 393 会话转换输出，gitignored，重新生成命令见 §4 |
| 1013-AgentX.md | AgentX 背景调研报告，介绍 AgentX 基准测试整体情况 |

---

## 1. AgentX trace 提供了什么数据

### 1.1 数据来源与文件结构

数据集托管于 HuggingFace：semianalysisai/cc-traces-weka-062126，共 393 行，每行是一个完整的 Claude Code 编程会话，格式来自 Weka KV-Cache-Tester 项目。每个 JSON 文件分三层：会话级元信息、requests 数组内的请求级记录、以及嵌套在 subagent 标记内的子会话请求。

所有原始内容（prompt、源代码、工具参数与结果）均已被移除，trace 只保留流量结构信息，可以公开分发。

### 1.2 会话级字段

| 字段 | 类型 | 含义 |
|---|---|---|
| id | string | 会话唯一标识 |
| models | list | 该会话用到的模型列表 |
| block_size | int | KV 缓存块大小，token 数。该数据集所有会话均为 64 |
| hash_id_scope | string | 固定为 local，表示块编号命名空间按会话隔离，每个会话独立从 0 开始编号，详见 1.6 |
| tool_tokens | int | 工具定义前缀的 token 数（可选，缺省 0） |
| system_tokens | int | 系统提示前缀的 token 数（可选，缺省 0） |
| requests | list | 请求序列，按时间排列，三种类型混排（见下） |

### 1.3 请求级字段

requests 数组中的每条记录，type 为 n 表示普通调用，type 为 s 表示流式调用，两者字段结构相同：

| 字段 | 类型 | 含义 |
|---|---|---|
| t | float | 请求时间戳，单位秒，相对会话开始 |
| type | string | n 或 s |
| model | string | 该请求实际使用的模型 |
| in | int | 输入 token 数 |
| out | int | 输出 token 数 |
| hash_ids | list | KV 缓存块编号列表，每 64 token 一块，表达该请求的完整前缀结构，详见 1.6 |
| input_types | list | 输入内容类型标注，如 tool_result、text |
| output_types | list | 输出内容类型标注 |
| stop | string | 停止原因：空串、tool_use、end_turn |
| api_time | float | 服务端处理耗时，单位秒 |
| think_time | float | 客户端在该请求前的等待时间，单位秒 |
| ttft | float | 记录的首 token 延迟，单位秒，仅流式请求有 |

### 1.4 subagent 级字段

requests 数组中 type 为 subagent 的记录表示一次子代理派生：

| 字段 | 类型 | 含义 |
|---|---|---|
| t | float | 派生时间戳，单位秒 |
| agent_id | string | 子代理标识 |
| subagent_type | string | 子代理类型，如 Explore、Subagent |
| duration_ms | int | 子代理运行时长，毫秒 |
| total_tokens | int | 子代理全部内部请求的 token 总量 |
| tool_use_count | int | 子代理的工具调用次数 |
| status | string | 终态，如 completed |
| requests | list | 子代理的内部请求，字段结构与 1.3 完全相同 |
| models | list | 子代理使用的模型 |

父会话中 subagent 标记之后的下一个请求，语义上表示父会话等待子代理完成后才继续。

### 1.5 真实数据示例

会话级（截取自 原始trace样本/006c98de37d819e95b0840e25426bb7ca99d.json）：

```json
{
  "id": "006c98de37d819e95b0840e25426bb7ca99d",
  "models": ["claude-opus-4-8"],
  "block_size": 64,
  "hash_id_scope": "local",
  "requests": [ ... ]
}
```

请求级（该会话前两条请求，hash_ids 已截断）：

```json
{"t": 0.0,   "model": "claude-opus-4-8", "in": 448,   "out": 14,
 "hash_ids": [0, 1, 2, 3, 4, 5, 6],
 "api_time": 1.45, "type": "s", "ttft": 1.192}

{"t": 1.014, "model": "claude-opus-4-8", "in": 64640, "out": 65,
 "hash_ids": [7, 8, 9, "... 共 1010 项"],
 "api_time": 3.55, "think_time": 0.0, "type": "s", "ttft": 3.271}
```

subagent 级（截取自 002001296e8a8c38ad9d7cc436d691afc602.json）：

```json
{"t": 86541.149, "type": "subagent", "agent_id": "subagent_001_f552fe6f",
 "subagent_type": "Subagent", "duration_ms": 108127, "total_tokens": 331009,
 "status": "completed", "models": ["claude-haiku-4-5-20251001"],
 "requests": [ ... ]}
```

### 1.6 hash_ids 机制：前缀结构如何表达

每个 hash_id 编号代表一个 64 token 的内容块，内容相同的块编号相同（按内容寻址）。一条请求与之前请求共享多少前缀，完全由 hash_ids 序列的公共前缀表达，无需任何文本内容。

以样本会话 006c98de 的第 2、3 条请求为例（真实数据，中间编号省略）：

| 请求 | in | hash_ids |
|---|---|---|
| 第 2 条 | 64640 | [7, 8, 9, ..., 1016]，共 1010 块 |
| 第 3 条 | 64960 | [7, 8, 9, ..., 1016, 1017, 1018, 1019, 1020, 1021]，共 1015 块 |

第 3 条请求的前 1010 个编号与第 2 条完全相同，表示 64640 token 的旧上下文原样复用；末尾追加 5 个新块共 320 token，是新一轮用户输入与工具结果。多轮会话因此天然呈现高前缀复用。

由此得出四条要点：

1. **按内容寻址**：编号相同即内容相同，所以只比较编号序列就能判断任意两个请求共享多少前缀，这也是 mooncake 格式用 hash_ids 表达前缀命中的原理
3. **subagent 与父会话共用编号空间**：同一会话文件内，subagent 的请求与主链使用同一套编号。兄弟 subagent 的首请求从相同的系统提示块开始（内容相同所以编号相同），随后各自分叉
4. **跨会话编号独立**：每个会话从 0 重新编号，会话 A 的块 0 与会话 B 的块 0 内容无关。合并多个会话时必须重新编号，否则会产生虚假的跨会话前缀共享，这正是转换脚本重编号的原因，见 3.3.1

### 1.7 转换量级

20260915 的 2 个样本会话实测转换规模：

| 指标 | 数值 | 含义 |
|---|---|---|
| 输出请求数 | 231 | JSONL 行数，每行一个请求 |
| 输出会话数 | 6 | 2 条主链加 4 条 subagent 子会话 |
| 全局唯一块数 | 13381 | 去重后的真实内容量，约 85.6 万 token |
| 输入 token 总量 | 约 3919 万 | 全部请求 input 累加，含大量重复前缀 |
| 输出 token 总量 | 约 20.7 万 | 全部请求 output 累加 |
| input_length | 最小 448，中位数 119360，p90 353408，最大 396224 | 单请求输入规模分布 |
| output_length | 中位数 260，最大 8261 | 单请求输出规模分布 |

subagent 的内部请求会被展平为独立会话，时间戳保持原始值（与父会话同一时间轴），父会话与子会话的并行关系通过时间戳重叠自然表达。

全量数据集为 393 个会话，20260919 完成全量拉取与转换，实测规模见 3.4。

---

## 2. 转换脚本使用方法

### 2.1 依赖

- 本地文件模式：Python 3.8 及以上，仅使用标准库，无需安装任何第三方包
- HuggingFace 模式：额外执行 pip install datasets（推荐改用下述"下载 + --input-jsonl"两步法，可完全绕开该依赖）

### 2.2 命令

本地 Weka JSON 文件目录转换（小样本）：

```bash
python convert_agentx_to_mooncake.py --input-dir 原始trace样本 -o 转换结果/agentx_mooncake_sample.jsonl --prefix-hit --keep-model
```

全量 393 会话：先从 HuggingFace 下载打包文件（注意必须加 -L 跟随重定向，否则只会存下 1 KB 的跳转页），再流式转换：

```bash
curl -L -C - -o 原始trace全量/traces.jsonl \
  "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl"
python convert_agentx_to_mooncake.py --input-jsonl 原始trace全量/traces.jsonl -o 转换结果/agentx_mooncake_full.jsonl --prefix-hit --keep-model
```

或用 datasets 库一步完成（会额外生成 arrow 缓存，占用约两倍磁盘）：

```bash
python convert_agentx_to_mooncake.py --hf semianalysisai/cc-traces-weka-062126 -o agentx_mooncake_full.jsonl --prefix-hit
```

### 2.3 参数说明

| 参数 | 说明 |
|---|---|
| --input-dir | 本地 Weka trace JSON 文件目录，三种输入方式之一 |
| --input-jsonl | 打包 JSONL 文件（每行一个会话，即数据集的 traces.jsonl 原始格式），逐行流式读取，全量转换首选 |
| --hf | HuggingFace 数据集仓库 ID，经 datasets 库拉取，三种输入方式之一 |
| -o / --output | 输出 JSONL 路径，缺省 agentx_mooncake.jsonl |
| --prefix-hit | 额外输出 prefix_hit_tokens 字段，即理想前缀缓存命中 token 数，口径见 3.3.3 |
| --keep-model | 额外保留每条请求的 model 字段 |

运行结束会在终端输出统计：会话数、请求数、subagent 子会话数、input_length 与块数不一致的请求数、全局唯一块数。

---

## 3. 转换后数据说明

### 3.1 字段映射与来源

| 转换后字段 | 来源与推导 |
|---|---|
| timestamp | 原始字段 t 乘 1000，秒转毫秒，四舍五入取整 |
| input_length | 原始字段 in，直接映射 |
| output_length | 原始字段 out，直接映射 |
| hash_ids | 原始 hash_ids 按会话重新编号后输出。同一会话内编号映射关系固定，前缀共享关系完全保留；不同会话的块编号互不冲突，详见 3.3.1 |
| session_id | 主链请求为原始会话 id；subagent 内部请求在会话 id 后拼接后缀 ::sa: 与 agent_id。sa 是 subagent 的缩写，双冒号加 sa 是脚本自定义的命名约定而非格式要求，作用是在同一文件内区分主链与子会话 |
| model | 原始字段 model，仅在 --keep-model 时输出 |
| prefix_hit_tokens | 脚本计算，仅在 --prefix-hit 时输出，计算口径见 3.3.3 |

subagent 的内部请求会被展平为独立会话，时间戳保持原始值（与父会话同一时间轴），父会话与子会话的并行关系通过时间戳重叠自然表达。

### 3.2 转换前后对照示例

原始请求（转换前，t 单位为秒）：

```json
{"t": 14.081, "model": "claude-opus-4-8", "in": 64960, "out": 173,
 "hash_ids": [7, 8, 9, "... 共 1015 项"], "type": "s"}
```

转换输出（转换后，timestamp 单位为毫秒）：

```json
{"timestamp": 14081, "input_length": 64960, "output_length": 173,
 "hash_ids": [11500, 11501, 11502, "... 共 1015 项"],
 "session_id": "006c98de37d819e95b0840e25426bb7ca99d",
 "model": "claude-opus-4-8", "prefix_hit_tokens": 64640}
```

subagent 内部请求转换示例，注意 session_id 后缀与 haiku 模型：

```json
{"timestamp": 86541149, "input_length": 35008, "output_length": 153,
 "hash_ids": [3103, 3104, 3105, "... 共 547 项"],
 "session_id": "002001296e8a8c38ad9d7cc436d691afc602::sa:subagent_001_f552fe6f",
 "model": "claude-haiku-4-5-20251001", "prefix_hit_tokens": 0}
```

### 3.3 与 Mooncake 官方 trace 的差异分析

Mooncake 官方公开 trace（FAST25 论文，arxiv-trace/mooncake_trace.jsonl）每行格式：

```json
{"timestamp": 0, "input_length": 6755, "output_length": 500, "hash_ids": [0, 1, "..."]}
```

转换结果与其逐字段兼容。差异分三类：转换已解决的、使用时需注意的、负载特征层面的。

#### 3.3.1 转换已解决的差异：hash 命名空间

问题：AgentX 每个会话的块编号独立从 0 开始（见 1.6 第 4 条），而 mooncake 官方 trace 的编号全局唯一。若把多个会话直接拼进一个文件，不同会话中编号相同的块会被当作同一内容，产生虚假的跨会话前缀共享，污染前缀命中统计与回放行为。

脚本处理：逐会话转换时建立旧编号到新编号的映射，用全局递增计数器分配新编号。会话内的共享关系（主链前后轮延续、兄弟 subagent 共享系统前缀）原样保留，跨会话编号互不冲突。

结论：转换结果的 hash_ids 已是全局命名，使用时无需任何额外处理。

#### 3.3.2 使用时需注意的差异

**块大小：64 对 512，回放必须显式指定**

AgentX 数据每块 64 token，mooncake 官方 trace 约每块 512 token（由其 input_length 与 hash_ids 数量的比值推算）。AIPerf 的 mooncake loader 从 hash_ids 重建 prompt 时按每块 N token 生成，并校验 (len(hash_ids) - 1) * N < input_length <= len(hash_ids) * N，N 默认取 512。64 token 块的数据按 512 重建必然校验失败，直接报 ConfigurationError。正确回放命令：

```bash
aiperf profile \
  --input-file agentx_mooncake_sample.jsonl \
  --custom-dataset-type mooncake_trace \
  --isl-block-size 64 \
  --fixed-schedule \
  --url <推理服务地址> --model <模型名>
```

**时间戳：会话内相对时间，非全局时间线**

AgentX 的 t 从每个会话开始计秒，mooncake 官方 trace 是全局到达时间线。转换后多个会话的时间戳都从 0 附近开始，按 fixed-schedule 回放时全部会话并发启动。做单会话负载分析不受影响；如需模拟真实到达节奏，按会话加偏移量即可。

**session_id：官方 arxiv trace 无此字段**

转换结果带 session_id 用于区分主链与 subagent 子会话。AIPerf 的 mooncake loader 原生支持该字段（多轮会话分组依赖它），不影响兼容性。

**附加字段：model 与 prefix_hit_tokens**

官方格式仅 timestamp、input_length、output_length、hash_ids 四个核心字段。AIPerf 的 MooncakeTrace 模型允许额外字段，直接回放不受影响；喂给严格校验的其他工具前，建议剥离这两个字段。

**输出长度：记录值不保证复现**

output_length 来自录制时的实际输出 token 数，回放时推理服务可能提前停止生成。需要精确控制输出长度时配合 ignore_eos 类参数，这是 mooncake 回放路径的通用已知问题，非转换引入。

#### 3.3.3 负载特征差异（信息性）

AgentX 转换结果与 mooncake 官方 arxiv trace 面向完全不同的负载，容量与缓存结论不可直接互推：

| 维度 | AgentX 转换结果（2 会话实测） | mooncake 官方 arxiv trace |
|---|---|---|
| 上下文长度 | input 中位数 119360，最大 396224 | 单请求 input 数千 token 量级 |
| 轮次结构 | 多轮会话，主链最长 131 个请求，另含并行 subagent | 单轮请求为主 |
| 输入输出比 | 输入总量约 3919 万对输出约 20.7 万，输入占绝对主体 | 输入输出量级接近 |
| 前缀复用 | 前缀命中率中位数 98.8%，均值 94.7% | 前缀复用有限 |
| 会话节奏 | 主链跨度最长约 56 小时，含最长约 18.9 小时的空闲间隔 | 无会话概念 |

其中前缀命中率与 prefix_hit_tokens 的口径说明：脚本对每个会话维护一个已见块编号集合，模拟无限容量、按内容寻址的 KV 缓存；每条请求的命中等于其 hash_ids 开头连续出现在集合中的块数乘 64。该口径是理想上界，不等于任何具体实现（如 LRU 有限缓存）的命中率。98.8% 的含义是：若缓存容量无限，几乎所有请求的输入主体都无需重新计算 prefill。做容量规划或缓存策略对比时，应按目标策略重新仿真。

### 3.4 全量转换实测（20260919）

对 HF 全量打包文件（393 行，sha256 29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276）执行 `--input-jsonl` 流式转换，独立脚本复核的实测规模：

| 指标 | 数值 | 含义 |
|---|---|---|
| 输出请求数 | 98827 | JSONL 行数 |
| 输出会话数 | 2090 | 393 条主链 + 1697 条 subagent 子会话 |
| 全局唯一块数 | 5717185 | 去重后的真实内容量，约 3.66 亿 token |
| 输入 token 总量 | 约 216.4 亿 | 全部请求 input 累加，含大量重复前缀 |
| 输出 token 总量 | 约 1.06 亿 | 全部请求 output 累加 |
| 理想命中 token 总量 | 约 212.7 亿 | 占输入总量 98.3%（总量口径） |
| input_length | 最小 128，中位数 142016，p90 549504，最大 989824 | 单请求输入规模分布 |
| output_length | 中位数 444，最大 59903 | 单请求输出规模分布 |
| 前缀命中率（理想口径） | 中位数 99.1%，均值 90.0%，p10 83.2% | 按请求平均，与 3.3.3 样本口径一致 |
| 块数一致性 | 0 条不一致 | (len-1)*64 < in <= len*64 全部满足 |
| 产物大小 | 约 2.99 GB | agentx_mooncake_full.jsonl，gitignored |

与 2 会话样本相比的两个量级修正：全量请求均摊到主链约 251 条/会话（样本仅约 115 条/会话，样本偏小偏短）；理想命中率分布的长尾在 83% 附近（p10），均值被首请求拉低到 90%，做缓存收益评估时应看分布而非只看中位数。

时间戳仍是会话内相对时间，单会话最大跨度 254.8 小时；全量合并回放前须按 3.3.2 的说明给各会话加偏移，否则 2090 个会话会全部在 t≈0 并发启动。

---

## 4. 完整复现步骤

```bash
# 1. 用本目录样本转换（Python 3.8+，无第三方依赖）
python convert_agentx_to_mooncake.py --input-dir 原始trace样本 -o 转换结果/agentx_mooncake_sample.jsonl --prefix-hit --keep-model

# 2. 下载全量打包文件（1.85 GB，393 行；-L 必须加，-C - 支持断点续传）
curl -L -C - -o 原始trace全量/traces.jsonl \
  "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/main/traces.jsonl"
shasum -a 256 原始trace全量/traces.jsonl   # 应为 29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276

# 3. 流式转换全量 393 会话（无第三方依赖）
python convert_agentx_to_mooncake.py --input-jsonl 原始trace全量/traces.jsonl -o 转换结果/agentx_mooncake_full.jsonl --prefix-hit --keep-model

# 4. 校验输出与块数一致性（应无输出）
python -c "import json; [None for l in open('转换结果/agentx_mooncake_full.jsonl') for r in [json.loads(l)] if not ((len(r['hash_ids'])-1)*64 < r['input_length'] <= len(r['hash_ids'])*64) and print('inconsistent:', r['session_id'], r['timestamp'])]"

# 5. 通过 AIPerf 回放（块大小必须为 64，原因见 3.3.2）
aiperf profile --input-file 转换结果/agentx_mooncake_full.jsonl --custom-dataset-type mooncake_trace --isl-block-size 64 --fixed-schedule --url <地址> --model <模型>
```

原始打包文件与全量转换产物体积过大（合计约 2.6 GB），均已 gitignored，删掉后按上述步骤即可完整重建。
