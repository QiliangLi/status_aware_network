# AgentX（Semianalysis）调研报告

信息来源：Semianalysis InferenceX 官网、GitHub 仓库 SemiAnalysisAI/agentx-harness、PyPI

## 1. AgentX 是什么

AgentX 是调研机构 Semianalysis 旗下 InferenceX 开源基准测试体系的子项目，定位为面向 agentic 工作负载的推理系统基准测试（当前版本 v1.0）。它不评测 AI 模型或 agent 的能力，而是评测推理基础设施（GPU/TPU 服务器与推理服务栈）在真实编程 agent 负载下的性能表现。

项目主页：[https://inferencex.semianalysis.com/agentx](https://inferencex.semianalysis.com/agentx)（提供中文页面）

**数据来源**：AgentX v1.0 数据集基于参与者自愿提供的 393 个真实 Claude Code 会话 trace 构建，原始调用价值超过 300 万美元。入选会话需满足：至少 20 个请求、Claude Code 版本不低于 2.1.139、并行 subagent 不超过 10 个。处理时移除重复请求、客户端安全监控与标题生成请求，以及重建后 input 长度超过 990k token 的请求。

**隐私处理与构建流程**：公开 trace 移除全部原始内容（prompt、源代码、tool 参数与 tool 结果），仅保留请求 token 长度、prefix 复用关系、subagent 分支与时间信息；随后用确定性生成的合成 coding 与 tool-use token 填充重建，将每个会话重建为包含主 agent 轮次、并行 subagent 与轮次间 tool 执行时间的有向无环图（DAG）。回放采用固定 seed，先执行 warmup 建立缓存状态，再在不同并发客户端数下进行一小时 profiling，保证结果可复现。

**数据集统计**：单请求 input token 中位数 142k，单请求 output token 中位数 444，44% 的会话包含 subagent。数据集提供两个版本：full 版包含最长 1M token 上下文的完整回放集；256k 版适用于最大上下文配置为 256k token 的模型与推理引擎。

**官网排行榜已覆盖的硬件与模型**（即官方已公开发布测试结果的硬件平台与被测模型范围）：硬件平台包括 Google TPUv7 Ironwood、OpenAI Jalapeño、AMD MI355X、NVIDIA GB300 NVL72、GB200 NVL72、B200、H200、H100、RTX Pro，即将支持 TPUv8i、SambaNova SN50、Rubin NVL72 与 MI455X UALoE72；被测模型包括 Kimi K3、DeepSeek V4.1 Flash、GLM 5.3、MiniMax M3、Qwen 3.5、DeepSeek R1 等。所有结果通过公开的 GitHub Actions 流程生成，平均每周新增约 1000 个数据点；测试配置以 shell 脚本形式保存在公开仓库，完整基准数据库每周作为 GitHub Release 发布快照，全程可复现、可审计。

## 2. 与编程 Benchmark 的区别及新特征

核心区别：SWE-bench 等编程 benchmark 评估模型能力，即任务完成的正确性；AgentX 评估推理系统性能，即硬件与推理服务栈在 agentic 负载下的吞吐与延迟。两者不在同一个评测层面，属于互补关系而非竞争关系。

### 2.1 评测对象不同

SWE-bench 的被测对象是模型或 agent：给定真实 GitHub issue，让被测系统生成 patch，再运行测试用例验证正确性，输出 pass rate，关注任务结果的质量。

AgentX 的被测对象是推理基础设施：服务器硬件与推理框架（vLLM、SGLang、TensorRT-LLM 等）。模型生成的代码内容完全不重要，请求中的 token 是确定性合成的填充文本，没有语义价值。官方文档明确说明合成 payload 不能用于评估模型质量。

### 2.2 负载构造方式不同

SWE-bench 构造任务：每个实例是一个待修复的 bug，负载即一次或少数几次代码生成请求。

AgentX 构造流量模式：从真实 Claude Code 会话提取 trace，剥离全部内容，只保留结构特征，包括每轮请求的 input/output token 长度、跨轮次的 prefix 复用关系（决定 KV cache 命中模式）、subagent 并行分支拓扑（会话 DAG）、轮次之间 tool 执行的等待时间。回放时用合成 token 按这些结构特征重建请求序列。

### 2.3 指标体系不同

SWE-bench 的核心指标是任务解决率，属于质量维度。

AgentX 的核心指标属于性能维度：TTFT（首 token 延迟）、interactivity（交互响应性）、output throughput（输出吞吐）。AgentX 采用 closed-loop 模式，concurrency 表示同时运行的 agent 会话数而非固定 batch size，因此更快的配置会自然完成更多请求，实际遇到的工作负载组合可能略有差异，低并发时最明显。官方要求吞吐必须与 TTFT、interactivity 组合报告，单一延迟指标不构成完整结论。

### 2.4 新特征

AgentX 相对于既有推理 benchmark 的增量价值：

1. **负载真实性**：传统推理测试用固定序列长度（如 8K input / 1K output），无法反映 agentic 编码场景中上下文逐轮累积（中位数 142k，上限约 1M）的负载形态
2. **系统压力建模**：显式建模 prefix 复用与并行 subagent，这两点是 KV cache、prefix caching、调度器等系统优化的关键压力来源
3. **隐私合规的数据公开**：通过 64 token 块 hash 与合成填充，在不暴露任何源代码与 prompt 的前提下公开数据集
4. **确定性与可复现**：固定 seed 决定会话采样、起点与合成内容；报告指标仅覆盖一小时 profiling 窗口，不含 warmup；投机解码按 model、speculator、draft length、thinking mode 的组合固定 acceptance length（采用 SPEED-Bench 实测值），消除合成 token 对 draft 接受率的干扰
5. **产业影响**：AgentX 上线后，合作伙伴已向 vLLM、SGLang、TensorRT-LLM、AMD ATOM、ROCm AITER、NVIDIA Dynamo、LMCache、Mooncake、MoRI UMBP 等推理栈提交 50 个以上面向真实 agent 负载的上游优化 PR

## 3. 是否有公开框架

有，全部开源（Apache-2.0 协议）：

| 资源 | 地址 | 说明 |
|---|---|---|
| AgentX 测试框架 | [https://github.com/SemiAnalysisAI/agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness) | 负载回放器，fork 自 NVIDIA 的 ai-dynamo/aiperf（AIPerf） |
| InferenceX 主仓库 | [https://github.com/SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) | 基准测试配置与公开 GitHub Actions 运行记录 |
| 可视化工具 | [https://github.com/SemiAnalysisAI/InferenceX-app](https://github.com/SemiAnalysisAI/InferenceX-app) | 结果仪表板前端 |
| PyPI 包 | [https://pypi.org/project/agentx-harness](https://pypi.org/project/agentx-harness) | 包名 agentx-harness，命令为 agentx，当前版本 1.0.7 |
| 公开数据集 | semianalysis_cc_traces_weka_062126 | full 与 256k 两个版本，随框架自动下载 |

补充说明：agentx-harness fork 自 NVIDIA 的 AIPerf（ai-dynamo/aiperf），底层保留了 aiperf 命令用于兼容。AIPerf 本身是通用推理基准测试工具，支持 OpenAI 兼容的 chat、completions、embeddings 等端点，支持多种 trace 回放模式（Bailian、Baseten、BurstGPT、SageMaker、Weka），支持 MLflow、OpenTelemetry、Weights and Biases 遥测集成。AgentX 在此基础上增加了 agentic 场景配置（--scenario agentx）与公开数据集支持。

## 4. 运行依赖与步骤

### 4.1 依赖清单

1. Python 3.11 至 3.13
2. 专用虚拟环境：该发行版包含 fork 后的 aiperf 包，不能与上游 aiperf 发行版共存，必须隔离安装
3. OpenAI 兼容推理服务端：需提供 /v1/chat/completions 端点（vLLM、SGLang、TensorRT-LLM、Ollama 等均可）
4. HuggingFace tokenizer：--tokenizer 需指向与服务端模型匹配的 tokenizer；--tokenizer-trust-remote-code 仅对可信仓库使用，不需要自定义 tokenizer 代码时可省略
5. 平台差异：Linux ARM64 上 crick 依赖只有 sdist，安装时需要 C 编译器（Debian/Ubuntu 安装 build-essential，RHEL/CentOS 安装 Development Tools）；Linux x86_64、macOS、Windows 使用预编译 wheel，无需工具链
6. 可选遥测集成：底层 AIPerf 支持 MLflow 上传、OpenTelemetry 流式遥测、Weights and Biases 结果上传，通过对应的 pip extras 安装

### 4.2 运行步骤

第一步，启动 OpenAI 兼容的推理服务器并部署被测模型。

第二步，安装并运行 AgentX：

```bash
pip install 'agentx-harness==1.0.7'

export HUGGINGFACE_MODEL_NAME="org/model"
export CONC_SESSIONS=16

agentx profile \
  --url "http://127.0.0.1:8000" --endpoint /v1/chat/completions \
  --model "$HUGGINGFACE_MODEL_NAME" --tokenizer "$HUGGINGFACE_MODEL_NAME" \
  --tokenizer-trust-remote-code \
  --concurrency "$CONC_SESSIONS" \
  --public-dataset "semianalysis_cc_traces_weka_062126" \
  --output-artifact-dir "output_dir_results"
```

第三步（可选），如路由器需要会话 header，运行前导出以下变量：

```bash
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
```

第一个变量发送 X-Dynamo-Session-ID，subagent 额外携带 X-Dynamo-Parent-Session-ID，保留 fork 出的 agent 血缘信息；第二个变量同时发送 X-Session-ID。

结果写入 output_dir_results 目录。测试结果不会自动发布，需检查输出中的 errors 与 submission_valid 字段，并遵循官方基准测试方法后再比较硬件。

### 4.3 注意事项

1. agentx profile 命令自动选择 AgentX 场景，显式 flag 会覆盖默认值；--config 或 -f 可使用 YAML 文件中的场景与设置
2. 官方 profiling 窗口为一小时，之前还有 primer 请求与 warmup 阶段，完整跑一轮需预留足够时间
3. 官方排行榜数据来自真实硬件上的 GitHub Actions 运行；自行测试时结果的可比性取决于是否遵循相同方法与配置

## 5. 参考链接

- AgentX 方法论与数据集：[https://inferencex.semianalysis.com/agentx](https://inferencex.semianalysis.com/agentx)
- 快速入门：[https://inferencex.semianalysis.com/agentx/quickstart](https://inferencex.semianalysis.com/agentx/quickstart)
- 完整测试方法：[https://inferencex.semianalysis.com/agentx/methodology](https://inferencex.semianalysis.com/agentx/methodology)
- AgentX 框架仓库：[https://github.com/SemiAnalysisAI/agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness)
- AgentX 场景说明文档：[https://github.com/SemiAnalysisAI/agentx-harness/blob/master/docs/tutorials/agentx-mvp.md](https://github.com/SemiAnalysisAI/agentx-harness/blob/master/docs/tutorials/agentx-mvp.md)
- InferenceX 主仓库：[https://github.com/SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
- 性能结果仪表板：[https://inferencex.semianalysis.com/inference](https://inferencex.semianalysis.com/inference)
