# AGENTS.md — 本仓库协作与代码修改规范

本仓库是 KV 存储压力感知调度的仿真研究仓库（v1 单/多 backend + v2 共享分布式 KV 存储拓扑）。
所有 agent / 协作者在修改本仓库前先读本文；与本文冲突的操作以本文为准。

---

## 0. 总工作流（顺序不可颠倒）

```text
调研（含联网检索） → 设计文档 → 实现 + 单元测试 → 冒烟验证场景 → 全量实验
→ 结果文档（嵌图） → 提交推送
```

任何一步没完成，不进入下一步。

## 1. 调研与信息获取

- 涉及**外部系统现状**（NVIDIA Dynamo、vLLM、SGLang、LMCache、Mooncake、MindIE Motor 等调度器 / 推理引擎 / KV 数据层）的方案、默认行为、配置项、API 时，**必须先联网搜索最新官方文档或论文原文**，不得凭训练记忆直接断言版本行为。
- 检索结论写进文档时**逐条标注来源**（官方文档链接 / arXiv 号），无法核实的内容显式标记"待核实"。
- 涉及论文机制对比时，参考 `docs/五工作的Scheduler-Engine修改与主流默认实现-20260904.md` 的既有映射，不要另起炉灶重定义基线语义。

## 2. 文档先行

- 任何非平凡修改（新机制、新实验、架构改动、场景调整）**先在 `docs/` 写设计文档再写代码**：包含问题定义、方案、实验设计（场景/策略/指标）、验收标准。
- 文档命名：`主题-YYYYMMDD.md`，日期为创建日。设计文档放 `docs/`；**实验结果文档放仓库根**（保证图片相对链接在 GitHub 与本地预览均正确，见 commit `04a1161` 的教训）。
- 修改既有结论时，若推翻了先前文档的判断（如 Cascade 边界的修正），在新文档中显式记录"修正了什么、为什么"，不悄悄改口径。

## 3. 实现规范

- **单位约定**：时间秒、字节 GB（十进制）、带宽 GB/s（见 `sim/config.py` 头注）。
- 配置一律 `frozen dataclass`；分段常数背景负载用 `((t0, v), ...)` schedule 表示。
- **信息边界铁律**：信息边界必须匹配研究问题，且对所有被比较的策略一致。存储侧资源内部状态（队列/带宽/背景负载、`hypothetical_*`）只有 Oracle 策略可访问，其余策略只能读可观测视图（`StorageObservable` / `quote.AccessCostQuery`）；GPU 侧当前为真值可见（`GpuPool.remaining_service/drain_est`，v1 既定简化：计算侧与 GPU 同信任域）。若实验对象涉及 GPU 侧预测误差（如 Cascade 式 budget），须对称引入 GPU observable 层，不得单侧开洞。
- 新实验必须注册进 `sim/run.py` 的 CLI choices，并保证可单实验冒烟（`--exp X --seeds 2 --duration 120~200`）。
- 代码注释为中文 docstring，风格与现有模块一致；不引入新依赖除非文档中说明理由。

## 4. 测试与回归

- 新机制必须配**不变量单元测试**（解析解吻合、守恒、确定性、策略单调性、行为断言），放 `tests/`。
- 每次提交前跑全量 pytest（mac 用 `.venv/bin/python -m pytest tests/ -q`，win 用 `.venv/Scripts/python -m pytest tests/ -q`），全绿才提交。纯文档/配置类改动且不碰 Python 时可豁免，但须在提交信息中注明。
- 改动 v1/v2 共享代码（`metrics.py`、`request.py`、`simrun.py`、`config.py`、`storage.py` 等）时，**以 1–2 种子、缩短 duration 跑全 v1 四件套（e1a/e1b/e2/e3/e4）**，核对结论方向不变。不得按"改了哪个文件"挑选实验子集——文件↔实验的映射会过时且易漏（v2 期间的真实风险源是重写的 `metrics.py`，而非未动的 `storage.py`）。

## 5. 实验执行与场景调优

- 全量前先冒烟并**检查区分度**：确认关键策略不在 goodput 天花板（SLO 全满足）、瓶颈落在被研究的资源上。历史上两次返工（本地缓存短路副本问题、GPU 被缺失请求打满导致复制无收益）都是冒烟阶段暴露的——不要跳过。
- 同一实验内做机制对比时**控制变量**：各策略共享相同动作空间 / 引擎能力，只差决策信息源（参照 E7 的设计）。
- 负载生成使用 CRN（同种子同 trace）；多结论取多种子中位数。

## 6. 实验结果文档（硬性要求）

- **图必须嵌入文档**：每个实验的汇总表配对应 `docs/figures/fig_*.png` 的 Markdown 相对链接（`docs/figures/...`，仓库根文档用此前缀）。只有数字没有图的实验结果文档视为未完成。
- 结构参照 `实验结果-E1E4-20260901.md` / `实验结果-E5E9-20260904.md`：开头元信息表 → **总判决（go/no-go + headline 数字）** → 分实验（表 + 图 + 读法）→ 与前序结论衔接 → 局限。
- 面向汇报的文档末尾附**通俗版解读**（一套贯穿的类比，每个实验一句话结论）。

## 7. 提交与推送（及时自动提交）

- **每完成一个逻辑单元（一个文档 / 一个机制 / 一组实验）立即自动提交并推送**，不要攒大批量。用户已授权常态推送，无需逐次询问。
- Commit message 用中文 conventional 风格（`feat(sim):` / `docs:` / `fix:` / `tests:`），正文说明动机与要点，参照 `git log` 既有风格。
- 提交粒度示例：v2 仿真器核心与 E5–E9 实验分属两个 commit（`aa9d479`、`8f95cc5`）。
- `results/` 不入库（gitignored）；`docs/figures/` 入库。**严禁**用少量种子的重跑覆盖已入库的正式图——回归验证后若图被改写，用 `git checkout --` 恢复。
- 推送通道按机器各配各的（git 配置是机器本地的，不随仓库同步）：mac 走 https 远端 + 本机代理 `127.0.0.1:6478`（仅 mac 本地 git config，20260908 更新）；win 走 SSH 直连 `github.com`（20261022 验证）。失败时可用 `git push git@github.com:QiliangLi/status_aware_network.git main` 兜底。

## 8. 命名与放置速查

| 内容 | 位置 | 命名 |
|---|---|---|
| 调研/设计文档 | `docs/` | `主题-YYYYMMDD.md` |
| 实验结果文档 | 仓库根 | `实验结果-EXX-YYYYMMDD.md` |
| 实验图 | `docs/figures/` | `fig_{a..k}_{实验名}.png`（v1 用字母，v2 用 e5–e9） |
| 实验 | `sim/experiments/` | `e{N}_{主题}.py` |
| v1 策略 / v2 策略 | `sim/policies/` / `sim/policies2.py` | v2 策略名以 `2` 结尾（如 `joint2`） |
| prefill 计算时间估算（GLM5+D950PR，独立工具，不入 sim/） | `prefill_time/` | `prefill_time_single.py`（单请求）/ `prefill_time_batch.py`（任意每卡 bs）/ `validate_prefill_time.py`（验证） |

注意：`prefill_time/` 的口径与外部仿真器不同——字节用十进制 GB（1e9，非 1024^3），
注意力路径已按 prefill 语义修正（F1–F11，见脚本文件头），输出是 kernel 级 roofline
下界（真实生产约再乘 2–3）。与外部 sim 对数前先读 `prefill_time/README.md`。

## 9. trace 资产（两套）

| trace | 位置 | 要点 |
|---|---|---|
| Mooncake 官方（FAST'25 公开 trace） | `mooncake_trace/` | conversation / toolagent / synthetic 三文件 + `derived/` 前缀命中派生。**512 token 块**，hash_ids 全局唯一；行数与 sha256 校验登记在 `sim/cq/trace.py` 的 `MOONCAKE_FILES`。E23 等 trace 实验的输入 |
| AgentX（SemiAnalysis cc-traces-weka，Claude Code 编程会话，已转 Mooncake JSONL 格式） | `AgentX及trace转换分析/` | `原始trace样本/` 为 Weka JSON（**64 token 块**、会话内局部编号）；`convert_agentx_to_mooncake.py` 转换（hash_ids 重编号为全局命名空间、subagent 展平为独立会话，`--input-jsonl` 流式模式）。全量已落地（20260919）：`原始trace全量/traces.jsonl`（393 会话打包，1.85 GB）→ `转换结果/agentx_mooncake_full.jsonl`（98827 请求 / 2090 会话，约 2.99 GB），两者均 gitignored，重建步骤与 sha256 见转换说明 §4 |

- 两套 trace 负载特征差异大（AgentX 长上下文多轮、理想前缀命中率中位数 99.1%（全量 98827 请求实测，样本期为 98.8%）；Mooncake 短上下文单轮），**容量/缓存结论不可互推**。
- 接入 AgentX trace 前先读 `AgentX及trace转换分析/AgentX trace转换说明.md` §3.3：块大小 64 ≠ Mooncake 的 512（AIPerf 回放须 `--isl-block-size 64`）、时间戳为会话内相对时间（多会话起点近 0，需加偏移才构成全局到达节奏）、hash_ids 已全局重编号无需再处理。AgentX 背景调研见同目录 `AgentX调研.md`；Mooncake 命中口径见 `docs/MooncakeTrace命中模型首现判定-变更设计-20260915.md`。

## 10. 双机协作（mac + win，经 GitHub 同步）

仓库由两台机器交替开发（常规白天/夜间各一台，交接点可能有长仿真仍在跑）。任务由用户统一下发、用户知晓各机状态，重叠开发时用户主动开分支隔离——**机制不做任何自动协调，唯一要求是在 main 上工作前有意识地取到最新代码**：

- **单元边界同步铁律**：每个 session 开始、每个逻辑单元开始前与结束后 `git fetch origin`；若 `HEAD..origin/main` 非空，先 `git pull --rebase origin main` 再继续。push 被拒同样先 pull --rebase 再推。
- 日常都在 main（trunk-based）；临时分支的创建与合入时机由用户显式驱动，**未获用户命令不得合入 main**。分支上同样遵守同步铁律，并在单元结束处 rebase 到最新 `origin/main`。
- 长仿真启动前依赖代码必须已 commit+push（记录运行基线；运行基于工作区快照，不受 main 后续更新影响）。`results/` 不跨机：谁启动谁收尾；兜底可同种子重跑（CRN 确定性）。
- 换机交接前确保无未推送提交（收不了尾的打 `wip(scope):` 提交推上去）。

**机器清单**（新机器接入时在此登记一行；机制细节与被否决方案见 `docs/双机协作机制-20261022.md`）：

| 代号 | hostname | 平台 | venv 解释器 | 推送通道 |
|---|---|---|---|---|
| mac | （待补） | macOS | `.venv/bin/python` | https + 本机代理 `127.0.0.1:6478` |
| win | DESKTOP-TJL427K | Windows 11 | `.venv/Scripts/python` | SSH 直连（20261022 验证） |
