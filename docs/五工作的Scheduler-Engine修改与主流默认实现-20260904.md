# 五个相关工作的 Scheduler / Inference Engine 修改点，与主流系统默认实现

> 回答两个问题：
> 1. TensorCast、Kairos、Cascade、AAFLOW+、CacheFlow 五个工作，在端到端请求流程中分别对 **Scheduler** 和 **Inference Engine** 修改了哪些部分？
> 2. 主流的 Scheduler（NVIDIA Dynamo、MindIE Motor）和 Inference Engine（vLLM、SGLang，含 LMCache 数据层）在同样的流程中**默认是怎么做的**？

---

## 0. 一页结论（TL;DR）

| 工作 | 改哪层 | 对 Scheduler 的修改 | 对 Inference Engine 的修改 | 一句话定位 |
|---|---|---|---|---|
| **TensorCast** | Global Scheduler（+ 数据面边界重划） | 可编程 request router：计算负载 ↔ KV locality 联合打分选 worker；新增 request+KV migration 动作 | KV 生命周期从 engine 中剥离，KV/权重成为独立统一服务（engine 不再独占管理 KV） | "请求去哪算" |
| **Kairos** | Global Scheduler（PD dispatcher） | 每 100ms 收节点状态；判定是否把 Prefill **deflect** 到 Decode 节点（比较 KV transfer 成本 vs 本地 Prefill 成本 + TBT 约束） | Decode 实例支持 chunked prefill 就地执行（P 能力下沉到 D） | "在哪算 Prefill" |
| **Cascade** | Serving Scheduler + KV Manager（实例内） | latency budget（SLO − 预测剩余服务时间）驱动的**优先级调度**替换 FCFS：Tier-1/Tier-2 队列、dispatch 顺序、batching | per-request KV 恢复策略：restore from tier / prefetch / retain in HBM / **recompute**，与调度联合优化（σ + φ） | "谁先算、KV 怎么取" |
| **AAFLOW+** | Inference Engine / KV Runtime（workflow 层） | 基本不动全局调度器 | KV 成为分布式一等对象，提供 transfer / materialization / fork / composition / eviction 算子；逐请求比较 T_transfer vs T_recompute 选小者 | "取还是重算" |
| **CacheFlow** | Inference Engine（KV 恢复调度器） | 无（请求/批内决策） | KV restoration 拆到 token/layer/GPU 三维并行：哪些 token/层 Fetch、哪些重算、I/O 与 GPU overlap | "恢复时 I/O 和计算怎么混" |

**主流默认实现的共同画像**：整条链路是 **"命中即取（hit → fetch）+ 到达序 FCFS + overlap/load 路由"** ——没有任何一个默认决策点消费"存储侧实时运行状态"（队列深度、有效带宽、拥塞）。这正是本仓库方向（Storage Pressure → Compute 反馈闭环）的切入点。

---

## 1. 参照系：主流系统的端到端请求流程与决策点

先定义一条"参照流程"，后面所有对比都基于它。场景：多实例 LLM serving，开启前缀缓存，KV 可下沉到 CPU/SSD/远端存储（LMCache / Mooncake 类）。

### 1.1 系统分层

```text
┌─────────────────────────────────────────────┐
│  Global Scheduler（Dynamo / MindIE Motor）   │  请求路由、P/D 选择
├─────────────────────────────────────────────┤
│  Inference Engine（vLLM / SGLang）           │  admission、prefix 匹配、
│  内含 Serving Scheduler 与 KV Manager        │  prefill/decode、KV 加载
├─────────────────────────────────────────────┤
│  KV 数据层（LMCache / Mooncake / NIXL…）     │  L1/L2 分层、传输
├─────────────────────────────────────────────┤
│  KV Storage（HBM / CPU DRAM / SSD / Remote） │  物理存放
└─────────────────────────────────────────────┘
```

注：单机小部署里 Global Scheduler 可以缺失（请求直连 engine）；Cascade 讨论的"Serving Scheduler"位于 engine 进程内的调度控制面。分层是逻辑上的，不是进程边界。

### 1.2 端到端流程与决策点（D1–D7）

```text
请求到达
   │
   ▼
[D1] 选实例/worker（PD 场景还包括 P/D pairing）
   │
   ▼
[D2] Engine admission：进 running/waiting 队列，决定加入/执行顺序
   │
   ▼
[D3] Prefix 匹配：本地 HBM 前缀缓存（block hash / radix tree）查询
   │
   ├─ 部分命中在低层/远端 ──►  [D4] KV 恢复：发起 fetch 并等待
   │
   ▼
[D5] Prefill：计算未命中剩余 tokens（可选 chunked prefill）
   │
   ▼
[D6] Decode：continuous batching 迭代生成
   │
   ▼
[D7] KV 生命周期：写回 / 下沉（offload 到 L1/L2）、淘汰（LRU/水位）、
    事件上报（如 Dynamo KV events）
```

### 1.3 各决策点的默认行为

| 决策点 | 主流默认行为 | 默认决策输入 | 缺失的输入 |
|---|---|---|---|
| D1 worker 选择 | Dynamo：tiered prefix overlap + load 联合打分；Motor：round-robin / load-balance | KV 位置事件、GPU 负载 | 存储后端实时压力（queue、有效带宽） |
| D2 admission 顺序 | **FCFS**（vLLM 按到达序；SGLang 默认 fcfs，可选 lpm） | 到达时间（lpm 另看 prefix 长度） | SLO 预算、KV 恢复成本、存储压力 |
| D3/D4 命中处理 | **命中即取**：远端/低层命中 → 通过 KV connector 异步加载，加载期间请求只能等待 | 命中长度、所在 tier | 本次 fetch 的实时代价（排队 + 有效带宽） |
| D5 prefill | 只算未命中部分；chunked prefill 是配置项 | token 数 | fetch 与 recompute 的联合权衡 |
| D6 decode | continuous batching | — | — |
| D7 KV 下沉/淘汰 | 水位/LRU；LMCache 按 L1/L2 分层管理 | 最近使用时间 | 全局热度、存储侧容量压力反馈 |
| （旁路）抢占/重算 | 仅 **显存压力** 触发（vLLM preemption：recompute 或 swap 模式） | HBM 水位 | **不存在**"存储压力驱动的主动重算" |

---

## 2. 主流 Scheduler 的默认实现

### 2.1 NVIDIA Dynamo

Dynamo 是当前最接近"KV 感知全局调度"的主流实现，其 KV-aware router 的默认机制：

1. **状态来源：KV 事件**。worker（vLLM/SGLang 集成）发布 `BlockStored` / `BlockRemoved` 事件；router 据此维护每个 worker 持有哪些 prefix、命中在哪一层（GPU HBM / Host memory / Disk / shared cache）。
2. **路由打分：overlap + load**。对每个候选 worker 计算 tiered prefix overlap 分数与 projected active load（prefill / decode 各自的负载模型），选综合分最高的 worker。
3. **分层权重（credit）**：不同 tier 的命中给不同信用——GPU 命中最高，Host 次之，**Disk 等低层默认 credit 更小**。也就是说 Dynamo 已经内置了"低层 KV 价值更低"的静态认知。
4. **回退模式**：无 KV 事件时退化为 load-aware / round-robin 路由。

**默认不做的**：routing cost model 里**没有**远端存储的有效带宽、队列深度、预测 fetch 完成时间等运行时状态。它知道"KV 在哪"，不知道"现在取它要多久"。此外 Dynamo 通过 NIXL 等传输引擎做 worker 间 KV 搬运，也有基于 KV 传输事件的过载保护，但这些是传输层面的保护，不构成"存储压力 → 调度"的反馈闭环。

### 2.2 MindIE Motor（原 PyMotor，3.1.0 起更名）

定位是 **PD 分离的请求调度框架**：

1. **Coordinator 调度**：把请求分配到 P / D 实例，文档描述的默认策略以 load-balance / round-robin 为主。
2. **`recompute_enabled` 配置**：注意语义——它处理的是**异常恢复**：engine 返回 `stop_reason=recomputed` 后，Coordinator 允许对 token cache 重算并重试。这是被动兜底，**不是**"存储压力高 → scheduler 主动决定重算"。

**默认不做的**：调度输入里没有 KV 存储侧状态；重算不是调度动作而是失败处理。

### 2.3 Scheduler 层的共同盲区

- 决策输入 = **KV 位置 + 计算负载**（Dynamo）或纯负载（Motor）。
- "Cache hit 就值得取"是隐含假设；tier 差异最多体现为**静态**权重（Dynamo credit）。
- 没有任何机制消费存储后端的 queue depth / inflight / 有效带宽 / 拥塞事件。

---

## 3. 主流 Inference Engine 的默认实现

### 3.1 vLLM

1. **调度**：continuous batching；admission 与执行顺序默认 **FCFS**（按到达序），没有 SLO / 预算概念。
2. **前缀缓存（Automatic Prefix Caching）**：按 KV block 哈希匹配，命中即直接复用 blocks，不做 fetch-vs-recompute 权衡。
3. **外部 KV（KV Connector 框架）**：LMCache、Mooncake、NIXL 等 connector 把 KV 扩展到 CPU/SSD/远端。远端命中 → 异步加载 KV，加载完成才进 running，**等待时间不可控也没有替代动作**。
4. **引擎内唯一的"重算"**：preemption。显存不足时抢占请求，模式为 recompute 或 swap——触发条件是 **HBM 水位**，与存储侧压力无关。

### 3.2 SGLang

1. **RadixAttention**：radix tree 前缀缓存，按 LRU 淘汰叶子。
2. **调度策略 `schedule_policy`**：默认 `fcfs`；可选 `lpm`（longest-prefix-match，优先调度前缀匹配更长的请求，cache-aware）等。即便 lpm 也只用 prefix 长度，不用恢复成本。
3. **分层缓存（HiRadixCache 等）**：write-through / write-back 把 KV 下沉到 CPU/磁盘 tier；低层命中 → 拉回 GPU。同样**命中即取**。
4. 有基于负载估计的过载保护 / 限流，但输入是请求速率与队列，不是存储状态。

### 3.3 LMCache（KV 数据层，非 engine，但位于同一流程）

1. **分层**：L1（本机 GPU/CPU）、L2（远端），提供 prefetch / move / pin 等控制原语。
2. **可观测性（重要）**：已公开 L0↔L1、L1↔L2 的 load/store throughput（**从 submit 统计到 complete，包含 adapter queue + 网络 + 磁盘 I/O，即端到端有效吞吐**）、active prefetch jobs、L1/L2 usage、inflight L2 loads/stores 等指标，并有 `/status`、`/metrics` 与 HTTP API。

**含义**：LMCache 已经是一个现成的 **storage pressure sensor**，但默认这些指标只用于监控，不进入任何调度决策——这正是本方向"补最后一条边"的基础设施前提。

### 3.4 Engine 层的共同盲区

- **D3/D4 是开环**：命中 → 取；取多慢都得等。
- **重算只有失败语义**（preemption / Motor 的 recompute_enabled），没有"取不如算"的主动决策。
- engine 与外部存储之间没有 cost quote / backpressure 接口。

---

## 4. 五个工作分别改了什么

以下每个工作按统一结构展开：定位 → Scheduler 修改 → Engine 修改 → 对照 D1–D7 的流程变化 → 边界（不改什么）→ 报告效果。

### 4.1 TensorCast（arXiv:2608.06007）

**定位**：补上分布式推理缺失的"Tensor 管理层"，让 KV/权重成为独立于 engine 的统一服务，路由与放置可联合编程。

**对 Scheduler 的修改**：

1. **可编程 request router**：周期性收集 instance 状态与 tensor/KV placement、replica 状态，对"计算负载均衡 ↔ KV cache locality"做**联合**决策（而非只看 load 或只看 hit）。
2. **新增迁移动作**：不只路由请求，还能 **迁移 request + KV**——把过载实例上的会话 KV 迁到低载实例，使后续请求既获得 locality 又获得空闲算力。
3. **Signal / Plan 抽象**：存储层状态以 runtime Signal 暴露，调度策略以可编程 Plan 描述——这是它作为"跨层抽象"最有价值的部分。

**对 Inference Engine 的修改**：

- 核心是**职责边界重划**而非 engine 内算法：KV 的 placement / migration / materialization 从 engine 中剥离，由 TensorCast 层统一管理。engine 不再独占 KV 生命周期。

**对照 D1–D7**：改 D1（联合路由）、D7（KV placement/migration 成为一等动作），并重划了 engine 与数据面的边界。D2–D6 的 engine 内逻辑基本不动。

**边界**：KV"存在 ≈ 可用"，不做 fetch-vs-recompute 权衡；不感知存储后端运行压力（queue / 带宽）。

**效果**：高并发多轮 workload 下，相对 load-aware + Mooncake 基线 median TTFT 最多降低 **93.2%**，并能更好维持 cache hit rate。

### 4.2 Kairos（arXiv:2607.02043）

**定位**：PD 分离 serving 中，排队 + KV transfer 占 P95 TTFT 的 **77%–98%**；与其搬 KV，不如把计算挪过去。

**对 Scheduler 的修改**：

1. **状态上报环**：节点每 **100ms** 向 dispatcher 汇报状态（decode batch size、KV-cache occupancy、P 侧队列等）。
2. **Prefill deflection 决策**：dispatcher 判断"P 排队长 + KV transfer 贵 + D 有余量 + 不破坏 Decode TBT SLO"时，不再走 `P → transfer KV → D`，直接把该请求的 prefill 指派到 Decode 节点。
3. **决策模型显式使用 KV 状态**：判定 D 节点能否接 prefill 时，输入包含 **decode batch size + 当前 KV-cache occupancy + 新增 prefill chunk size**——KV 占用成为计算调度的输入变量。比较项为 `T_KV-transfer = KVBytes/Bandwidth` vs 本地 prefill 时间。

**对 Inference Engine 的修改**：

- Decode 实例支持 **chunked prefill 就地执行**（P 能力下沉到 D），并与既有 decode batch 共享资源、受 TBT 约束。

**对照 D1–D7**：改 D1 的 PD pairing 部分（deflection 是一种"换计算位置"的路由决策）与 D5（prefill 的执行节点）；新增 dispatcher↔节点的状态环。D4（fetch vs recompute）不动。

**边界**：它感知的是 **P→D 链路的传输代价与实例内 KV occupancy**，不是共享远端 KV 存储的实时压力。

**效果**：bursty workload 下 P95 TTFT 最多降低 **81%**。

### 4.3 Cascade（arXiv:2608.06557）

**定位**：单 serving instance（PD co-located）内，用 SLO latency budget 统一驱动请求调度与多层 KV（HBM → CPU DRAM → NVMe/网络存储）管理。

**对 Scheduler（Serving Scheduler）的修改**：

1. **预算计算**：`Budget = SLO − Lr`，其中 `Lr = ½(L_actual + L_max) × γ`（γ>1）为带 guardband 的保守剩余服务时间预测；TTFT estimator 离线训练（类似 Vidur 的 profiling 校准），输入为 **请求特征（context length）+ prefix KV 命中 profile（跨 tier）+ 当前 serving 负载**。
2. **优先级调度替换 FCFS**：按 budget 紧迫度排序；Tier-1 / Tier-2 两级队列；决定 dispatch 顺序与 batching——让有余量的请求承担等待，把资源留给快超时的请求。
3. **联合优化表述**：`max Goodput over (σ, φ)`，σ = request scheduling（顺序/batching），φ = prefix retrieval policy——调度与 KV 恢复是同一个优化问题的两个变量。

**对 Inference Engine / KV Manager 的修改**：

1. **per-request 恢复策略 φ**：从哪个 tier restore、是否 prefetch、是否 retain in HBM、是否 **recompute prefix**。
2. **显式建模负载下的有效带宽**：tier 间恢复时间用 effective operational bandwidth under load 而非静态峰值带宽估算（"storage 太慢 → 不 fetch → recompute"的机制 Cascade 已具备）。

**对照 D1–D7**：改 D2（预算优先级 + 分层队列）、D4（retrieval policy：tier 选择 / prefetch / retain / recompute）。不动 D1（不做跨实例路由），存储被建模为**实例本地的 memory hierarchy**。

**边界**：request-level、单实例；KV 恢复状态被感知，但共享存储系统作为独立资源池（跨 worker 竞争、全局压力）未建模。这是与本仓库方向的关键分界（见 §6）。

**效果**：三个大模型 production traces 上，相对 vLLM FCFS goodput 最多 **2.4×**，SLO violation 降低 **40%**。

### 4.4 AAFLOW+（arXiv:2607.10987）

**定位**：多 Agent workflow 场景，把 KV 当作分布式一等对象编排；核心贡献是"**cache hit ≠ 一定 fetch**"的决策原语。

**对 Scheduler 的修改**：

- 基本不动全局调度器。编排发生在 workflow / KV runtime 层（算子图级别），不做 worker 选择或 P/D 放置。

**对 Inference Engine / KV Runtime 的修改**：

1. **KV 算子集**：transfer / materialization / fork / composition / eviction，zero-copy 编排。
2. **逐请求二选一**：比较 `T_transfer = KVBytes/Bandwidth + latency` 与 `T_recompute = T_prefill`，选更小者执行。10Gbps 条件下 8 个场景中 **5 个重算更便宜**。

**对照 D1–D7**：只改 D4（fetch vs recompute），把它从"永远 fetch"变成显式 cost model 二选一。

**边界**：带宽是**配置/扫描值**，不是实时反馈——没有在线 storage backpressure；也没有路由、放置等全局动作。

**效果**：证明低带宽下重算常优于读取；提供的是 decision primitive，不是完整控制系统。

### 4.5 CacheFlow（arXiv:2604.25080）

**定位**：KV restoration 不必二选一——把恢复过程本身做成 I/O + 计算的联合调度问题。

**对 Scheduler 的修改**：

- 无。决策都在请求/批内的 KV 恢复调度器里。

**对 Inference Engine 的修改**：

1. **三维并行的恢复调度器**：把 KV restoration 拆到 **token / layer / GPU** 三个维度，逐粒度决定哪些部分 Fetch、哪些部分重算。
2. **I/O 与 GPU overlap**：恢复期间磁盘/网络读与 GPU 计算流水重叠；batched serving 下联合分配两类资源，报告达到约 **88% GPU、78% I/O 利用率**。

**对照 D1–D7**：把 D4 和 D5 合并成一个联合调度问题（部分 fetch + 部分重算 + overlap）。

**边界**：优化"给定 I/O 与计算资源下的最优恢复"，I/O 能力仍视为已知静态资源；不看共享存储动态拥塞，也不做全局路由。

**效果**：相对已有方案 TTFT 降低 **10%–62%**。

---

## 5. 决策点 × 工作对照总表

行 = 端到端流程中的决策点；列 = 默认实现与五个工作。

| 决策点 | 主流默认 | TensorCast | Kairos | Cascade | AAFLOW+ | CacheFlow |
|---|---|---|---|---|---|---|
| D1 worker/PD 选择 | overlap+load（Dynamo）/ RR（Motor） | **联合 load↔locality 路由 + KV migration** | **deflection 到 Decode 节点** | —（单实例） | — | — |
| D2 admission 顺序 | FCFS（SGLang 可选 lpm） | — | — | **latency budget 优先级 + Tier-1/2 队列** | — | — |
| D3/D4 命中 → 取还是算 | 命中即取，等 connector 加载 | —（存在≈可用） | — | **retrieval policy φ：tier/prefetch/retain/recompute** | **T_transfer vs T_recompute 二选一** | —（交给 D4'） |
| D4'+D5 恢复执行 | prefill 只算未命中部分；无混合 | — | chunked prefill 在 D 节点执行 | 与 σ 联合 | 二选一后执行 | **token/layer/GPU 三维 fetch+recompute 混合并 overlap** |
| D7 KV 下沉/放置 | 水位/LRU | **一等 placement/migration/materialization** | — | retain 决策 | eviction 算子 | — |
| 新增状态环 | Dynamo KV 事件（位置）；LMCache 指标（仅监控） | Signal（tensor/replica/worker 状态） | **100ms 节点状态上报（含 KV occupancy）** | TTFT estimator（KV 命中 profile + 负载） | —（静态带宽） | —（静态 I/O 资源） |
| 消费存储**实时**压力？ | ❌ | ❌ | ❌（P→D 链路，非共享存储） | △（恢复成本动态，但 request-level） | ❌ | ❌ |

---

## 6. 对本仓库研究方向的含义

把 §5 最后一行连起来看：

- 五个工作分别替换了默认流程的**某一个**决策点（路由 / 顺序 / 取舍 / 恢复执行），并各自证明了收益；
- 但没有一个工作把"**共享 KV 存储的实时服务能力**"（queue depth、有效带宽、跨 worker 竞争、拥塞趋势）作为一等反馈信号，同时供给 Global Scheduler（D1：routing / admission / P-D 放置）和 Engine（D2/D4：顺序 / fetch-vs-recompute）；
- 主流系统的现状恰好留出了这条边：Dynamo 已有 KV 位置事件与分层 credit，LMCache 已有端到端有效吞吐与 inflight 指标——**传感器和执行器都在，缺的是把压力信号接进 cost function 的闭环**。

因此本方向与五个工作的差异化定位（与调研对话 9/1 晚间的修正一致）：

> Cascade 证明了"KV 恢复成本可进入 request 级调度"；本方向进一步研究**多 worker 共享 KV 存储场景下，存储集群级压力如何作为跨请求、跨节点的反馈信号，驱动 Global Scheduler 与 Engine 的联合调度**（对应仿真中的 E1 动态压力收益边界、E2 存储热点下 routing、E3 信号/接口设计、E4 incast 保护）。

---

## 7. 信息来源与边界说明

- 五个论文的机制描述依据 `discussions/` 下两份调研对话导出（20260902-2040），其中的论文细节（数字、公式、机制）均来自对话中带引用的核查（arXiv 2608.06007 / 2607.02043 / 2608.06557 / 2607.10987 / 2604.25080）。
- Dynamo / MindIE Motor / LMCache 的默认行为依据对话中引用的官方文档（docs.dynamo.nvidia.com、mindie-motor.readthedocs.io、docs.lmcache.ai）及 Ascend/MindIE-PyMotor 仓库配置说明。
- vLLM / SGLang 的默认行为（FCFS、APC、RadixAttention、schedule_policy、preemption、KV connector）为公开文档与源码的通行描述；具体版本细节以对应版本文档为准。
- 本文只描述"默认实现 + 五个工作"；PTStore、Lightstorm、SmartGen 等次要相关工作未展开。
