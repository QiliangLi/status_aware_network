# 通俗介绍Cascade

**User:** Anonymous  
**Created:** 9/1/2026 15:57:18  
**Updated:** 9/2/2026 20:31:03  
**Exported:** 9/2/2026 20:40:22  
**Link:** [https://chatgpt.com/c/6a96855d-e048-83ec-a23d-c0a81534856c](https://chatgpt.com/c/6a96855d-e048-83ec-a23d-c0a81534856c)  

## Prompt:
2026/9/1 15:57:17

用最通俗易懂的方式介绍一下：Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving

## Response:
2026/9/1 15:57:18

> **正在搜索网页**
>
> [Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://arxiv.org/abs/2608.06557)
> [(PDF) Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving)
> [Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://www.catalyzex.com/paper/cascade-exploiting-slo-aware-latency-budget)
> [Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving | Arxiv - DeepPaper](https://jp.ibbac.eu.org/papers/2608.06557v1)
> [[2608.06557] Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://wegamans.net/?_=%2Fabs%2F2608.06557%23BaQ2cyRFYLjjLXYuzGgU%2Fgo%3D)
> [Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving | Cool Papers - Immersive Paper Discovery](https://papers.cool/arxiv/2608.06557)
>
> **已搜索 6 个网站**
>
> 
>
> 思考了两秒

这篇 **Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving** 的核心思想其实非常漂亮：**不要简单地按照“谁先来谁先服务”，而是计算每个请求还有多少“时间余量”，把有限的 GPU、KV Cache 搬运资源优先给最紧急的请求。** ([arXiv](https://arxiv.org/abs/2608.06557?utm_source=chatgpt.com))

我先不用论文语言，用一个生活例子解释。

---

# 1. 背景：为什么现在 LLM 推理调度很难？

假设你经营一家医院：

来了三个病人：

| 病人 | 要求 | 当前状态 |
|-|-|-|
| A | 10分钟内治疗 | 很严重，已经等了8分钟 |
| B | 10分钟内治疗 | 刚来，还能等 |
| C | 10分钟内治疗 | 手术复杂，需要8分钟 |

如果医院只按照排队：

> 谁先来谁先治（FCFS）

可能出现：

```
A 先来
↓
等待
↓
B 做简单治疗
↓
C 做复杂治疗
↓
A 超时死亡
```

问题：

**虽然所有人 SLO 都是10分钟，但是他们真正能等待的时间完全不同。**

LLM 推理也是一样。

一个请求：

```
用户请求
 ↓
排队时间
 ↓
Prefill
 ↓
KV Cache加载
 ↓
Decode生成
 ↓
返回
```

最终要求：

```
TTFT < 5s
TPOT < 100ms
```

但是：

请求之间差异巨大：

- prompt 长度不同
- output 长度不同
- KV Cache命中情况不同
- 当前GPU负载不同

所以：

> 同样5秒SLO，不代表所有请求还有同样的等待空间。

Cascade抓住了这个问题。([arXiv](https://arxiv.org/abs/2608.06557?utm_source=chatgpt.com))

---

# 2. 核心概念：Latency Budget（延迟预算）

这是整篇论文最大的贡献。

它提出：

不要看：

> 这个请求什么时候deadline？

而应该看：

> 它距离deadline还有多少可浪费时间？

公式：

```
Latency Budget

=
SLO要求时间
-
预计剩余执行时间
```

举例：

## 请求A

SLO：

```
TTFT < 10s
```

预测：

```
prefill需要3s
```

那么：

```
Budget = 10 - 3 = 7s
```

意思：

> A还可以接受7秒额外开销。

---

## 请求B

SLO：

```
TTFT < 10s
```

但是：

prompt很长

预测：

```
prefill需要9s
```

那么：

```
Budget = 10 - 9 = 1s
```

意思：

> B已经非常危险，只能接受1秒额外等待。

---

于是调度器知道：

```
B:
剩余1秒
🔥 非常紧急

A:
剩余7秒
可以等等
```

所以：

```
优先调度 B
```

而不是：

```
谁先来谁先服务
```

---

# 3. 它解决的第一个问题：请求调度

现在很多系统类似：

```
Request Queue

A
B
C
D

↓

FCFS

↓

GPU执行
```

问题：

Head-of-Line Blocking（队头阻塞）

例如：

```
队列：

长prompt请求
短prompt请求
短prompt请求

↓

第一个请求卡住

↓

后面的全部等待
```

结果：

短请求也超时。

---

Cascade：

给每个请求计算budget：

```
Queue

A  budget=8s
B  budget=0.5s
C  budget=5s
D  budget=1s

        ↓

按照budget排序

B
D
C
A

```

优先救快超时的。

---

# 4. 但是论文真正有意思的是第二点：

## 它把 KV Cache 管理也纳入 budget

这一点和普通调度论文不一样。

以前：

调度系统：

```
决定谁先跑
```

KV Cache系统：

```
决定KV放哪里
```

两个系统割裂。

例如：

一个请求：

```
KV Cache在NVMe
```

现在需要：

```
NVMe
 ↓
CPU DRAM
 ↓
GPU HBM
```

搬运。

但是搬运需要时间：

```
500ms
```

问题：

这个500ms应该给谁？

以前：

不知道。

Cascade：

看budget。

---

比如：

请求A:

```
budget=8s

KV在NVMe

搬运需要1s

没问题
```

请求B:

```
budget=0.5s

KV在NVMe

搬运需要1s

必炸
```

于是：

```
A:
可以提前prefetch

B:
直接重新计算或者放弃深层KV恢复
```

---

所以 Cascade实际上统一优化：

```
             Latency Budget

                 |
        --------------------
        |                  |
   Request Scheduling   KV Cache Management

        |                  |
 谁先执行？          KV放哪里？
```

这是论文最核心的地方。([arXiv](https://arxiv.org/abs/2608.06557?utm_source=chatgpt.com))

---

# 5. 它里面的 Cascade 为什么叫 Cascade？

简单理解：

KV Cache有多级：

```
GPU HBM
  |
  |
CPU DRAM
  |
  |
NVMe
```

像瀑布(cascade)一样。

距离GPU越远：

越便宜：

```
容量 ↑
速度 ↓
```

例如：

|位置|容量|访问|
|-|-|-|
|HBM|小|最快|
|DRAM|中|慢|
|NVMe|大|最慢|

Cascade根据budget决定：

这个KV应该：

```
留在HBM?
↓
放DRAM?
↓
放NVMe?
↓
重新计算?
```

---

# 6. 一个完整流程

假设来了一个Agent请求：

```
用户:
帮我修改代码
```

历史上下文：

```
100k token
```

发现：

部分KV Cache存在。

Cascade:

---

## Step1

预测执行时间：

考虑：

- prompt长度
- KV命中率
- GPU当前负载

得到：

```
预计执行:
4s
```

---

## Step2

看SLO：

```
要求:
8s
```

计算：

```
Budget=4s
```

---

## Step3

决定：

### 调度

当前：

```
A budget=0.2s
B budget=4s
```

先跑A。

---

### KV管理

发现：

B的KV：

```
NVMe
```

搬运：

```
3s
```

可以接受：

```
4s budget > 3s
```

开始prefetch。

---

# 7. 它和 vLLM 的区别是什么？

vLLM：

核心思想：

```
最大化GPU利用率

+
PagedAttention管理KV
```

但是默认：

```
请求来了
↓
排队
↓
batch
↓
执行
```

类似：

FCFS。

Cascade认为：

问题不是GPU利用率，而是：

> 如何在SLO约束下最大化有效吞吐(goodput)。

Goodput不是：

```
生成多少token
```

而是：

```
满足SLO的token数量
```

例如：

1000 tokens/s

但是50%请求超时：

实际goodput：

500。

Cascade实验：

相比vLLM FCFS：

- goodput最高提升2.4倍
- SLO violation降低40% ([arXiv](https://arxiv.org/abs/2608.06557?utm_source=chatgpt.com))

---

# 8. 这篇论文和你之前关注的 KVCache 感知调度有什么关系？

我觉得它和你之前问的：

> “存储侧状态感知是否可以反过来影响计算侧调度？”

高度相关。

因为传统：

```
计算调度
    |
    ↓
决定资源分配

KV Cache只是被动提供
```

Cascade：

反过来：

```
KV状态
(在哪里、有多少、恢复成本)

        ↓

影响计算调度
        ↓

决定谁先跑
```

也就是说：

它实现了一种：

**KV-aware compute scheduling**

或者：

**storage state → compute scheduling feedback loop**

这和你之前讨论的方向非常接近。

---

# 9. 但是它还有几个明显局限

我觉得值得注意：

## (1) Budget预测依赖准确性

它需要预测：

```
prefill时间
KV恢复时间
系统负载影响
```

如果预测错：

budget错。

例如：

认为：

```
budget=5s
```

实际：

```
budget=1s
```

可能导致SLO失败。

---

## (2) 它主要解决单集群内部

它考虑：

```
HBM
DRAM
NVMe
```

但是没有深入：

```
跨GPU
跨节点
跨地域
```

所以对于：

跨域KVCache

还有空间。

---

## (3) 它优化的是请求级别

但是Agent场景可能更复杂：

例如：

```
Agent step1
 |
tool call
 |
step2
 |
tool call
 |
step3
```

未来可能需要：

```
session-level budget
workflow-level budget
```

而不是单request。

---

# 一句话总结

**Cascade的核心思想：**

> LLM推理调度不应该只看“请求来了多久”，而应该实时计算“这个请求距离SLO还剩多少时间余量”，然后用这个统一指标同时指导请求调度和KV Cache搬运，让有时间余量的请求承担等待成本，把资源留给真正快超时的请求。

它实际上提出了一种：

**SLO-aware + KV-aware + compute-memory co-optimization 的LLM serving架构。** ([arXiv](https://arxiv.org/abs/2608.06557?utm_source=chatgpt.com))

结合你最近关注的 **KVCache跨域访问、存储状态感知调度、PD解耦**，这篇论文值得重点看，因为它提供了一个很清晰的范式：**“KV状态不是纯存储问题，而应该成为计算调度决策输入。”**

---

**Sources:**

- [Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://arxiv.org/abs/2608.06557?utm_source=chatgpt.com)

## Prompt:
2026/9/1 16:01:22

当前是怎么预测这个budget的？

## Response:
2026/9/1 16:01:24

> **正在搜索网页**
>
> [Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://arxiv.org/abs/2608.06557)
> [Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving | Arxiv - DeepPaper](https://jp.ibbac.eu.org/papers/2608.06557v1)
> [(PDF) Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving)
> [Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://www.catalyzex.com/paper/cascade-exploiting-slo-aware-latency-budget)
> [[2608.06557] Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://wegamans.net/?_=%2Fabs%2F2608.06557%23BaQ2cyRFYLjjLXYuzGgU%2Fgo%3D)
>
> **已搜索 5 个网站**
>
> 
>
> 思考了两秒

这个问题其实是 **Cascade 最关键的技术细节之一**。我前面讲“预测执行时间”比较粗略，论文里并不是简单估算：

> prompt token 数 × 一个固定 token/s

而是做了一个 **TTFT latency estimator（首 token 延迟预测器）**，综合考虑请求特征、KV Cache 状态和当前 GPU 负载。([ResearchGate](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving?utm_source=chatgpt.com))

它的计算逻辑可以拆成：

```
Latency Budget
=
SLO目标时间
-
预测剩余服务时间
```

而预测剩余服务时间主要就是：

```
预测TTFT latency
```

因为 Cascade 主要针对 TTFT SLO 调度。([ResearchGate](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving?utm_source=chatgpt.com))

---

## 1. 输入哪些信息？

论文考虑三个维度：

```
                 TTFT预测

        -------------------------
        |           |           |
   请求特征     KV状态      系统负载
```

([ResearchGate](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving?utm_source=chatgpt.com))

### ① 请求特征

最核心：

### 输入长度

比如：

```
request A:

prompt:
2K tokens

request B:

prompt:
100K tokens
```

显然：

B 的 prefill 时间更长。

---

### 输出长度

为什么输出长度影响 TTFT？

严格来说：

TTFT主要是prefill。

但是输出长度影响：

- batch调度
- decode占用GPU
- 当前实例压力

所以也作为负载预测因素。

---

### ② KV Cache状态

这个是 Cascade 和普通scheduler最大的区别。

例如：

请求：

```
prompt:

system message
+
历史聊天100K tokens
+
当前问题
```

如果：

```
100K prefix KV已经存在
```

那么：

需要：

```
重新计算:
10K tokens
```

而不是：

```
110K tokens
```

所以：

预测时间：

```
prefill cost

=
未命中KV token计算
+
KV恢复时间
```

---

KV所在位置也不同：

```
GPU HBM

↓

CPU DRAM

↓

NVMe
```

恢复成本：

```
HBM:
0

DRAM:
几百us~ms

NVMe:
ms级甚至更高
```

所以：

同样100K KV：

状态不同：

```
A:
KV全部在HBM

预测:
1s

B:
KV在NVMe

预测:
5s
```

budget完全不同。

---

### ③ 当前GPU负载

这是非常重要的一点。

LLM不是单请求执行。

例如：

GPU现在：

```
decode batch:

request1
request2
request3
...
```

来了新prefill：

它不能直接执行。

因为：

decode阶段对延迟敏感。

所以：

Cascade考虑：

```
active decode requests
```

也就是：

当前GPU有多少decode压力。([ResearchGate](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving?utm_source=chatgpt.com))

---

# 2. 它具体怎么预测？

论文里面用了一个离线训练的预测模型。

大致流程：

---

## Offline阶段

先profiling：

例如：

不同模型：

```
Qwen-72B

prompt长度:
1K
2K
4K
8K
16K
...

KV hit:
0%
25%
50%
75%
100%

GPU load:
0%
50%
90%
```

跑实验：

得到：

```
输入状态

↓

真实TTFT
```

然后训练预测器。

论文提到使用类似 Vidur 的执行profiling/simulation框架进行校准。([ResearchGate](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving?utm_source=chatgpt.com))

---

## Runtime阶段

请求来了：

例如：

```
Request A:

prompt:
50K token

KV hit:
80%

KV位置:
DRAM

当前decode:
20个request
```

输入预测器：

得到：

```
L_actual = 2.5s
```

---

但是论文发现：

如果只预测当前负载：

容易过于乐观。

例如：

现在GPU：

```
50%负载
```

预测：

```
2.5s
```

但是过一会：

```
90%负载
```

实际：

```
4s
```

所以它加了一个保守机制。

---

# 3. Guardband：防止过度乐观

论文用了：

```
Lr = 1/2(L_actual + L_max) × γ
```

([ResearchGate](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving?utm_source=chatgpt.com))

其中：

## L_actual

当前负载预测：

```
当前GPU状态
+
当前KV状态
```

比如：

```
2.5s
```

---

## L_max

最坏情况：

假设：

```
GPU接近满载

但是还有一个slot可以执行
```

预测：

```
5s
```

---

平均：

```
(2.5 + 5)/2

=
3.75s
```

再乘：

```
γ > 1
```

例如：

```
γ=1.2
```

得到：

```
Lr=4.5s
```

这就是最终认为：

```
这个请求至少需要4.5秒
```

然后：

```
Budget=SLO-Lr
```

---

# 4. 举一个完整例子

假设：

SLO：

```
TTFT < 10s
```

来了两个请求。

---

## Request A

状态：

```
prompt:
20K

KV:
HBM hit 90%

GPU:
轻载
```

预测：

```
Lr=2s
```

那么：

```
Budget:

10-2

=8s
```

意思：

它可以承担：

```
8秒等待
```

---

## Request B

状态：

```
prompt:
100K

KV:
NVMe

GPU:
decode很忙
```

预测：

```
Lr=8s
```

那么：

```
Budget:

10-8

=2s
```

所以：

B更紧急。

调度：

```
B先执行

A等待
```

---

# 5. 这里有一个非常值得讨论的问题

其实你之前一直关注：

> KV Cache状态能不能反过来影响计算调度？

Cascade就是一个很典型例子。

传统：

```
Scheduler:

只看request

↓

决定GPU执行顺序

KV Manager:

被动提供cache
```

Cascade：

```
KV状态

(hit率 / tier位置 / transfer cost)

        ↓

预测执行时间

        ↓

计算budget

        ↓

影响GPU调度
```

也就是：

**storage state → latency prediction → compute scheduling**

这正是你之前说的“存储侧状态反向影响计算侧”的方向。([ResearchGate](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving?utm_source=chatgpt.com))

---

不过我觉得这里还有一个很值得质疑的地方：

它预测的是：

> “这个request单次服务还需要多久”

但是对于 Agent 场景：

```
Agent step1
 ↓
tool call
 ↓
step2
 ↓
tool call
 ↓
step3
```

真正的问题可能不是：

```
request latency budget
```

而是：

```
session/workflow budget
```

例如：

一个coding agent已经跑了20轮，它下一轮是否值得继续保留KV、是否应该压缩上下文、是否应该迁移cache，这个Cascade还没有解决。

这和你之前讨论的 **Agent Harness + KVCache + Memory Policy** 其实是一个更上层的问题。

---

**Sources:**

- [(PDF) Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving](https://www.researchgate.net/publication/412057864_Cascade_Exploiting_SLO-Aware_latency_budget_for_fair_and_high_goodput_LLM_inference_serving?utm_source=chatgpt.com)

## Prompt:
2026/9/1 16:03:33

> ### ② KV Cache状态
> 
> 这个是 Cascade 和普通scheduler最大的区别。
> 
> 例如：
> 
> 请求：
> 
>     prompt:
>     
>     system message
>     +
>     历史聊天100K tokens
>     +
>     当前问题
> 
> 如果：
> 
>     100K prefix KV已经存在
> 
> 那么：
> 
> 需要：
> 
>     重新计算:
>     10K tokens
> 
> 而不是：
> 
>     110K tokens
> 
> 所以：
> 
> 预测时间：
> 
>     prefill cost
>     
>     =
>     未命中KV token计算
>     +
>     KV恢复时间
> 
> * * *
> 
> KV所在位置也不同：
> 
>     GPU HBM
>     
>     ↓
>     
>     CPU DRAM
>     
>     ↓
>     
>     NVMe
> 
> 恢复成本：
> 
>     HBM:
>     0
>     
>     
>     DRAM:
>     几百us~ms
>     
>     
>     NVMe:
>     ms级甚至更高
> 
> 所以：
> 
> 同样100K KV：
> 
> 状态不同：
> 
>     A:
>     KV全部在HBM
>     
>     预测:
>     1s
>     
>     
>     B:
>     KV在NVMe
>     
>     预测:
>     5s
> 
> budget完全不同。

但是这个对于KVCache状态的考虑没有考虑存储的状态啊，例如存储的队列深度、可用带宽、等等。

## Response:
2026/9/1 16:03:34

你这个质疑是对的，而且实际上指出了 **Cascade 这类工作目前 KV-aware scheduling 的一个边界**。

我前面说“考虑 KV Cache 状态”容易让人误解成：

> 它感知了完整的 KV Cache 存储系统状态。

实际上不是。

更准确地说：

**Cascade 感知的是 KV Cache 的“静态位置/可恢复成本（placement & retrieval cost）”，而不是 KV Cache 存储系统的“动态运行状态（storage system state）”。**

这两个层次差别很大。

---

## 1. Cascade 考虑的是哪类 KV 状态？

它关注类似：

```
这个请求需要的 KV 在哪里？
```

例如：

```
Request A:

需要恢复100GB KV

当前：

GPU HBM
 ↓
直接使用

或者：

CPU DRAM
 ↓
copy到GPU

或者：

NVMe
 ↓
读取
 ↓
copy GPU
```

所以它估算：

```
KV recovery latency

≈

KV size / bandwidth(position)
```

例如：

|位置|估计成本|
|-|-|
|GPU HBM|接近0|
|CPU DRAM|GB/s级搬运|
|NVMe|更慢|

它回答的是：

> 如果我要拿这个 KV，需要付出多少成本？

---

但是你说的：

> 存储队列深度、可用带宽、竞争情况呢？

这是另外一个问题：

> 现在拿这个 KV，到底什么时候能拿回来？

---

## 2. 举一个例子

假设：

两个请求：

### Request A

KV：

```
100GB
```

位置：

```
NVMe
```

系统状态：

```
NVMe bandwidth:
20GB/s
```

预测：

```
100/20 = 5s
```

---

但是突然：

同时来了10个KV恢复任务。

现在：

```
NVMe queue depth = 10
```

实际：

```
effective bandwidth:

20GB/s / 10

=2GB/s
```

那么：

```
100/2=50s
```

完全不同。

---

所以真正应该是：

```
KV recovery latency

=

f(
 KV size,
 KV location,
 storage bandwidth,
 storage queue depth,
 competing IO,
 network congestion,
 cache tier pressure
)
```

而不是简单：

```
KV location → latency
```

---

## 3. 为什么 Cascade 没做到这个？

因为它主要解决的问题是：

```
单个 LLM serving instance 内：

GPU scheduling
+
KV placement
```

它的优化目标：

```
maximize SLO satisfied requests
```

所以它默认：

KV tier：

```
GPU
CPU
Local SSD
```

之间的访问成本相对稳定。

但是你关注的是：

```
KVCache as a distributed storage system
```

也就是：

```
              Scheduler

                  ↑
                  |
        storage state feedback

                  ↑

KV Cache Storage System

  bandwidth
  queue depth
  congestion
  eviction
  remote fetch
```

这个层次更接近：

**storage-aware inference scheduling**

---

## 4. 如果扩展 Cascade，应该怎么做？

我觉得你之前提出的方向其实就是自然扩展。

现在 Cascade：

```
Request

 ↓

Latency estimator

 ↓

Budget

 ↓

Scheduler
```

其中：

Latency estimator：

```
request feature
+
KV location
+
GPU load
```

---

你希望增加：

```
+
Storage state
```

变成：

```
Latency estimator

=

f(
 request tokens,

 KV hit ratio,

 KV placement,

 KV size,

 GPU load,

 Storage queue depth,

 Storage bandwidth,

 Network RTT,

 Network congestion

)
```

然后：

```
Budget

=

SLO

-

Predicted latency
```

---

例如：

请求 A：

```
KV:
remote memory cluster

size:
50GB

storage:

bandwidth:
100GB/s

queue:
0

RTT:
10us
```

预测：

```
0.5s
```

budget:

```
10-0.5=9.5s
```

---

请求 B：

一样：

```
KV:
remote memory cluster

size:
50GB
```

但是：

```
queue depth=100

bandwidth collapse

RTT:
10ms
```

预测：

```
8s
```

budget:

```
2s
```

那么 scheduler：

```
先执行B
```

否则B可能超时。

---

## 5. 其实这和你之前问的那个问题高度一致

你之前问：

> “正常来说，调度是根据计算侧的信息决定存储侧怎么调度，但是这里想考虑存储侧状态感知是否反过来影响计算侧调度。”

Cascade 是：

```
KV placement
        ↓
compute scheduling
```

已经迈了一步。

但是你想的是：

```
KV storage runtime state
        ↓
latency prediction
        ↓
compute scheduling
```

更进一步。

区别：

| |Cascade|你的方向|
|-|-|-|
|KV是否存在|✓|✓|
|KV在哪里|✓|✓|
|恢复成本估计|✓|✓|
|存储带宽|弱/固定|✓|
|IO队列深度|×|✓|
|跨节点网络状态|×|✓|
|远端KV服务负载|×|✓|
|存储拥塞反馈|×|✓|

---

## 6. 我觉得这里有一个很有潜力的研究问题

甚至可以总结成：

> **Current LLM serving systems treat KVCache as a passive memory hierarchy, while future systems should expose KVCache storage states as first-class scheduling signals.**

也就是：

现在：

```
GPU scheduler
      |
      |
   KV manager
```

未来：

```
             Scheduler

          ↑        ↑

   GPU state   KV storage state

          ↑        ↑

     Compute    Storage
```

这其实和数据库里面：

- buffer manager影响query optimizer
- storage congestion影响query scheduling

是类似思想。

所以你的质疑不是挑刺，反而指出了 Cascade 没覆盖的下一层问题。你之前关注的“跨域 KVCache / DPU / 存储侧状态反馈调度”，其实就在这个方向上。

## Prompt:
2026/9/2 16:12:47

在当前这个工作中，存储是不是仅仅被考虑为一个单点？

## Response:
2026/9/2 16:12:51

是的，如果从**系统架构抽象层次**来看，你可以基本这么理解：

> **Cascade 中的 KV Cache 存储被抽象成一个本地、单机内的分层存储（GPU HBM / CPU DRAM / 本地 SSD），而不是一个独立的分布式 KV Cache 存储系统。**

所以它考虑的是 **memory hierarchy**，不是 **storage system**。

这个区别非常关键。

---

## 1. Cascade里的“存储”模型是什么？

它隐含的架构大概是：

```
              LLM Serving Instance

                    GPU
                    |
                    |
             GPU HBM KV Cache
                    |
                    |
              CPU Memory
                    |
                    |
              Local SSD

Scheduler
                 |
                 |
          KV placement decision
```

也就是说：

KV Cache属于这个 inference worker 自己管理的资源。

例如：

请求来了：

```
Request A

需要历史KV:
50GB

当前:
GPU HBM没有

CPU DRAM有

或者SSD有
```

Cascade考虑：

```
把KV搬回来需要多久？
```

然后影响调度。

---

## 2. 它没有考虑什么？

你说的：

> 存储队列深度、可用带宽

其实属于另一类系统。

例如：

假设：

```
Inference Worker 1
        |
        |
        Remote KV Store
        |
        |
+----------------+
| KV Servers     |
|                |
| SSD/NVMe       |
| RDMA network   |
+----------------+
```

这里的问题：

### 存储节点负载

例如：

```
KV Server:

queue depth:
200

active reads:
1000

bandwidth:
90% consumed
```

那么：

同样100GB KV：

情况A：

```
读取:
1s
```

情况B：

```
读取:
30s
```

---

Cascade没有这种反馈。

---

## 3. 更准确地说，它假设 KV access latency 是可预测的静态函数

大概类似：

```
KV latency = f(size, tier)
```

例如：

```
HBM:
   0.x ms

DRAM:
   x ms

SSD:
   xx ms
```

但是现实分布式存储：

应该是：

```
KV latency = f(

size,

placement,

storage load,

network congestion,

queue depth,

IO scheduling,

remote server state

)
```

复杂很多。

---

## 4. 为什么它会这样设计？

其实不是论文作者没有意识到，而是研究边界不同。

Cascade关注的问题：

> 在一个 LLM serving worker 内，当 KV Cache 分布在不同介质时，如何利用 SLO budget 做调度？

它解决：

```
GPU compute
+
local KV hierarchy
```

优化目标：

```
maximize goodput
```

---

而你关注的问题：

> 如果 KV Cache 本身成为一个共享的分布式资源，存储状态如何反馈影响计算调度？

更接近：

```
Disaggregated KV Cache Serving
```

或者：

```
KV Cache as a Service
```

架构：

```
                 Scheduler

              ↑          ↑
              |          |
        GPU state     KV storage state

|
              |

        Remote KV System

    +----------------+
    | KV nodes       |
    | Memory         |
    | SSD            |
    | RDMA           |
    +----------------+
```

---

## 5. 两者最大的区别

可以这样比较：

| |Cascade|你关注的方向|
|-|-|-|
|KV位置|GPU/CPU/SSD|远程KV节点|
|存储范围|单机|分布式|
|访问模型|load KV|network fetch|
|带宽|固定profile|动态变化|
|队列|不考虑|核心因素|
|网络|基本忽略|核心因素|
|调度反馈|KV tier|KV system state|

---

## 6. 其实这个问题和现在 LMCache / Mooncake / KVCache offloading 的方向更接近

现在很多 KV Cache 系统正在走：

```
GPU
 |
 |
Local memory
 |
 |
Remote memory pool
 |
 |
Distributed KV storage
```

例如：

- LMCache
- Mooncake
- vLLM KV connector
- CacheGen 类工作

这时候真正的问题就是：

> scheduler 是否应该知道 KV backend 当前状态？

比如：

一个请求：

```
Prefix KV hit = yes
```

传统scheduler：

```
很好，cache hit
优先调度
```

但是实际：

```
KV server:
90% busy

network:
congested

RDMA queue:
full
```

那么：

这个 cache hit 可能反而比重新计算更慢。

---

所以你的理解可以总结为：

> **Cascade把KV Cache看成“计算节点内部的一种分层内存”，而不是“独立的存储系统”。因此它能做KV-aware scheduling，但还没有做到storage-state-aware scheduling。**

我觉得这也是一个很自然的后续研究方向：

**Storage-aware KV Cache Scheduling：让KV存储系统的实时状态反向参与LLM inference scheduling。**

这和你之前一直讨论的“存储侧状态感知影响计算侧调度”基本是同一个问题。

## Prompt:
2026/9/2 16:26:07

如果换成共享的分布式存储，整个架构会有什么变化，需要新考虑哪些问题？

## Response:
2026/9/2 16:26:09

这个问题其实正好切到当前 **LLM 推理系统下一阶段的核心挑战**。

如果把 Cascade 里的：

```
GPU HBM / CPU DRAM / Local SSD
```

替换成：

```
共享分布式 KV Cache Storage
```

整个系统的抽象会发生根本变化。

最大的变化是：

> **KV Cache 不再是某个推理实例的“本地内存”，而变成了多个推理实例共享访问的“分布式状态”。**

这会让调度问题从：

> 如何安排 GPU 执行顺序？

变成：

> 如何联合调度计算资源、网络资源和 KV 存储资源？

---

# 1. 架构变化

## 原来的 Cascade 架构

可以抽象成：

```
              Scheduler

                  |
                  |

              GPU Worker

                  |
        --------------------
        |        |          |
       HBM     DRAM       SSD
```

特点：

- KV 和计算绑定
- 存储是本地的
- latency 主要由 tier 决定

所以：

```
KV location
     |
     v
latency estimate
     |
     v
scheduler
```

---

# 2. 换成共享 KV Store 后

架构变成：

```
                         Scheduler
                             |
             +---------------+---------------+
             |                               |
             v                               v

       GPU Workers                 Distributed KV Store

   +---------------+             +----------------+
   | GPU1          |             | KV Server 1    |
   | GPU2          |<----------->| KV Server 2    |
   | GPU3          |    RDMA     | KV Server 3    |
   +---------------+             +----------------+

```

此时：

KV访问路径：

```
GPU Worker

    |
    |
 RDMA / TCP

    |
    |

KV Server

    |
    |

Memory / SSD
```

原来的：

```
HBM → DRAM → SSD
```

变成：

```
GPU
 |
 |
Network
 |
 |
KV Service
 |
 +-- Memory
 |
 +-- SSD
```

---

# 3. 第一个变化：Latency模型完全变了

Cascade：

```
latency =
KV tier cost
+
compute cost
```

共享存储：

应该变成：

```
latency =
compute cost

+
network latency

+
KV lookup latency

+
storage queue latency

+
data transfer latency
```

也就是：

$$
T =
T_{compute}
+
T_{network}
+
T_{KV lookup}
+
T_{storage}
$$

---

举例：

请求 A：

需要：

```
50GB KV
```

情况1：

```
KV server idle

RDMA:
100GB/s

读取:
0.5s
```

---

情况2：

同一个 KV server：

```
100个请求同时读取

queue depth=100

bandwidth被打满
```

变成：

```
5s
```

---

所以：

以前：

```
KV在哪里？
```

现在：

必须知道：

```
KV在哪里？

谁正在访问？

存储节点忙不忙？

网络堵不堵？
```

---

# 4. 第二个变化：Scheduler必须感知Storage State

这是最核心变化。

原来：

Scheduler：

```
request
 |
 |
GPU状态

```

现在：

应该：

```
                 Scheduler

↑          ↑          ↑

       GPU状态    网络状态    KV状态

↓          ↓          ↓

       GPU pool   Network    KV Store

```

例如：

来了两个请求：

---

Request A:

```
KV hit

位置:
KV Server 1
```

但是：

```
KV Server 1:

queue depth=100
```

---

Request B:

```
KV hit

位置:
KV Server 2
```

但是：

```
KV Server 2:

idle
```

传统：

A、B一样。

新的scheduler：

应该：

```
B优先
```

---

# 5. 第三个变化：KV placement成为动态问题

单机：

```
KV放哪里？
```

答案：

```
HBM?
DRAM?
SSD?
```

---

分布式：

变成：

```
KV应该放哪个节点？
```

考虑：

## 数据局部性

例如：

```
GPU cluster A

距离:

KV node1:
10us

KV node2:
100us
```

---

## 热点

一个热门system prompt：

例如：

```
ChatGPT system prompt
```

大量请求访问。

如果：

只有：

```
KV Server 1
```

会：

```
hotspot
```

所以需要：

```
replication
```

---

## 负载均衡

例如：

```
KV node1:

90%

KV node2:

20%
```

新的prefix：

应该放：

```
node2
```

---

这已经接近传统分布式存储问题。

---

# 6. 第四个变化：一致性问题出现

单机：

没有。

共享KV：

必须考虑：

## KV更新

例如：

Agent场景：

```
conversation history

step1

↓

step2

↓

step3
```

多个worker可能访问：

```
session KV
```

问题：

worker A：

```
append token
```

worker B：

```
read old KV
```

需要：

- version
- epoch
- lease
- invalidation

这些和你之前做MetaEC里的问题非常类似。

---

# 7. 第五个变化：故障处理

单机：

GPU挂：

请求失败。

共享KV：

KV Store可能：

```
KV node crash
```

怎么办？

例如：

```
KV Server 3

保存:
10000 sessions
```

挂了。

需要：

- replication
- erasure coding
- recovery
- metadata service

---

# 8. 第六个变化：Prefetch策略变复杂

Cascade：

可以：

```
SSD → GPU
```

提前搬。

共享：

变成：

```
KV server
   |
network
   |
GPU
```

问题：

什么时候搬？

搬多少？

搬到哪里？

例如Agent：

下一步可能：

```
调用工具A
```

那么：

未来需要：

```
tool A相关上下文KV
```

应该提前拉。

这和：

```
cache prefetch
```

很像。

---

# 9. 最终调度器会变成什么？

未来可能不是：

```
GPU scheduler
```

而是：

```
Joint Compute-KV-Network Scheduler
```

决策：

一次请求：

选择：

```
1. 哪个GPU执行？

2. 哪个KV副本读取？

3. 是否迁移KV？

4. 是否重新计算？

5. 是否prefetch？

6. 是否复制热点KV？
```

---

类似：

```
                  Request

                     |
                     v

          +--------------------+
          | Global Scheduler   |
          +--------------------+

             /        |        \

            /         |         \

        GPU Pool   KV Pool   Network

```

---

# 10. 和你之前关注的方向对应

你之前一直问：

> 存储侧状态是否可以反向影响计算调度？

如果是共享KV，这个问题会更加明显。

因为：

现在：

```
KV状态
 |
 |
影响latency
 |
 |
影响GPU调度
```

而且：

KV状态包括：

```
- cache hit
- KV位置
- replica数量
- storage queue depth
- bandwidth
- network RTT
- congestion
- eviction pressure
```

---

所以从研究角度看，可以形成一个非常清晰的问题：

**Cascade解决的是：**

> SLO-aware scheduling under hierarchical local KV cache

而进一步的问题：

> SLO-aware scheduling under distributed KV Cache service

核心创新点可能是：

**把KV Store从“被动缓存层”提升为“参与推理调度的一等公民”。**

这其实和你之前讨论的 **跨域 KVCache、DPU KV gateway、存储状态反馈计算调度** 是同一条演进路线。你关注的方向比 Cascade 再往前推进了一层。

## Prompt:
2026/9/2 20:30:49

应该不需要考虑第四和第五个变化。还有一个问题是，调度侧和推理引擎侧需要感知存储的哪些独有的状态呢？注意，“独有”的意思是只能存储侧去给，计算侧一方面很需要，另一方面又很难感知的到。

## Response:
2026/9/2 20:30:50

你这个限定非常关键。前面我提到的一致性、故障恢复确实偏向“分布式存储系统设计本身”，如果目标是研究 **共享 KV Cache 场景下，存储侧如何反向影响推理调度**，那么应该聚焦：

> **哪些 KV 存储运行时状态，对于推理决策非常重要，但计算侧天然不可见，只能由存储系统暴露。**

换句话说，不是“存储有什么状态”，而是：

**哪些状态形成了 compute scheduler 的信息盲区。**

我认为主要有下面几类。

---

# 1. KV访问排队状态（Queueing State）——最核心

这是你前面提到的 queue depth，我认为是第一优先级。

## 为什么计算侧不知道？

推理引擎看到：

```
Request A:
KV hit = true
KV size = 20GB
```

于是认为：

```
KV load latency ≈ 20GB / bandwidth
```

但是实际：

```
KV Storage Node:

Request queue:
----------------
req1 100GB
req2 50GB
req3 80GB
req4 20GB(A)
----------------

queue depth=4
```

A 实际需要：

```
等待前面的IO完成
+
自己的读取时间
```

---

所以存储侧需要提供：

### (1) 当前请求排队长度

例如：

```
kv_read_queue_depth = 30
```

### (2) 当前等待时间估计

比 queue depth 更直接：

```
estimated_wait_time = 800ms
```

### (3) service rate

例如：

```
current_read_bandwidth = 40GB/s
```

---

这个信息直接影响：

```
Budget计算:

SLO - compute_time - KV_wait_time - KV_transfer_time
```

---

# 2. KV实际可用带宽（Effective Bandwidth）

注意：

不是：

```
NIC bandwidth
SSD bandwidth
memory bandwidth
```

这些是静态能力。

推理真正关心：

> 我现在发起这个KV读取，实际能拿到多少吞吐？

---

例如：

存储节点：

硬件：

```
RDMA:
400Gbps

SSD:
20GB/s
```

但是当前：

```
10个prefetch
20个decode恢复
大量scan
```

实际：

```
effective bandwidth = 5GB/s
```

---

计算侧无法知道：

因为：

- 它不知道其他请求
- 它不知道存储内部调度
- 它不知道IO竞争

所以需要存储暴露：

```
available_read_bw
```

或者：

```
expected_transfer_time(size)
```

---

# 3. KV所在节点的访问路径状态（Path State）

共享KV以后：

一个KV可能有多个路径：

例如：

```
GPU A

 |
 | RDMA

KV node 1

GPU A

 |
 | RDMA switch

KV node 2
```

两个节点：

理论：

一样。

实际：

```
Node1:
RTT 10us

Node2:
RTT 200us
```

为什么？

可能：

- 网络拥塞
- switch queue
- oversubscription

计算侧通常不知道。

---

存储侧可以提供：

```
KV object:

nearest replica:
node1

estimated access latency:
50us
```

---

这个其实类似：

数据库：

optimizer不知道disk内部状态。

需要storage feedback。

---

# 4. KV Cache热度 / 竞争程度（Popularity & Contention）

这个非常有价值。

计算侧知道：

```
这个prefix cache hit
```

但是不知道：

```
这个prefix是不是热点
```

---

举例：

一个system prompt：

```
10000个请求共享
```

KV Store知道：

```
prefix_id=xxx

access QPS:
50000/s
```

计算侧不知道。

为什么重要？

因为调度：

可能应该：

```
优先把热点KV放近GPU
```

或者：

```
增加replica
```

---

存储侧可以暴露：

```
KV popularity score
KV contention level
```

---

# 5. KV驻留状态（Residency Stability）

这个我觉得特别容易被忽略。

计算侧看到：

```
KV hit=true
```

但是不知道：

这个KV：

```
是不是马上要被evict？
```

---

例如：

请求A：

```
KV hit
```

但是：

```
memory pressure=95%

eviction probability high
```

请求B：

```
KV hit
```

但是：

```
hot cache
```

两个完全不同。

存储侧知道：

- eviction pressure
- remaining capacity
- replacement priority

---

这个会影响：

prefetch策略：

```
现在拉KV

还是重新计算？
```

---

# 6. KV布局状态（Fragmentation / Placement Quality）

这个更偏存储系统。

例如：

逻辑：

```
KV size = 100GB
```

但是：

物理：

```
block1
block2
block3
...
```

分散。

导致：

```
random IO
```

或者：

```
RDMA request数量增加
```

---

计算侧看到：

```
100GB KV
```

无法知道：

实际上：

```
需要10000次IO
```

---

存储侧可以提供：

```
read amplification
fragmentation ratio
```

---

# 7. 数据压缩/编码状态（如果存储做优化）

未来共享KV存储可能：

- compression
- quantization
- dedup
- EC

例如：

KV：

逻辑：

```
100GB
```

实际：

```
compressed size=30GB
```

但是：

恢复：

需要：

```
decompression cost
```

---

计算侧不知道：

```
effective transfer cost
```

---

存储侧可以提供：

```
compressed_size
decode_cost
```

---

# 总结一下：真正“存储独有”的状态

如果只挑最有价值的，我会排序：

|优先级|状态|为什么计算侧不知道|影响|
|-|-|-|-|
|⭐⭐⭐⭐⭐|KV read queue depth / waiting time|存储内部排队|直接影响latency budget|
|⭐⭐⭐⭐⭐|effective available bandwidth|动态竞争|影响KV恢复时间|
|⭐⭐⭐⭐|KV访问路径RTT/拥塞|网络+存储内部|影响placement选择|
|⭐⭐⭐⭐|KV popularity/contention|全局访问模式|影响replication/prefetch|
|⭐⭐⭐⭐|eviction pressure/residency|存储缓存策略|影响cache可信度|
|⭐⭐⭐|fragmentation/layout|物理布局|影响IO效率|
|⭐⭐⭐|compression/encoding状态|存储优化细节|影响恢复成本|

---

# 如果重新设计 Cascade，Latency Predictor 应该变成：

现在：

```
Latency =
f(
 request tokens,
 KV location,
 GPU load
)
```

共享KV时代：

应该：

```
Latency =
f(

request feature,

KV metadata,

GPU state,

+
KV queue state,
KV bandwidth,
KV RTT,
KV contention,
KV eviction pressure

)
```

然后：

```
Budget

↓

Scheduler决定：

1. 选哪个GPU

2. 先恢复哪个KV

3. 是否prefetch

4. 是否等待cache

5. 是否重新计算
```

---

所以我觉得你的问题实际上把研究问题进一步收敛了：

**不是“把KV放到分布式存储”这么简单，而是：**

> 在共享KV Cache服务中，如何设计一套 storage-to-inference feedback interface，让推理系统获得传统调度无法观测的存储状态，从而做更准确的SLO-aware scheduling。

其中最核心的 interface，我认为就是：

**KV access latency prediction API**：

存储侧不要暴露几十个底层指标，而直接告诉 scheduler：

```
For KV object X:

current fetch latency prediction = 800ms
confidence = 90%
```

这可能比暴露 queue depth/bandwidth 更适合作为系统设计。你之前关注的“存储状态反向影响计算调度”，核心接口其实就在这里。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)