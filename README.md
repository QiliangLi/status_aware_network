# status_aware_network

KV 存储压力感知调度（KV Storage Pressure-Aware Cross-Layer Inference Scheduling）的仿真研究仓库。

## 目录

- `discussions/` — 调研对话导出与架构图
- `docs/` — 调研记录、[v1 仿真设计](docs/KV存储压力感知调度仿真设计-20260901.md)、[相关工作对比](docs/五工作的Scheduler-Engine修改与主流默认实现-20260904.md)、[v2 架构与实验](docs/共享KV感知调度仿真v2-架构与实验-20260904.md)、[局限问题分析与改进方案](docs/仿真v2局限问题分析与改进方案-20260904.md)
- [实验结果-E1E4-20260901.md](实验结果-E1E4-20260901.md) — v1 实验结论与图表
- [实验结果-E5E9-20260904.md](实验结果-E5E9-20260904.md) — v2 实验结论与图表（Q1–Q5，含 rate-aware 修正补记）
- [实验结果-E9bE10E11E12-20260904.md](实验结果-E9bE10E11E12-20260904.md) — 改进方案四实验（问题②③④⑤）
- [实验结果-E13E14E15E16-20260904.md](实验结果-E13E14E15E16-20260904.md) — 改进方案 II 四实验（问题⑥⑦⑧⑨，含参数现实性核查）
- [实验结果-E17-20260904.md](实验结果-E17-20260904.md) — 改进方案 III 实验（问题⑩⑪：会话预取证伪+机制重归因、副本可见性闭环）
- [实验结果-E18-20260904.md](实验结果-E18-20260904.md) — 改进方案 IV 实验（问题⑫⑬：预取保护零浪费字节、流体最优背书协同定位）
- `tools/calibrate.py` — 真实系统标定脚手架（LMCache 指标 / prefill 曲线 / RDMA 路径）
- `sim/` — 离散事件仿真器（Python + SimPy），v1 与 v2 两套拓扑
- `tests/` — 不变量单元测试（解析解吻合 / 字节守恒 / 确定性 / 策略单调性 / v2 拓扑与策略）
- `results/` — 实验输出（gitignored）
- `docs/figures/` — 实验图

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install simpy numpy pandas matplotlib pytest
.venv/bin/python -m pytest tests/ -q                  # 不变量测试
.venv/bin/python -m sim.run --exp all --seeds 10      # v1: E1–E4
.venv/bin/python -m sim.run --exp v2  --seeds 8       # v2: E5–E9（共享分布式 KV 存储拓扑）
.venv/bin/python -m sim.run --exp v3  --seeds 8       # v3: E9b/E10/E11/E12（改进方案实验）
.venv/bin/python -m sim.run --exp v4  --seeds 8       # v4: E13–E16（改进方案 II 实验）
.venv/bin/python -m sim.run --exp smoke               # 快速冒烟
```

单实验：`--exp e1a|…|e9|e9b|e10|e11|e12`，并行度 `--procs N`，仿真时长 `--duration`。

## 核心结构

### v1（单/多 backend，E1–E4）

- `sim/storage.py` — 共享 KV 存储：nominal 带宽固定 + 动态背景负载 + processor sharing（双世界：ground truth vs 陈旧/EMA/噪声观测）
- `sim/gpu.py` — prefill 曲线 + 每 worker FCFS 流体队列（背景负载 reserved-capacity）
- `sim/policies/` — P0 AlwaysFetch / P1 StaticCost / P2 Dynamic / P3 Dynamic+Routing / P4 Oracle，及 E2 routing baseline
- `sim/experiments/` — E1a/E1b（压力×余量网格）、E2（热点路由）、E3（信号设计）、E4（burst incast）

### v2（共享分布式 KV 存储拓扑，E5–E9，对应架构图）

- `sim/topology.py` — 计算节点本地缓存（LRU）、元数据目录（副本放置/容量压力）、多存储节点（mem/ssd 双层）+ 共享 fabric
- `sim/quote.py` — 访问成本查询接口：bytes+worker+node+tier → 预计完成时间/压力等级（滞回）/置信度
- `sim/engine2.py` — local / fetch（路径延迟→tier→fabric 三级链路）/ partial（F 比例取回+I/O 计算重叠）/ recompute
- `sim/policies2.py` — 主流默认（alwaysfetch2/default2）、TensorCast（tensorcast2）、AAFLOW+（static2）、CacheFlow（partial_static2）、本方向（joint2/coord2）、Oracle（oracle2）
- `sim/experiments/e5..e9` — Q1 路由 / Q2 副本与路径 / Q3 取回与部分重算 / Q4 复制 / Q5 协同抗振荡

指标与实验定义见设计文档；结果解读见 `实验结果-*.md`。
