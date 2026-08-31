# status_aware_network

KV 存储压力感知调度（KV Storage Pressure-Aware Cross-Layer Inference Scheduling）的仿真研究仓库。

## 目录

- `docs/` — 调研记录、[仿真设计文档](docs/KV存储压力感知调度仿真设计-20260901.md)、实验结果
- `sim/` — 离散事件仿真器（Python + SimPy）
- `tests/` — 不变量单元测试（解析解吻合 / 字节守恒 / 确定性 / 策略单调性）
- `results/` — 实验输出（gitignored）
- `docs/figures/` — 实验图（Fig A–D）

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install simpy numpy pandas matplotlib pytest
.venv/bin/python -m pytest tests/ -q                  # 不变量测试
.venv/bin/python -m sim.run --exp all --seeds 10      # 全部实验 + 出图
.venv/bin/python -m sim.run --exp smoke               # 快速冒烟
```

单实验：`--exp e1a|e1b|e2|e3|e4`，并行度 `--procs N`，仿真时长 `--duration`。

## 核心结构

- `sim/storage.py` — 共享 KV 存储：nominal 带宽固定 + 动态背景负载 + processor sharing（双世界：ground truth vs 陈旧/EMA/噪声观测）
- `sim/gpu.py` — prefill 曲线 + 每 worker FCFS 流体队列（背景负载 reserved-capacity）
- `sim/policies/` — P0 AlwaysFetch / P1 StaticCost / P2 Dynamic / P3 Dynamic+Routing / P4 Oracle，及 E2 routing baseline
- `sim/experiments/` — E1a/E1b（压力×余量网格）、E2（热点路由）、E3（信号设计）、E4（burst incast）

指标与实验定义见设计文档；结果解读见 `docs/实验结果-*.md`。
