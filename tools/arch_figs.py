"""架构分析文档配图（docs/仿真架构分析-20260921.md）。

生成 5 张 PNG 到 docs/figures/fig_arch_*.png：
  A 总览（四族仿真 + 共享支撑层）
  B 信息边界双世界（ground truth vs observable vs Oracle 通道）
  C v2 共享分布式拓扑与决策/取回链路
  D cq 引擎（批层状态机 + 同刻五步闭包 + FCFS 存储）
  E 实验执行与产物流水线

纯 matplotlib 手绘（box/arrow），不依赖仿真代码；只反映代码现状。
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "figures")

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Songti SC",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配色：入口灰 / M1紫 / v1蓝 / v2绿 / cq橙 / 共享黄 / 真值红 / 观测绿
C_CLI = "#ececec"
C_M1 = "#ece4f3"
C_V1 = "#dce9f5"
C_V2 = "#e3f0dc"
C_CQ = "#fbe8d3"
C_SHARE = "#faf3d8"
C_TRUE = "#f6dcdc"
C_OBS = "#dff0e4"
C_POL = "#e8e8f8"

EDGE = "#555555"


def box(ax, x, y, w, h, text, fc, fs=8.5, ec=EDGE, bold=False, align="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                       fc=fc, ec=ec, lw=1.0, mutation_aspect=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2 if align == "center" else x + 0.008, y + h / 2, text,
            ha=align if align != "center" else "center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)
    return (x, y, w, h)


def lane(ax, x, y, w, h, title, fc, fs=10):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
                       fc=fc, ec=EDGE, lw=1.2, alpha=0.45)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top",
            fontsize=fs, fontweight="bold")


def arrow(ax, xy, xytext, color=EDGE, style="-", lw=1.2, rad=0.0, mut=12):
    a = FancyArrowPatch(xy, xytext, arrowstyle="-|>", mutation_scale=mut,
                        color=color, lw=lw, linestyle=style,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- 图 A 总览
def fig_a():
    fig, ax = new_ax(14.5, 9.8)
    ax.text(0.5, 0.985, "status_aware_network 仿真架构总览：四族仿真 + 共享支撑层（基线 commit db0b1c6）",
            ha="center", va="top", fontsize=13, fontweight="bold")

    # CLI 入口条
    box(ax, 0.04, 0.905, 0.92, 0.052,
        "sim/run.py — CLI 入口：--exp all|v2|v3|v4|all2|cq|smoke*|单实验（e1a…e25, g1, g2）；--seeds/--procs/--duration/--stage/--manifest",
        C_CLI, fs=9.5, bold=True)

    # 四泳道
    lx = [0.025, 0.245, 0.475, 0.745]
    lw_ = [0.205, 0.215, 0.255, 0.235]
    lane(ax, lx[0], 0.175, lw_[0], 0.705, "M0 / M1 支线（G 系列）", C_M1)
    lane(ax, lx[1], 0.175, lw_[1], 0.705, "v1 族：请求级流体仿真（E1–E4）", C_V1)
    lane(ax, lx[2], 0.175, lw_[2], 0.705, "v2 族：共享分布式 KV 拓扑（E5–E18）", C_V2)
    lane(ax, lx[3], 0.175, lw_[3], 0.705, "cq 族：统一队列批+错峰（E19–E25）", C_CQ)

    def stack(x, w, ytop, items, fc, gap=0.012, fs=8.0):
        y = ytop
        for txt, hh in items:
            box(ax, x, y - hh, w, hh, txt, fc, fs=fs)
            y -= hh + gap
        return y

    stack(lx[0] + 0.008, lw_[0] - 0.016, 0.845, [
        ("m0.py — 解析层\n取回/重算边界 b*=B/(ΔC−ℓ)", 0.085),
        ("m0sim.py — 事件微仿真（simpy）\n单 GPU/单 I/O 串行，解析↔仿真对拍", 0.085),
        ("m1.py — 迭代级引擎\nchunked prefill + decode 优先\nHBM 保守准入 + 命中即取", 0.105),
        ("policies_g.py — 路由/排序策略", 0.05),
        ("实验 G0/G1/G2（象限扫描、混合负载）", 0.05),
    ], "#f6f1fb")

    stack(lx[1] + 0.008, lw_[1] - 0.016, 0.845, [
        ("workload.py — CRN 负载\nPoisson/类别/burst/会话/热度漂移", 0.085),
        ("storage.py — SharedKVStorage\nPS 流体 + 分段背景负载 + t_base 门控\n（双世界：配 StorageObservable）", 0.10),
        ("gpu.py — PrefillCurve 查表 + GpuPool\nFCFS 流体队列（rate=1−bg）", 0.075),
        ("scheduler.py + worker.py\nroute→engine_decide→执行", 0.06),
        ("policies/ — P0 AlwaysFetch / P1 Static\nP2 Dynamic / P3 Route / P4 Oracle\n+ rr / load / kv 基线", 0.09),
        ("实验 E1a/E1b/E2/E3/E4", 0.045),
    ], "#f0f6fc")

    stack(lx[2] + 0.008, lw_[2] - 0.016, 0.845, [
        ("topology.py — World：LocalKVCache(LRU)\nMetadataDirectory(副本+容量)\n多节点 mem/ssd + 共享 fabric", 0.10),
        ("quote.py — AccessCostQuery\n压力档位(滞回) + 完成时间估计", 0.06),
        ("gpu_obs.py — GpuObservable（对称观测）", 0.045),
        ("engine2.py — local / fetch(三级链路)\npartial(F 比例+overlap) / recompute", 0.075),
        ("policies2.py — 17 策略\n主流映射 + joint2/coord2 + oracle/先知", 0.06),
        ("prefetch.py + _ctrl_loop\n预取/回写/保护 + 复制/迁移/淘汰", 0.07),
        ("实验 E5–E9, E9b–E18（v2/v3/v4）", 0.045),
    ], "#f2f9ef")

    stack(lx[3] + 0.008, lw_[3] - 0.016, 0.845, [
        ("types/config — 三域对象 + Fraction 配置\n（输入 | 运行真值 | 可观测快照）", 0.075),
        ("profile.py — synthetic-v1 画像\nc=t_launch+aN/η(N)+bA、T0 解析式", 0.07),
        ("storage.py — StorageSim\nB(t) 阶梯 + FCFS 按申请上限 + due 提升", 0.07),
        ("engine.py — CqEngine 自研事件循环\n五步同刻闭包 + 批层状态机", 0.065),
        ("observable.py — 报价采样/滞后/噪声\nObservableSnapshot（类型化信息边界）", 0.07),
        ("policies + search + exact\n简单排序策略 / MPC 双预算 / B&B 金标", 0.065),
        ("trace.py — Mooncake 三 trace\n首现命中派生 + 窗口缩放", 0.06),
        ("实验 E19–E25（含时间序列 timeseries.py）", 0.045),
    ], "#fdf4ea")

    # 共享支撑层
    box(ax, 0.025, 0.028, 0.955, 0.115,
        "共享支撑层：config.py 单位约定(秒/GB十进制/GBps)与 frozen dataclass ｜ request.py 请求对象 ｜ metrics.py Collector(v1/v2 通用)\n"
        "simrun.py 装配(v1 run_once / v2 run_once_v2) ｜ experiments/ 装配框架(common/v2common/cq_common/g_common) + run_pool 多进程\n"
        "tests/ 不变量单测(解析吻合/守恒/确定性/单调性) ｜ results/(gitignored) ｜ docs/figures/ 入库",
        C_SHARE, fs=8.2)

    # 依赖箭头：CLI → 各泳道；泳道 → 共享层
    for x, w in zip(lx, lw_):
        arrow(ax, (x + w / 2, 0.905), (x + w / 2, 0.882), rad=0.0)
    for x, w in zip(lx, lw_):
        arrow(ax, (x + w / 2, 0.175), (x + w / 2, 0.146), rad=0.0, style="--",
              color="#888888")

    ax.text(0.5, 0.16, "四族均落在同一支撑层上；v1/v2 共享 SimPy+流体资源内核，cq 为独立自研内核（Fraction/事件循环）",
            ha="center", fontsize=8, color="#666666")
    save(fig, "fig_arch_overview.png")


# ------------------------------------------------------- 图 B 信息边界双世界
def fig_b():
    fig, ax = new_ax(14.0, 8.6)
    ax.text(0.5, 0.985, "信息边界双世界：ground truth ｜ 可观测层 ｜ 策略（三族统一的方法论）",
            ha="center", va="top", fontsize=13, fontweight="bold")

    # 左：ground truth
    box(ax, 0.025, 0.20, 0.30, 0.70, "", C_TRUE)
    ax.text(0.175, 0.875, "ground truth（物理真值世界）", ha="center", fontsize=11, fontweight="bold")
    box(ax, 0.04, 0.735, 0.27, 0.115,
        "v1/v2：SharedKVStorage\n内部队列/背景负载/带宽\ngpu.GpuPool.queue\nhypothetical_* 精确预测", "white", fs=8.2)
    box(ax, 0.04, 0.595, 0.27, 0.115,
        "v2 拓扑真值：World.resources\n(mem/ssd/fabric 流体)\nMetadataDirectory 副本集\nLocalKVCache 实际内容", "white", fs=8.2)
    box(ax, 0.04, 0.455, 0.27, 0.115,
        "cq：WorldState/BatchRuntime\nFlowState(剩余字节/速率)\nB(t) 真带宽、c_layers 真值\n（engine.py 物理内核私有）", "white", fs=8.2)
    box(ax, 0.04, 0.315, 0.27, 0.115,
        "Oracle/先知专属通道：\nhypothetical_fetch_time /\nhypothetical / _hyp_fetch\nclairvoyant future(cls,t,H)", "white", fs=8.2, ec="#b55a5a")

    # 中：observable
    box(ax, 0.40, 0.20, 0.30, 0.70, "", C_OBS)
    ax.text(0.55, 0.875, "可观测层（普通策略的唯一接口）", ha="center", fontsize=11, fontweight="bold")
    box(ax, 0.415, 0.735, 0.27, 0.115,
        "StorageObservable\ninterval 采样 / EMA 平滑\n乘性 lognormal 噪声\nsignal=quote|bw|util|queue", "white", fs=8.2)
    box(ax, 0.415, 0.595, 0.27, 0.115,
        "quote.AccessCostQuery\nest = path_lat+tier+fabric\n压力档位 NORMAL/WARM/\nHOT/CRITICAL（滞回防抖）", "white", fs=8.2)
    box(ax, 0.415, 0.455, 0.27, 0.115,
        "GpuObservable（问题⑤）\nGPU 排队 drain_est 的\n陈旧/EMA/带噪视图\n默认 0/0=真值（v1 简化）", "white", fs=8.2)
    box(ax, 0.415, 0.315, 0.27, 0.115,
        "cq ObservableSnapshot\n报价 Quote(采样/滞后/噪声)\n公共请求/流账本/worker 事件\nest_bw/c_hat 预测辅助", "white", fs=8.2)

    # 右：策略
    box(ax, 0.755, 0.20, 0.225, 0.70, "", C_POL)
    ax.text(0.8675, 0.875, "策略", ha="center", fontsize=11, fontweight="bold")
    box(ax, 0.77, 0.695, 0.195, 0.115, "普通策略（只读右栏）\nP2/P3、static2dyn、\njoint2/coord2、cascade2、\ncq_fcfs…cq_mpc", "white", fs=8.2)
    box(ax, 0.77, 0.545, 0.195, 0.09, "Oracle 家族（读左栏）\nP4 / oracle2 / clairvoyant2 /\nclairfluid2 / cq_oracle", "white", fs=8.2, ec="#b55a5a")

    # 边界虚线
    ax.plot([0.355, 0.355], [0.18, 0.92], ls="--", lw=2.2, color="#b55a5a")
    ax.plot([0.725, 0.725], [0.18, 0.92], ls="--", lw=2.2, color="#4a7d59")
    ax.text(0.355, 0.945, "信息边界（存储侧）", ha="center", fontsize=9.5, color="#b55a5a", fontweight="bold")
    ax.text(0.725, 0.945, "接口边界", ha="center", fontsize=9.5, color="#4a7d59", fontweight="bold")

    # 箭头
    arrow(ax, (0.415, 0.79), (0.31, 0.79), color="#4a7d59")   # 采样
    ax.text(0.3625, 0.815, "采样", ha="center", fontsize=7.5, color="#4a7d59")
    arrow(ax, (0.77, 0.75), (0.685, 0.75), color="#4a7d59")
    arrow(ax, (0.77, 0.585), (0.685, 0.37), color="#b55a5a", rad=-0.25)   # oracle 通道
    ax.text(0.72, 0.47, "特权\n通道", ha="center", fontsize=7.5, color="#b55a5a")
    arrow(ax, (0.55, 0.315), (0.55, 0.20), color="#888888", style="--")

    # 底部铁律
    box(ax, 0.025, 0.035, 0.955, 0.135,
        "信息边界铁律（AGENTS.md §3）：边界必须匹配研究问题，且对所有被比较的策略一致。\n"
        "· v1/v2：以对象引用边界实现 —— 普通策略只拿 WorkerView.obs / V2Ctx.quote，Oracle 直接拿资源对象；\n"
        "· cq：以类型系统实现 —— ObservableSnapshot 不携带 world/engine 引用，物理真值只在 include_oracle=True 的诊断快照出现；\n"
        "· GPU 侧默认真值可见（v1 既定简化：计算侧与 GPU 同信任域）；涉及 GPU 预测误差的实验（E12）用 GpuObservable 对称开洞，不单侧开洞。",
        "#fdf6e3", fs=8.4)
    save(fig, "fig_arch_dual_world.png")


# ------------------------------------------------------- 图 C v2 拓扑
def fig_c():
    fig, ax = new_ax(14.0, 9.2)
    ax.text(0.5, 0.985, "v2 共享分布式 KV 拓扑与一次 fetch 决策的完整链路（sim/topology.py + engine2.py + policies2.py）",
            ha="center", va="top", fontsize=12.5, fontweight="bold")

    # workers
    for i in range(4):
        x = 0.035 + i * 0.24
        box(ax, x, 0.78, 0.20, 0.135,
            f"worker {i}\nGpuPool(FCFS 流体, rate=1−bg)\nLocalKVCache(LRU, 12GB)", "#eef4fb", fs=8)

    # 决策层
    box(ax, 0.035, 0.575, 0.44, 0.115,
        "policies2.V2Ctx.decide(req) → Decision\n(worker, action∈{local,fetch,partial,recompute,prefill},\n node, tier, F, overlap) — 遍历 worker×副本×动作×F 网格", C_POL, fs=8.2)
    box(ax, 0.52, 0.575, 0.44, 0.115,
        "quote.AccessCostQuery.estimate(bytes,w,node,tier)\n= path_lat[w][node] + tier 段 + fabric 段\n（各段读 StorageObservable：陈旧/EMA/带噪）", C_OBS, fs=8.2)

    # directory
    box(ax, 0.035, 0.44, 0.925, 0.095,
        "MetadataDirectory：class → {(node, tier)} 副本集合；节点占用 held[n] → 容量压力；remove() 内建孤儿检查（每类至少保留一副本）",
        "#f7f2df", fs=8.5)

    # nodes + fabric
    ny = 0.20
    for i, (mm, ss) in enumerate([(60, 25), (60, 25), (40, 15)]):
        x = 0.045 + i * 0.30
        box(ax, x, ny, 0.26, 0.175, f"存储节点 n{i}（E5–E9 典型）\nmem SharedKVStorage {mm} GB/s\nssd SharedKVStorage {ss} GB/s\ncap 512 GB",
            "#eef7ec", fs=8.2)
    box(ax, 0.035, 0.075, 0.925, 0.075,
        "共享 fabric：SharedKVStorage 120 GB/s（所有取回/预取/复制流量共用的第三个流体资源）", "#eef7ec", fs=8.5)

    # arrows: workers <-> decision; decision -> directory; quote -> nodes
    arrow(ax, (0.135, 0.78), (0.135, 0.69), rad=0.0)
    ax.text(0.145, 0.735, "gpu_wait/drain_est", fontsize=7.2, color="#666666")
    arrow(ax, (0.25, 0.575), (0.25, 0.535), rad=0.0)
    arrow(ax, (0.74, 0.575), (0.74, 0.535), rad=0.0, style="--", color="#4a7d59")
    arrow(ax, (0.52, 0.6325), (0.47, 0.6325), rad=0.0, color="#4a7d59")

    # fetch 链路标注（红）
    arrow(ax, (0.83, 0.78), (0.83, 0.535), color="#c0392b", lw=1.6)
    arrow(ax, (0.80, 0.44), (0.70, 0.375), color="#c0392b", lw=1.6, rad=0.15)
    arrow(ax, (0.63, 0.20), (0.63, 0.15), color="#c0392b", lw=1.6)
    box(ax, 0.565, 0.24, 0.40, 0.10,
        "fetch 三级顺序链路（engine2._chain）：\n① path_lat 延迟 → ② tier 资源 submit → ③ fabric submit\npartial：链路与 GPU 剩余计算并行，完成取 max（overlap）", "#fdeceb", fs=8.2, ec="#c0392b")

    # 闭环控制器标注
    box(ax, 0.035, 0.005, 0.44, 0.062,
        "_ctrl_loop（CtrlConfig）：滞回判 HOT → 复制/跨层降级/冷迁移/容量淘汰", "#f2f2f2", fs=7.8)
    box(ax, 0.52, 0.005, 0.44, 0.062,
        "Prefetcher：会话预取(gated/predictive/session) + 重算回写 + 活跃保护", "#f2f2f2", fs=7.8)

    # local 命中标注
    box(ax, 0.045, 0.245, 0.30, 0.115,
        "local 动作：本地缓存持有该类\n→ 仅 suffix prefill；完成后\ninsert() 写回本 worker 缓存", "#f7f9fc", fs=8.0)
    save(fig, "fig_arch_v2_topo.png")


# ------------------------------------------------------- 图 D cq 引擎
def fig_d():
    fig, ax = new_ax(14.0, 9.0)
    ax.text(0.5, 0.985, "cq 引擎：批层状态机 + 同刻五步闭包 + FCFS 聚合存储（sim/cq/engine.py、storage.py）",
            ha="center", va="top", fontsize=12.5, fontweight="bold")

    # 左：请求/批/worker 状态机
    box(ax, 0.03, 0.70, 0.42, 0.235, "", "#fdf4ea")
    ax.text(0.24, 0.905, "批的生命周期（BatchRuntime，L 层）", ha="center", fontsize=10, fontweight="bold")
    box(ax, 0.05, 0.845, 0.10, 0.045, "FUTURE", "white", fs=8)
    box(ax, 0.175, 0.845, 0.10, 0.045, "QUEUED", "white", fs=8)
    box(ax, 0.30, 0.845, 0.10, 0.045, "ACTIVE", "white", fs=8)
    box(ax, 0.395, 0.845, 0.048, 0.045, "DONE", "white", fs=7.5)
    arrow(ax, (0.15, 0.8675), (0.175, 0.8675))
    arrow(ax, (0.275, 0.8675), (0.30, 0.8675))
    arrow(ax, (0.40, 0.8675), (0.395, 0.8675))
    ax.text(0.2225, 0.888, "DISPATCH 领取", ha="center", fontsize=7.2, color="#666666")

    # 层循环
    yb = 0.725
    box(ax, 0.05, yb, 0.115, 0.075, "wait_read(ℓ)\n读取流 V_ℓ GB\n(q=V_ℓ/c, due=Z_ℓ)", "white", fs=7.6)
    box(ax, 0.195, yb, 0.105, 0.075, "wait_prev(ℓ)\nC=max(R_ℓ, Z_ℓ₋₁)", "white", fs=7.6)
    box(ax, 0.325, yb, 0.105, 0.075, "computing(ℓ)\nZ_ℓ=C_ℓ+c_β", "white", fs=7.6)
    arrow(ax, (0.165, yb + 0.0375), (0.195, yb + 0.0375))
    arrow(ax, (0.30, yb + 0.0375), (0.325, yb + 0.0375))
    arrow(ax, (0.3775, yb), (0.1075, yb - 0.035), rad=0.30, color="#c0392b")
    ax.text(0.24, yb - 0.055, "ℓ < L−1：提交 ℓ+1 层读取（C_{β,ℓ+1} 时刻，q=V_{ℓ+1}/c_β，due=Z_ℓ）；ℓ = L−1：末层 Z 即全批 F",
            ha="center", fontsize=7.4, color="#c0392b")
    ax.text(0.24, 0.712, "", ha="center", fontsize=7)

    # worker 三态
    box(ax, 0.05, 0.585, 0.115, 0.055, "IDLE", "#eef4fb", fs=8.5)
    box(ax, 0.195, 0.585, 0.105, 0.055, "STALL(等读)", "#fdeceb", fs=8.5)
    box(ax, 0.325, 0.585, 0.105, 0.055, "COMPUTE", "#eef7ec", fs=8.5)
    arrow(ax, (0.165, 0.6125), (0.195, 0.6125))
    arrow(ax, (0.30, 0.6125), (0.325, 0.6125))
    arrow(ax, (0.3775, 0.6125), (0.1075, 0.6125), rad=-0.3, color="#888888", style="--")
    ax.text(0.24, 0.558, "WorkerState 三态积分（compute_s / stall_s / idle_s，段日志供时间序列）",
            ha="center", fontsize=7.4, color="#666666")

    # 数值后端
    box(ax, 0.03, 0.455, 0.42, 0.075,
        "数值双后端：E20/金标全程 Fraction（严格相等）\n生产仿真 float64（tol 1e-12）；同 run 不混用", "#f2f2f2", fs=8.2)

    # 左下：控制器
    box(ax, 0.03, 0.24, 0.42, 0.185, "", "#fdf4ea")
    ax.text(0.24, 0.405, "控制器成本三模式（ControllerConfig）", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 0.045, 0.315, 0.12, 0.065, "zero\n决策同刻生效", "white", fs=7.8)
    box(ax, 0.180, 0.315, 0.12, 0.065, "fixed\n固定延迟 Δ", "white", fs=7.8)
    box(ax, 0.315, 0.315, 0.12, 0.065, "measured\n实测 CPU 秒", "white", fs=7.8)
    box(ax, 0.045, 0.26, 0.39, 0.042,
        "过期 WAIT 剔除；动作验证失败 → GuardedEDF 回退（n_fallback 审计）", "white", fs=7.6)

    # 右：五步闭包
    box(ax, 0.50, 0.45, 0.47, 0.485, "", "#fdf4ea")
    ax.text(0.735, 0.915, "同刻闭包（§4.4，显式循环，不依赖回调顺序）", ha="center", fontsize=9.8, fontweight="bold")
    steps = [
        ("① 积分结算：读取/计算完成、请求到达、带宽变更", 0.055),
        ("② 旧批闭包：按 worker 升序推进依赖已满足的层、提交下层读取", 0.065),
        ("③ 发布观测 → 控制器返回 → 统一新批领取（联合动作按 worker 升序原子落实）", 0.075),
        ("④ 到期流提升：due 未读完的流 q→q_max（保留序号，不可撤回）", 0.065),
        ("⑤ 新空闲 worker + 有队列 → 同刻再派发（固定点，每轮至少领取一个请求）", 0.075),
    ]
    y = 0.885
    for txt, hh in steps:
        box(ax, 0.515, y - hh, 0.44, hh, txt, "white", fs=7.9)
        y -= hh + 0.012
    ax.text(0.735, 0.47, "时间推进：下一时刻 = min(读完成, 计算完成, 到达, 带宽断点, 观测采样, 控制返回, 唤醒)",
            ha="center", fontsize=7.4, color="#666666")

    # 右下：FCFS 存储示意
    box(ax, 0.50, 0.075, 0.47, 0.335, "", "#eef7ec")
    ax.text(0.735, 0.385, "StorageSim：B(t) 阶梯带宽 + FCFS 按申请上限分配", ha="center", fontsize=9.5, fontweight="bold")
    # 带宽条
    bx, bw, bh = 0.525, 0.42, 0.075
    segs = [(0.16, "流1\nmin(q₁,B)"), (0.13, "流2\nmin(q₂,余)"), (0.09, "流3\n余=0 停"), (0.04, "…")]
    cx = bx
    for wfrac, lab in segs:
        ww = bw * (wfrac / 0.42)
        box(ax, cx, 0.27, ww, bh, lab if wfrac > 0.08 else "", "#c6e0c0", fs=6.8)
        cx += ww
    ax.text(0.735, 0.245, "按 submit_seq 升序逐流分：rate=min(q_i, 剩余带宽)；B(t) 分段常数（右连续）",
            ha="center", fontsize=7.6)
    box(ax, 0.515, 0.095, 0.44, 0.115,
        "守恒账本：∫B dt、∫Σrate dt、∫Σq dt 与区间日志\n（record_intervals，E25 时间序列 + BU11 零影响对拍）", "white", fs=7.9)
    save(fig, "fig_arch_cq_engine.png")


# ------------------------------------------------------- 图 E 流水线
def fig_e():
    fig, ax = new_ax(14.5, 7.2)
    ax.text(0.5, 0.985, "实验执行与产物流水线：CLI → 实验装配 → 仿真 → 指标 → 图 → 文档（含质量门）",
            ha="center", va="top", fontsize=12.5, fontweight="bold")

    row1 = [
        (0.025, 0.155, "CLI\nsim.run --exp X\n--seeds N --procs P", C_CLI),
        (0.215, 0.19, "实验装配 experiments/e*.py\nRunSpec / CqScenario 网格\n× 策略集合 × 种子", C_SHARE),
        (0.44, 0.185, "simrun.run_once / run_case\nCRN trace 缓存（同种子同 trace）\nrun_pool 多进程 spawn", C_SHARE),
        (0.66, 0.155, "事件内核\nSimPy（v1/v2）｜\nCqEngine（cq）", C_SHARE),
        (0.86, 0.115, "Collector /\nsummarize", C_SHARE),
    ]
    y1 = 0.76
    for x, w, t, c in row1:
        box(ax, x, y1, w, 0.115, t, c, fs=8.2)
    for i in range(len(row1) - 1):
        x0 = row1[i][0] + row1[i][1]
        arrow(ax, (x0 + 0.004, y1 + 0.0575), (row1[i + 1][0] - 0.004, y1 + 0.0575))

    row2 = [
        (0.025, 0.20, "results/<exp>/summary.csv\n+ requests/ts CSV（gitignored）", "#f2f2f2"),
        (0.265, 0.17, "tools/ 分析与绘图\ne25_* / cq_report / g_report …", "#f2f2f2"),
        (0.475, 0.175, "docs/figures/*.png\n（入库；严禁小样本重跑覆盖正式图）", "#f2f2f2"),
        (0.69, 0.165, "实验结果-EXX-YYYYMMDD.md\n（仓库根，嵌图 + 读法 + 通俗版）", "#f2f2f2"),
        (0.89, 0.085, "提交推送", C_CLI),
    ]
    y2 = 0.545
    for x, w, t, c in row2:
        box(ax, x, y2, w, 0.105, t, c, fs=8.2)
    for i in range(len(row2) - 1):
        x0 = row2[i][0] + row2[i][1]
        arrow(ax, (x0 + 0.004, y2 + 0.0525), (row2[i + 1][0] - 0.004, y2 + 0.0525))
    arrow(ax, (0.9175, y1), (0.9175, y2 + 0.105), rad=0.0)  # summary → results
    arrow(ax, (0.115, y2), (0.115, 0.42), style="--", color="#888888")

    # 质量门
    box(ax, 0.025, 0.10, 0.60, 0.275, "", "#fdf6e3")
    ax.text(0.325, 0.345, "质量门（AGENTS.md 工作流）", ha="center", fontsize=10, fontweight="bold")
    box(ax, 0.045, 0.265, 0.55, 0.058, "tests/ 不变量单测：解析解吻合 / 字节守恒 / 确定性 / 策略单调性（每次提交前全绿）", "white", fs=8.0)
    box(ax, 0.045, 0.195, 0.55, 0.058, "冒烟先行：--exp smoke*（1–2 种子、短 duration），检查区分度（策略不在 goodput 天花板）", "white", fs=8.0)
    box(ax, 0.045, 0.125, 0.55, 0.058, "改动共享代码 → 以 1–2 种子跑全 v1 四件套 + 对应族回归，核对结论方向不变", "white", fs=8.0)

    box(ax, 0.655, 0.10, 0.32, 0.275, "", "#f4f0f8")
    ax.text(0.815, 0.345, "可复现性手段", ha="center", fontsize=10, fontweight="bold")
    box(ax, 0.672, 0.255, 0.285, 0.068, "CRN：同种子同 trace（workload.py / Mooncake 窗口缩放）", "white", fs=7.9)
    box(ax, 0.672, 0.185, 0.285, 0.068, "Fraction 精确后端 + trace/manifest 规范哈希（canonical JSON SHA256）", "white", fs=7.9)
    box(ax, 0.672, 0.115, 0.285, 0.068, "种子流水 rng=default_rng([seed, salt, stream])（观测噪声、决策抖动独立流）", "white", fs=7.9)
    save(fig, "fig_arch_pipeline.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    fig_e()
    print("all figures done")
