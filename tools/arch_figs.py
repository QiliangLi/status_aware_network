"""架构分析文档配图（docs/仿真架构分析-20260921.md）。

生成 5 张 PNG 到 docs/figures/fig_arch_*.png：
  A 总览（四族仿真 + 共享支撑层）
  B 信息边界双世界（ground truth vs observable vs Oracle 通道）
  C v2 共享分布式拓扑与决策/取回链路
  D cq 引擎（批层状态机 + 同刻五步闭包 + FCFS 存储）
  E 实验执行与产物流水线

纯 matplotlib 手绘（box/arrow），不依赖仿真代码；只反映代码现状。

布局防裂机制（v2，20260921 修复）：
  1) 文本自适应：每个 box 的文字用 renderer 实测包围盒，超出盒宽/高则自动
     降字号（至 6.5pt），仍放不下直接抛错——不允许静默溢出；
  2) 盒子两两重叠断言：除"泳道/面板背景 vs 其子盒"的祖先关系外，
     任何两盒重叠超过容差即抛错——不允许标注压图。
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "figures")

# 字体链以实测覆盖为准（Heiti TC 覆盖本文全部字形含 ℓ/−/↔；PingFang SC 未被
# matplotlib 注册、Hiragino 缺 ℓ−↔，均不能排首位）
plt.rcParams["font.sans-serif"] = ["Heiti TC", "Songti SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

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

REG: list = []          # [{id, rect, parent, text}]，verify 用


def box(ax, x, y, w, h, text, fc, fs=8.5, ec=EDGE, bold=False, parent=None):
    """画一个圆角盒。parent=所属泳道/面板的 REG id，允许被其包含。"""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
                       fc=fc, ec=ec, lw=1.0)
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)
    rid = len(REG)
    REG.append({"id": rid, "rect": (x, y, w, h), "parent": parent, "text": t,
                "fs0": fs, "label": text.split("\n")[0][:24]})
    return rid


def lane(ax, x, y, w, h, title, fc, fs=10):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
                       fc=fc, ec=EDGE, lw=1.2, alpha=0.45)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top",
            fontsize=fs, fontweight="bold")
    return box(ax, x, y, w, h, "", fc, fs=1)  # 占位登记矩形（无文字）


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


def verify(fig, ax, name):
    """防裂验收：文本不溢出（自动缩字号兜底）+ 盒子不重叠（祖先除外）。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    n_shrunk = 0
    for e in REG:
        t = e["text"]
        if t is None or not t.get_text():
            continue
        for _ in range(10):
            bb = t.get_window_extent(renderer=renderer)
            (x0, y0) = inv.transform((bb.x0, bb.y0))
            (x1, y1) = inv.transform((bb.x1, bb.y1))
            tw, th = x1 - x0, y1 - y0
            bx, by, bw, bh = e["rect"]
            if tw <= bw - 0.004 and th <= bh - 0.002:
                break
            fs = max(6.5, t.get_fontsize() - 0.5)
            t.set_fontsize(fs)
            n_shrunk += 1
            fig.canvas.draw()
        else:
            pass
        bb = t.get_window_extent(renderer=renderer)
        (x0, y0) = inv.transform((bb.x0, bb.y0))
        (x1, y1) = inv.transform((bb.x1, bb.y1))
        tw, th = x1 - x0, y1 - y0
        bx, by, bw, bh = e["rect"]
        if tw > bw + 0.003 or th > bh + 0.003:
            raise RuntimeError(
                f"[{name}] 文本溢出: {e['label']!r} box={e['rect']} "
                f"text=({tw:.3f},{th:.3f}) fs={t.get_fontsize()}")

    def is_anc(a, b):
        while b is not None:
            if b == a:
                return True
            b = REG[b]["parent"]
        return False

    def overlap(a, b):
        ax_, ay, aw, ah = a
        bx, by, bw, bh = b
        return (min(ax_ + aw, bx + bw) - max(ax_, bx) > 0.004
                and min(ay + ah, by + bh) - max(ay, by) > 0.004)

    for i in range(len(REG)):
        for j in range(i + 1, len(REG)):
            if is_anc(i, j) or is_anc(j, i):
                continue
            if overlap(REG[i]["rect"], REG[j]["rect"]):
                raise RuntimeError(f"[{name}] 盒子重叠: {REG[i]['label']!r} "
                                   f"{REG[i]['rect']} vs {REG[j]['label']!r} {REG[j]['rect']}")
    if n_shrunk:
        detail = [(e["label"], e["text"].get_fontsize()) for e in REG
                  if e["text"] is not None and e["text"].get_text()
                  and e["text"].get_fontsize() < e["fs0"]]
        print(f"  [{name}] {n_shrunk} 步缩号: " + ", ".join(f"{l}@{f:.1f}" for l, f in detail))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    REG.clear()
    print("wrote", path)


def stack(ax, x, w, ytop, items, fc, parent, gap=0.012, fs=8.0):
    y = ytop
    for txt, hh in items:
        box(ax, x, y - hh, w, hh, txt, fc, fs=fs, parent=parent)
        y -= hh + gap
    return y


# ---------------------------------------------------------------- 图 A 总览
def fig_a():
    fig, ax = new_ax(14.5, 9.8)
    ax.text(0.5, 0.985, "仿真架构总览：一个研究问题、三台仿真器、一套共用底座",
            ha="center", va="top", fontsize=13, fontweight="bold")

    box(ax, 0.04, 0.905, 0.92, 0.052,
        "sim/run.py — CLI 入口：--exp all|v2|v3|v4|all2|cq|smoke*|单实验（e1a…e25, g1, g2）；--seeds/--procs/--duration/--stage/--manifest",
        C_CLI, fs=9.5, bold=True)

    lx = [0.025, 0.245, 0.475, 0.745]
    lw_ = [0.205, 0.215, 0.255, 0.235]
    titles = ["解析与迭代级支线工具", "仿真器① 取回决策（请求级）",
              "仿真器② 分布式拓扑（请求级+放置）", "仿真器③ 组批错峰（批级·精确内核）"]
    lanes = [lane(ax, lx[i], 0.175, lw_[i], 0.705, titles[i], c)
             for i, c in enumerate([C_M1, C_V1, C_V2, C_CQ])]

    stack(ax, lx[0] + 0.008, lw_[0] - 0.016, 0.845, [
        ("m0.py — 解析层\n取回/重算边界 b*=B/(ΔC−ℓ)", 0.085),
        ("m0sim.py — 事件微仿真（simpy）\n单 GPU/单 I/O 串行，解析↔仿真对拍", 0.085),
        ("m1.py — 迭代级引擎\nchunked prefill + decode 优先\nHBM 保守准入 + 命中即取", 0.105),
        ("policies_g.py — 路由/排序策略", 0.05),
        ("实验 G0/G1/G2（象限扫描、混合负载）", 0.05),
    ], "#f6f1fb", lanes[0])

    stack(ax, lx[1] + 0.008, lw_[1] - 0.016, 0.845, [
        ("workload.py — CRN 负载\nPoisson/类别/burst/会话/热度漂移", 0.085),
        ("storage.py — SharedKVStorage\nPS 流体 + 背景负载 + t_base 门控\n（配 StorageObservable 双世界）", 0.10),
        ("gpu.py — PrefillCurve 查表 + GpuPool\nFCFS 流体队列（rate=1−bg）", 0.075),
        ("scheduler.py + worker.py\nroute→engine_decide→执行", 0.06),
        ("policies/ — P0 AlwaysFetch / P1 Static\nP2 Dynamic / P3 Route / P4 Oracle\n+ rr / load / kv 基线", 0.09),
        ("实验 E1a/E1b/E2/E3/E4", 0.045),
    ], "#f0f6fc", lanes[1])

    stack(ax, lx[2] + 0.008, lw_[2] - 0.016, 0.845, [
        ("topology.py — World：LocalKVCache(LRU)\nMetadataDirectory(副本+容量)\n多节点 mem/ssd + 共享 fabric", 0.10),
        ("quote.py — AccessCostQuery\n压力档位(滞回) + 完成时间估计", 0.06),
        ("gpu_obs.py — GpuObservable（对称观测）", 0.045),
        ("engine2.py — local / fetch(三级链路)\npartial(F 比例+overlap) / recompute", 0.075),
        ("policies2.py — 17 策略\n主流映射 + joint2/coord2 + oracle/先知", 0.06),
        ("prefetch.py + _ctrl_loop\n预取/回写/保护 + 复制/迁移/淘汰", 0.07),
        ("实验 E5–E18（含三批改进系列）", 0.045),
    ], "#f2f9ef", lanes[2])

    stack(ax, lx[3] + 0.008, lw_[3] - 0.016, 0.845, [
        ("types/config — 三域对象 + Fraction 配置\n（输入 | 运行真值 | 可观测快照）", 0.075),
        ("profile.py — synthetic-v1 画像\nc=t_launch+aN/η(N)+bA、T0 解析式", 0.07),
        ("storage.py — StorageSim\nB(t) 阶梯 + FCFS 按申请上限 + due 提升", 0.07),
        ("engine.py — CqEngine 自研事件循环\n五步同刻闭包 + 批层状态机", 0.065),
        ("observable.py — 报价采样/滞后/噪声\nObservableSnapshot（类型化信息边界）", 0.07),
        ("policies + search + exact\n简单排序 / MPC 双预算 / B&B 金标", 0.065),
        ("trace.py — Mooncake 三 trace\n首现命中派生 + 窗口缩放", 0.06),
        ("实验 E19–E25（含 timeseries.py）", 0.045),
    ], "#fdf4ea", lanes[3])

    box(ax, 0.025, 0.028, 0.955, 0.115,
        "共享支撑层：config.py 单位约定(秒/GB十进制/GBps)与 frozen dataclass ｜ request.py 请求对象 ｜ metrics.py Collector(v1/v2 通用)\n"
        "simrun.py 装配(v1 run_once / v2 run_once_v2) ｜ experiments/ 装配框架(common/v2common/cq_common/g_common) + run_pool 多进程\n"
        "tests/ 不变量单测(解析吻合/守恒/确定性/单调性) ｜ results/(gitignored) ｜ docs/figures/ 入库",
        C_SHARE, fs=8.2)

    for x, w in zip(lx, lw_):
        arrow(ax, (x + w / 2, 0.905), (x + w / 2, 0.884))
    ax.text(0.5, 0.155, "仿真器①②共用同一 SimPy+流体内核与运行装配；③为独立精确内核（Fraction）。历史代号（v1/v2/cq）见文档代码地图一节",
            ha="center", fontsize=8, color="#666666")
    verify(fig, ax, "A")
    save(fig, "fig_arch_overview.png")


# ------------------------------------------------------- 图 B 信息边界双世界
def fig_b():
    fig, ax = new_ax(14.0, 8.6)
    ax.text(0.5, 0.985, "信息边界双世界：ground truth ｜ 可观测层 ｜ 策略（三族统一的方法论）",
            ha="center", va="top", fontsize=13, fontweight="bold")

    rt = lane(ax, 0.025, 0.32, 0.30, 0.60, "ground truth（物理真值世界）", C_TRUE, fs=10.5)
    ro = lane(ax, 0.40, 0.32, 0.30, 0.60, "可观测层（普通策略的唯一接口）", C_OBS, fs=10.5)
    rp = lane(ax, 0.755, 0.32, 0.225, 0.60, "策略", C_POL, fs=10.5)

    box(ax, 0.04, 0.755, 0.27, 0.115,
        "资源真值（仿真器①②共用）：\nSharedKVStorage 内部队列/背景负载\nGpuPool.queue\nhypothetical_* 精确推演接口", "white", parent=rt, fs=8.2)
    box(ax, 0.04, 0.615, 0.27, 0.115,
        "拓扑真值（仿真器②）：World.resources\n(mem/ssd/fabric 流体)\nMetadataDirectory 副本集\nLocalKVCache 实际内容", "white", parent=rt, fs=8.2)
    box(ax, 0.04, 0.475, 0.27, 0.115,
        "组批内核真值（仿真器③）：\nWorldState/BatchRuntime\nFlowState(剩余字节/速率)\nB(t) 真带宽、c_layers 计算真值", "white", parent=rt, fs=8.2)
    box(ax, 0.04, 0.335, 0.27, 0.105,
        "Oracle/先知专属数据：\nhypothetical_*、future(cls,t,H)", "white", parent=rt, fs=8.2, ec="#b55a5a")

    box(ax, 0.415, 0.755, 0.27, 0.115,
        "StorageObservable\ninterval 采样 / EMA 平滑\n乘性 lognormal 噪声\nsignal=quote|bw|util|queue", "white", parent=ro, fs=8.2)
    box(ax, 0.415, 0.615, 0.27, 0.115,
        "quote.AccessCostQuery\nest = path_lat + tier + fabric\n压力档位 NORMAL/WARM/\nHOT/CRITICAL（滞回防抖）", "white", parent=ro, fs=8.2)
    box(ax, 0.415, 0.475, 0.27, 0.115,
        "GpuObservable（GPU 侧对称仪表盘）\nGPU 排队 drain_est 的\n陈旧/EMA/带噪视图\n默认 0/0=真值（请求级既定简化）", "white", parent=ro, fs=8.2)
    box(ax, 0.415, 0.335, 0.27, 0.105,
        "组批可观测快照 ObservableSnapshot\n报价 Quote + 流账本 + est_bw", "white", parent=ro, fs=8.2)

    box(ax, 0.77, 0.655, 0.195, 0.21,
        "普通策略（只读中栏）\nP2/P3、static2dyn、\njoint2/coord2、cascade2、\ncq_fcfs…cq_mpc", "white", parent=rp, fs=8.2)
    box(ax, 0.77, 0.44, 0.195, 0.155,
        "Oracle 家族（读左栏）\nP4 / oracle2 /\nclairvoyant2 / cq_oracle", "white", parent=rp, fs=8.2, ec="#b55a5a")

    # 边界虚线（不进入下方特权通道区）
    ax.plot([0.355, 0.355], [0.31, 0.935], ls="--", lw=2.2, color="#b55a5a")
    ax.plot([0.725, 0.725], [0.31, 0.935], ls="--", lw=2.2, color="#4a7d59")
    ax.text(0.355, 0.948, "信息边界（存储侧）", ha="center", fontsize=9.5, color="#b55a5a", fontweight="bold")
    ax.text(0.725, 0.948, "接口边界", ha="center", fontsize=9.5, color="#4a7d59", fontweight="bold")

    # 采样箭头（真值 → 可观测）
    arrow(ax, (0.31, 0.8125), (0.415, 0.8125), color="#4a7d59")
    ax.text(0.3625, 0.832, "采样", ha="center", fontsize=7.5, color="#4a7d59")
    arrow(ax, (0.31, 0.77), (0.415, 0.535), color="#4a7d59", rad=0.05)
    arrow(ax, (0.31, 0.51), (0.415, 0.39), color="#4a7d59", rad=0.05)
    ax.text(0.345, 0.63, "采样/报价", ha="center", fontsize=7.5, color="#4a7d59", rotation=90)
    # O1 → O2（AccessCostQuery 读各资源 Observable）
    arrow(ax, (0.55, 0.755), (0.55, 0.732), color="#4a7d59", style="--")
    # 策略 ← 可观测层
    arrow(ax, (0.69, 0.7575), (0.77, 0.7575), color="#4a7d59")
    ax.text(0.73, 0.775, "只读", ha="center", fontsize=7.5, color="#4a7d59")

    # Oracle 特权通道：从 Oracle 策略盒绕开可观测层、贴底横穿到左栏
    arrow(ax, (0.8675, 0.44), (0.8675, 0.265), color="#b55a5a", lw=1.6)
    arrow(ax, (0.8675, 0.265), (0.175, 0.265), color="#b55a5a", lw=1.6)
    arrow(ax, (0.175, 0.265), (0.175, 0.332), color="#b55a5a", lw=1.6)
    ax.text(0.52, 0.283, "特权通道：绕过可观测层直读 ground truth（hypothetical_* / 未来视线）",
            ha="center", fontsize=8, color="#b55a5a")

    box(ax, 0.025, 0.02, 0.955, 0.165,
        "信息边界铁律（AGENTS.md §3）：边界必须匹配研究问题，且对所有被比较的策略一致。\n"
        "· 仿真器①②：以对象引用实现边界 —— 普通策略只拿 WorkerView.obs / V2Ctx.quote，Oracle 直接拿资源对象；\n"
        "· 仿真器③：以类型系统实现 —— ObservableSnapshot 不携带 world/engine 引用，真值只在 include_oracle=True 诊断快照出现；\n"
        "· GPU 侧默认真值可见（请求级既定简化）；研究 GPU 侧误差的实验用 GpuObservable 对称开洞。",
        "#fdf6e3", fs=8.4)
    verify(fig, ax, "B")
    save(fig, "fig_arch_dual_world.png")


# ------------------------------------------------------- 图 C v2 拓扑
def fig_c():
    fig, ax = new_ax(14.0, 9.5)
    ax.text(0.5, 0.985, "分布式拓扑仿真器：存储拓扑与一次取回决策的完整链路（sim/topology.py + engine2.py + policies2.py）",
            ha="center", va="top", fontsize=12.5, fontweight="bold")

    # workers（左列宽度内 4 个）
    for i in range(4):
        x = 0.035 + i * 0.181
        box(ax, x, 0.80, 0.161, 0.13,
            f"worker {i}\nGpuPool(FCFS 流体\nrate=1−bg)\nLocalKVCache(LRU 12GB)", "#eef4fb", fs=7.8)

    box(ax, 0.035, 0.635, 0.565, 0.12,
        "policies2.decide(req) → Decision(worker, action, node, tier, F, overlap)\n"
        "动作：local（本地命中仅 suffix）｜fetch｜partial(F 比例+overlap)｜recompute —— 遍历 worker×副本×动作×F",
        C_POL, fs=8.2)
    box(ax, 0.775, 0.635, 0.165, 0.12,
        "quote.AccessCostQuery\nest=path_lat\n+tier+fabric\n（读 Observable）", C_OBS, fs=7.8)

    box(ax, 0.035, 0.50, 0.685, 0.095,
        "MetadataDirectory：class → {(node, tier)} 副本集合；节点占用 → 容量压力；remove() 内建孤儿检查",
        "#f7f2df", fs=8.5)

    for i, (mm, ss) in enumerate([(60, 25), (60, 25), (40, 15)]):
        box(ax, 0.035 + i * 0.235, 0.27, 0.215, 0.175,
            f"存储节点 n{i}（默认实验配置）\nmem SharedKVStorage\n{mm} GB/s\nssd SharedKVStorage {ss} GB/s\ncap 512 GB",
            "#eef7ec", fs=7.9)

    box(ax, 0.775, 0.27, 0.185, 0.19,
        "fetch 三级顺序链路\n（engine2._chain）\n① path_lat 延迟\n② tier 资源服务\n③ fabric 传输\npartial：与 GPU 剩余\n计算重叠取 max",
        "#fdeceb", fs=7.8, ec="#c0392b")

    box(ax, 0.035, 0.14, 0.925, 0.08,
        "共享 fabric：SharedKVStorage 120 GB/s（所有取回/预取/复制流量共用的第三个流体资源）",
        "#eef7ec", fs=8.5)

    box(ax, 0.035, 0.02, 0.44, 0.075,
        "_ctrl_loop（CtrlConfig）：滞回判 HOT\n→ 复制/跨层降级/冷迁移/容量淘汰", "#f2f2f2", fs=7.8)
    box(ax, 0.52, 0.02, 0.44, 0.075,
        "Prefetcher：会话预取(gated/predictive/\nsession) + 重算回写 + 活跃保护", "#f2f2f2", fs=7.8)

    # 纵向主干箭头（左列）
    arrow(ax, (0.30, 0.80), (0.30, 0.757))
    ax.text(0.315, 0.778, "gpu_wait/drain_est", fontsize=7.0, color="#666666")
    arrow(ax, (0.30, 0.635), (0.30, 0.597))
    arrow(ax, (0.30, 0.50), (0.30, 0.447))
    arrow(ax, (0.30, 0.27), (0.30, 0.222))
    # quote → decision / directory
    arrow(ax, (0.775, 0.695), (0.602, 0.695), color="#4a7d59")
    arrow(ax, (0.80, 0.635), (0.72, 0.555), color="#4a7d59", style="--")
    # 红色取回链路：worker3 → 右缘竖直下行 → 说明盒 → n2 / fabric
    arrow(ax, (0.739, 0.865), (0.955, 0.865), color="#c0392b", lw=1.6)
    arrow(ax, (0.955, 0.865), (0.955, 0.465), color="#c0392b", lw=1.6)
    arrow(ax, (0.775, 0.365), (0.723, 0.365), color="#c0392b", lw=1.6)
    arrow(ax, (0.955, 0.27), (0.955, 0.223), color="#c0392b", lw=1.6)
    ax.text(0.847, 0.885, "fetch 取回流", fontsize=7.8, color="#c0392b", ha="center")
    verify(fig, ax, "C")
    save(fig, "fig_arch_v2_topo.png")


# ------------------------------------------------------- 图 D cq 引擎
def fig_d():
    fig, ax = new_ax(14.0, 9.5)
    ax.text(0.5, 0.985, "组批错峰仿真器：批的层状态机、同刻处理顺序、共享带宽分配（sim/cq/engine.py、storage.py）",
            ha="center", va="top", fontsize=12.5, fontweight="bold")

    # 左上：批生命周期面板
    p1 = lane(ax, 0.03, 0.655, 0.42, 0.28, "批的生命周期（BatchRuntime，L 层）", C_CQ, fs=9.5)
    box(ax, 0.05, 0.845, 0.10, 0.05, "FUTURE", "white", fs=8, parent=p1)
    box(ax, 0.175, 0.845, 0.10, 0.05, "QUEUED", "white", fs=8, parent=p1)
    box(ax, 0.30, 0.845, 0.10, 0.05, "ACTIVE", "white", fs=8, parent=p1)
    box(ax, 0.40, 0.845, 0.042, 0.05, "DONE", "white", fs=7.2, parent=p1)
    arrow(ax, (0.15, 0.87), (0.173, 0.87))
    arrow(ax, (0.275, 0.87), (0.298, 0.87))
    arrow(ax, (0.40, 0.87), (0.398, 0.87))
    ax.text(0.2375, 0.898, "DISPATCH 领取", ha="center", fontsize=6.8, color="#666666")

    yb = 0.75
    box(ax, 0.05, yb, 0.115, 0.075, "wait_read(ℓ)\n读取流 V_ℓ GB\n(q=V_ℓ/c, due=Z_ℓ)", "white", fs=7.4, parent=p1)
    box(ax, 0.195, yb, 0.105, 0.075, "wait_prev(ℓ)\nC=max(R_ℓ, Z_(ℓ-1))", "white", fs=7.4, parent=p1)
    box(ax, 0.325, yb, 0.105, 0.075, "computing(ℓ)\nZ_ℓ=C_ℓ+c_β", "white", fs=7.4, parent=p1)
    arrow(ax, (0.165, yb + 0.0375), (0.193, yb + 0.0375))
    arrow(ax, (0.30, yb + 0.0375), (0.323, yb + 0.0375))
    arrow(ax, (0.3775, yb), (0.1075, yb - 0.028), rad=0.30, color="#c0392b")
    ax.text(0.24, 0.688, "ℓ<L−1：提交 ℓ+1 层读取（q=V_{ℓ+1}/c_β，due=Z_ℓ）；ℓ=L−1：末层 Z 即全批 F",
            ha="center", fontsize=7.2, color="#c0392b")

    # 左中：worker 三态
    p2 = lane(ax, 0.03, 0.525, 0.42, 0.095, "", C_V1, fs=1)
    box(ax, 0.05, 0.555, 0.115, 0.055, "IDLE", "#eef4fb", fs=8.5, parent=p2)
    box(ax, 0.195, 0.555, 0.105, 0.055, "STALL(等读)", "#fdeceb", fs=8.5, parent=p2)
    box(ax, 0.325, 0.555, 0.105, 0.055, "COMPUTE", "#eef7ec", fs=8.5, parent=p2)
    arrow(ax, (0.165, 0.5825), (0.193, 0.5825))
    arrow(ax, (0.30, 0.5825), (0.323, 0.5825))
    arrow(ax, (0.3775, 0.575), (0.1075, 0.549), rad=-0.3, color="#888888", style="--")
    ax.text(0.24, 0.535, "WorkerState 三态积分（compute_s/stall_s/idle_s，段日志供时间序列）",
            ha="center", fontsize=7.2, color="#666666")

    box(ax, 0.03, 0.435, 0.42, 0.07,
        "数值双后端：E20/金标全程 Fraction（严格相等）｜生产 float64（tol 1e-12）",
        "#f2f2f2", fs=8.2)

    # 左下：控制器
    p3 = lane(ax, 0.03, 0.235, 0.42, 0.175, "控制器成本三模式（ControllerConfig）", C_CQ, fs=9.5)
    box(ax, 0.045, 0.30, 0.12, 0.065, "zero\n决策同刻生效", "white", fs=7.8, parent=p3)
    box(ax, 0.18, 0.30, 0.12, 0.065, "fixed\n固定延迟 Δ", "white", fs=7.8, parent=p3)
    box(ax, 0.315, 0.30, 0.12, 0.065, "measured\n实测 CPU 秒", "white", fs=7.8, parent=p3)
    box(ax, 0.045, 0.248, 0.39, 0.042,
        "过期 WAIT 剔除；动作验证失败 → GuardedEDF 回退（n_fallback 审计）", "white", fs=7.6, parent=p3)

    # 右上：五步闭包
    p4 = lane(ax, 0.50, 0.45, 0.47, 0.485, "同一时刻的处理顺序（显式循环，不依赖回调注册顺序）", C_CQ, fs=9.8)
    steps = [
        ("① 积分结算：读取/计算完成、请求到达、带宽变更", 0.055),
        ("② 旧批闭包：按 worker 升序推进依赖已满足的层、提交下层读取", 0.065),
        ("③ 发布观测 → 控制器返回 → 统一新批领取（联合动作按 worker 升序原子落实）", 0.075),
        ("④ 到期流提升：due 未读完的流 q→q_max（保留序号，不可撤回）", 0.065),
        ("⑤ 新空闲 worker + 有队列 → 同刻再派发（固定点，每轮至少领取一个请求）", 0.075),
    ]
    y = 0.885
    for txt, hh in steps:
        box(ax, 0.515, y - hh, 0.44, hh, txt, "white", fs=7.9, parent=p4)
        y -= hh + 0.012
    ax.text(0.735, 0.468, "时间推进：下一时刻 = min(读完成, 计算完成, 到达, 带宽断点, 观测采样, 控制返回, 唤醒)",
            ha="center", fontsize=7.4, color="#666666")

    # 右下：FCFS 存储
    p5 = lane(ax, 0.50, 0.065, 0.47, 0.35, "StorageSim：B(t) 阶梯带宽 + FCFS 按申请上限分配", "#eef7ec", fs=9.5)
    bx, bw, bh = 0.525, 0.42, 0.075
    segs = [(0.16, "流1\nmin(q1,B)"), (0.13, "流2\nmin(q2,余)"), (0.09, "流3\n余=0 停"), (0.04, "…")]
    cx = bx
    for wfrac, lab in segs:
        ww = bw * (wfrac / 0.42)
        box(ax, cx, 0.27, ww, bh, lab if wfrac > 0.08 else "", "#c6e0c0", fs=6.8, parent=p5)
        cx += ww
    ax.text(0.735, 0.243, "按 submit_seq 升序逐流分：rate=min(q_i, 剩余带宽)；B(t) 分段常数（右连续）",
            ha="center", fontsize=7.6)
    box(ax, 0.515, 0.085, 0.44, 0.115,
        "守恒账本：∫B dt、∫Σrate dt、∫Σq dt 与区间日志\n（record_intervals，E25 时间序列 + BU11 零影响对拍）",
        "white", fs=7.9, parent=p5)
    verify(fig, ax, "D")
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
        (0.66, 0.155, "事件内核\nSimPy（仿真器①②）｜\n自研精确内核（③）", C_SHARE),
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
        (0.475, 0.175, "docs/figures/*.png\n（入库；严禁小样本重跑覆盖）", "#f2f2f2"),
        (0.69, 0.165, "实验结果-EXX-YYYYMMDD.md\n（仓库根，嵌图+读法+通俗版）", "#f2f2f2"),
        (0.89, 0.085, "提交推送", C_CLI),
    ]
    y2 = 0.545
    for x, w, t, c in row2:
        box(ax, x, y2, w, 0.105, t, c, fs=8.2)
    for i in range(len(row2) - 1):
        x0 = row2[i][0] + row2[i][1]
        arrow(ax, (x0 + 0.004, y2 + 0.0525), (row2[i + 1][0] - 0.004, y2 + 0.0525))
    arrow(ax, (0.9175, y1), (0.9175, y2 + 0.107))
    arrow(ax, (0.115, y2), (0.115, 0.422), style="--", color="#888888")

    p1 = lane(ax, 0.025, 0.10, 0.60, 0.275, "质量门（AGENTS.md 工作流）", "#fdf6e3", fs=10)
    box(ax, 0.045, 0.265, 0.55, 0.058, "tests/ 不变量单测：解析解吻合 / 字节守恒 / 确定性 / 策略单调性（每次提交前全绿）", "white", fs=8.0, parent=p1)
    box(ax, 0.045, 0.195, 0.55, 0.058, "冒烟先行：--exp smoke*（1–2 种子、短 duration），检查区分度（策略不在 goodput 天花板）", "white", fs=8.0, parent=p1)
    box(ax, 0.045, 0.125, 0.55, 0.058, "改动共享代码 → 以 1–2 种子跑全 v1 四件套 + 对应族回归，核对结论方向不变", "white", fs=8.0, parent=p1)

    p2 = lane(ax, 0.655, 0.10, 0.32, 0.275, "可复现性手段", "#f4f0f8", fs=10)
    box(ax, 0.672, 0.255, 0.285, 0.068, "CRN：同种子同 trace\n（workload.py / Mooncake 窗口缩放）", "white", fs=7.9, parent=p2)
    box(ax, 0.672, 0.185, 0.285, 0.068, "Fraction 精确后端 + 规范哈希\n（trace/manifest SHA256）", "white", fs=7.9, parent=p2)
    box(ax, 0.672, 0.115, 0.285, 0.068, "种子流水 rng=default_rng\n([seed, salt, stream])，噪声/抖动独立", "white", fs=7.9, parent=p2)
    verify(fig, ax, "E")
    save(fig, "fig_arch_pipeline.png")


if __name__ == "__main__":
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    if not ({"Heiti TC", "Songti SC", "Hiragino Sans GB", "Arial Unicode MS"} & names):
        raise SystemExit("未找到任何已注册的中文字体，中止出图")
    os.makedirs(OUT, exist_ok=True)
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    fig_e()
    print("all figures done")
