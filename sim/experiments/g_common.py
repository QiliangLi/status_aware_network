"""G 系列实验公共：S-G 场景构建（拓扑/压力象限/负载档位/SLO 合成）与运行循环。

S-G 锚定（设计 §2）：W=4 同构 worker、K=2 存储节点（mem 双副本）+ 共享 fabric、
local_cache_gb=0、目录全局可见、副本放置冻结。
SLO 为合成档位：slo_ttft = clip(4×(nominal取回 + suffix prefill + d0), 0.25, 8)、
slo_tpot = 6×(d0+d1)——公式冻结于正式运行前，不按结果回调（纲领 §4.2）。
"""
from __future__ import annotations

import json
import os

from ..config import (GpuConfig, ModelConfig, NodeConfig, ObsConfig,
                      StorageConfig, TopoConfig)
from ..gpu import PrefillCurve
from ..m1 import M1Class, M1Config, M1Spec, M1System

MODEL = ModelConfig()                     # 80 层 → kv ≈ 0.000328 GB/token
# G 系列 prefill 曲线：轻微超线性（attention 上下文效应的机制级替身）。
# 注意：曲线形状是取/算交叉的敏感变量（纲领 §3.1 线性反例），凹曲线会令长命中
# 取回几乎永不可取——故选超线性并作为模型变体维度在报告中声明。
G_TABLE = ((4096, 0.060), (8192, 0.123), (16384, 0.254),
           (32768, 0.530), (65536, 1.110))
G_GPU = GpuConfig(prefill_table=G_TABLE)
CURVE = PrefillCurve(G_TABLE)
M1 = M1Config(hbm_gb=96.0)
KVT = MODEL.kv_gb_per_token
PATH_LAT = ((0.002, 0.004), (0.003, 0.002),
            (0.006, 0.003), (0.008, 0.004))   # 4 worker × 2 node
MIN_LAT = 0.002
NOM_BW = 0.8 * min(60.0, 120.0)           # nominal 有效带宽（SLO 公式用）

# 压力象限（G1）：背景负载 schedule 常数
QUADRANTS = {
    "loose": dict(mem_bg=8.0, fabric_bg=12.0, gpu_bg=0.10),   # 双松
    "io":    dict(mem_bg=42.0, fabric_bg=78.0, gpu_bg=0.10),  # I/O 紧 GPU 松
    "gpu":   dict(mem_bg=8.0, fabric_bg=12.0, gpu_bg=0.30),   # GPU 紧 I/O 松
    "both":  dict(mem_bg=42.0, fabric_bg=78.0, gpu_bg=0.30),  # 双紧
}


def build_topo(classes, quad: dict, n_workers: int = 4) -> TopoConfig:
    nodes = tuple(
        NodeConfig(f"n{i}", mem=StorageConfig(b_total=60.0,
                                              bg_schedule=((0.0, quad["mem_bg"]),)),
                   ssd=StorageConfig(b_total=25.0))
        for i in range(2))
    replicas = tuple((c.name, ((0, "mem"), (1, "mem"))) for c in classes if c.H > 0)
    return TopoConfig(
        n_workers=n_workers, nodes=nodes,
        fabric=StorageConfig(b_total=120.0, bg_schedule=((0.0, quad["fabric_bg"]),)),
        path_lat=PATH_LAT, local_cache_gb=0.0, replicas=replicas,
        gpu_bgs=tuple(((0.0, quad["gpu_bg"]),) for _ in range(n_workers)),
    )


def _slo(H: int, U: int):
    if H > 0:
        nom = MIN_LAT + H * KVT / NOM_BW + 2 * 0.005
        t = nom + CURVE(U) + M1.decode_t0
    else:
        t = CURVE(U) + M1.decode_t0
    return min(8.0, max(0.25, 4.0 * t))


def make_classes(defs: list) -> tuple:
    """defs: (name, H, U, O, share, prio) → M1Class（SLO 按公式冻结）。"""
    return tuple(M1Class(name=n, H=h, U=u, O=o, share=s, prio=p,
                         slo_ttft=_slo(h, u), slo_tpot=6.0 * (M1.decode_t0 + M1.decode_t1))
                 for (n, h, u, o, s, p) in defs)


# G1 负载：长短命中 × 长短后缀 × 长短输出四角 + 无命中对照（设计 §6 G1）
G1_DEFS = [
    ("S3", 512,   512,  128, 0.25, 0),    # 短命中短后缀短输出
    ("L1", 16384, 512,  128, 0.15, 1),    # 长命中短后缀短输出（取/算张力最大）
    ("S1", 512,   8192, 2048, 0.25, 1),   # 短命中长后缀长输出（decode 大户）
    ("L3", 16384, 8192, 2048, 0.15, 1),   # 全大
    ("M0", 0,     4096, 256, 0.20, 0),    # 无命中对照
]

# G2 负载：纲领 §7.2 第二正交组 H×U×O 全 8 类等份额
G2_DEFS = [
    (f"h{h//1024}k_u{u//1024}k_o{o}", h, u, o, 1 / 8, 0 if o <= 128 else 1)
    for h in (512, 16384) for u in (512, 8192) for o in (128, 2048)
]

# 策略清单：(router, order, gate, label)
G1_POLICIES = [
    ("rr", "fcfs", False, "B-RR"),
    ("ll", "fcfs", False, "B-LL"),
    ("dynkv", "fcfs", False, "B-DynKV"),
    ("moons", "fcfs", False, "B-Moon-S"),
    ("hist", "fcfs", False, "B-Hist"),
    ("histj", "fcfs", False, "Hist+Joint"),
    ("staticar", "fcfs", False, "E-StaticAR"),
    ("d1b", "fcfs", False, "D1b 源选择"),
    ("dynar", "fcfs", False, "D2b 动态取算"),
    ("d1a", "fcfs", False, "D1a 联合"),
    ("truest", "fcfs", False, "True-State 诊断"),
]
G2_POLICIES = [
    ("hist", "fcfs", False, "E-FCFS(B-Hist)"),
    ("hist", "lpm", False, "E-LPM"),
    ("hist", "prio", False, "E-Prio"),
    ("hist", "ready", False, "D2a 就绪排序"),
    ("hist", "budget", False, "D2c 预算排序"),
    ("hist", "fcfs", True, "D2d 门控"),
    ("dynar", "fcfs", False, "D2b 动态取算"),
    ("dynar", "fcfs", True, "D2b+D2d 组合"),
    ("d1a", "fcfs", False, "D1a 联合"),
]


def run_grid(exp: str, policies, quadrants, seeds, defs, lam: float,
             duration: float, out_path: str, timeline_marks=()) -> list:
    """运行 策略 × 象限 × 种子 网格，落盘 JSON。timeline_marks: {(label, quad, seed)}."""
    classes = make_classes(defs)
    results = []
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for qname in quadrants:
        topo = build_topo(classes, QUADRANTS[qname])
        for (router, order, gate, label) in policies:
            for seed in seeds:
                tl = (label, qname, seed) in timeline_marks
                # True-State 诊断：d1a 决策器 + live 无噪声观测（真值反馈，仅诊断用）
                rname = router
                obs = ObsConfig()
                if router == "truest":
                    rname = "d1a"
                    obs = ObsConfig(interval=0.0, noise_sigma=0.0, signal="quote")
                spec = M1Spec(exp=exp, router=rname, order=order, gate=gate,
                              seed=seed, duration=duration, warmup=30.0, lam=lam,
                              classes=classes, topo=topo, obs=obs,
                              m1=M1, model=MODEL, gpu=G_GPU, timeline=tl)
                res = M1System(spec).run()
                res["label"] = label
                res["quadrant"] = qname
                results.append(res)
                o = res["overall"]
                print(f"[{exp}] {qname:5s} {label:14s} seed={seed} "
                      f"ttft_p95={o['ttft_p95']:.3f} attain={o['attain_mean']:.2f} "
                      f"min={o['attain_min']:.2f} undone={o['n_undone']}", flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"saved {len(results)} runs -> {out_path}", flush=True)
    return results


def slo_table(defs) -> list:
    return [(n, h, u, o, round(_slo(h, u), 3)) for (n, h, u, o, s, p) in defs]
