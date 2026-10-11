"""候选评分进程池（fork-per-decision）——MPC 推演零结果影响的并行加速。

设计：docs/MPC推演并行加速-候选评分进程池-20261011.md。等价性三支柱：
①同一决策点内各候选从同一 est0 出发、独立副本推演（天然并行）；
②hist 写入幂等（policies.bw_demand 的 update 同 rid 同常数，决策点内视图
恒定）；③事件预算池的顺序依赖由"满预算推演+原序重放"精确复现（重放在
search.MPCPolicy._decide_par，本模块不做任何裁决）。

本模块只做机械执行：fork N 个 child（继承 est0 与全部已 import 模块，
零序列化），各自对分到的候选做 clone+满预算 forecast_drain，结果 pickle
落临时文件（避开 pipe 缓冲上限），父进程 waitpid 后读取汇总。

异常契约：任何 child 失败（非零退出/结果损坏/下标不完整）抛 ParEvalError，
调用方（decide）捕获后整决策点回退串行重算——基础设施故障只损失该决策
点的加速，不改变行为。

child 安全注意：推演路径为纯 Python 单线程（不触碰 numpy BLAS 线程）；
退出用 os._exit 绕过 atexit 与 stdout 缓冲刷新（防止子进程重复刷父进程
的输出缓冲）；结果文件在 waitpid 返回后读取，同页缓存立即可见。
"""
from __future__ import annotations

import os
import pickle
import tempfile
from typing import Callable, Dict, List, Tuple


class ParEvalError(RuntimeError):
    """fork 批评估基础设施故障（非仿真语义事件）。"""


def par_eval_batch(clone_fn: Callable, drain_fn: Callable, est0, items: List,
                   n_workers: int, max_events: int) -> Dict[int, Tuple]:
    """fork 批评估一个决策点的全部推演项（含第 0 项=base 动作）。

    clone_fn(est0) -> 推演副本（=search._clone_forecast_eng）；
    drain_fn(est, act, [budget]) -> Fs（=forecast_drain 签名，budget 为
    单元素列表、drain 内部原位扣减）。items=[(idx, act), ...]。
    返回 {idx: (events_used, truncated, Fs)}：
    - events_used = 满预算推演实际消耗（截断时 =max_events；undrainable
      等提前 return None 的路径 =0，与 forecast_drain 的扣减语义一致）；
    - truncated 仅在"预算耗尽型 None"为 True（budget 被清零所致）。
    """
    items = list(items)
    if n_workers < 2 or len(items) < 2:
        raise ParEvalError("并行度不足（n_workers>=2 且 items>=2）")
    chunks = [items[k::n_workers] for k in range(n_workers)]
    chunks = [c for c in chunks if c]
    tmpdir = tempfile.mkdtemp(prefix="cq_par_")
    jobs = []                               # (path, pid, chunk)
    try:
        for w, chunk in enumerate(chunks):
            path = os.path.join(tmpdir, f"r{w}.pkl")
            pid = os.fork()
            if pid == 0:                    # ---- child ----
                status = 0
                try:
                    out = []
                    for idx, act in chunk:
                        est = clone_fn(est0)
                        b = [max_events]    # 满预算（重放裁决交给主进程）
                        Fs = drain_fn(est, act, b)
                        used = max_events - b[0]
                        trunc = Fs is None and used >= max_events
                        out.append((idx, used, trunc, Fs))
                    with open(path, "wb") as f:
                        pickle.dump(out, f)
                except BaseException:
                    status = 1
                os._exit(status)
            jobs.append((path, pid, chunk))
        results: Dict[int, Tuple] = {}
        for path, pid, chunk in jobs:
            _pid, st = os.waitpid(pid, 0)
            if st != 0:
                raise ParEvalError(f"child pid={pid} 异常退出 st={st}")
            try:
                with open(path, "rb") as f:
                    rows = pickle.load(f)
            except Exception as e:
                raise ParEvalError(f"结果读取失败 {path}: {e}")
            if {r[0] for r in rows} != {i for i, _a in chunk}:
                raise ParEvalError(f"child 结果下标不完整 {path}")
            for idx, used, trunc, Fs in rows:
                results[idx] = (used, trunc, Fs)
        return results
    finally:
        for path, _pid, _c in jobs:
            try:
                os.unlink(path)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
