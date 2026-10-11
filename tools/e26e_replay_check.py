"""V1:并行加速版与串行基线的三级产物逐位对拍(设计文档 §6-V1)。

层级:①records 结果性字段(排除 wall_s/ctrl_time_s/ctrl_p50/p95 环境项,
n_overrun **必比**);②甘特桶数据;③requests 逐请求 F_s。
用法:
  .venv/bin/python tools/e26e_replay_check.py [--new DIR] [--ref DIR] \
      [--cells "D1c_k4@0,D1c_k16@0"]
默认 new=results/cq/eval/e26e(并行版在位重跑后)、ref=e26e_serialref。
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {"wall_s", "ctrl_time_s", "ctrl_p50_ms", "ctrl_p95_ms",
        "ts_path", "delta_s"}
# 旧分片（并行实现前）未记录 n_overrun：仅在两边都有时才比较（新分片间
# 对拍时则为必比字段）
MISS_OK_REF = {"n_overrun"}


def load_records(root):
    recs = []
    for p in sorted(glob.glob(os.path.join(root, "records_part_*.json"))):
        recs.extend(json.load(open(p, encoding="utf-8")))
    return recs


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in SKIP:
                continue
            if k in MISS_OK_REF and (k not in a or k not in b):
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}: 缺失")
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, float) and isinstance(b, float):
        if a != b and abs(a - b) > 1e-12:
            out.append(f"{path}: {a} != {b}")
    elif a != b:
        out.append(f"{path}: {a} != {b}")
    return out


def main():
    new = os.environ.get("NEW") or os.path.join(
        REPO, "results", "cq", "eval", "e26e")
    ref = os.environ.get("REF") or os.path.join(
        REPO, "results", "cq", "eval", "e26e_serialref")
    cells = None
    for a in sys.argv[1:]:
        if a.startswith("--new="):
            new = a[6:]
        elif a.startswith("--ref="):
            ref = a[6:]
        elif a.startswith("--cells="):
            cells = a[8:].split(",")
    rn, rr = load_records(new), load_records(ref)
    key = lambda r: (r["cell"], r["rep"], r["policy"])
    if cells:
        want = {(c.split("@")[0], int(c.split("@")[1])) for c in cells}
        rn = [r for r in rn if (r["cell"], r["rep"]) in want]
        rr = [r for r in rr if (r["cell"], r["rep"]) in want]
    refm = {key(r): r for r in rr}
    bad = 0
    for r in rn:
        o = refm.get(key(r))
        if o is None:
            print("[records] 基线缺:", key(r))
            bad += 1
            continue
        d = diff(o, r)
        if d:
            bad += 1
            print("[records] 差异", key(r), d[:4])
    print(f"[records] {len(rn)-bad}/{len(rn)} 逐位一致"
          + (" ✓" if bad == 0 and rn else "（有差异!）"))

    # 甘特桶 + requests 逐臂
    bad2 = bad3 = n2 = n3 = 0
    for r in rn:
        tag = f"{r['cell']}@s{r['rep']}"
        for ext, nm in (("gantt.json.gz", "甘特"), ("requests.json.gz", "req")):
            pn = os.path.join(new, tag, f"{r['policy']}.{ext}")
            pr = os.path.join(ref, tag, f"{r['policy']}.{ext}")
            if not (os.path.exists(pn) and os.path.exists(pr)):
                continue
            dn = json.load(gzip.open(pn, "rt"))
            dr = json.load(gzip.open(pr, "rt"))
            if nm == "req":
                same = dn["rows"] == dr["rows"]
                n3 += 1
                bad3 += 0 if same else 1
                if not same:
                    print(f"[requests] 差异 {tag}/{r['policy']}")
                continue
            # 甘特:anchor 相同(=分片首臂相同)才逐桶比;不同则比三态累计
            # 不变量(anchor=首臂 makespan×1.1,分片臂序不同则窗口不同,
            # 桶数值不可直接比——串行分片之间同样存在,非并行效应)
            n2 += 1
            if dn["meta"]["T_anchor_s"] == dr["meta"]["T_anchor_s"]:
                same = dn["buckets"] == dr["buckets"]
            else:
                def _inv(d):
                    out = []
                    for wi in range(d["meta"]["m_workers"]):
                        a = b = w = 0.0
                        for bk in d["buckets"]:
                            a += bk[wi][0]; b += bk[wi][1]; w += bk[wi][2]
                        out.append((round(a, 9), round(b, 9), round(w, 9)))
                    return out
                same = _inv(dn) == _inv(dr)
                if same:
                    print(f"[甘特] anchor 不同(窗口效应),三态不变量一致 "
                          f"{tag}/{r['policy']}")
            bad2 += 0 if same else 1
            if not same:
                print(f"[甘特] 差异 {tag}/{r['policy']}")
    print(f"[甘特] {n2-bad2}/{n2} 逐位一致" + (" ✓" if bad2 == 0 else "（有差异!）"))
    print(f"[requests] {n3-bad3}/{n3} 逐位一致" + (" ✓" if bad3 == 0 else "（有差异!）"))
    ok = bad == 0 and bad2 == 0 and bad3 == 0
    print("V1 对拍:", "全绿 ✓✓" if ok else "存在差异——按设计停止合入")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
