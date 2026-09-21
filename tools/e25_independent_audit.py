"""E25 三问独立核查：只读记录跨流竞争，重放既有配置并生成独立证据。

不导入此前同题分析工具，不更改仿真内核或策略可见信息。
用法：--stage smoke/full/figures；full 为 10 格×5 窗 FCFS 诊断。
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.cq.engine import CqEngine
from sim.cq.observable import Observable
from sim.cq.policies import GuardedEDF
from sim.cq.profile import batch_compute_s, singleton_K, layer_read_gb
from sim.cq.config import ProfileConfig
from sim.cq.types import RequestSpec
from sim.cq.simrun import run_case
from sim.cq.timeseries import aggregate_run, anchor_T
from sim.cq.trace import sha256_file
from sim.experiments.cq_common import MooncakeSource, default_scenario, TRACE_WIDE_LIMITS, build_policy, setup_matplotlib

OUT = ROOT / 'results/cq/e25_independent'
# 同一 trace/rho 的 specs 在所有 B 和 m 配置间直接复用，避免重新锚定。
CELLS = [(key, b, .6, 4) for key in ('conversation', 'toolagent', 'synthetic') for b in (80, 20)] + [
    ('toolagent', 15, .6, 4), ('conversation', 10, .9, 4),
    ('synthetic', 15, .6, 4), ('toolagent', 20, .6, 8)]


def observed_run(scn, specs):
    """仅在物理世界 advance 前采集区段；原生 advance 原样执行。"""
    obs = Observable(scn, numeric=float, seed=0)
    eng = CqEngine(scn, specs, build_policy('cq_fcfs'), numeric=float,
                    fallback_policy=GuardedEDF(), observable=obs)
    obs.attach(eng)
    storage = eng.w.storage
    storage.record_intervals = True
    native_advance = storage.advance
    rows = []

    def advance(t0, t1):
        if t1 > t0:
            b = float(storage.b_at(t0))
            flows = list(storage.flows.values())
            q = sum(float(f.q_gbps) for f in flows)
            independent = sum(min(float(f.q_gbps), b) for f in flows)
            actual = sum(float(f.rate_gbps) for f in flows)
            assert actual <= b + 1e-8
            assert abs(actual - min(q, b)) < 1e-7
            rows.append([float(t0), float(t1), actual, q, independent])
        native_advance(t0, t1)
    storage.advance = advance
    eng.run(drain_deadline=max(float(s.arrival_s) for s in specs) + 600)
    assert eng.status == 'done'
    expected = sum(float(layer_read_gb(s, scn.profile)) * scn.profile.L for s in specs)
    assert abs(float(storage.actual_integral_gb) - expected) < 1e-7 * max(1, expected)
    return eng, rows


def metrics(eng, rows, scn, specs):
    """分别记录申请超限、共享争用、实际饱和和完整字节工作量。"""
    t = float(eng.w.t)
    b = float(scn.storage.b_at(0))
    m = scn.m_workers
    ts = aggregate_run(eng, anchor_T(t))
    comp = sum(float(w.compute_s) for w in eng.w.workers.values()) / (m*t)
    stall = sum(float(w.stall_s) for w in eng.w.workers.values()) / (m*t)
    share = sum(y-x for x,y,r,q,i in rows if i > b+1e-8) / t
    over = sum(y-x for x,y,r,q,i in rows if q > b+1e-8) / t
    sat = sum(y-x for x,y,r,q,i in rows if r >= .9*b-1e-8) / t
    spans, cur = [], 0
    for x,y,r,q,i in rows:
        if i > b+1e-8:
            cur += y-x
        elif cur:
            spans.append(cur)
            cur = 0
    if cur:
        spans.append(cur)
    rr = ts['rows']
    return dict(T=t, n=len(specs), comp=comp, stall=stall, contention=share,
                request_over=over, sat90=sat,
                contention_longest=max(spans, default=0),
                contention_seconds=share*t,
                served_gb=float(eng.w.storage.actual_integral_gb),
                mean_gbps=float(eng.w.storage.actual_integral_gb)/t,
                bucket_peak=max(r['served_gb']/r['width_s'] for r in rr if r['width_s']>0),
                bucket_width=ts['meta']['delta_s'],
                bucket_sat90=sum(r['width_s'] for r in rr if r['width_s']>0 and r['served_gb']>=.9*r['capacity_gb'])/t,
                ttft=mean(float(r.F_s)-float(r.spec.arrival_s) for r in eng.w.requests.values()))


def self_check():
    """观察器零影响及单流限速/双流竞争定义的独立小世界核查。"""
    scn = default_scenario(B_gbps=20, limits=TRACE_WIDE_LIMITS)
    specs = [RequestSpec(i, 0, 8192, 128, 'audit', F(1), F(5)) for i in range(10)]
    eng, rows = observed_run(scn, specs)
    native = run_case(scn, specs, build_policy('cq_fcfs'), record_intervals=True)
    assert [r.F_s for r in eng.w.requests.values()] == [r.F_s for r in native.w.requests.values()]
    assert eng.w.storage.interval_log == native.w.storage.interval_log
    assert min(200, 20) <= 20  # 单流申请200：两分配器都给20。
    assert min(15, 20) + min(15, 20) > 20  # 两流：独立各15，共享合计20。
    assert metrics(eng, rows, scn, specs)['contention'] > 0


def run(stage):
    """先单窗冒烟，再重放预先列出的完整五窗；随格保存防止中断丢失。"""
    self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    records = []
    profiles = {}
    cells = CELLS if stage == 'full' else [('toolagent',20,.6,4), ('toolagent',15,.6,4)]
    windows = [0,4,9,14,19] if stage == 'full' else [9]
    specs_cache = {}
    for key,b,rho,m in cells:
        if key not in sources:
            src = sources[key] = MooncakeSource(key+'_trace.jsonl')
            p = ProfileConfig()
            raw = [RequestSpec(i,0,r.hit_tokens,r.u_tokens,'audit',1,1) for i,r in enumerate(src.imp.rows)]
            profiles[key] = dict(n=len(raw), mean_compute=mean(float(batch_compute_s([s],p)*p.L) for s in raw),
                mean_T0=mean(float(singleton_K(s,p,F(80),F(200))) for s in raw),
                mean_full_read=mean(float(layer_read_gb(s,p)*p.L) for s in raw),
                lam0=float(src.lam0), derived_sha256=src.imp.derived_sha256,
                raw_sha256=sha256_file(str(ROOT/'mooncake_trace'/src.fname)))
        src=sources[key]
        for w in windows:
            skey=(key,rho,w)
            if skey not in specs_cache:
                specs_cache[skey]=src.window_specs(w,src.lam0*F(str(rho)),F(4),limits=TRACE_WIDE_LIMITS)
            specs,duration=specs_cache[skey]
            scn=default_scenario(B_gbps=b,m=m,alpha=F(4),limits=TRACE_WIDE_LIMITS)
            eng,rows=observed_run(scn,specs)
            rec=dict(trace=key,B=b,rho=rho,m=m,window=w,arrival_span=duration,**metrics(eng,rows,scn,specs))
            records.append(rec)
            print(json.dumps(rec),flush=True)
            if w==9:
                cid=f'{key}_B{b}_r{rho}_m{m}'
                ts=aggregate_run(eng,anchor_T(float(eng.w.t)))
                (OUT/(cid+'_ts.json')).write_text(json.dumps(ts))
                # 合并完全相同速率区段，保留所有争用变化，减小结果体积。
                merged=[]
                for row in rows:
                    if merged and merged[-1][1]==row[0] and merged[-1][2:]==row[2:]:
                        merged[-1][1]=row[1]
                    else:
                        merged.append(row[:])
                (OUT/(cid+'_intervals.json')).write_text(json.dumps(merged))
            (OUT/(stage+'.json')).write_text(json.dumps(dict(profiles=profiles,records=records),indent=2))


def figures():
    """新文件名出图，既有正式图不触碰。"""
    plt=setup_matplotlib()
    data=json.loads((OUT/'full.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharey=True)
    for ri,key in enumerate(('conversation','toolagent','synthetic')):
        for ci,b in enumerate((80,20)):
            ts=json.loads((OUT/f'{key}_B{b}_r0.6_m4_ts.json').read_text())
            rows=[r for r in ts['rows'] if r['width_s']>0]
            ax=axes[ri,ci]
            ax.plot([r['t_start_s'] for r in rows],[r['served_gb']/r['width_s'] for r in rows],color='#20639b',lw=1,label='实际速率（200 桶）')
            ax.axhline(b,color='#ac3b30',ls='--',label='物理容量')
            ax.set_title(f'{key}  /  B={b} GB/s')
            ax.set_ylim(0,85)
            ax.set_ylabel('GB/s')
            ax.set_xlabel('仿真时间（秒）')
            if ri==0 and ci==0: ax.legend(fontsize=8)
    fig.suptitle('独立重放：实际吞吐被容量截住，峰值低于容量不等于没有瞬时竞争\nFCFS · OL · ρ=0.6 · m=4 · 窗口9（各行同一份到达流）',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(ROOT/'docs/figures/fig_e25_independent_bandwidth.png',dpi=160)
    plt.close(fig)
    labels=[]
    vals={k:[] for k in ('contention','stall','comp')}
    ranges={k:[] for k in vals}
    for key,b,rho,m in CELLS:
        rr=[r for r in data['records'] if (r['trace'],r['B'],r['rho'],r['m'])==(key,b,rho,m)]
        labels.append(f'{key[:4]} B{b} ' + (f'ρ{rho} m{m}' if m == 4 else '固定λ m8'))
        for k in vals:
            v=[100*r[k] for r in rr]
            med=median(v)
            vals[k].append(med)
            ranges[k].append((med-min(v),max(v)-med))
    fig,ax=plt.subplots(figsize=(13,6))
    for j,(k,color,title) in enumerate((('contention','#b64b35','跨流竞争时间'),('stall','#dd9a35','NPU 等数据时间'),('comp','#20639b','NPU 计算时间'))):
        xs=[i+(j-1)*.25 for i in range(len(labels))]
        ax.bar(xs,vals[k],width=.24,color=color,label=title,yerr=list(zip(*ranges[k])),capsize=2,error_kw={'lw':.7})
    ax.set_xticks(range(len(labels)),labels,rotation=32,ha='right',fontsize=9)
    ax.set_ylabel('时间占比（%）')
    ax.set_ylim(0,100)
    ax.legend(loc='upper right',fontsize=9)
    ax.set_title('独立核查：存储争用与计算负载分开量\n5 窗中位数；误差线是窗口最小—最大值（不是置信区间）')
    fig.tight_layout()
    fig.savefig(ROOT/'docs/figures/fig_e25_independent_regimes.png',dpi=160)
    plt.close(fig)
    archive_pairs(plt)


def archive_pairs(plt):
    """直接从主矩阵逐窗口配对，保留原始记录指纹，不读取旧分析输出。"""
    path = ROOT/'results/cq/eval/e25/e25_records.json'
    records = json.loads(path.read_text())
    groups = [(key, 'OL', b, 'T') for key in ('conversation','toolagent','synthetic') for b in (80,20)]
    groups += [('synthetic','FW',b,'S') for b in (80,20)]
    pairs = []
    for key,mode,b,theta in groups:
        rr = [r for r in records if r['file']==key+'_trace.jsonl' and r['mode']==mode
              and r['B']==b and r['theta']==theta and r['alpha']==4 and r['cap']=='C'
              and r['profile']=='default' and r['rho']==(.6 if mode=='OL' else None)
              and r['window'] in [0,4,9,14,19] and r['policy'] in ('cq_mpc','cq_local')]
        lookup = {(r['window'],r['policy']):r for r in rr}
        assert len(rr)==len(lookup)==10
        for w in [0,4,9,14,19]:
            m,l = lookup[w,'cq_mpc'],lookup[w,'cq_local']
            pairs.append(dict(trace=key,mode=mode,B=b,theta=theta,window=w,
                ttft_gain_pct=100*(l['ttft_mean_lower']-m['ttft_mean_lower'])/l['ttft_mean_lower'],
                slo_gain_pp=100*(m['slo_rate_interval'][0]-l['slo_rate_interval'][0]),
                local_slo=l['slo_rate_interval'][0],mpc_slo=m['slo_rate_interval'][0],
                code_rev=m['code_rev']))
    (OUT/'archive_pairs.json').write_text(json.dumps(dict(source=str(path.relative_to(ROOT)),
        source_sha256=sha256_file(str(path)),pairs=pairs),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,mode,field,title in [(axes[0],'OL','ttft_gain_pct','OL · θ=T · TTFT 改善（%）'),
                                (axes[1],'FW','slo_gain_pp','Synthetic FW · θ=S · SLO 改善（pp）')]:
        gg=[g for g in groups if g[1]==mode]
        for i,(key,_,b,theta) in enumerate(gg):
            vals=[p[field] for p in pairs if p['trace']==key and p['mode']==mode and p['B']==b]
            ax.scatter([i+(.04*(j-2)) for j in range(len(vals))],vals,color='#20639b',s=28)
            ax.plot([i-.2,i+.2],[median(vals)]*2,color='#ac3b30',lw=2)
        ax.axhline(0,color='gray',lw=.7)
        ax.set_xticks(range(len(gg)),[f'{g[0][:4]} B{g[2]}' for g in gg],rotation=25)
        ax.set_title(title)
        ax.set_ylabel('MPC 相对 local，正值为改善')
    fig.suptitle('主矩阵重新配对：每点一个窗口，红线为配对中位数\n历史归档结果；未重跑 MPC，不构成统计显著性检验',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.87))
    fig.savefig(ROOT/'docs/figures/fig_e25_independent_pairs.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['smoke','full','figures'],required=True)
    stage=p.parse_args().stage
    if stage=='figures': figures()
    else: run(stage)
