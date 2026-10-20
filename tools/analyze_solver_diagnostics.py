"""逐次 NMPC 诊断；按当前请求剩余时间分箱，区分数值失败与执行验收。"""
import argparse, collections, csv, hashlib, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
FONT=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():matplotlib.rcParams.update({'font.family':FontProperties(fname=str(FONT)).get_name(),'axes.unicode_minus':False})
import matplotlib.pyplot as plt

BINS=[(0,.05,'0–50'),(.05,.1,'50–100'),(.1,.2,'100–200'),(.2,.5,'200–500'),(.5,1,'500–1000'),(1,float('inf'),'>1000')]
REASONS={0:'accepted',1:'loose',2:'solver_failed',3:'path_height',4:'solution_jump',5:'angle_window',6:'position_or_speed',7:'constraint_residual',8:'invalid_input',9:'terminal_error'}
def stats(values):
    a=np.array([v for v in values if v is not None and math.isfinite(v)],float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else dict(n=0)
def norm(v):return math.sqrt(sum(x*x for x in v))
def read_solves(p):
    with p.open() as f:
        return [e for line in f if (e:=json.loads(line)).get('kind')=='solve' and e.get('mode')=='nmpc6']
def aggregate(rows):
    failed=[r for r in rows if not r['applied']]
    return dict(attempts=len(rows),failed=len(failed),failure_rate=len(failed)/len(rows) if rows else None,
        statuses=dict(collections.Counter(r['ipopt_status'] for r in rows)),
        rejected_statuses=dict(collections.Counter(r['ipopt_status'] for r in failed)),
        reject_reasons=dict(collections.Counter(r['reject_reason'] for r in failed)),
        worst_violated_constraints=dict(collections.Counter(r['worst_name'] for r in failed if r['residual']>1e-4)),
        iterations=stats([r['iterations'] for r in rows if r['iterations']>=0]),
        failed_iterations=stats([r['iterations'] for r in failed if r['iterations']>=0]),
        residual=stats([r['residual'] for r in rows]),failed_residual=stats([r['residual'] for r in failed]),
        accepted_base_tracking_m=stats([r['base_tracking_m'] for r in rows if r['applied']]),
        failed_base_tracking_m=stats([r['base_tracking_m'] for r in failed]),
        accepted_arm_tracking_m=stats([r['arm_tracking_m'] for r in rows if r['applied']]),
        failed_arm_tracking_m=stats([r['arm_tracking_m'] for r in failed]),
        target_step_m=stats([r['target_step_m'] for r in rows]),
        failed_target_step_m=stats([r['target_step_m'] for r in failed]),
        failed_target_vs_accepted_m=stats([r['target_vs_accepted_m'] for r in failed]))
def analyze(root,out,previous):
    out.mkdir(parents=True,exist_ok=True);rows=[];runs=[];streaks=[];comparisons=[]
    for run in sorted((root/'runs').glob('*/nmpc6/*/run.json')):
        info=json.loads(run.read_text());events=read_solves(run.parent/'diagnostics.jsonl')
        assert all('ipopt_status' in e for e in events)
        v=info['variant'];sample=info['sample_id'];oldpath=previous/'runs'/v/'nmpc6'/sample
        old=read_solves(oldpath/'diagnostics.jsonl') if (oldpath/'diagnostics.jsonl').exists() else []
        same=len(old)==len(events)
        if same:
            for a,b in zip(old,events):
                for key in ['t','target','target_time','state','accepted','applied','reason']:
                    if isinstance(a[key],(list,int,float)):
                        same=same and bool(np.allclose(a[key],b[key],rtol=0,atol=1e-10))
                    else:same=same and a[key]==b[key]
        oldinfo=json.loads((oldpath/'run.json').read_text()) if (oldpath/'run.json').exists() else {}
        comparisons.append(dict(variant=v,sample=sample,attempts=len(events),original_attempts=len(old),same_decisions_targets_states=same,
            same_input=info['input_sha256']==oldinfo.get('input_sha256'),same_cache=info['cache_sha256']==oldinfo.get('cache_sha256')))
        start=None;last=None;count=0
        def close_streak(recovered):
            nonlocal start,last,count
            if start is not None:
                streaks.append(dict(variant=v,sample=sample,start_s=start['t'],end_s=last['t'],attempts=count,
                    span_s=last['t']-start['t'],start_remaining_s=start['remaining_s'],end_remaining_s=last['remaining_s'],
                    recovered=recovered,includes_last200ms=any(0<e['remaining_s']<=.2 for e in events if start['t']<=e['t']<=last['t'])))
            start=None;last=None;count=0
        for e in events:
            assert e['remaining_s']>0
            # 从全部 g 和决策变量重新算残差；null 是 JSON 对无穷界的表示。
            residuals=[]
            for prefix in ['constraint','control']:
                vals=e[prefix+'_values'];lo=e[prefix+'_lower'];hi=e[prefix+'_upper']
                assert len(vals)==len(lo)==len(hi)
                for x,l,u in zip(vals,lo,hi):
                    residuals.append(1e9 if x is None else max(0,(l-x) if l is not None else 0,(x-u) if u is not None else 0))
            if residuals:assert abs(max(residuals)-e['constraint_violation'])<1e-9
            if e['applied']:assert e['solver_ok'] and e['constraint_violation']<=1e-4
            valid=e['tracking_valid']
            row=dict(variant=v,sample=sample,t=e['t'],remaining_s=e['remaining_s'],locked=e['locked'],
                applied=e['applied'],ipopt_status=e['ipopt_status'],iterations=e['ipopt_iterations'],residual=e['constraint_violation'],
                reject_reason=REASONS.get(e['reason'],str(e['reason'])),worst_name=e['worst_constraint']['name'],
                worst_side=e['worst_constraint']['side'],worst_value=e['worst_constraint']['value'],
                worst_lower=e['worst_constraint']['lower'],worst_upper=e['worst_constraint']['upper'],
                base_tracking_m=norm(e['base_tracking_vector']) if valid else None,
                arm_tracking_m=norm(e['arm_tracking_vector']) if valid else None,
                target_step_m=e.get('target_step_m'),target_time_step_s=e.get('target_time_step_s'),
                target_vs_accepted_m=e.get('target_vs_accepted_m'),plan_age_s=e.get('plan_age_s'),solve_ms=e['ms'])
            rows.append(row)
            if not e['applied']:
                if start is None:start=e
                last=e;count+=1
            else:close_streak(True)
        close_streak(False);runs.append(info)
    assert rows
    result=dict(protocol='current request remaining time; (low,high]; attempts weighted; tracking only valid old-plan coverage; FK base/arm components, not sum of norms',groups={})
    for v in sorted({r['variant'] for r in rows}):
        rr=[r for r in rows if r['variant']==v];ss=[s for s in streaks if s['variant']==v]
        last=[r for r in rr if r['remaining_s']<=.2];early=[r for r in rr if r['remaining_s']>.2]
        result['groups'][v]=dict(overall=aggregate(rr),bins={label:aggregate([r for r in rr if lo<r['remaining_s']<=hi]) for lo,hi,label in BINS},
            last200=aggregate(last),before200=aggregate(early),
            fraction_all_failures_in_last200=sum(not r['applied'] for r in last)/sum(not r['applied'] for r in rr),
            streak_attempts=stats([s['attempts'] for s in ss]),streak_span_s=stats([s['span_s'] for s in ss]),
            trajectories_with_3plus_failures=len({s['sample'] for s in ss if s['attempts']>=3}),
            trajectories_with_last200_failure=len({r['sample'] for r in last if not r['applied']}),
            longest_streaks=sorted(ss,key=lambda s:s['attempts'],reverse=True)[:10])
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    (out/'audit.json').write_text(json.dumps(dict(runs=len(runs),attempts=len(rows),residual_recomputed=True,accepted_gates_checked=True,comparisons=comparisons),indent=2)+'\n')
    for name,data in [('attempts',rows),('failure_streaks',streaks)]:
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for v,g in result['groups'].items():
        labels=list(g['bins']);bs=list(g['bins'].values());xx=np.arange(len(bs))
        axs[0,0].plot(xx,[100*b['failure_rate'] if b['failure_rate'] is not None else np.nan for b in bs],'-o',label=v)
        axs[0,1].plot(xx,[b['failed_iterations'].get('median',np.nan) for b in bs],'-o',label=v)
        axs[1,0].plot(xx,[100*b['failed_base_tracking_m'].get('median',np.nan) for b in bs],'-o',label=v+' base')
        axs[1,0].plot(xx,[100*b['failed_arm_tracking_m'].get('median',np.nan) for b in bs],'--o',label=v+' arm')
        axs[1,1].plot(xx,[100*b['failed_target_step_m'].get('p95',np.nan) for b in bs],'-o',label=v)
    for ax,title in zip(axs.flat,['拒收率（%）','拒收尝试：IPOPT迭代次数中位数','拒收尝试：跟踪分量中位数（cm）','拒收尝试：相邻请求目标变化P95（cm）']):
        ax.set_title(title);ax.set_xticks(xx,labels,rotation=25);ax.set_xlabel('距本次请求击球时刻的剩余时间（ms）');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(out/'remaining_time.png',dpi=180);plt.close(fig)
    intro='<h2>主要发现</h2><p>底盘途中走廊越界是最突出的数值冲突；末段更容易拒收，但大多数拒收已在200ms之前发生。最后100ms目标变化为零，拒收仍持续。IPOPT判定不可行是当前非线性问题的求解结果，不是全局不可达证明。</p><table border="1" cellpadding="7"><tr><th>候选</th><th>总体拒收</th><th>最后200ms拒收率</th><th>200ms之前拒收率</th><th>全部拒收中末200ms占比</th></tr>'
    for v,g in result['groups'].items():intro+=f'<tr><td>{v}</td><td>{g["overall"]["failed"]}/{g["overall"]["attempts"]}</td><td>{g["last200"]["failure_rate"]:.1%}</td><td>{g["before200"]["failure_rate"]:.1%}</td><td>{g["fraction_all_failures_in_last200"]:.1%}</td></tr>'
    intro+='</table><p>节点1/2底盘位置为固定绝对时钟走廊，节点3为终端锚；最大残差表示最严重原始越界，不证明删除该约束即可恢复可行。各箱详细迭代、残差、偏差和目标变化分布见summary.json；每次具体上下界见CSV。连续失败跨度是首末失败尝试的仿真时间差，末尾未恢复的记录为截尾。</p>'
    html='<meta charset="utf-8"><h1>nmpc6逐次求解诊断</h1><p>按当前请求剩余时间分箱；失败指未采用，不能全部解释为数学不可行。残差为原始单位，跨约束类型不可当作物理严重度排名。</p><img style="max-width:100%" src="remaining_time.png"><p><a href="attempts.csv">逐次诊断CSV</a> · <a href="failure_streaks.csv">连续失败CSV</a> · <a href="summary.json">完整统计</a> · <a href="audit.json">审计</a></p>'
    html+=intro
    for v,g in result['groups'].items():
        html+=f'<h2>{v}</h2><table border="1" cellpadding="7"><tr><th>剩余ms</th><th>尝试</th><th>拒收</th><th>拒收率</th><th>原始退出状态（拒收）</th><th>最严重越界约束</th></tr>'
        for label,b in g['bins'].items():html+=f'<tr><td>{label}</td><td>{b["attempts"]}</td><td>{b["failed"]}</td><td>{b["failure_rate"]:.1%}</td><td>{b["rejected_statuses"]}</td><td>{b["worst_violated_constraints"]}</td></tr>' if b['attempts'] else ''
        html+='</table>'
    (out/'index.html').write_text(html)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();analyze(a.root,a.out,a.previous)
