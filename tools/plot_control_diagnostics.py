"""控制诊断科学曲线：所有样本，明确目标、下发参考、实际状态、回退与拒解。"""
import argparse,collections,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':9,'figure.dpi':130})
STATUS={0:'严格',1:'宽松',2:'求解异常',3:'路径低于0.35m',4:'解跳变',5:'角窗',6:'精度/速度门'}

def norm(a):return float(np.linalg.norm(a))
def read_case(path,item):
 rows=[json.loads(s) for s in (path/'diagnostics.jsonl').read_text().splitlines()]
 states=[r for r in rows if r['kind']=='state'];events=[r for r in rows if r['kind']=='solve'];applies=[r for r in rows if r['kind']=='wb_apply']
 assert states
 times=np.array([r['t'] for r in states]);assert np.all(np.diff(times)>0)
 vec=lambda key:np.array([r[key] for r in states],float)
 ref=vec('p_reference');actual=vec('p_actual');base=vec('p_command_actualbase');nominal=vec('p_nominal_actual')
 np.testing.assert_allclose((base-ref)+(nominal-base)+(actual-nominal),actual-ref,atol=1e-12)
 raw,cap,torque=vec('tau_raw'),vec('tau_cap'),vec('tau');np.testing.assert_allclose(np.clip(raw,-cap,cap),torque,atol=1e-9)
 assert np.isfinite(actual).all() and np.all(cap>=0)
 planned=np.array([r['planned'] for r in states]);last=states[-1];th=last['goal_time'] if planned.any() else None
 scope=min(item['input_end_s'],item['bounce_labels'][1] if len(item['bounce_labels'])>1 else item['input_end_s'])
 covered=bool(th is not None and times[0]<=th<=min(times[-1]+.003,scope))
 mode=path.parent.name;sample=path.name;metric={'mode':mode,'sample_id':sample,'trajectory_id':item['trajectory_id'],'planned':bool(planned.any()),'hit_time_s':th,'hit_time_covered':covered,'fallback':any(r['fallback'] for r in states),'fallback_time_s':next((r['t'] for r in states if r['fallback']),None),'solver_calls':len(events),'gateway_accepted':sum(r['accepted'] for r in events),'applied_solutions':sum(r.get('applied',False) for r in events) if mode=='nmpc6' else len(applies),'reject_reasons':dict(collections.Counter(str(STATUS.get(r['reason'],r['reason'])) for r in events if not r['accepted']))}
 applied_events=[r for r in events if r.get('applied')] if mode=='nmpc6' else [r for r in events if r['accepted'] and any(abs(r['t']-z['t'])<1e-9 for z in applies)]
 if covered:
  i=int(np.argmin(abs(times-th)));s=states[i];goal=np.array(s['goal']);true=np.loadtxt(item['csv'],delimiter=',',skiprows=1);ball=np.array([np.interp(th,true[:,0],true[:,j]) for j in (1,2,3)])
  eb=base[i]-ref[i];ea=nominal[i]-base[i];er=actual[i]-nominal[i];eplan=ref[i]-goal
  metric.update(at_t_s=float(times[i]),active_at_hit=s['active'],goal_m=goal.tolist(),ball_at_hit_m=ball.tolist(),prediction_target_error_m=norm(ball-goal),actual_to_goal_m=norm(actual[i]-goal),reference_to_goal_m=norm(eplan),actual_to_reference_m=norm(actual[i]-ref[i]),base_tracking_m=norm(eb),arm_tracking_m=norm(ea),body_pose_m=norm(er),max_joint_error_deg=float(np.max(abs(vec('q_cmd')[i]-vec('q')[i]))*180/np.pi),plan_age_s=None if s['plan_issued_t']<0 else float(times[i]-s['plan_issued_t']),lead_s=s['tau_lead'],contact_node_offset_s=None if s['contact_time']<0 else s['contact_time']-th)
  window=planned&(times>=th-.15)&(times<=th+.05)
  clipped=abs(raw)>cap+1e-8;metric['saturation_fraction_per_joint']=np.mean(clipped[window],axis=0).tolist() if window.any() else [0]*5
  labels=['参考未到任务点','底盘执行','机械臂执行','机身姿态'];metric['largest_vector_term']=labels[int(np.argmax([norm(eplan),norm(eb),norm(ea),norm(er)]))]
  if s['plan_issued_t']>=0:
   metric.update(accepted_goal_staleness_m=norm(np.array(s['accepted_goal'])-goal),reference_to_accepted_goal_m=norm(ref[i]-s['accepted_goal']))
  same_plan=np.array([z['active']==mode and abs(z['plan_issued_t']-s['plan_issued_t'])<1e-9 for z in states])&(times>=th-.25)&(times<=th+.10)
  if same_plan.any():
   ids=np.flatnonzero(same_plan);best=ids[np.argmin(np.linalg.norm(ref[ids]-goal,axis=1))]
   metric.update(reference_closest_time_offset_s=float(times[best]-th),reference_closest_gap_m=norm(ref[best]-goal))
  before=[r for r in applied_events if r['t']<=th]
  if before:
   e=before[-1];metric.update(last_applied_t_s=e['t'],last_applied_ms=e['ms'],last_applied_endpoint_error_m=norm(np.array(e['endpoint_fk'])-e['target']))
   if 'dense_endpoint_fk' in e:metric.update(last_dense_endpoint_error_m=norm(np.array(e['dense_endpoint_fk'])-e['target']),last_dense_vs_solver_m=norm(np.array(e['dense_endpoint_fk'])-e['endpoint_fk']),last_applied_status=STATUS.get(e['reason'],e['reason']))
 metric['upper_gate_rejected']=sum(r['accepted'] for r in events)-len(applied_events)
 return states,events,applies,metric

def draw_case(states,events,applies,metric,out):
 t=np.array([r['t'] for r in states]);th=metric['hit_time_s'];x=t-(th or 0);planned=np.array([r['planned'] for r in states]);fallback=np.array([r['fallback'] for r in states]);mode=metric['mode'];name=metric['sample_id']
 arr=lambda key:np.array([r[key] for r in states],float)
 def decorate(ax):
  if th is not None:ax.axvline(0,color='#333',ls=':',lw=1)
  if th is not None:ax.set_xlim(x[0],min(x[-1],.25))
  if fallback.any():ax.axvspan(x[np.flatnonzero(fallback)[0]],x[-1],color='#ffbd69',alpha=.18,label='已回退pfly')
  ax.grid(alpha=.25);ax.set_xlabel('相对计划击球时刻 / s' if th is not None else '时间 / s')
 def done(fig,suffix):
  fig.suptitle(f'{mode} / {name}  |  击球时实际执行：{metric.get("active_at_hit","未覆盖/无计划")}',fontsize=13);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/(name+'_'+suffix+'.png'));plt.close(fig)
 target=arr('goal');target[~planned]=np.nan;accepted=arr('accepted_goal');accepted[np.array([r['plan_issued_t']<0 for r in states])]=np.nan
 actual,ref=arr('p_actual'),arr('p_reference');fig,axs=plt.subplots(4,1,figsize=(12,11),sharex=True)
 for j,axis in enumerate('XYZ'):
  axs[j].plot(x,target[:,j],color='#d1495b',lw=1.4,label='击球目标（预测/锁止）');axs[j].plot(x,accepted[:,j],color='#8f64b0',ls=':',label='最后被采用解的目标');axs[j].plot(x,ref[:,j],color='#e69f00',label='下发参考的拍心FK');axs[j].plot(x,actual[:,j],color='#087e8b',label='实际拍心');axs[j].set_ylabel(f'{axis} / m')
  if metric['hit_time_covered']:axs[j].scatter([0],[metric['ball_at_hit_m'][j]],c='black',s=35,marker='x',label='实测球@击球时刻（离线）')
  decorate(axs[j])
 axs[0].legend(ncol=3,fontsize=8)
 parts=[('p_reference','goal','参考-目标','#d1495b'),('p_command_actualbase','p_reference','底盘跟踪','#7868e6'),('p_nominal_actual','p_command_actualbase','臂跟踪','#e69f00'),('p_actual','p_nominal_actual','机身姿态','#777')]
 for a,b,label,c in parts:axs[3].plot(x,np.linalg.norm(arr(a)-arr(b),axis=1)*100,label=label,color=c)
 axs[3].plot(x,np.linalg.norm(actual-arr('goal'),axis=1)*100,label='实际-目标',c='#087e8b',lw=1.8);axs[3].set_ylim(0,min(150,max(30,np.nanpercentile(np.linalg.norm(actual-arr('goal'),axis=1)*100,95))));axs[3].set_ylabel('向量范数 / cm（不可相加）');decorate(axs[3]);axs[3].legend(ncol=5,fontsize=8);done(fig,'task')
 fig,axs=plt.subplots(3,2,figsize=(13,10));cmd=arr('q_cmd')*180/np.pi;q=arr('q')*180/np.pi
 for j,ax in enumerate(axs.flat[:5]):
  delayed=np.interp(t-.009,t,cmd[:,j]);ax.plot(x,cmd[:,j],label='下发参考（含前移）',c='#e69f00');ax.plot(x,delayed,label='参考延迟9ms',ls=':',c='#888');ax.plot(x,q[:,j],label='实际',c='#087e8b');ax.set_ylabel(f'J{j+1} / °');decorate(ax)
 axs[0,0].legend(fontsize=8)
 ax=axs[2,1]
 for e in events:ax.scatter(e['t']-(th or 0),e['ms'],c='#218c74' if e['accepted'] else '#c44536',marker='o' if e.get('applied',e['accepted']) else 'x',s=13)
 ax.axhline(20,c='#333',ls=':',label='50Hz周期20ms');ax.set_ylabel('单次求解 / ms');decorate(ax);ax.legend(fontsize=8);done(fig,'joints')
 fig,axs=plt.subplots(4,2,figsize=(13,12));base=arr('base');bref=arr('base_ref');v=arr('v_base_world');vref=arr('v_base_ref')
 for j in range(2):
  axs[0,j].plot(x,bref[:,j],c='#e69f00',label='位置参考');axs[0,j].plot(x,base[:,j],c='#087e8b',label='实际');axs[0,j].set_ylabel(f'底盘 {"XY"[j]} / m');decorate(axs[0,j]);axs[0,j].legend(fontsize=8)
  axs[1,j].plot(x,vref[:,j],c='#e69f00',label='速度参考');axs[1,j].plot(x,v[:,j],c='#087e8b',label='实际');axs[1,j].set_ylabel(f'底盘 V{"XY"[j]} / m/s');decorate(axs[1,j])
 for j,ax in enumerate([axs[2,0],axs[2,1],axs[3,0]]):
  ax.plot(x,arr('tau_raw')[:,j],c='#e69f00',label='延迟后PD所需力矩');ax.plot(x,arr('tau')[:,j],c='#087e8b',label='限幅后力矩');ax.plot(x,arr('tau_cap')[:,j],c='#aaa',ls=':',label='当步力矩上限');ax.plot(x,-arr('tau_cap')[:,j],c='#aaa',ls=':');ax.set_ylabel(f'J{j+1} / Nm');decorate(ax);ax.legend(fontsize=7)
 ax=axs[3,1];ax.plot(x,arr('tau_lead')*1000,label='臂采样前移',c='#d1495b');age=np.array([r['t']-r['plan_issued_t'] if r['plan_issued_t']>=0 else np.nan for r in states]);ax.plot(x,age*1000,label='采用计划年龄',c='#7868e6');ax.set_ylabel('ms');decorate(ax);ax.legend(fontsize=8);done(fig,'execution')

 # 全部五关节速度与实际力矩，避免只看位置而遗漏加速段失配。
 fig,axs=plt.subplots(5,2,figsize=(13,15))
 for j in range(5):
  axs[j,0].plot(x,arr('dq_cmd')[:,j],label='速度参考（含前移）',c='#e69f00');axs[j,0].plot(x,arr('dq')[:,j],label='实际速度',c='#087e8b');axs[j,0].set_ylabel(f'J{j+1} / rad/s');decorate(axs[j,0])
  axs[j,1].plot(x,arr('tau_raw')[:,j],label='限幅前需求',c='#e69f00');axs[j,1].plot(x,arr('tau')[:,j],label='实际输出',c='#087e8b');axs[j,1].plot(x,arr('tau_cap')[:,j],ls=':',c='#888',label='当步上限');axs[j,1].plot(x,-arr('tau_cap')[:,j],ls=':',c='#888');axs[j,1].set_ylabel(f'J{j+1} / Nm');decorate(axs[j,1])
 axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8);done(fig,'speed_effort')

def stats(xs):
 a=np.array([v for v in xs if v is not None],float);return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p95':float(np.percentile(a,95)) if len(a) else None}

def draw_examples(examples,out):
 fig,axs=plt.subplots(4,2,figsize=(14,11),sharex=True)
 for col,(states,metric) in enumerate(examples):
  t=np.array([r['t'] for r in states])-metric['hit_time_s'];arr=lambda k:np.array([r[k] for r in states])
  for j,axis in enumerate('XYZ'):
   ax=axs[j,col];ax.plot(t,arr('goal')[:,j],c='#d1495b',label='击球目标');ax.plot(t,arr('p_reference')[:,j],c='#e69f00',label='下发参考FK');ax.plot(t,arr('p_actual')[:,j],c='#087e8b',label='实际拍心');ax.set_ylabel(axis+' / m')
  axs[0,col].set_title(f"{metric['mode']} / {metric['sample_id']}（诊断实例）");axs[0,col].legend(ncol=3,fontsize=8)
  ax=axs[3,col]
  for key,label,color in [('p_reference','参考-目标','#e69f00'),('p_actual','实际-目标','#087e8b')]:ax.plot(t,np.linalg.norm(arr(key)-arr('goal'),axis=1)*100,label=label,c=color)
  ax.set_ylabel('拍心位置差 / cm');ax.set_xlabel('相对球到达计划击球点的时间 / s');ax.legend(fontsize=8)
  for row in range(4):axs[row,col].axvline(0,c='#333',ls=':');axs[row,col].set_xlim(-.25,.15);axs[row,col].grid(alpha=.2)
 fig.suptitle('击球附近：目标 → 下发参考 → 实际状态\n竖线为球到达时刻；击球前后无需持续停留在目标点',fontsize=13);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'examples_zoom.png');plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--metrics-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 items={r['sample_id']:r for r in json.loads((a.source/'input_manifest.json').read_text())['items']};metrics=[];examples=[]
 for mode in ('nmpc6','wbmpc'):
  out=a.out/'plots'/mode;out.mkdir(parents=True,exist_ok=True)
  for sample,item in items.items():
   dest=a.root/mode/sample
   if not (dest/'run.json').exists():continue
   s,e,ap,m=read_case(dest,item);metrics.append(m)
   if (mode,sample) in (('nmpc6','test_01'),('wbmpc','test_08')):examples.append((s,m))
   if not a.metrics_only:draw_case(s,e,ap,m,out)
   print('processed',mode,sample,flush=True)
 groups={}
 for mode in ('nmpc6','wbmpc'):
  rr=[r for r in metrics if r['mode']==mode];covered=[r for r in rr if r['hit_time_covered']];pure=[r for r in covered if r['active_at_hit']==mode]
  g={'trials':len(rr),'planned':sum(r['planned'] for r in rr),'covered':len(covered),'mpc_active_at_hit':len(pure),'fallback_trials':sum(r['fallback'] for r in rr),'largest_term_all':dict(collections.Counter(r['largest_vector_term'] for r in covered)),'largest_term_mpc_active':dict(collections.Counter(r['largest_vector_term'] for r in pure)),'reject_reasons':dict(sum((collections.Counter(r['reject_reasons']) for r in rr),collections.Counter()))}
  for subgroup,rs in [('all_covered',covered),('mpc_active',pure)]:g[subgroup]={k:stats([r.get(k) for r in rs]) for k in ('prediction_target_error_m','actual_to_goal_m','reference_to_goal_m','actual_to_reference_m','base_tracking_m','arm_tracking_m','body_pose_m','plan_age_s','last_applied_endpoint_error_m','last_dense_vs_solver_m','contact_node_offset_s','max_joint_error_deg','reference_closest_time_offset_s','reference_closest_gap_m','accepted_goal_staleness_m','reference_to_accepted_goal_m')}
  groups[mode]=g
 (a.out/'summary.json').write_text(json.dumps({'groups':groups,'cases':metrics,'protocol':{'samples':'same historical35 per controller, serial diagnostic rerun','state_period_s':.005,'state_phase':'before current control action/physical step','frame_decomposition':'signed vectors add exactly; component norms do not add','labels':'second-bounce/end only offline scope; not solver input','original_video_results':'not replaced; solver time gates may change diagnostic run outcomes'}},indent=2)+'\n')
 draw_examples(examples,a.out)
 fig,axs=plt.subplots(2,2,figsize=(13,10));colors={'nmpc6':'#087e8b','wbmpc':'#d1495b'}
 for mode,c in colors.items():
  rs=[r for r in metrics if r['mode']==mode and r['hit_time_covered']];axs[0,0].scatter([r['prediction_target_error_m']*100 for r in rs],[r['actual_to_goal_m']*100 for r in rs],c=c,label=mode,alpha=.8)
  pure=[r for r in rs if r['active_at_hit']==mode]
  data=[[r[k]*100 for r in pure] for k in ('reference_to_goal_m','base_tracking_m','arm_tracking_m','body_pose_m')]
  means=[np.mean(v) if v else 0 for v in data];positions=np.arange(4)+(-.18 if mode=='nmpc6' else .18);axs[0,1].bar(positions,means,width=.36,label=f'{mode}（击球时MPC仍执行）',color=c)
 axs[0,0].set(xlabel='预测目标与实测球差 / cm',ylabel='实际拍心与目标差 / cm',title='预测误差与机器人控制误差');axs[0,0].legend();axs[0,1].set_xticks(range(4),['参考-目标','底盘跟踪','机械臂跟踪','机身姿态']);axs[0,1].set_ylabel('向量范数均值 / cm（不可相加）');axs[0,1].legend(fontsize=8)
 for j,mode in enumerate(('nmpc6','wbmpc')):
  g=groups[mode];axs[1,j].bar(['全部','有计划','有时刻覆盖','MPC仍执行','发生回退'],[g['trials'],g['planned'],g['covered'],g['mpc_active_at_hit'],g['fallback_trials']],color=colors[mode]);axs[1,j].set_title(mode+'：先区分是否实际由MPC执行');axs[1,j].set_ylabel('样本数')
 for ax in axs.flat:ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(a.out/'overview.png');plt.close(fig)
 html='''<!doctype html><meta charset="utf-8"><title>MPC控制诊断</title><style>body{font:16px sans-serif;max-width:1500px;margin:24px auto;background:#f2f5f8;color:#17233a}select{font-size:18px;margin:8px}img{width:100%;background:white;margin:10px 0}</style><h1>实测test：MPC目标、参考与实际状态</h1><p>橙色阴影表示已经回退pfly；原视频结果与本次串行诊断重跑分开。图中分量范数不能相加。</p><select id="mode"><option>nmpc6</option><option>wbmpc</option></select><select id="sample"></select><div id="figures"></div><script>for(let n=1;n<=35;n++){let o=document.createElement('option');o.value='test_'+String(n).padStart(2,'0');o.textContent=o.value;sample.append(o)}function show(){figures.innerHTML=['task','joints','execution','speed_effort'].map(s=>`<img src="plots/${mode.value}/${sample.value}_${s}.png">`).join('')}mode.onchange=sample.onchange=show;show();</script>'''
 (a.out/'index.html').write_text(html);print(json.dumps(groups,indent=2),flush=True)
if __name__=='__main__':main()
