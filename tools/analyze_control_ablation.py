"""配对消融的统一误差、有限拍面几何拦截、17轴状态与目标曲线。"""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
import mujoco
from interception_metrics import InterceptionProtocol,InterceptionSample,evaluate_interception
VARIANTS=['baseline','feasible','clock','feedforward','gains','base_feedback','servo_mpc','chassis_model','chassis_recovery','chassis_guard']
LABELS={'baseline':'原基线','feasible':'可行性与验收','clock':'＋统一快照/时钟','feedforward':'＋完整逆动力学前馈','gains':'＋辨识增益','base_feedback':'＋底盘位置反馈','servo_mpc':'＋执行器状态预测','chassis_model':'底盘执行动态预测','chassis_recovery':'＋途中走廊恢复','chassis_guard':'＋旧计划失效回退'}

def contact(model,data,states,item,hit,step=.001,*,protocol=None):
 if hit is None:return {'contact':False,'approaching':False,'min_gap_m':None}
 gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'racket_face');bgid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ball_geom')
 if protocol is None:protocol=InterceptionProtocol(tuple(model.geom_size[gid]),float(model.geom_size[bgid,0]),step,endpoint_step_fraction=.1)
 raw=np.loadtxt(item['csv'],delimiter=',',skiprows=1)
 times=np.array([r['t'] for r in states]);poses=np.array([r['qpos'] for r in states]);vels=np.array([r['qvel'] for r in states]);tan=np.empty(model.nv);rv=np.empty(6)
 end=min(item['input_end_s'],item['bounce_labels'][1] if len(item['bounce_labels'])>1 else item['input_end_s'],times[-1])
 def racket_velocity():
  mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_GEOM,gid,rv,0)
  return rv[3:]
 def sample(now):
  i=int(np.clip(np.searchsorted(times,now,side='right')-1,0,len(times)-2));delta=times[i+1]-times[i];u=float(np.clip((now-times[i])/delta,0,1))
  data.qpos[:]=poses[i];mujoco.mj_differentiatePos(model,tan,delta,poses[i],poses[i+1]);mujoco.mj_integratePos(model,data.qpos,tan,u*delta);data.qvel[:]=(1-u)*vels[i]+u*vels[i+1]
  mujoco.mj_forward(model,data)
  ball=np.array([np.interp(now,raw[:,0],raw[:,j]) for j in (1,2,3)]);k=int(np.clip(np.searchsorted(raw[:,0],now,side='right')-1,0,len(raw)-2));vb=(raw[k+1,1:4]-raw[k,1:4])/(raw[k+1,0]-raw[k,0])
  return InterceptionSample(ball,vb,data.geom_xpos[gid],data.geom_xmat[gid].reshape(3,3),racket_velocity,states[i]['planned'])
 geometry=evaluate_interception(sample,hit_time_s=hit,start_s=times[0],end_s=end,protocol=protocol,minimum_initial_m=1e9)
 return {'contact':geometry.contact,'approaching':geometry.approaching,'min_gap_m':geometry.min_gap_m,'first_contact_s':geometry.first_contact_s,'step_s':protocol.interpolation_step_s}

def motor_audit(states):
 for s in states:
  assert len(s['motors'])==17
  for m in s['motors']:
   assert np.isfinite([m[k] for k in ('q','dq','q_target','dq_target','raw','applied','cap')]).all()
   assert abs(np.clip(m['raw'],-m['cap'],m['cap'])-m['applied'])<1e-9

def plots(states,metric,out):
 from plot_control_diagnostics import plt
 t=np.array([s['t'] for s in states]);hit=metric['hit_time_s'];x=t-(hit or 0);sel=(x>=-.8)&(x<=.2) if hit is not None else np.ones(len(t),bool)
 if not sel.any():sel=np.ones(len(t),bool)
 x=x[sel];ss=[s for s,k in zip(states,sel) if k]
 def arr(k):return np.array([s[k] for s in ss])
 def decoration(ax):
  ax.grid(alpha=.2)
  if hit is not None:ax.axvline(0,c='#555',ls=':')
  ax.set_xlabel('相对计划击球时刻 / s' if hit else '时间 / s')
 def finish(fig,name):
  fig.suptitle(f"{metric['mode']} / {metric['sample_id']} / {LABELS[metric['variant']]}\n实际控制：{metric.get('active_at_hit','无计划/未覆盖')}；位置/速度为真实状态，力矩为仿真执行器输出",fontsize=12);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/f'{metric["sample_id"]}_{name}.png');plt.close(fig)
 fig,axs=plt.subplots(4,1,figsize=(12,10),sharex=True)
 for j in range(3):
  for key,label,c in [('goal','击球目标','#d1495b'),('p_reference','规划拍心（含底盘响应）' if metric['variant'].startswith('chassis_') else '下发参考FK','#e69f00'),('p_actual','实际拍心','#087e8b')]:axs[j].plot(x,arr(key)[:,j],label=label,c=c)
  axs[j].set_ylabel('XYZ'[j]+' / m');decoration(axs[j])
 for key,label,c in [('p_reference','参考-目标','#e69f00'),('p_actual','实际-目标','#087e8b')]:axs[3].plot(x,np.linalg.norm(arr(key)-arr('goal'),axis=1)*100,label=label,c=c)
 axs[3].set_ylabel('距离 / cm');decoration(axs[3]);axs[0].legend(ncol=3);axs[3].legend();finish(fig,'task')
 for name,axes in [('arm',list(range(12,17))),('steer',[0,3,6,9]),('wheels',[1,2,4,5,7,8,10,11])]:
  ncols=2 if name=='wheels' else 3;fig,axs=plt.subplots(len(axes),ncols,figsize=(15,len(axes)*2.1),sharex=True,squeeze=False)
  for row,a in enumerate(axes):
   m=[s['motors'][a] for s in ss];marr=lambda key:np.array([r[key] for r in m]);offset=0
   if name!='wheels':
    ax=axs[row,0];ax.plot(x,marr('q_target')*180/np.pi,label='下发位置');ax.plot(x,marr('q_effective')*180/np.pi,ls=':',label='延迟后目标');ax.plot(x,marr('q')*180/np.pi,label='实际位置');ax.set_ylabel(m[0]['name']+' / °');decoration(ax);offset=1
   ax=axs[row,offset];ax.plot(x,marr('dq_target'),label='下发速度');
   if name!='wheels':ax.plot(x,marr('dq_effective'),ls=':',label='延迟后目标')
   ax.plot(x,marr('dq'),label='实际速度');ax.set_ylabel(m[0]['name']+' / rad/s');decoration(ax)
   ax=axs[row,offset+1];ax.plot(x,marr('tau_ff'),label='PI力矩命令（接口tau_ff）' if name=='wheels' else '前馈分量',alpha=.7);ax.plot(x,marr('raw'),label='延迟后总需求',lw=.8);ax.plot(x,marr('applied'),label='限幅后输出');ax.plot(x,marr('cap'),ls=':',c='#888',label='原限矩');ax.plot(x,-marr('cap'),ls=':',c='#888');ax.set_ylabel('Nm');decoration(ax)
  for ax in axs[0]:ax.legend(fontsize=7,ncol=2)
  finish(fig,name)

def main(a):
 from plot_control_diagnostics import read_case,stats,plt
 a.out.mkdir(parents=True,exist_ok=True);items={r['sample_id']:r for r in json.loads((a.source/'input_manifest.json').read_text())['items']};model=mujoco.MjModel.from_xml_path(str(a.team/'sckd_sim/models/generated/x4_tennis_demo.mjcf'));data=mujoco.MjData(model);records=[];prior={}
 face=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'racket_face');ball=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ball_geom')
 geometry_protocol=InterceptionProtocol(tuple(model.geom_size[face]),float(model.geom_size[ball,0]),a.geometry_step,getattr(a,'window_before',.35),getattr(a,'window_after',.35),getattr(a,'approach_threshold',-.30),.1)
 if (a.out/'summary.json').exists():prior={(r['variant'],r['mode'],r['sample_id']):r for r in json.loads((a.out/'summary.json').read_text())['cases']}
 for run in sorted((a.root/'runs').glob('*/*/*/run.json')):
  p=run.parent;variant=p.parent.parent.name;mode=p.parent.name;sample=p.name
  if a.variants and variant not in a.variants.split(','):continue
  info=json.loads(run.read_text());states,events,applies,metric=read_case(p,items[sample]);metric['variant']=variant;motor_audit(states)
  if mode=='pfly' and metric['hit_time_covered']:
   s=min(states,key=lambda r:abs(r['t']-metric['hit_time_s']));metric['active_at_hit']='pfly' if s['arm_qp_active'] else 'wait'
  metric['qp_attempts']=states[-1].get('qp_attempts',0);metric['qp_accepts']=states[-1].get('qp_accepts',0)
  metric['request_candidates']=states[-1]['planner_checked'];metric['request_candidates_rejected']=states[-1]['planner_rejected']
  adopted_times={s['plan_issued_t'] for s in states if s['plan_issued_t']>=0};metric['observed_adopted_plan_updates']=len(adopted_times)
  if variant!='baseline':
   accepted=[e for e in events if e.get('applied')];assert all(e['solver_ok'] and e['constraint_violation']<=1e-4 for e in accepted)
  metric['accepted_without_solver_success']=sum(e.get('applied',False) and not e['solver_ok'] for e in events)
  old=prior.get((variant,mode,sample));metric['geometric']=old['geometric'] if old and old['binary_sha256']==info['binary_sha256'] and old['cache_sha256']==info['cache_sha256'] and old.get('geometric_protocol')==geometry_protocol.as_dict() and old['geometric'].get('step_s')==a.geometry_step else contact(model,data,states,items[sample],metric['hit_time_s'],a.geometry_step,protocol=geometry_protocol)
  metric['geometric_protocol']=geometry_protocol.as_dict()
  if metric['hit_time_covered']:
   th=metric['hit_time_s'];near=[s for s in states if th-.15<=s['t']<=th+.05]
   metric['arm_q_error_rmse_deg']=[float(np.sqrt(np.mean([(r['motors'][12+j]['q']-r['motors'][12+j]['q_target'])**2 for r in near]))*180/np.pi) for j in range(5)]
   metric['arm_dq_error_rmse_rad_s']=[float(np.sqrt(np.mean([(r['motors'][12+j]['dq']-r['motors'][12+j]['dq_target'])**2 for r in near]))) for j in range(5)]
  metric['cache_sha256']=info['cache_sha256'];metric['binary_sha256']=info['binary_sha256']
  # same input cannot silently change across stage/controller
  for old in records:
   if old['sample_id']==sample:assert old['cache_sha256']==metric['cache_sha256'] and old['binary_sha256']==metric['binary_sha256']
  records.append(metric)
  dest=a.out/'plots'/variant/mode;dest.mkdir(parents=True,exist_ok=True)
  if a.plot and variant in a.plot_variants.split(',') and (not a.plot_samples or sample in a.plot_samples.split(',')) and not (dest/f'{sample}_wheels.png').exists():plots(states,metric,dest)
  print(variant,mode,sample,metric['geometric']['approaching'],flush=True)
 groups={}
 for variant in VARIANTS:
  for mode in ('nmpc6','pfly'):
   rr=[r for r in records if r['variant']==variant and r['mode']==mode]
   if not rr:continue
   assert len(rr)==35,(variant,mode,len(rr));covered=[r for r in rr if r['hit_time_covered']];active=[r for r in covered if r.get('active_at_hit')==mode]
   candidate=sum(r['request_candidates'] for r in rr);rejected=sum(r['request_candidates_rejected'] for r in rr);calls=sum(r['solver_calls'] for r in rr);adopted=sum(r['applied_solutions'] for r in rr)
   g={'trials':len(rr),'planned':sum(r['planned'] for r in rr),'covered':len(covered),'active_at_hit':len(active),'fallback':sum(r['fallback'] for r in rr),'geometric_contacts':sum(r['geometric']['contact'] for r in rr),'approaching_contacts':sum(r['geometric']['approaching'] for r in rr),'candidate_checks':candidate,'candidate_feasible_fraction':None if candidate==0 else (candidate-rejected)/candidate,'solver_calls':calls,'adopted_solves':adopted if mode=='nmpc6' else sum(r['qp_accepts'] for r in rr),'qp_attempts':sum(r['qp_attempts'] for r in rr),'qp_accepts':sum(r['qp_accepts'] for r in rr),'solver_adoption_fraction':adopted/calls if calls and mode=='nmpc6' else None,'qp_plan_adoption_fraction':sum(r['qp_accepts'] for r in rr)/max(1,sum(r['qp_attempts'] for r in rr)),'observed_adopted_plan_updates':sum(r['observed_adopted_plan_updates'] for r in rr),'accepted_without_solver_success':sum(r['accepted_without_solver_success'] for r in rr)}
   for name,rs in [('all_covered',covered),('active',active)]:g[name]={key:stats([r.get(key) for r in rs]) for key in ['actual_to_goal_m','reference_to_goal_m','actual_to_reference_m','prediction_target_error_m','base_tracking_m','arm_tracking_m','plan_age_s']}
   groups[variant+'/'+mode]=g
 paired={}
 for mode in ('nmpc6','pfly'):
  available=[v for v in VARIANTS if v+'/'+mode in groups]
  common=set(items)
  for v in available:common&={r['sample_id'] for r in records if r['mode']==mode and r['variant']==v and r['hit_time_covered']}
  paired[mode]={'common_covered_count':len(common),'sample_ids':sorted(common),'stages':{v:{key:stats([r[key] for r in records if r['mode']==mode and r['variant']==v and r['sample_id'] in common]) for key in ('actual_to_goal_m','reference_to_goal_m','actual_to_reference_m','base_tracking_m','arm_tracking_m')} for v in available}}
 result={'paired_common':paired,'groups':groups,'cases':records,'protocol':{'same35':True,'fixed_causal_snapshots':True,'state_period_s':.005,'geometry_step_s':a.geometry_step,'geometry':'finite box-sphere intersection with relative normal approach <=-.30m/s, final planned t +/-350ms, before second bounce; all35 denominator' if geometry_protocol.window_before_s==geometry_protocol.window_after_s==.35 and geometry_protocol.approach_threshold_m_s==-.30 else 'finite box-sphere intersection; explicit window and approach threshold in geometry_protocol; before second bounce; all35 denominator','geometry_protocol':geometry_protocol.as_dict(),'physical_hit_rate':None,'pfly_plan_adoption':'QP trajectory planning attempts incl IK/path checks; accepted count at coefficient commit; separate from native QP numerical success','candidate_checks':'all considered planner candidates incl refinements, depends on policy; not a true physical reachability probability','selection':'exploratory repeated historical test, not new blind validation'}}
 (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 visible=[v for v in VARIANTS if any(key.startswith(v+"/") for key in groups)]
 fig,axs=plt.subplots(1,3,figsize=(16,5))
 for mode,c in [('nmpc6','#087e8b'),('pfly','#d1495b')]:
  vv=[v for v in VARIANTS if v+'/'+mode in groups]
  if not vv:continue
  x=np.array([visible.index(v) for v in vv]);gg=[groups[v+'/'+mode] for v in vv]
  axs[0].plot(x,[g['approaching_contacts'] for g in gg],'-o',label=mode,c=c)
  axs[1].plot(x,[g['all_covered']['actual_to_goal_m']['median']*100 for g in gg],'-o',label=mode,c=c)
  axs[2].plot(x,[g['fallback'] for g in gg],'-o',label=mode,c=c)
 for ax,title in zip(axs,['几何拦截（含接近速度门）/ 全35条','有计划且覆盖：实际目标差中位数 / cm','发生回退的试次 / 全35条']):ax.set_xticks(range(len(visible)),[LABELS[v] for v in visible],rotation=30,ha='right',fontsize=8);ax.set_title(title);ax.grid(alpha=.2);ax.legend()
 axs[0].set_ylim(0,35);axs[0].set_yticks(range(0,36,5));axs[1].set_ylim(bottom=0);axs[2].set_ylim(bottom=0)
 fig.tight_layout();fig.savefig(a.out/'overview.png');plt.close(fig)
 html='''<!doctype html><meta charset="utf-8"><title>nmpc6 / pfly 配对消融</title><style>body{font:16px sans-serif;max-width:1500px;margin:20px auto;background:#f2f5f8}select{font-size:18px;margin:8px}img{width:100%;margin:8px 0}p{line-height:1.6}</style><h1>nmpc6 / pfly：目标与实际、电机状态</h1><p>红/橙为目标或参考，蓝绿为实际。五轴机械臂、四舵、八轮分别展示；驱动轮使用速度控制，不画无意义的零位置目标。力矩无独立轨迹目标：展示前馈、延迟后总需求、原限幅后的输出。黑竖线为计划击球时刻。</p><select id="variant"></select><select id="mode"><option>nmpc6</option><option>pfly</option></select><select id="sample"></select><div id="figures"></div><script>const vs=VARIANTS;for(const [v,l] of vs){let o=document.createElement('option');o.value=v;o.textContent=l;variant.append(o)}for(const sid of SAMPLES){let o=document.createElement('option');o.value=sid;o.textContent=sid;sample.append(o)}function show(){if((['base_feedback','chassis_model','chassis_recovery','chassis_guard'].includes(variant.value)&&mode.value!=='nmpc6')){figures.textContent='中间阶段已保存统计与原始状态；完整逐样本电机曲线先展示原基线和辨识增益候选。';return}figures.innerHTML=['task','arm','steer','wheels'].map(k=>`<img src="plots/${variant.value}/${mode.value}/${sample.value}_${k}.png">`).join('')}variant.onchange=mode.onchange=sample.onchange=show;variant.value=vs[vs.length-1][0];show();</script>'''.replace('VARIANTS',json.dumps([(v,LABELS[v]) for v in a.plot_variants.split(',')],ensure_ascii=False)).replace('SAMPLES',json.dumps(a.plot_samples.split(',') if a.plot_samples else sorted(items)))
 if not any(r['mode']=='pfly' for r in records):html=html.replace('<option>pfly</option>','')
 (a.out/'index.html').write_text(html);print(json.dumps(groups,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--team',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--plot',action='store_true');p.add_argument('--plot-samples');p.add_argument('--plot-variants',default='baseline,gains,base_feedback');p.add_argument('--variants');p.add_argument('--geometry-step',type=float,default=.001);p.add_argument('--window-before',type=float,default=.35);p.add_argument('--window-after',type=float,default=.35);p.add_argument('--approach-threshold',type=float,default=-.30);a=p.parse_args();a.team=a.team.resolve();main(a)
