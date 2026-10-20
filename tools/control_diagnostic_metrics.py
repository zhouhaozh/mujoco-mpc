"""控制诊断读取与统计；只依赖 NumPy，不导入绘图库或初始化字体。"""
import collections
import json
import numpy as np

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


def stats(xs):
 a=np.array([v for v in xs if v is not None],float);return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p95':float(np.percentile(a,95)) if len(a) else None}
