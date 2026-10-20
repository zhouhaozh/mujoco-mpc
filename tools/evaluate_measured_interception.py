"""对已录制实测test回放计算几何拦截率；不修改视频/求解器，不宣称真实出球命中。"""
import argparse,json,re
from pathlib import Path
import numpy as np
import mujoco
from interception_metrics import (InterceptionProtocol, InterceptionSample,
                                 box_sphere_gap, evaluate_interception)

def recorded_interception(model,data,rows,item,hit_time,step=.001,*,protocol=None):
 """录像状态的MuJoCo插值适配；数值判定与协议交给纯Evaluator。"""
 gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'racket_face')
 bgid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ball_geom')
 jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'ball_free');qadr=model.jnt_qposadr[jid];vadr=model.jnt_dofadr[jid]
 if protocol is None:protocol=InterceptionProtocol(tuple(model.geom_size[gid]),float(model.geom_size[bgid,0]),step,endpoint_step_fraction=.5)
 raw=np.loadtxt(item['csv'],delimiter=',',skiprows=1);source_times=raw[:,0]
 end=min(item['input_end_s'],item['bounce_labels'][1] if len(item['bounce_labels'])>1 else item['input_end_s'])
 stamps=np.array([r['t'] for r in rows]);poses=np.array([r['qpos'] for r in rows]);vels=np.array([r['qvel'] for r in rows]);tangent=np.empty(model.nv);rv=np.empty(6)
 def racket_velocity():
  mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_GEOM,gid,rv,0)
  return rv[3:]
 def sample(now):
  index=max(0,min(len(rows)-2,int(np.searchsorted(stamps,now,side='right')-1)))
  a,b=stamps[index:index+2];u=float(np.clip((now-a)/(b-a),0,1))
  data.qpos[:]=poses[index];mujoco.mj_differentiatePos(model,tangent,b-a,poses[index],poses[index+1]);mujoco.mj_integratePos(model,data.qpos,tangent,u*(b-a))
  data.qvel[:]=(1-u)*vels[index]+u*vels[index+1]
  ball=np.array([np.interp(now,source_times,raw[:,j]) for j in (1,2,3)])
  k=max(0,min(len(raw)-2,int(np.searchsorted(source_times,now,side='right')-1)));vb=(raw[k+1,1:]-raw[k,1:])/(source_times[k+1]-source_times[k])
  data.qpos[qadr:qadr+3]=ball;data.qvel[vadr:vadr+3]=vb
  mujoco.mj_forward(model,data)
  return InterceptionSample(ball,vb,data.geom_xpos[gid],data.geom_xmat[gid].reshape(3,3),racket_velocity,rows[index]['target'] is not None)
 # 保留录像入口从0开始、端点额外裁到参考结束、无计划也统计checked_steps的口径。
 return evaluate_interception(sample,hit_time_s=hit_time,start_s=0,end_s=end,protocol=protocol,maximum_sample_time_s=end)

def evaluate(team,root,step,*,window_before_s=.35,window_after_s=.35,approach_threshold_m_s=-.30):
 model=mujoco.MjModel.from_xml_path(str(team/'sckd_sim/models/generated/x4_tennis_demo.mjcf'))
 data=mujoco.MjData(model);gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'racket_face')
 jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'ball_free');qadr=model.jnt_qposadr[jid];vadr=model.jnt_dofadr[jid]
 ball_gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ball_geom');radius=float(model.geom_size[ball_gid,0]);half=model.geom_size[gid].copy()
 protocol=InterceptionProtocol(tuple(half),radius,step,window_before_s,window_after_s,approach_threshold_m_s,.5)
 items=json.loads((root/'input_manifest.json').read_text())['items'];results=[]
 for mode in ('pfly','nmpc6','wbmpc'):
  for item in items:
   name=item['sample_id'];dest=root/'.records'/mode/name;rows=[json.loads(line) for line in (dest/'frames.jsonl').read_text().splitlines()]
   log=(dest/'control.log').read_text();match=re.search(r'\(plan t_hit=([\d.]+)\)',log);hit_time=float(match[1]) if match else None
   has_plan=any(r['target'] is not None for r in rows)
   geometry=recorded_interception(model,data,rows,item,hit_time,step,protocol=protocol)
   r=dict(mode=mode,sample_id=name,trajectory_id=item['trajectory_id'],planned=has_plan,planned_hit_time_s=hit_time,geometry_contact=geometry.contact,approaching_contact=geometry.approaching,min_surface_gap_m=geometry.min_gap_m,closest=geometry.closest,first_contact_time_s=geometry.first_contact_s,step_s=step,checked_steps=geometry.checked_steps,fallback=any(row['fallback'] for row in rows))
   results.append(r);print(mode,name,int(geometry.contact),int(geometry.approaching),r['min_surface_gap_m'],flush=True)
 groups={}
 for mode in ('pfly','nmpc6','wbmpc'):
  rs=[r for r in results if r['mode']==mode];assert len(rs)==35
  groups[mode]={'trials':35,'planned':sum(r['planned'] for r in rs),'geometry_contacts':sum(r['geometry_contact'] for r in rs),'approaching_contacts':sum(r['approaching_contact'] for r in rs),'fallback_trials':sum(r['fallback'] for r in rs),'geometry_contact_rate':sum(r['geometry_contact'] for r in rs)/35,'approaching_contact_rate':sum(r['approaching_contact'] for r in rs)/35}
 return {'groups':groups,'records':results,'protocol':{'source':'existing measured test video runs, unchanged controller states','robot_state_recording_s':.01,'evaluation_step_s':step,'robot_interpolation':'MuJoCo configuration-space interpolation between recorded qpos; linear qvel','ball':'native measured reference piecewise linear, unchanged alignment','ball_radius_m':radius,'racket_halfsize_m':half.tolist(),'window':'planned final t_hit +/- 0.35s, after plan exists; stop before second annotated ground event' if window_before_s==window_after_s==.35 else 'planned final t_hit window from geometry_protocol; after plan exists; stop before second annotated ground event','criterion':'exact sphere versus finite racket box distance <= ball radius','approach_filter_m_s':approach_threshold_m_s,'physical_hit_rate':None,'denominator':'all35; unplanned counted as failure','geometry_protocol':protocol.as_dict()}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--videos',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--step',type=float,default=.001);p.add_argument('--window-before',type=float,default=.35);p.add_argument('--window-after',type=float,default=.35);p.add_argument('--approach-threshold',type=float,default=-.30);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
 result=evaluate(a.team.resolve(),a.videos.resolve(),a.step,window_before_s=a.window_before,window_after_s=a.window_after,approach_threshold_m_s=a.approach_threshold);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['groups'],indent=2),flush=True)
if __name__=='__main__':main()
