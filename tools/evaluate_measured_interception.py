"""对已录制实测test回放计算几何拦截率；不修改视频/求解器，不宣称真实出球命中。"""
import argparse,json,re
from pathlib import Path
import numpy as np
import mujoco

def box_sphere_gap(local,halfsize,radius):
 return float(np.linalg.norm(np.maximum(np.abs(local)-halfsize,0))-radius)

def evaluate(team,root,step):
 model=mujoco.MjModel.from_xml_path(str(team/'sckd_sim/models/generated/x4_tennis_demo.mjcf'))
 data=mujoco.MjData(model);gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'racket_face')
 jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'ball_free');qadr=model.jnt_qposadr[jid];vadr=model.jnt_dofadr[jid]
 ball_gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'ball_geom');radius=float(model.geom_size[ball_gid,0]);half=model.geom_size[gid].copy()
 items=json.loads((root/'input_manifest.json').read_text())['items'];results=[]
 for mode in ('pfly','nmpc6','wbmpc'):
  for item in items:
   name=item['sample_id'];dest=root/'.records'/mode/name;rows=[json.loads(line) for line in (dest/'frames.jsonl').read_text().splitlines()]
   log=(dest/'control.log').read_text();match=re.search(r'\(plan t_hit=([\d.]+)\)',log);hit_time=float(match[1]) if match else None
   has_plan=any(r['target'] is not None for r in rows)
   raw=np.loadtxt(item['csv'],delimiter=',',skiprows=1);source_times=raw[:,0]
   end=min(item['input_end_s'],item['bounce_labels'][1] if len(item['bounce_labels'])>1 else item['input_end_s'])
   rr=[r for r in rows if r['t']<=end];stamps=np.array([r['t'] for r in rows]);poses=np.array([r['qpos'] for r in rows]);vels=np.array([r['qvel'] for r in rows])
   gap_min=float('inf');global_min=float('inf');contact=False;moving_contact=False;first_t=None;best=None;steps=0
   begin=max(0,hit_time-.35) if hit_time is not None else 0;finish=min(end,hit_time+.35) if hit_time is not None else end
   tangent=np.empty(model.nv);rv=np.empty(6)
   # 10ms状态记录之间用MuJoCo配置空间插值；不把这些插值作为观测输入。
   for now in np.arange(begin,finish+step*.5,step):
    if now>end:continue
    index=max(0,min(len(rows)-2,int(np.searchsorted(stamps,now,side='right')-1)))
    a,b=stamps[index:index+2];u=float(np.clip((now-a)/(b-a),0,1))
    data.qpos[:]=poses[index];mujoco.mj_differentiatePos(model,tangent,b-a,poses[index],poses[index+1]);mujoco.mj_integratePos(model,data.qpos,tangent,u*(b-a))
    data.qvel[:]=(1-u)*vels[index]+u*vels[index+1]
    ball=np.array([np.interp(now,source_times,raw[:,j]) for j in (1,2,3)])
    k=max(0,min(len(raw)-2,int(np.searchsorted(source_times,now,side='right')-1)));vb=(raw[k+1,1:]-raw[k,1:])/(source_times[k+1]-source_times[k])
    data.qpos[qadr:qadr+3]=ball;data.qvel[vadr:vadr+3]=vb
    mujoco.mj_forward(model,data);center=data.geom_xpos[gid];R=data.geom_xmat[gid].reshape(3,3);local=R.T@(ball-center)
    gap=box_sphere_gap(local,half,radius);global_min=min(global_min,gap);steps+=1
    planned=rows[index]['target'] is not None
    if not planned or hit_time is None:continue
    if gap<gap_min:gap_min=gap;best={'time_s':float(now),'local_m':local.tolist(),'ball_m':ball.tolist(),'racket_center_m':center.tolist()}
    if gap<=0:
     contact=True
     if first_t is None:first_t=float(now)
     mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_GEOM,gid,rv,0);normal=R[:,1]*(1 if local[1]>=0 else -1);closing=float((vb-rv[3:])@normal)
     if closing<=-.30:moving_contact=True
   r=dict(mode=mode,sample_id=name,trajectory_id=item['trajectory_id'],planned=has_plan,planned_hit_time_s=hit_time,geometry_contact=contact,approaching_contact=moving_contact,min_surface_gap_m=None if not np.isfinite(gap_min) else gap_min,closest=best,first_contact_time_s=first_t,step_s=step,checked_steps=steps,fallback=any(row['fallback'] for row in rows))
   results.append(r);print(mode,name,int(contact),int(moving_contact),r['min_surface_gap_m'],flush=True)
 groups={}
 for mode in ('pfly','nmpc6','wbmpc'):
  rs=[r for r in results if r['mode']==mode];assert len(rs)==35
  groups[mode]={'trials':35,'planned':sum(r['planned'] for r in rs),'geometry_contacts':sum(r['geometry_contact'] for r in rs),'approaching_contacts':sum(r['approaching_contact'] for r in rs),'fallback_trials':sum(r['fallback'] for r in rs),'geometry_contact_rate':sum(r['geometry_contact'] for r in rs)/35,'approaching_contact_rate':sum(r['approaching_contact'] for r in rs)/35}
 return {'groups':groups,'records':results,'protocol':{'source':'existing measured test video runs, unchanged controller states','robot_state_recording_s':.01,'evaluation_step_s':step,'robot_interpolation':'MuJoCo configuration-space interpolation between recorded qpos; linear qvel','ball':'native measured reference piecewise linear, unchanged alignment','ball_radius_m':radius,'racket_halfsize_m':half.tolist(),'window':'planned final t_hit +/- 0.35s, after plan exists; stop before second annotated ground event','criterion':'exact sphere versus finite racket box distance <= ball radius','approach_filter_m_s':-.30,'physical_hit_rate':None,'denominator':'all35; unplanned counted as failure'}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--videos',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--step',type=float,default=.001);a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
 result=evaluate(a.team.resolve(),a.videos.resolve(),a.step);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['groups'],indent=2),flush=True)
if __name__=='__main__':main()
