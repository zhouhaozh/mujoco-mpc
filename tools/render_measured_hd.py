"""重渲染已有实测test回放：1080p、紧凑取景、机器人后方近景、首次拍面接触结束。"""
import argparse,concurrent.futures,csv,hashlib,json,os,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from evaluate_measured_interception import box_sphere_gap
from record_prediction_demo import prediction_until_second_contact

W,H,FPS,SPEED=1920,1080,60,.25
TOP,BOTTOM=84,36

class Playback:
 def __init__(self,team,rows,raw):
  self.m=mujoco.MjModel.from_xml_path(str(team/'sckd_sim/models/generated/x4_tennis_demo.mjcf'))
  self.d=mujoco.MjData(self.m);self.rows=rows;self.raw=raw
  self.times=np.array([r['t'] for r in rows]);self.q=np.array([r['qpos'] for r in rows]);self.v=np.array([r['qvel'] for r in rows])
  self.face=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'racket_face');ball=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'ball_geom')
  self.radius=float(self.m.geom_size[ball,0]);self.half=self.m.geom_size[self.face].copy()
  jid=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,'ball_free');self.qa=self.m.jnt_qposadr[jid];self.va=self.m.jnt_dofadr[jid]
  self.tangent=np.empty(self.m.nv);self.m.geom_contype[ball]=0;self.m.geom_conaffinity[ball]=0
 def at(self,t):
  i=max(0,min(len(self.rows)-2,int(np.searchsorted(self.times,t,side='right')-1)))
  a,b=self.times[i:i+2];u=float(np.clip((t-a)/(b-a),0,1));self.d.qpos[:]=self.q[i]
  mujoco.mj_differentiatePos(self.m,self.tangent,b-a,self.q[i],self.q[i+1]);mujoco.mj_integratePos(self.m,self.d.qpos,self.tangent,u*(b-a))
  self.d.qvel[:]=(1-u)*self.v[i]+u*self.v[i+1]
  p=np.array([np.interp(t,self.raw[:,0],self.raw[:,j]) for j in (1,2,3)])
  self.d.qpos[self.qa:self.qa+3]=p;mujoco.mj_forward(self.m,self.d)
  source=max(0,int(np.searchsorted(self.times,t,side='right')-1))
  return self.rows[source],p,source
 def gap(self):
  R=self.d.geom_xmat[self.face].reshape(3,3)
  return box_sphere_gap(R.T@(self.d.qpos[self.qa:self.qa+3]-self.d.geom_xpos[self.face]),self.half,self.radius)

def first_contact(play,end,step=.0005):
 for t in np.arange(0,end+step*.5,step):
  t=min(float(t),end);play.at(t)
  if play.gap()<=0:return t
 return None

def fit_camera(points,aspect):
 cam=mujoco.MjvCamera();cam.azimuth=90;cam.elevation=-38
 # 相机高度足够，避免把地平线上方的空白/天空纳入画面。
 angle=np.deg2rad(38);right=np.array([1.,0,0]);forward=np.array([0,-np.cos(angle),-np.sin(angle)]);up=np.array([0,-np.sin(angle),np.cos(angle)])
 center=(points.min(axis=0)+points.max(axis=0))/2
 q=points-center
 for axis in (right,up):center+=axis*(float((q@axis).min()+(q@axis).max())/2);q=points-center
 tan_y=np.tan(np.deg2rad(22.5));tan_x=tan_y*aspect
 distance=max(3.3,float(np.max(1.10*np.abs(q@right)/tan_x-q@forward)),float(np.max(1.10*np.abs(q@up)/tan_y-q@forward)))
 cam.lookat[:]=center;cam.distance=distance
 return cam,(right,up,forward,tan_x,tan_y)

def render_one(job):
 team,source,out,mode,item=job;name=item['sample_id'];target=out/mode/(name+'.mp4');meta=target.with_suffix('.json')
 if target.exists() and meta.exists():
  result=json.loads(meta.read_text())
  if result.get('validated') and result.get('render_version')==2 and hashlib.sha256(target.read_bytes()).hexdigest()==result['sha256']:return result
 rows=[json.loads(s) for s in (source/'.records'/mode/name/'frames.jsonl').read_text().splitlines()]
 raw=np.loadtxt(item['csv'],delimiter=',',skiprows=1);play=Playback(team,rows,raw)
 original_end=min(float(raw[-1,0]),item['bounce_labels'][1] if len(item['bounce_labels'])>1 else float(raw[-1,0]))
 contact=first_contact(play,original_end);end=contact if contact is not None else original_end
 # 整条有效球路和机器人包围盒共同决定主镜头，不为无效远场/天空留空。
 track=raw[raw[:,0]<=end,1:];last_p=np.array([np.interp(end,raw[:,0],raw[:,j]) for j in (1,2,3)])
 robot=[]
 for r in rows:
  if r['t']>end:break
  root=np.array(r['qpos'][:3])
  robot.extend([root+[x,y,z] for x in (-.65,.85) for y in (-.65,.65) for z in (0,1.9)])
 points=np.vstack([track,last_p,np.array(robot)]);scene_h=H-TOP-BOTTOM
 cam,basis=fit_camera(points,W/scene_h)
 near=mujoco.MjvCamera();near.distance=3.0;near.elevation=-28
 play.m.vis.global_.offwidth=W;play.m.vis.global_.offheight=scene_h
 renderer=mujoco.Renderer(play.m,scene_h,W,max_geom=5000)
 near_renderer=mujoco.Renderer(play.m,360,640,max_geom=5000)
 fonts=[ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',n) for n in (34,25,23)]
 count=int(np.ceil(end/SPEED*FPS))+1
 target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_suffix('.partial.mp4')
 ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','17','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE)
 # 选择对有效球路遮挡最少的角落，整个视频保持位置不跳动。
 right,up,forward,tan_x,tan_y=basis;rel=points-np.array(cam.lookat);depth=cam.distance+rel@forward
 pixels=np.column_stack(((.5+.5*(rel@right)/(depth*tan_x))*W,(.5-.5*(rel@up)/(depth*tan_y))*scene_h))
 corners=[(W-658,scene_h-378),(W-658,18),(18,scene_h-378),(18,18)]
 inset=min(corners,key=lambda xy:sum((pixels[:,0]>=xy[0])&(pixels[:,0]<=xy[0]+640)&(pixels[:,1]>=xy[1])&(pixels[:,1]<=xy[1]+360)))
 def line(sc,a,b,color,r=.017):
  if sc.ngeom>=sc.maxgeom:return
  if np.linalg.norm(np.array(a)-b)<1e-9:return
  g=sc.geoms[sc.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
  mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,r,np.array(a),np.array(b));sc.ngeom+=1
 def decorate(r,p,now):
  sc=r.scene;g=sc.geoms[sc.ngeom]
  # 接触结束采用真实球半径，避免放大的球心标记看似提前撞拍。
  mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,play.radius),p,np.eye(3).ravel(),np.array([1,1,.02,1],dtype=np.float32));sc.ngeom+=1
  history=raw[raw[:,0]<=now,1:][::3];history=np.vstack([history,p])
  for a,b in zip(history,history[1:]):line(sc,a,b,[.05,.85,.95,1])
  row,_,index=play.at(now);grid=(row['forecast'] or {}).get('grid',[])
  anchor_count=sum(t<=grid[0][0] for t in item['bounce_labels']) if grid else 0
  forecast=prediction_until_second_contact(grid,now,anchor_count)
  for a,b in zip(forecast,forecast[1:]):line(sc,a,b,[1,.52,.08,1])
  if row['target']:
   p0=np.array(row['target']);line(sc,p0-[.10,0,0],p0+[.10,0,0],[1,.25,.8,1]);line(sc,p0-[0,0,.10],p0+[0,0,.10],[1,.25,.8,1])
 try:
  for i in range(count):
   now=min(i/FPS*SPEED,end);row,p,index=play.at(now)
   # 跟随当前球与机器人。远端历史曲线仍保留，不强制镜头始终显示发球起点。
   root=play.d.qpos[:3].copy()
   active=raw[(raw[:,0]>=max(0,now-.20))&(raw[:,0]<=now),1:]
   current_robot=np.array([root+[x,y,z] for x in (-.65,.85) for y in (-.65,.65) for z in (0,1.9)])
   grid=(row['forecast'] or {}).get('grid',[])
   future=np.array([a[1:4] for a in grid if now<=a[0]<=now+.50 and a[3]>=.033]).reshape(-1,3)
   desired,_=fit_camera(np.vstack([active,p,current_robot,future]),W/scene_h)
   cam.lookat[:]=.82*np.array(cam.lookat)+.18*np.array(desired.lookat)
   cam.distance=max(desired.distance,.90*cam.distance+.10*desired.distance)
   renderer.update_scene(play.d,camera=cam);decorate(renderer,p,now);main=Image.fromarray(renderer.render())
   root=play.d.qpos[:3];quat=play.d.qpos[3:7];yaw=np.arctan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
   near.azimuth=180+np.rad2deg(yaw);near.lookat[:]=root+[.35*np.cos(yaw),.35*np.sin(yaw),.90]
   near_renderer.update_scene(play.d,camera=near);decorate(near_renderer,p,now);close=Image.fromarray(near_renderer.render())
   frame=Image.new('RGB',(W,H),(16,24,36));frame.paste(main,(0,TOP));frame.paste(close,(inset[0],TOP+inset[1]))
   dr=ImageDraw.Draw(frame);dr.rectangle((inset[0],TOP+inset[1],inset[0]+640,TOP+inset[1]+360),outline='#e0e8ee',width=2)
   dr.text((inset[0]+12,TOP+inset[1]+10),'机器人后方近景',font=fonts[2],fill='white')
   dr.text((18,5),f'{mode.upper()} · 实测 test / {name}',font=fonts[0],fill='white')
   seq=(row['forecast'] or {}).get('seq',0);grid=(row['forecast'] or {}).get('grid',[])
   age=f"{1000*(now-grid[0][0]):.0f}ms" if grid else '预热'
   dr.text((18,48),f'{now:.3f}s  |  0.25×  |  预测版本 {seq} / 龄期 {age}     青色：实测参考  橙色：预测  粉色：目标',font=fonts[1],fill='#dce8f4')
   status='球拍接触 · 动画结束（几何判定）' if contact is not None and i==count-1 else ('控制已回退 pfly' if row['fallback'] else '实测球路与机器人拦截回放')
   dr.text((18,H-BOTTOM+3),status,font=fonts[2],fill='#7df7ae' if contact is not None and i==count-1 else '#dce8f4')
   if name=='test_01' or (mode=='pfly' and name=='test_02'):
    if i in (0,count//2,count-1):frame.save(target.with_name(target.stem+f'_preview_{i}.png'))
   ff.stdin.write(frame.tobytes())
 finally:
  ff.stdin.close();code=ff.wait();renderer.close();near_renderer.close()
 if code:raise RuntimeError('encode failed')
 partial.replace(target)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(target)],text=True))
 assert probe['streams'][0]['width']==W and probe['streams'][0]['height']==H and int(probe['streams'][0]['nb_frames'])==count
 subprocess.run(['ffmpeg','-threads','1','-v','error','-i',str(target),'-f','null','-'],check=True,timeout=120)
 # 对最后一帧重算，确认没有在首次拍面接触后继续播放。
 play.at(end);gap=play.gap()
 if contact is not None:assert gap<=0
 result={'render_version':2,'mode':mode,'sample_id':name,'trajectory_id':item['trajectory_id'],'source_run':str(source/'.records'/mode/name),'contact_time_s':contact,'end_time_s':end,'original_scope_end_s':original_end,'end_reason':'racket_geometric_contact' if contact is not None else 'second_ground_or_data_end','last_frame_surface_gap_m':gap,'physical_hit_rate':None,'contact_detection_step_s':.0005,'camera_main':{'tracking':'current ball, robot, recent 200ms and 500ms predicted path','lookat':list(cam.lookat),'distance':cam.distance,'azimuth':cam.azimuth,'elevation':cam.elevation},'camera_close':'robot rear, follows base yaw; azimuth=180deg+yaw','resolution':[W,H],'fps':FPS,'frames':count,'playback_speed':SPEED,'crf':17,'video':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'probe':probe,'validated':True}
 meta.write_text(json.dumps(result,indent=2)+'\n');return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--samples');p.add_argument('--modes',default='pfly,nmpc6,wbmpc');a=p.parse_args()
 a.team=a.team.resolve();a.source=a.source.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
 items=json.loads((a.source/'input_manifest.json').read_text())['items']
 if a.samples:items=[x for x in items if x['sample_id'] in a.samples.split(',')]
 jobs=[(a.team,a.source,a.out,mode,item) for mode in a.modes.split(',') for item in items];records=[]
 with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
  for f in concurrent.futures.as_completed([pool.submit(render_one,j) for j in jobs]):
   r=f.result();records.append(r);print(f"video {len(records)}/{len(jobs)}: {r['mode']} {r['sample_id']} {r['end_reason']}",flush=True)
   (a.out/'progress.json').write_text(json.dumps({'expected':len(jobs),'validated':len(records)},indent=2)+'\n')
 for mode in a.modes.split(','):
  with (a.out/mode/'index.csv').open('w') as f:
   fields=['sample_id','trajectory_id','contact_time_s','end_reason','end_time_s','sha256'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(sorted([r for r in records if r['mode']==mode],key=lambda r:r['sample_id']))
 (a.out/'manifest.json').write_text(json.dumps({'validated':len(records),'expected':len(jobs),'source_runs':str(a.source),'records':records},indent=2)+'\n');print('COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
