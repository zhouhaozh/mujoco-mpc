"""对侧发球闭环录像：先保存运行状态，再用 MuJoCo 离屏回放。
画面预测只使用运行时已记录的预测；离线渲染不触发额外预测。
"""
import argparse,json,os,subprocess,time
from pathlib import Path

def run(team,prediction,out,modes):
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 for key in ('M2_VIEW','M2_TIMING_PATH','BALL_DEDUP','BALL_PREDICTOR_SOCKET'):
  env.pop(key,None)
 for mode in modes:
  dest=out/mode;dest.mkdir(parents=True,exist_ok=False)
  sock=team/'.tmp'/f'video-{mode}-{os.getpid()}.sock'
  with (dest/'service.log').open('w') as log:
   service=subprocess.Popen([str(prediction/'.venv-tennis/bin/python'),'-m','tennis_calibration.controller_service','--selection',str(prediction/'artifacts/tennis_controller/20261014/selected.json'),'--socket',str(sock),'--synthetic-world'],cwd=prediction,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:
    for _ in range(400):
     if service.poll() is not None:raise RuntimeError('service exited')
     if sock.exists():break
     time.sleep(.05)
    else:raise TimeoutError('service startup')
    cmd=[str(team/'sckd_sim/build-predictor/m2_intercept'),'models/generated/x4_tennis_demo.mjcf','1','qp','.25','2026',mode,'opposite','xb']
    with (dest/'control.log').open('w') as control:
     result=subprocess.run(cmd,cwd=team/'sckd_sim',env=dict(env,BALL_PREDICTOR_SOCKET=str(sock),M2_VIDEO_TRACE=str(dest/'frames.jsonl')),stdout=control,stderr=subprocess.STDOUT,timeout=600)
    # M2 的未达到命中率门槛返回 1，录像必须保留失败试次。
    if result.returncode not in (0,1) or '=== M2 拦截验收 ===' not in (dest/'control.log').read_text():
     raise RuntimeError('simulation failed before evaluation summary')
   finally:
    service.terminate()
    try:service.wait(5)
    except subprocess.TimeoutExpired:service.kill();service.wait()
    sock.unlink(missing_ok=True)
  print('captured',mode,flush=True)

def trim_second_contact(rows):
 """保留第二次真实触地帧及之前；旧录像用高度谷值兼容检测。"""
 if any('ground_contacts' in r for r in rows):
  stop=next((i for i,r in enumerate(rows) if r.get('ground_contacts',0)>=2),len(rows)-1)
 else:
  bounces=[i for i in range(1,len(rows)-1) if rows[i]['ball'][2]<.15 and rows[i-1]['ball'][2]>rows[i]['ball'][2]<=rows[i+1]['ball'][2]]
  stop=bounces[1] if len(bounces)>1 else len(rows)-1
 return rows[:stop+1]

def prediction_until_second_contact(grid,now,contacts):
 """仅显示预测第二次落地之前；不删除低点后把两段重新连起来。"""
 pts=[];last_event=-1e9
 for i,p in enumerate(grid):
  if i:
   a=grid[i-1]
   first_flag=len(p)>5 and bool(p[5]) and not bool(a[5])
   low_turn=len(p)>4 and a[4]<0<=p[4] and min(a[3],p[3])<.18
   down_cross=a[3]>.033>=p[3]
   if (first_flag or low_turn or down_cross) and p[0]-last_event>.06:
    contacts+=1;last_event=p[0]
    if contacts>=2:
     if p[0]>=now:pts.append([p[1],p[2],max(.033,p[3])])
     break
  if p[0]>=now:
   if p[3]<.03:break
   pts.append(p[1:4])
 return pts

def render(team,out,mode,speed,rows_data=None,output_path=None,label="对侧发球",previews=True):
 os.environ.setdefault('MUJOCO_GL','egl')
 import mujoco,numpy as np
 from PIL import Image,ImageDraw,ImageFont
 rows=rows_data if rows_data is not None else [json.loads(s) for s in (out/mode/'frames.jsonl').read_text().splitlines()]
 rows=trim_second_contact(rows)
 assert rows
 ts=np.array([r['t'] for r in rows]);assert np.all(np.diff(ts)>0)
 m=mujoco.MjModel.from_xml_path(str(team/'sckd_sim/models/generated/x4_tennis_demo.mjcf'))
 m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
 d=mujoco.MjData(m);renderer=mujoco.Renderer(m,720,1280,max_geom=5000)
 font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
 fonts=[ImageFont.truetype(font,n) for n in (30,23,20)]
 points=np.array([r['ball'] for r in rows]+[[0,0,0],[0,0,2]])
 lo=points.min(axis=0);hi=points.max(axis=0)
 full=mujoco.MjvCamera();full.lookat[:]=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(hi[2]+lo[2])/2-.15];full.distance=max(7.5,(hi[0]-lo[0])*1.10,(hi[2]-lo[2])*3.3);full.azimuth=90;full.elevation=-18
 near=mujoco.MjvCamera();near.lookat[:]=[1.8,0,.8];near.distance=4.6;near.azimuth=105;near.elevation=-23
 fps=60;duration=ts[-1]/speed;count=int(np.ceil(duration*fps))+1
 output=Path(output_path) if output_path else out/f'{mode}_prediction_opposite.mp4'
 output.parent.mkdir(parents=True,exist_ok=True)
 temp_output=output.with_suffix('.partial.mp4')
 final_hit=any(r['hit'] for r in rows)
 last_index=-1;base_frame=None
 ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','21','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(temp_output)],stdin=subprocess.PIPE)
 def line(a,b,color,width=.018):
  sc=renderer.scene
  if sc.ngeom>=sc.maxgeom:return
  g=sc.geoms[sc.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
  mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width,np.array(a),np.array(b));sc.ngeom+=1
 def draw_scene(row,index,camera):
  renderer.update_scene(d,camera=camera)
  # 黄色球心标记仅放大可视尺寸，绝不改变仿真球半径/碰撞。
  sc=renderer.scene;g=sc.geoms[sc.ngeom]
  mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.075,.075,.075]),np.array(row['ball']),np.eye(3).ravel(),np.array([1,1,.05,1],dtype=np.float32));sc.ngeom+=1
  history=[x['ball'] for x in rows[:index+1]]
  for a,b in zip(history[::2],history[2::2]):line(a,b,[.05,.9,.95,1],.022)
  # 只画当前时刻以后的已缓存预测；拍击后的旧来球预测隐藏，避免误称为出球预测。
  grid=(row['forecast'] or {}).get('grid',[])
  anchor_count=next((r.get('ground_contacts',0) for r in reversed(rows[:index+1]) if grid and r['t']<=grid[0][0]),0)
  pts=prediction_until_second_contact(grid,row['t'],anchor_count) if not row['hit'] else []
  for a,b in zip(pts,pts[1:]):line(a,b,[1,.55,.10,1],.024)
  if row['target'] and not row['hit']:
   p=np.array(row['target']);line(p-[.13,0,0],p+[.13,0,0],[1,.25,.8,1],.025);line(p-[0,0,.13],p+[0,0,.13],[1,.25,.8,1],.025)
  return Image.fromarray(renderer.render())
 try:
  for i in range(count):
   t=min(i/fps*speed,ts[-1]);index=max(0,int(np.searchsorted(ts,t,side='right')-1));row=rows[index]
   if index!=last_index:
    d.qpos[:]=row['qpos'];d.qvel[:]=row['qvel'];mujoco.mj_forward(m,d)
    base_frame=draw_scene(row,index,full)
    # 跟随底盘，扩大机器人近景。仅改变显示相机。
    near.lookat[:]=[float(d.qpos[0])+.5,float(d.qpos[1]),.9]
    inset=draw_scene(row,index,near).resize((512,288));base_frame.paste(inset,(750,386))
    last_index=index
   frame=base_frame.copy()
   dr=ImageDraw.Draw(frame);dr.rectangle((0,0,1280,122),fill=(17,25,39));dr.rectangle((0,678,1280,720),fill=(17,25,39))
   dr.text((22,8),f'{mode.upper()} · {label} / 在线预测更新',font=fonts[0],fill='white')
   grid=(row['forecast'] or {}).get('grid',[]);seq=(row['forecast'] or {}).get('seq',0)
   age=f"{(row['t']-grid[0][0])*1000:.0f} ms" if grid else '预热中'
   dr.text((24,53),f"仿真时间 {row['t']:.2f}s   |   {speed:g}× 慢放   |   预测版本 {seq}   |   预测锚点龄期 {age}",font=fonts[1],fill='#dbe4f0')
   legend='青色：已发生的回放轨迹' if row.get('measurement_replay',False) else '青色：已发生的真值轨迹'
   dr.text((24,86),legend+'     橙色：当前缓存预测     粉色：规划击球点',font=fonts[1],fill='#e3eaf4')
   measured_view=row.get('measurement_replay',False)
   status='已判定命中，隐藏旧来球预测' if row['hit'] else ('已回退 pfly' if row['fallback'] else '闭环运行')
   if i>=count-60 and not final_hit:status='本次未命中（如实保留）'
   if measured_view:status='轨迹参考回放：不判定物理击球命中'
   dr.text((24,684),f'{status}   |   离线按仿真时间回放，不代表墙钟实时运行速度',font=fonts[2],fill='white')
   dr.text((764,394),'机器人近景',font=fonts[2],fill='white')
   if previews and i in (0,count//2,count-1):frame.save(output.with_name(output.stem+f'_preview_{i}.png'))
   ff.stdin.write(frame.tobytes())
 finally:
  ff.stdin.close();code=ff.wait();renderer.close()
 if code:raise RuntimeError('ffmpeg failed')
 temp_output.replace(output)
 seqs=sorted({(r['forecast'] or {}).get('seq',0) for r in rows}-{0})
 crossings=[r for a,r in zip(rows,rows[1:]) if a['ball'][0]>=11.885>r['ball'][0]]
 result={'mode':mode,'frames':count,'fps':fps,'playback_speed':speed,'simulation_seconds':float(ts[-1]),'prediction_versions':len(seqs),'initial_position':rows[0]['ball'],'net_crossing_height':crossings[0]['ball'][2] if crossings else None,'hit':None if rows[0].get('measurement_replay') else any(r['hit'] for r in rows),'fallback':any(r['fallback'] for r in rows),'video':str(output)}
 result.update(measurement_replay=bool(rows[0].get('measurement_replay')),sample=label,ground_contacts=rows[-1].get('ground_contacts'),cut_at_second_contact=True)
 output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--prediction',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--modes',default='pfly,nmpc6,wbmpc');p.add_argument('--render-only',action='store_true');p.add_argument('--speed',type=float,default=.25);a=p.parse_args()
 if not .1<=a.speed<=4:raise ValueError('speed must be 0.1..4')
 a.team=a.team.resolve();a.prediction=a.prediction.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True);modes=a.modes.split(',')
 if not set(modes)<= {'pfly','nmpc6','wbmpc'}:raise ValueError('invalid modes')
 if not a.render_only:run(a.team,a.prediction,a.out,modes)
 for mode in modes:render(a.team,a.out,mode,a.speed)
if __name__=='__main__':main()
