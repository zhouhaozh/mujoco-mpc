"""对侧发球闭环录像：先保存运行状态，再用 MuJoCo 离屏回放。
画面预测只使用控制器实际请求过的缓存；录像不触发额外预测。
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

def render(team,out,mode,speed):
 os.environ.setdefault('MUJOCO_GL','egl')
 import mujoco,numpy as np
 from PIL import Image,ImageDraw,ImageFont
 rows=[json.loads(s) for s in (out/mode/'frames.jsonl').read_text().splitlines()]
 assert rows and rows[0]['ball'][0]>11.885
 ts=np.array([r['t'] for r in rows]);assert np.all(np.diff(ts)>0)
 m=mujoco.MjModel.from_xml_path(str(team/'sckd_sim/models/generated/x4_tennis_demo.mjcf'))
 m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
 d=mujoco.MjData(m);renderer=mujoco.Renderer(m,720,1280,max_geom=5000)
 font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
 fonts=[ImageFont.truetype(font,n) for n in (30,23,20)]
 full=mujoco.MjvCamera();full.lookat[:]=[8,0,1];full.distance=23;full.azimuth=90;full.elevation=-30
 near=mujoco.MjvCamera();near.lookat[:]=[1.8,0,.8];near.distance=7;near.azimuth=105;near.elevation=-23
 fps=60;duration=ts[-1]/speed;count=int(np.ceil(duration*fps))+1
 output=out/f'{mode}_prediction_opposite.mp4'
 ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
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
  pts=[x[1:4] for x in grid if x[0]>=row['t'] and x[3]>=.03] if not row['hit'] else []
  for a,b in zip(pts,pts[1:]):line(a,b,[1,.55,.10,1],.024)
  if row['target'] and not row['hit']:
   p=np.array(row['target']);line(p-[.13,0,0],p+[.13,0,0],[1,.25,.8,1],.025);line(p-[0,0,.13],p+[0,0,.13],[1,.25,.8,1],.025)
  return Image.fromarray(renderer.render())
 try:
  for i in range(count):
   t=min(i/fps*speed,ts[-1]);index=max(0,int(np.searchsorted(ts,t,side='right')-1));row=rows[index]
   d.qpos[:]=row['qpos'];d.qvel[:]=row['qvel'];mujoco.mj_forward(m,d)
   frame=draw_scene(row,index,full)
   inset=draw_scene(row,index,near).resize((448,252));frame.paste(inset,(814,422))
   dr=ImageDraw.Draw(frame);dr.rectangle((0,0,1280,122),fill=(17,25,39));dr.rectangle((0,678,1280,720),fill=(17,25,39))
   dr.text((22,8),f'{mode.upper()} · 对侧发球 / 在线预测更新',font=fonts[0],fill='white')
   grid=(row['forecast'] or {}).get('grid',[]);seq=(row['forecast'] or {}).get('seq',0)
   age=f"{(row['t']-grid[0][0])*1000:.0f} ms" if grid else '预热中'
   dr.text((24,53),f"仿真时间 {row['t']:.2f}s   |   {speed:g}× 慢放   |   预测版本 {seq}   |   预测锚点龄期 {age}",font=fonts[1],fill='#dbe4f0')
   dr.text((24,86),'青色：已发生的真值轨迹     橙色：当前缓存预测     粉色：规划击球点',font=fonts[1],fill='#e3eaf4')
   status='已判定命中，隐藏旧来球预测' if row['hit'] else ('已回退 pfly' if row['fallback'] else '闭环运行')
   if i>=count-60 and not any(r['hit'] for r in rows):status='本次未命中（如实保留）'
   dr.text((24,684),f'{status}   |   离线按仿真时间回放，不代表墙钟实时运行速度',font=fonts[2],fill='white')
   dr.text((827,430),'机器人近景',font=fonts[2],fill='white')
   if i in (0,count//2,count-1):frame.save(out/f'{mode}_preview_{i}.png')
   ff.stdin.write(frame.tobytes())
 finally:
  ff.stdin.close();code=ff.wait();renderer.close()
 if code:raise RuntimeError('ffmpeg failed')
 seqs=sorted({(r['forecast'] or {}).get('seq',0) for r in rows}-{0})
 crossings=[r for a,r in zip(rows,rows[1:]) if a['ball'][0]>=11.885>r['ball'][0]]
 result={'mode':mode,'frames':count,'fps':fps,'playback_speed':speed,'simulation_seconds':float(ts[-1]),'prediction_versions':len(seqs),'initial_position':rows[0]['ball'],'net_crossing_height':crossings[0]['ball'][2] if crossings else None,'hit':any(r['hit'] for r in rows),'fallback':any(r['fallback'] for r in rows),'video':str(output)}
 (out/f'{mode}_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--prediction',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--modes',default='pfly,nmpc6,wbmpc');p.add_argument('--render-only',action='store_true');p.add_argument('--speed',type=float,default=.25);a=p.parse_args()
 if not .1<=a.speed<=4:raise ValueError('speed must be 0.1..4')
 a.team=a.team.resolve();a.prediction=a.prediction.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True);modes=a.modes.split(',')
 if not set(modes)<= {'pfly','nmpc6','wbmpc'}:raise ValueError('invalid modes')
 if not a.render_only:run(a.team,a.prediction,a.out,modes)
 for mode in modes:render(a.team,a.out,mode,a.speed)
if __name__=='__main__':main()
