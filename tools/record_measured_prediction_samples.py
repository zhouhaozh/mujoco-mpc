"""固定历史测试集35条实测参考，三种控制器各一视频；不会追加观测噪声。"""
import argparse,concurrent.futures,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from record_prediction_demo import render
from record_all_prediction_samples import validate_video

def job(args):
 team,prediction,out,mode,item=args
 name=item['sample_id'];dest=out/'.records'/mode/name;target=out/mode/(name+'.mp4');meta=target.with_suffix('.json')
 if meta.exists() and target.exists():
  r=json.loads(meta.read_text())
  if r.get('validated') and hashlib.sha256(target.read_bytes()).hexdigest()==r['sha256']:return r
 dest.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 for key in ('M2_VIEW','M2_TIMING_PATH','BALL_PREDICTOR_SOCKET','M2_MEASURED_REPLAY','M2_VIDEO_TRACE'):env.pop(key,None)
 source=dest/'frames.jsonl'
 if not source.exists():
  sock=team/'.tmp'/f'measured-{mode}-{name}.sock';transform=out/'transform.json'
  with (dest/'service.log').open('w') as log:
   service=subprocess.Popen([str(prediction/'.venv-tennis/bin/python'),'-m','tennis_calibration.controller_service','--selection',str(prediction/'artifacts/tennis_controller/20261014/selected.json'),'--socket',str(sock),'--transform',str(transform)],cwd=prediction,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:
    for _ in range(200):
     if service.poll() is not None:raise RuntimeError('service exited')
     if sock.exists():break
     time.sleep(.05)
    else:raise TimeoutError('service startup')
    cmd=[str(team/'sckd_sim/build-predictor/m2_intercept'),'models/generated/x4_tennis_demo.mjcf','1','qp','.25','2026',mode,'measured','xb']
    with (dest/'control.log').open('w') as f:r=subprocess.run(cmd,cwd=team/'sckd_sim',env=dict(env,BALL_PREDICTOR_SOCKET=str(sock),M2_MEASURED_REPLAY=item['csv'],M2_VIDEO_TRACE=str(source)),stdout=f,stderr=subprocess.STDOUT,timeout=600)
    if r.returncode not in (0,1) or '[MEASURED REPLAY]' not in (dest/'control.log').read_text():raise RuntimeError('incomplete measured replay')
   finally:
    service.terminate()
    try:service.wait(5)
    except subprocess.TimeoutExpired:service.kill();service.wait()
    sock.unlink(missing_ok=True)
 rows=[json.loads(line) for line in source.read_text().splitlines()]
 assert rows and all(r['measurement_replay'] for r in rows)
 # 标签仅用于离线画面裁切和实测参考触地计数，不曾写入控制输入CSV。
 for row in rows:row['ground_contacts']=sum(t<=row['t'] for t in item['bounce_labels'])
 result=render(team,out,mode,.25,rows_data=rows,output_path=target,label=f"实测测试集 / {name}",previews=item['sample_id']=='test_01')
 result.update(validate_video(target));result.update(validated=True,trajectory_id=item['trajectory_id'],sample_id=name,source_sha256=item['source_sha256'],input_frames=item['input_frames'],input_end_s=item['input_end_s'],split='test',added_noise=False,physical_hit_rate=None)
 meta.write_text(json.dumps(result,indent=2)+'\n');return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--prediction',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int);a=p.parse_args()
 for k in ('team','prediction','out'):setattr(a,k,getattr(a,k).resolve())
 sys.path.insert(0,str(a.prediction));from tennis_calibration.online_data import load_dataset
 traces=load_dataset(a.prediction/'artifacts/new_ubuntu/tennis_calibration');eligible=sorted([t for t in traces if t['split']=='test' and t['bounce_reference_s'] is not None],key=lambda r:r['id'])
 assert len(eligible)==35
 if a.limit:eligible=eligible[:a.limit]
 a.out.mkdir(parents=True,exist_ok=True);inputs=a.out/'.inputs';inputs.mkdir(exist_ok=True)
 height=json.loads((a.prediction/'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json').read_text())['upstream']['height'];dz=.033-height
 transform={'R_world_from_model':[[1,0,0],[0,1,0],[0,0,1]],'translation_world_from_model_m':[0,0,dz],'note':'visual alignment of effective contact height; camera/court extrinsics uncalibrated'}
 (a.out/'transform.json').write_text(json.dumps(transform,indent=2)+'\n')
 items=[]
 for n,t in enumerate(eligible,1):
  name=f'test_{n:02d}';path=inputs/(name+'.csv');start=float(t['raw_times'][0]);times=t['raw_times']-start;xyz=t['raw_xyz'].copy();xyz[:,2]+=dz
  with path.open('w') as f:
   w=csv.writer(f,lineterminator='\n');w.writerow(['t_s','x_m','y_m','z_m']);w.writerows([[format(float(stamp),'.17g'),*[format(float(x),'.17g') for x in pt]] for stamp,pt in zip(times,xyz)])
  items.append(dict(sample_id=name,trajectory_id=t['id'],csv=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_frames=len(times),input_end_s=float(times[-1]),bounce_labels=[v-start for v in t['bounce_event_times_s']]))
 (a.out/'input_manifest.json').write_text(json.dumps({'items':items,'excluded_test_ids':[t['id'] for t in traces if t['split']=='test' and t['bounce_reference_s'] is None],'transform':transform},indent=2)+'\n')
 for mode in ('pfly','nmpc6','wbmpc'):(a.out/mode).mkdir(exist_ok=True)
 records=[];jobs=[(a.team,a.prediction,a.out,mode,item) for mode in ('pfly','nmpc6','wbmpc') for item in items]
 with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
  for r in pool.map(job,jobs):
   records.append(r);print(f"video {len(records)}/{len(jobs)}: {r['mode']} {r['sample_id']}",flush=True)
   (a.out/'progress.json').write_text(json.dumps({'validated':len(records),'expected':len(jobs)},indent=2)+'\n')
 for mode in ('pfly','nmpc6','wbmpc'):
  rr=[r for r in records if r['mode']==mode];assert len(rr)==len(items)
  with (a.out/mode/'index.csv').open('w') as f:
   fields=['sample_id','trajectory_id','split','prediction_versions','fallback','simulation_seconds','sha256'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
 (a.out/'manifest.json').write_text(json.dumps({'expected':len(jobs),'validated':len(records),'source':'historical measured test trajectories','observations':'original chronological device positions; no added noise; no label filtering','physical_hit_rate':None,'records':records},indent=2)+'\n');print('COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
