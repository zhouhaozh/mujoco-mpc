"""对同一历史实测test集记录控制目标、参考、实际和求解门控；不重训或调参。"""
import argparse,json,os,subprocess,time
from pathlib import Path

def run(team,prediction,source,out,mode,item):
 dest=out/mode/item['sample_id'];done=dest/'run.json'
 if done.exists():return json.loads(done.read_text())
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'diagnostics.jsonl').exists():raise FileExistsError(f'incomplete diagnostic run preserved: {dest}')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',M2_CONTROL_VARIANT='baseline')
 for key in ('M2_VIDEO_TRACE','M2_VIEW','M2_TIMING_PATH','BALL_DEDUP','BALL_PREDICTOR_SOCKET','M2_CONTROL_DIAGNOSTICS','M2_MEASURED_REPLAY'):env.pop(key,None)
 sock=team/'.tmp'/f"diag-{mode}-{item['sample_id']}.sock"
 with (dest/'service.log').open('w') as log:
  service=subprocess.Popen([str(prediction/'.venv-tennis/bin/python'),'-m','tennis_calibration.controller_service','--selection',str(prediction/'artifacts/tennis_controller/20261014/selected.json'),'--socket',str(sock),'--transform',str(source/'transform.json')],cwd=prediction,env=env,stdout=log,stderr=subprocess.STDOUT)
  try:
   for _ in range(200):
    if service.poll() is not None:raise RuntimeError('prediction service failed')
    if sock.exists():break
    time.sleep(.05)
   else:raise TimeoutError('prediction startup')
   cmd=[str(team/'sckd_sim/build-predictor/m2_intercept'),'models/generated/x4_tennis_demo.mjcf','1','qp','.25','2026',mode,'measured','xb']
   started=time.monotonic()
   with (dest/'control.log').open('w') as f:r=subprocess.run(cmd,cwd=team/'sckd_sim',env=dict(env,BALL_PREDICTOR_SOCKET=str(sock),M2_MEASURED_REPLAY=item['csv'],M2_CONTROL_DIAGNOSTICS=str(dest/'diagnostics.jsonl')),stdout=f,stderr=subprocess.STDOUT,timeout=600)
   if r.returncode not in (0,1) or '[MEASURED REPLAY]' not in (dest/'control.log').read_text() or not (dest/'diagnostics.jsonl').exists():raise RuntimeError('diagnostic run incomplete')
   result={'mode':mode,'sample_id':item['sample_id'],'trajectory_id':item['trajectory_id'],'source_csv':item['csv'],'wall_s':time.monotonic()-started,'returncode':r.returncode,'diagnostics':str(dest/'diagnostics.jsonl')};done.write_text(json.dumps(result,indent=2)+'\n')
   return result
  finally:
   service.terminate()
   try:service.wait(5)
   except subprocess.TimeoutExpired:service.kill();service.wait()
   sock.unlink(missing_ok=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',required=True,type=Path);p.add_argument('--prediction',required=True,type=Path);p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--samples');p.add_argument('--modes',default='nmpc6,wbmpc');a=p.parse_args()
 for k in ('team','prediction','source','out'):setattr(a,k,getattr(a,k).resolve())
 a.out.mkdir(parents=True,exist_ok=True);items=json.loads((a.source/'input_manifest.json').read_text())['items']
 if a.samples:items=[i for i in items if i['sample_id'] in a.samples.split(',')]
 results=[];total=len(items)*len(a.modes.split(','))
 for mode in a.modes.split(','):
  for item in items:
   r=run(a.team,a.prediction,a.source,a.out,mode,item);results.append(r);print(f"{len(results)}/{total} {mode} {item['sample_id']}",flush=True)
   (a.out/'progress.json').write_text(json.dumps({'done':len(results),'expected':total},indent=2)+'\n')
 (a.out/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
