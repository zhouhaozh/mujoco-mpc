"""Ubuntu integration evaluation. Owns isolated predictor per job; no real hardware."""
import argparse,concurrent.futures,json,os,re,subprocess,time
from pathlib import Path

def run(job):
 a,variant,mode,family,seed=job
 team=Path(a.team).resolve();prediction=Path(a.prediction).resolve();out=Path(a.out).resolve()
 name=f'{variant}_{mode}_{family}_{seed}';dest=out/name;dest.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 for key in ('BALL_PREDICTOR_SOCKET','BALL_DEDUP','NMPC6_DEBUG','N6_TRACE','NMPC6_LEGACY_LIMITS'):env.pop(key,None)
 if a.profile:env['M2_TIMING_PATH']=str(dest/'stage_timing.json')
 else:env.pop('M2_TIMING_PATH',None)
 service=None;sf=None;sock=team/'.tmp'/f'forecast-{name}.sock'
 try:
  if variant=='external':
   sf=(dest/'service.log').open('w')
   service=subprocess.Popen([str(prediction/'.venv-tennis/bin/python'),'-m','tennis_calibration.controller_service','--selection',str(prediction/'artifacts/tennis_controller/20261014/selected.json'),'--socket',str(sock),'--synthetic-world','--telemetry',str(dest/'timing.jsonl')],cwd=prediction,env=env,stdout=sf,stderr=subprocess.STDOUT)
   for _ in range(200):
    if service.poll() is not None:raise RuntimeError('prediction service exited')
    if sock.exists():break
    time.sleep(.05)
   else:raise TimeoutError('prediction service startup')
   env['BALL_PREDICTOR_SOCKET']=str(sock)
  elif variant=='dedup':env['BALL_DEDUP']='1'
  cmd=[str(team/'sckd_sim/build-predictor/m2_intercept'),'models/generated/x4_tennis_demo.mjcf',str(a.trials),'qp','.25',str(seed),mode,family,'xb' if family=='gs' else 'v2']
  t=time.monotonic()
  with (dest/'control.log').open('w') as f:r=subprocess.run(cmd,cwd=team/'sckd_sim',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
  result=dict(variant=variant,mode=mode,family=family,seed=seed,trials=a.trials,returncode=r.returncode,wall_s=time.monotonic()-t,command=cmd)
  log=(dest/'control.log').read_text();m=re.search(r'命中 (\d+)/(\d+)\(.*?平均击球点偏差 (.*?)m',log)
  if not m:
   result['error']='missing control summary; see control.log'
   (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result
  result.update(hits=int(m[1]),mean_plan_error_m=None if m[3]=='n/a' else float(m[3]))
  m=re.search(r'随机发球族: 可达 (\d+)/(\d+)',log);result['feasible']=int(m[1]) if m else a.trials
  m=re.search(r'旧机NMPC: (\d+) 解 (\d+) 拒收 .*?回退 (\d+) trial',log)
  if m:result.update(nmpc_accepted=int(m[1]),nmpc_rejected=int(m[2]),fallback_trials=int(m[3]))
  m=re.search(r'全身MPC: (\d+) 解 (\d+) 拒收 (\d+) 失败,最大单解 ([\d.]+)ms.*?回退 (\d+) trial',log)
  if m:result.update(wbmpc_accepted=int(m[1]),wbmpc_rejected=int(m[2]),wbmpc_failed=int(m[3]),wbmpc_accepted_max_ms=float(m[4]),fallback_trials=int(m[5]))
  m=re.search(r'NMPC attempts=(\d+) mean_ms=([\d.]+) max_ms=([\d.]+)',log)
  if m:result.update(nmpc_attempts=int(m[1]),nmpc_attempt_mean_ms=float(m[2]),nmpc_attempt_max_ms=float(m[3]))
  if a.profile:result['stage_timing']=json.loads((dest/'stage_timing.json').read_text())
  if variant=='external':
   import numpy as np
   timings=[json.loads(x) for x in (dest/'timing.jsonl').read_text().splitlines()]
   result['service_timing']={op:{'n':len(v),'mean_ms':float(np.mean(v)),'p95_ms':float(np.percentile(v,95)),'max_ms':float(np.max(v))} for op in ('update','forecast') if (v:=[r['elapsed_ms'] for r in timings if r['op']==op])}
  (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result
 finally:
  if service:
   service.terminate()
   try:service.wait(5)
   except subprocess.TimeoutExpired:service.kill();service.wait()
   sock.unlink(missing_ok=True)
  if sf:sf.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',required=True);p.add_argument('--prediction',required=True);p.add_argument('--out',required=True);p.add_argument('--trials',type=int,default=10);p.add_argument('--workers',type=int,default=3);p.add_argument('--variants',default='legacy,dedup,external');p.add_argument('--seeds',default='2026,31,777');p.add_argument('--modes',default='pfly,nmpc6,wbmpc');p.add_argument('--profile',action='store_true');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 jobs=[(a,v,m,f,int(s)) for v in a.variants.split(',') for m in a.modes.split(',') for f in ('fixed','rand','gs') for s in a.seeds.split(',')];results=[]
 with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
  for fut in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
   r=fut.result();results.append(r);print(r['variant'],r['mode'],r['family'],r['seed'],r.get('hits',r.get('error')),r.get('nmpc_accepted'),flush=True)
   (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
