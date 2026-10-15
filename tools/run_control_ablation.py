"""nmpc6/pfly逐项配对消融；固定因果预测快照、相同实测输入与物理初态。"""
import argparse,hashlib,json,os,subprocess,time,sys
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 team=a.team.resolve();source=a.source.resolve();root=a.root.resolve();binary=team/'sckd_sim/build-predictor/m2_intercept'
 items=json.loads((source/'input_manifest.json').read_text())['items']
 if a.samples:items=[r for r in items if r['sample_id'] in a.samples.split(',')]
 variants=a.variants.split(',');modes=a.modes.split(',');runlist=[(v,m,i) for v in variants for m in modes if not(v=='servo_mpc' and m!='nmpc6') for i in items]
 for n,(v,mode,item) in enumerate(runlist,1):
  dest=root/'runs'/v/mode/item['sample_id'];dest.mkdir(parents=True,exist_ok=True)
  if (dest/'run.json').exists():continue
  diag=dest/'diagnostics.jsonl'
  if diag.exists():raise FileExistsError(f'incomplete run preserved: {dest}')
  cache=root/'cache'/(item['sample_id']+'.jsonl');assert cache.exists()
  sock=team/'.tmp'/f'ablate-{os.getpid()}.sock'
  env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
  for key in list(env):
   if key.startswith(('M2_','NMPC6_','N6_','BALL_')):env.pop(key,None)
  env.update(M2_CONTROL_VARIANT=v,M2_CONTROL_DIAGNOSTICS=str(diag),M2_MEASURED_REPLAY=item['csv'],BALL_PREDICTOR_SOCKET=str(sock),M2_SERVO_IDENTIFICATION=str(root/'identification/identified.json'))
  with (dest/'cache_service.log').open('w') as log:
   srv=subprocess.Popen([sys.executable,str(team/'tools/control_ablation_cache.py'),'--serve',str(cache),'--socket',str(sock)],env=env,stdout=log,stderr=subprocess.STDOUT)
   try:
    for _ in range(200):
     if srv.poll() is not None:raise RuntimeError('cache service died')
     if sock.exists():break
     time.sleep(.01)
    else:raise TimeoutError('cache service')
    started=time.monotonic()
    with (dest/'control.log').open('w') as f:
     r=subprocess.run([str(binary),'models/generated/x4_tennis_demo.mjcf','1','qp','.25','2026',mode,'measured','xb'],cwd=team/'sckd_sim',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    assert r.returncode in (0,1) and diag.exists(),f'{dest} failed {r.returncode}'
    result={'variant':v,'mode':mode,'sample_id':item['sample_id'],'trajectory_id':item['trajectory_id'],'wall_s':time.monotonic()-started,'returncode':r.returncode,'binary_sha256':digest(binary),'cache_sha256':digest(cache),'input_sha256':digest(Path(item['csv'])),'identified_sha256':digest(root/'identification/identified.json')}
    (dest/'run.json').write_text(json.dumps(result,indent=2)+'\n')
   finally:
    srv.terminate()
    try:srv.wait(3)
    except subprocess.TimeoutExpired:srv.kill();srv.wait()
    sock.unlink(missing_ok=True)
  print(f'{n}/{len(runlist)} {v} {mode} {item["sample_id"]} {result["wall_s"]:.2f}s',flush=True)
  (root/'progress.json').write_text(json.dumps({'done':sum(1 for _ in (root/'runs').rglob('run.json')),'expected_this_command':len(runlist),'current':[v,mode,item['sample_id']]})+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--team',required=True,type=Path);p.add_argument('--source',required=True,type=Path);p.add_argument('--root',required=True,type=Path);p.add_argument('--variants',default='baseline,feasible,clock,feedforward,gains,servo_mpc');p.add_argument('--modes',default='nmpc6,pfly');p.add_argument('--samples');main(p.parse_args())
