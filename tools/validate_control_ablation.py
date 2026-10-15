"""独立核对配对覆盖、固定输入、目标锁止、电机裁剪与已采用解约束。"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main(a):
 rows=[];cache_by_sample={};binary=set();motor_rows=0;bad=0;frozen=0
 for f in sorted((a.root/'runs').glob('*/*/*/run.json')):
  info=json.loads(f.read_text());states=[];last_time=-1;prevlock=None
  binary.add(info['binary_sha256']);sample=info['sample_id'];pair=(info['cache_sha256'],info['input_sha256']);assert sample not in cache_by_sample or cache_by_sample[sample]==pair;cache_by_sample[sample]=pair
  for line in (f.parent/'diagnostics.jsonl').read_text().splitlines():
   r=json.loads(line)
   if r['kind']=='solve' and info['variant']!='baseline' and r.get('applied'):
    assert r['solver_ok'] and r['constraint_violation']<=1e-4
   if r['kind']!='state':continue
   assert r['t']>last_time
   if last_time>=0:assert abs(r['t']-last_time-.005)<1e-9
   last_time=r['t'];states.append(r);assert len(r['motors'])==17
   for i,m in enumerate(r['motors']):
    assert i==m['axis'];assert abs(np.clip(m['raw'],-m['cap'],m['cap'])-m['applied'])<1e-9
    assert np.isfinite([m[k] for k in ('q','dq','q_target','dq_target','raw','applied','cap')]).all();motor_rows+=1
    if i<12 and i%3!=0:assert m['wheel_external_pi'] and m['wire_dq_target']==0
   if info['variant'] not in ('baseline','feasible') and r['locked']:
    snap=r['target_snapshot'];test=[snap['version'],snap['capture_time'],*snap['position'],*snap['racket_velocity'],*snap['normal'],snap['arrival_time'],*snap['base_anchor']]
    if prevlock is not None:np.testing.assert_allclose(test,prevlock,atol=1e-10);frozen+=1
    prevlock=test
  assert states and states[-1]['qp_accepts']<=states[-1]['qp_attempts']
  rows.append({'variant':info['variant'],'mode':info['mode'],'sample_id':sample,'diagnostics_sha256':hashlib.sha256((f.parent/'diagnostics.jsonl').read_bytes()).hexdigest()})
 assert len(rows)==a.expected and len(cache_by_sample)==35 and len(binary)==1
 groups={(r['variant'],r['mode']) for r in rows}
 for v,mode in groups:
  assert mode in ('nmpc6','pfly') and v in ('baseline','feasible','clock','feedforward','gains','servo_mpc','base_feedback')
  assert sum(r['variant']==v and r['mode']==mode for r in rows)==35
 assert len(groups)*35==len(rows)
 result={'date':'2026-10-15','platform':'Ubuntu ubuntu-11','runs':len(rows),'samples':len(cache_by_sample),'same_binary_sha256':next(iter(binary)),'same_input_and_forecast_hashes':f'passed across all{len(groups)} cells','motor_rows_checked':motor_rows,'torque_clipping':'passed every logged motor state','locked_snapshots_checked':frozen,'accepted_nmpc_solver_and_constraint_gate':'passed all corrected stages','source_hashes':rows,'physical_hit_rate':None}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected',type=int,default=385);main(p.parse_args())
