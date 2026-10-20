"""配对来源、原基线复现、约束/限矩和回退贡献审计。"""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
def main(a):
    summary=json.loads((a.root/'report/summary.json').read_text());original=json.loads(a.previous.read_text())
    old={r['sample_id']:r for r in original['cases'] if r['variant']=='gains' and r['mode']=='nmpc6'}
    binaries=set();model_resources=set()
    bysample=collections.defaultdict(list);out={'runs':0,'attempts':0,'variants':{},'baseline_equal':True,'constraint_counts':{},'same_input_cache':True,'frozen_target_checked':True}
    for run in sorted((a.root/'runs').glob('*/nmpc6/*/run.json')):
        meta=json.loads(run.read_text());bysample[meta['sample_id']].append((meta['input_sha256'],meta['cache_sha256']));v=meta['variant']
        binaries.add(meta['binary_sha256'])
        if 'chassis_resources' in meta:model_resources.add(json.dumps(meta['chassis_resources'],sort_keys=True))
        result=out['variants'].setdefault(v,{'attempts':0,'accepted':0,'solve_ms':[],'statuses':{},'stale_fallbacks':0});counter=collections.Counter(result['statuses']);lock=None
        with (run.parent/'diagnostics.jsonl').open() as f:
            for line in f:
                e=json.loads(line)
                if e['kind']=='state':
                    for m in e['motors']:
                        assert abs(float(np.clip(m['raw'],-m['cap'],m['cap']))-m['applied'])<1e-9
                if e['kind']=='stale_plan_fallback':result['stale_fallbacks']+=1
                if e['kind']!='solve':continue
                result['attempts']+=1;counter[e['ipopt_status']]+=1;result['solve_ms'].append(e['ms']);result['accepted']+=e['applied']
                vals=e['constraint_values'];out['constraint_counts'][str(len(vals))]=out['constraint_counts'].get(str(len(vals)),0)+1
                residual=0
                for prefix in ['constraint','control']:
                    for x,lo,hi in zip(e[prefix+'_values'],e[prefix+'_lower'],e[prefix+'_upper']):
                        residual=max(residual,1e9 if x is None else max(0,0 if lo is None else lo-x,0 if hi is None else x-hi))
                assert abs(residual-e['constraint_violation'])<1e-9
                if e['applied']:assert e['solver_ok'] and residual<=1e-4
                if e['locked']:
                    if lock is None:lock=(e['target'],e['target_time'])
                    np.testing.assert_allclose(e['target'],lock[0],atol=1e-10,rtol=0);assert abs(e['target_time']-lock[1])<1e-10
        result['statuses']=dict(counter);out['runs']+=1
    for k,items in bysample.items():assert len(items)==4 and len(set(items))==1
    assert len(bysample)==35 and out['runs']==140
    assert len(binaries)==1 and len(model_resources)==1
    out['binary_sha256']=next(iter(binaries));out['same_chassis_resources']=True
    for r in summary['cases']:
        if r['variant']=='gains':
            z=old[r['sample_id']]
            for k in ['solver_calls','applied_solutions','fallback','hit_time_covered','goal_m','actual_to_goal_m','base_tracking_m','arm_tracking_m']:
                if k not in r and k not in z:continue
                if r.get(k) is None or z.get(k) is None:assert r.get(k)==z.get(k)
                else:np.testing.assert_allclose(r[k],z[k],atol=1e-10,rtol=0)
            assert r['geometric']['approaching']==z['geometric']['approaching']
    for v,r in out['variants'].items():
        t=np.array(r.pop('solve_ms'));r['solve_ms']={'mean':float(t.mean()),'p95':float(np.percentile(t,95)),'max':float(t.max())};out['attempts']+=r['attempts']
        cases=[c for c in summary['cases'] if c['variant']==v];r['geometric_with_fallback']=sum(c['geometric']['approaching'] and c['fallback'] for c in cases);r['geometric_without_fallback']=sum(c['geometric']['approaching'] and not c['fallback'] for c in cases)
    files=[a.team/'sckd_sim/models/config/motors.json',a.team/'sckd_sim/models/config/chassis_m1.json',a.team/'sckd_sim/models/config/chassis_identified_20261015.json',a.team/'sckd_sim/third_party/nmpc6/nlp.so',a.team/'sckd_sim/third_party/nmpc6/nlp_chassis.so',a.team/'sckd_sim/third_party/nmpc6/chassis_rollout.so']
    out['resources']={str(p.relative_to(a.team)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (a.root/'report/audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);main(p.parse_args())
