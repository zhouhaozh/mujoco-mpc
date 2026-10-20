"""Aggregate same-protocol wbmpc and serial timing evidence without averaging quantiles."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
 wb=json.loads((root/'wbmpc_comparison/summary.json').read_text());assert len(wb)==27 and all('error' not in r for r in wb)
 groups=[]
 for variant in ('legacy','dedup','external'):
  for family in ('fixed','rand','gs'):
   rows=[r for r in wb if r['variant']==variant and r['family']==family]
   groups.append(dict(variant=variant,family=family,**{k:sum(r.get(k,0) for r in rows) for k in ('hits','trials','feasible','wbmpc_accepted','wbmpc_rejected','wbmpc_failed','fallback_trials')}))
 timing=json.loads((root/'serial_stage_timing/summary.json').read_text());assert len(timing)==18 and all('error' not in r for r in timing)
 timing_groups=[]
 for variant in ('dedup','external'):
  for mode in ('pfly','nmpc6','wbmpc'):
   runs=[r for r in timing if r['variant']==variant and r['mode']==mode];keys=set().union(*(r['stage_timing']['stages'].keys() for r in runs));stages={}
   for r in runs:
    st=r['stage_timing']['stages'];assert st['complete_tick']['n']==st['physics_and_motor_step']['n']
    if variant=='external':
     for op in ('forecast','update'):assert st['ipc_'+op]['n']==r['service_timing'][op]['n']
   for key in sorted(keys):
    data=[r['stage_timing']['stages'][key] for r in runs if key in r['stage_timing']['stages']];n=sum(r['n'] for r in data);total=sum(r['total_ms'] for r in data)
    stages[key]=dict(n=n,total_ms=total,mean_ms=total/n,max_ms=max(r['max_ms'] for r in data),worst_run_p95_ms=max(r['p95_ms'] for r in data),**{k:sum(r[k] for r in data) for k in ('over_1ms','over_2ms','over_20ms')})
   wall_s=stages['complete_tick']['total_ms']/1000
   rates={key:st['n']/wall_s for key,st in stages.items()}
   events={key:sum(r['stage_timing']['events'].get(key,0) for r in runs) for key in ('nmpc6_execution_ticks','wbmpc_execution_ticks')}
   row=dict(variant=variant,mode=mode,trials=sum(r['trials'] for r in runs),active_wall_s=wall_s,simulation_s=stages['complete_tick']['n']*.001,effective_wall_hz=rates,stages=stages,events=events)
   if variant=='external':
    row['ipc_extra_mean_ms']={op:stages['ipc_'+op]['mean_ms']-sum(r['service_timing'][op]['n']*r['service_timing'][op]['mean_ms'] for r in runs)/sum(r['service_timing'][op]['n'] for r in runs) for op in ('forecast','update')}
   timing_groups.append(row)
 result=dict(date='2026-10-14',wbmpc_shots=270,wbmpc_groups=groups,timing_shots=54,timing_workers=1,timing_groups=timing_groups,notes=['timing stages are inclusive and can nest; do not sum all stages','worst_run_p95_ms is max of run P95 values, NOT pooled P95','wall_hz excludes trial initialization; unthrottled synchronous simulated time; no real-time deadline guarantee','wbmpc and nmpc6 retain fallback policies; accepted solves are not identical to applied plans','ipc_extra includes scheduling, JSON, transport and telemetry overhead, not pure socket latency'])
 (root/'wbmpc_and_timing_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 for r in groups:print('WB',r['variant'],r['family'],r['hits'],r['fallback_trials'])
 for r in timing_groups:
  print('RATE',r['variant'],r['mode'],{k:round(r['effective_wall_hz'].get(k,0),2) for k in ('complete_tick','observation_update','ipc_forecast','target_refinement','nmpc6_solver','wbmpc_solver')})
  print('MS',{k:round(v['mean_ms'],4) for k,v in r['stages'].items()})
if __name__=='__main__':main()
