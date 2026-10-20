"""Fixed selected recipe sensitivity over five existing group-disjoint models.

Not nested selection or new blind data. Each residual is fitted within its own
fold; evaluation labels never enter dynamics/GP/noise/residual training.
"""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from . import robust_optimization as ro
from .controller_horizon_evaluation import ROOT,HORIZONS,dump,summarize
from .controller_contract import residual_domain_gain
from .residual_head import fit_residual_head
from .online_data import load_dataset
from .group_fold_evaluation import checked_model

def fold_run(task):
 number,out,selection=task;out=Path(out)/f'fold{number}';out.mkdir(parents=True,exist_ok=True)
 calibration=ROOT/'artifacts/new_ubuntu/tennis_calibration';folder,plan,trained,checkpoint,selected,hashes=checked_model(ROOT/'artifacts/tennis_hold_sweep/20261013_groupcv',calibration,number)
 noise=json.loads((ROOT/f'artifacts/tennis_robust/20261013/fold{number}/evaluation/noise_proxy.json').read_text())
 assert set(noise['allowed_training_ids'])==set(plan['train_ids'])
 ro.worker_init(checkpoint,selected,noise['sigma_xyz_m']);data=load_dataset(calibration)
 result={};oof={}
 for experiment in ['baseline',selection['experiment']]:
  chunks=[]
  for trace in data:
   evaluation=trace['id'] in plan['evaluation_ids'];training=trace['id'] in plan['train_ids']
   if trace['bounce_reference_s'] is None or not(evaluation or (training and experiment!='baseline')):continue
   trace=dict(trace,split='test' if evaluation else 'train')
   rows,event,x=ro.run_case(trace,experiment,model_id='gp',horizons=HORIZONS)
   keep=[i for i,r in enumerate(rows) if r['twenty_ms_anchor'] and r['elapsed_capture_s']>=.1-1e-12]
   chunks.append(ro.cache_arrays([rows[i] for i in keep],x[keep]))
  c={k:np.concatenate([a[k] for a in chunks]) for k in chunks[0]};p=c['pred'].copy();eligible=c['valid']&c['prediction_ok']
  assert np.all(c['prediction_ok'][c['valid']])
  if experiment!='baseline' and selection.get('head'):
   train=(c['split']=='train')&eligible&(c['gate']>=1-1e-12)&(c['H']<=1.5)&np.isfinite(c['X']).all(axis=1)
   kind=json.loads((ROOT/selection['head']).read_text())['kind']
   head=fit_residual_head(kind,c['X'][train],c['ref'][train]-p[train],c['ids'][train].tolist(),allowed_train_ids=set(plan['train_ids']))
   hmax=float(c['H'][train].max());head.metadata['maximum_trained_horizon_s']=hmax;head.save(out/'residual.json')
   mask=c['prediction_ok']&np.isfinite(c['X']).all(axis=1)
   p[mask]+=(c['gate'][mask]*residual_domain_gain(c['H'][mask],hmax))[:,None]*head.predict_bounded(c['X'][mask],c['H'][mask])
  # summarize expects both split keys. Empty validation mask is not a score.
  d=c.copy();d['split']=np.where(c['split']=='test','validation','train')
  summary=summarize(d,p)['validation'];result[experiment]=summary
  mask=c['split']=='test';oof[experiment]={k:v[mask] for k,v in c.items()};oof[experiment]['pred']=p[mask]
  np.savez_compressed(out/f'{experiment}_oof.npz',**oof[experiment])
 dump(out/'summary.json',{'fold':number,'model_hashes':hashes,'train_ids':plan['train_ids'],'evaluation_ids':plan['evaluation_ids'],'results':result})
 return number

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();out=args.out.resolve();selection=json.loads((out/'selected.json').read_text());dest=out/'groupcv'
 with ProcessPoolExecutor(3) as pool:
  for f in as_completed([pool.submit(fold_run,(i,str(dest),selection)) for i in range(5)]):print('FOLD',f.result(),flush=True)
 result={'selection_fixed_before_group_validation':selection,'scope':'group sensitivity, historically exposed data; no nested selection','folds':[json.loads((dest/f'fold{i}/summary.json').read_text()) for i in range(5)],'oof':{}}
 for exp in ['baseline',selection['experiment']]:
  chunks=[dict(np.load(dest/f'fold{i}/{exp}_oof.npz',allow_pickle=False)) for i in range(5)];c={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
  result['oof'][exp]=summarize(c,c['pred'])['test']
 dump(dest/'summary.json',result)
if __name__=='__main__':main()
