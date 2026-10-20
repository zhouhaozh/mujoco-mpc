"""Controller-horizon comparison; train-only fitting and validation-only selection.

Run on Ubuntu. Labels never enter observer replay. Frozen historical split is
not a new blind test. Long CSV/NPZ remain on the evaluation machine.
"""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from . import robust_optimization as ro
from .online_data import load_dataset
from .residual_head import fit_residual_head,ResidualHead
from .controller_contract import residual_domain_gain
ROOT=Path(__file__).resolve().parents[1]
HORIZONS=(.1,.2,.5,.7,.9,1.2,1.5,2.,2.5)
RECIPES=[(m,e) for m,e in [('gp','baseline'),('gp','huber_only'),('gp','adaptive_huber'),('gp','protected_only'),('gp','adaptive_protected'),('gp','smooth_only'),('gp','protected_smooth'),('gp','adaptive_protected_smooth'),('fixed','adaptive_protected'),('ridge1','adaptive_protected'),('ridge2','adaptive_protected')]]

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def replay(task):
 model,exp,out=task;out=Path(out);name=model+'__'+exp;path=out/'cache'/f'{name}.npz'
 if path.exists():return name
 selected=json.loads((ROOT/'artifacts/tennis_identified/20261009_training_prior/selected_model.json').read_text())
 noise=json.loads((ROOT/'artifacts/tennis_robust/20261013/validation_v2/noise_proxy.json').read_text())
 ro.worker_init(ROOT/'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json',selected,noise['sigma_xyz_m'])
 data=load_dataset(ROOT/'artifacts/new_ubuntu/tennis_calibration');chunks=[];events=[]
 for t in data:
  if t['bounce_reference_s'] is None:continue
  rows,event,features=ro.run_case(t,exp,model_id=model,horizons=HORIZONS)
  keep=[i for i,r in enumerate(rows) if r['twenty_ms_anchor'] and r['elapsed_capture_s']>=.1-1e-12]
  r=[rows[i] for i in keep]
  for x in r:x['candidate']=name
  chunks.append(ro.cache_arrays(r,features[keep]));events.append(event)
 result={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
 path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**result)
 dump(out/'events'/f'{name}.json',events)
 return name

def stats(c,pred,mask):
 requested=int(mask.sum());valid=mask&c['valid'];ok=valid&c['prediction_ok']&np.isfinite(pred).all(axis=1)
 if not ok.any():return dict(requested=requested,labeled=int(valid.sum()),predicted=0,trajectories=0,mean_m=None,rmse_m=None,p95_m=None,max_m=None)
 err=np.linalg.norm(pred[ok]-c['ref'][ok],axis=1);ids=c['ids'][ok]
 unique,inv,n=np.unique(ids,return_inverse=True,return_counts=True);w=1./n[inv]/len(unique)
 order=np.argsort(err);p95=float(err[order][min(np.searchsorted(np.cumsum(w[order]),.95),len(err)-1)])
 return dict(requested=requested,labeled=int(valid.sum()),predicted=int(ok.sum()),trajectories=len(unique),mean_m=float(w@err),rmse_m=float(np.sqrt(w@(err**2))),p95_m=p95,max_m=float(err.max()))

def summarize(c,pred):
 result={}
 for split in ('validation','test'):
  sm=c['split']==split;cells=[];curves=[]
  for h in HORIZONS:cells.append(dict(horizon_s=h,**stats(c,pred,sm&np.isclose(c['H'],h))))
  # Different target availability at each capture time is reported, not filled.
  for h in (.1,.5,.9,1.2,1.5):
   for bin_index in range(-20,21):
    low=bin_index*.05;mask=sm&np.isclose(c['H'],h)&(c['relative_capture_s']>=low-1e-12)&(c['relative_capture_s']<low+.05-1e-12)
    if mask.any():curves.append(dict(horizon_s=h,capture_relative_s=low+.025,**stats(c,pred,mask)))
  usable=[x for x in cells if x['horizon_s']<=1.5 and x['rmse_m'] is not None]
  recovery=stats(c,pred,sm&np.isclose(c['H'],.5)&(c['relative_capture_s']>=.1)&(c['relative_capture_s']<.2))
  result[split]={'by_horizon':cells,'by_capture_time':curves,'balanced_horizon_rmse_m':float(np.sqrt(np.mean([x['rmse_m']**2 for x in usable]))) if usable else None,'recovery_h500':recovery}
 return result

def score(out):
 out=Path(out).resolve();results={};meta={};coverage_reference=None
 for model,exp in RECIPES:
  name=model+'__'+exp;c=dict(np.load(out/'cache'/f'{name}.npz',allow_pickle=False));pred=c['pred']
  eligible=c['valid']&(c['elapsed_capture_s']>=.1)
  keys=np.column_stack((c['ids'],c['capture_t_s'].astype(str),c['H'].astype(str)))[eligible]
  if coverage_reference is None:coverage_reference=keys
  else:assert np.array_equal(coverage_reference,keys),'mismatched label coverage'
  assert np.all(c['prediction_ok'][eligible]),'candidate prediction failures must not be discarded'
  results[name]=summarize(c,pred);meta[name]={'model_id':model,'experiment':exp,'head':None}
  if model!='gp' or exp not in ('baseline','adaptive_huber','adaptive_protected'):continue
  train=(c['split']=='train')&eligible&c['prediction_ok']&(c['gate']>=1-1e-12)&(c['H']<=1.5)&np.isfinite(c['X']).all(axis=1)
  ids=set(c['ids'][c['split']=='train'])
  for kind in ('ridge','mlp'):
   head=fit_residual_head(kind,c['X'][train],c['ref'][train]-pred[train],c['ids'][train].tolist(),allowed_train_ids=ids)
   head.metadata['maximum_trained_horizon_s']=float(c['H'][train].max())
   dest=out/'models'/f'{name}__{kind}.json';dest.parent.mkdir(exist_ok=True);head.save(dest)
   # Same causal gate/cap as old experiment; no extrapolation past trained horizon.
   p=pred.copy();mask=c['prediction_ok']&np.isfinite(c['X']).all(axis=1)&(c['H']<=1.5)
   p[mask]+=(c['gate'][mask]*residual_domain_gain(c['H'][mask],head.metadata['maximum_trained_horizon_s']))[:,None]*head.predict_bounded(c['X'][mask],c['H'][mask])
   key=name+'__'+kind;results[key]=summarize(c,p);meta[key]={'model_id':model,'experiment':exp,'head':str(dest.relative_to(ROOT))}
 baseline=results['gp__baseline']['validation']['recovery_h500']
 ranking=sorted(results,key=lambda k:results[k]['validation']['balanced_horizon_rmse_m'])
 admissible=[k for k in ranking if results[k]['validation']['recovery_h500']['p95_m']<=baseline['p95_m'] and results[k]['validation']['recovery_h500']['max_m']<=baseline['max_m']]
 winner=admissible[0] if admissible else 'gp__baseline'
 selection={'schema':'CONTROLLER_PREDICTOR/1','candidate':winner,**meta[winner],'max_horizon_s':1.5,'hold_s':.1,'selection_split':'validation','criterion':'equal-horizon trajectory-equal RMSE at .1/.2/.5/.7/.9/1.2/1.5s subject to validation recovery H500 P95 and maximum not worse than original GP','ranking':ranking,'admissible':admissible,'history_note':'previously inspected historical splits; exploratory, not new blind data','warning':'2.0s sparse coverage; 2.5s no labels. First bounce only.'}
 dump(out/'selected.json',selection);dump(out/'summary.json',{'selection':selection,'candidates':results,'error_definition':'Euclidean position distance in metres against same-device retained observations; trajectory-equal within each cell; P95 weighted empirical quantile; not absolute sensor truth'})
 # Plain human-readable table for terminal and report generation.
 for k in ranking:print(k,round(results[k]['validation']['balanced_horizon_rmse_m']*100,2),round(results[k]['test']['balanced_horizon_rmse_m']*100,2),flush=True)
 print('SELECTED',winner,flush=True)
 plot(out,results,ranking,winner)

def plot(out,results,ranking,winner):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 names=list(dict.fromkeys(['gp__baseline','gp__adaptive_protected',winner,ranking[0]]))
 fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
 for ax,(split,metric) in zip(axs.ravel(),[(s,m) for s in ('validation','test') for m in ('mean_m','p95_m')]):
  for name in names:
   cells=results[name][split]['by_horizon'];valid=[r for r in cells if r[metric] is not None]
   ax.plot([r['horizon_s'] for r in valid],[100*r[metric] for r in valid],'o-',label=name)
  ax.axvspan(1.5,2.5,color='gray',alpha=.12);ax.set(xlabel='Prediction horizon (s)',ylabel=metric+' (cm)',title=split+' / horizon');ax.grid(alpha=.25);ax.legend(fontsize=7)
 fig.savefig(out/'error_vs_horizon.png',dpi=160);fig.savefig(out/'error_vs_horizon.pdf');plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
 for col,h in enumerate((.1,.5,1.2)):
  for row,metric in enumerate(('mean_m','p95_m')):
   ax=axs[row,col]
   for name in names:
    cells=[r for r in results[name]['test']['by_capture_time'] if r['horizon_s']==h and r[metric] is not None]
    ax.plot([r['capture_relative_s'] for r in cells],[100*r[metric] for r in cells],label=name)
   ax.axvspan(0,.1,color='gray',alpha=.15);ax.axvline(0,color='black',lw=.7);ax.set(xlabel='Observation time relative to reference bounce (s)',ylabel=metric+' (cm)',title=f'Historical test, H={h:g}s');ax.grid(alpha=.25);ax.legend(fontsize=6)
 fig.savefig(out/'error_vs_capture_time.png',dpi=160);fig.savefig(out/'error_vs_capture_time.pdf');plt.close(fig)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--score-only',action='store_true');a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if not a.score_only:
  dump(a.out/'protocol.json',{'horizons':HORIZONS,'recipes':RECIPES,'split':'original frozen train/validation/test','selection':'validation only','residual_fit':'train only; gate=1; max horizon1.5; original fixed hyperparameters','no_change':'original weights/split/references untouched'})
  with ProcessPoolExecutor(a.workers) as pool:
   for job in as_completed([pool.submit(replay,(m,e,str(a.out))) for m,e in RECIPES]):print('REPLAY',job.result(),flush=True)
 score(a.out)
if __name__=='__main__':main()
