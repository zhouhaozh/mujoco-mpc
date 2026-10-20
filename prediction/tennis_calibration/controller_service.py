"""Local Unix-socket bridge for the team pfly/nmpc6 simulator.

Causal observations only. Runtime never loads trajectory labels. Explicit
synthetic-frame alignment is not a real-camera extrinsic calibration.
"""
from __future__ import annotations
import argparse,json,os,socket,time
from pathlib import Path
import numpy as np
from .collision_model_bank import BounceModelBank
from .model_bundle import ModelBundle
from .robust_observer import RobustHeightPredictor
from .residual_head import ResidualHead,causal_features,residual_gate
from .identified_filter import _contact_dt
from .controller_contract import residual_domain_gain
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='BALL_FORECAST/1'

class ControllerPredictor:
 def __init__(self,selection=None,transform=None,*,bundle=None,forecast_max_horizon_s=None):
  if bundle is None:bundle=ModelBundle.from_legacy(selection,transform,root=ROOT)
  elif selection is not None or transform is not None:raise ValueError('supply a bundle or legacy selection/transform')
  if forecast_max_horizon_s is not None:bundle=bundle.with_forecast_horizon(forecast_max_horizon_s)
  self.bundle=bundle;self.selection=bundle.selection;self.settings=bundle.settings
  self.bundle_fingerprint=bundle.fingerprint;self.resolved_manifest=bundle.resolved_manifest
  effective=bundle.effective_config
  self.bank=BounceModelBank(bundle.resource_path('collision'),model_paths=bundle.collision_model_paths)
  self.config=effective['filter'];self.noise=effective['noise_sigma_xyz_m']
  transform=effective['transform']
  self.R=np.asarray(transform['R_world_from_model'],float);self.shift=np.asarray(transform['translation_world_from_model_m'],float)
  self.head=ResidualHead.load(bundle.resource_path('head')) if self.settings['residual']['enabled'] else None
  self.max_h=float(self.settings['forecast']['max_horizon_s'])
  self.reset()
 def reset(self):
  observer=self.settings['observer']
  self.tracker=RobustHeightPredictor(self.bank,model_id=observer['model_id'],policy=observer['policy'],hold_s=observer['hold_s'],config=self.config,experiment=observer['experiment'],noise_sigma_xyz=self.noise)
  self.times=[];self.xyz=[];self.seq=0;self.cache=None;self.snapshot=None
  return {'initialized':False}
 def update(self,seq,t,xyz):
  xyz=np.asarray(xyz,float)
  if xyz.shape!=(3,) or not np.isfinite(xyz).all() or not np.isfinite(t):raise ValueError('nonfinite observation')
  if seq<=self.seq or (self.times and t<=self.times[-1]):raise ValueError('duplicate/out-of-order observation')
  p=self.R.T@(xyz-self.shift);snap=self.tracker.update(float(t),p)
  self.times.append(float(t));self.xyz.append(p);self.seq=int(seq);self.snapshot=snap;self.cache=None
  state=None
  if snap['initialized']:state=np.r_[self.R@snap['state'][:3]+self.shift,self.R@snap['state'][3:6]].tolist()
  return {'initialized':bool(snap['initialized']),'seq':self.seq,'state':state,'hold_active':bool(snap.get('hold_active',False)),'reason':snap['reason']}
 def forecast(self,seq):
  if seq!=self.seq or self.snapshot is None or not self.snapshot['initialized']:raise ValueError('uninitialized/stale forecast request')
  if self.cache is not None:return self.cache
  start=time.perf_counter();snap=self.snapshot;t=self.times[-1]
  forecast=self.settings['forecast'];residual=self.settings['residual']
  hs=np.linspace(0,self.max_h,int(round(self.max_h/forecast['grid_step_s']))+1)
  # Dense mean rollout once. Extra +/-1ms samples yield the derivative of the
  # same bounded residual function, never differences between observed frames.
  eps=forecast['velocity_difference_step_s'];offsets=np.unique(np.r_[hs,np.maximum(0,hs-eps),np.minimum(self.max_h,hs+eps)]) if self.head else hs
  states,flags=self.tracker._forecast_with_flags(t+offsets)
  positions=states[:,:3].copy()
  gate=residual_gate(snap,wait_s=residual['wait_s'],ramp_s=residual['ramp_s']) if self.head else 0.
  if self.head and gate>0:
   # Snapshot/history features are common to every future horizon. Fit the
   # trailing history once, then evaluate the head as a batch (same formula).
   common=causal_features(snap,positions[0],float(offsets[0]),self.times,self.xyz)
   features=np.tile(common,(len(offsets),1))
   features[:,0]=offsets;features[:,1]=offsets**2
   features[:,9:12]=positions-np.asarray(snap['state'])[:3]
   gains=gate*residual_domain_gain(offsets,residual['maximum_trained_horizon_s'])
   positions+=gains[:,None]*self.head.predict_bounded(features,offsets,maximum_correction_m=residual['maximum_correction_m'])
  base_indices=np.searchsorted(offsets,hs);rows=[];next_bounce=-1.;last_safe=True
  for j,(h,i) in enumerate(zip(hs,base_indices)):
   if j and flags[i] and not flags[base_indices[j-1]]:
    a=base_indices[j-1];dt=offsets[i]-offsets[a]
    next_bounce=t+offsets[a]+float(_contact_dt(states[a:a+1],dt,self.bank.parameters,self.bank.height)[0])
   # This model only describes the first ground impact; never serve underground
   # extrapolation after its subsequent descending ground crossing.
   if flags[i] and states[i,2]<self.bank.height-forecast['second_contact_tolerance_m'] and states[i,5]<0:break
   v=states[i,3:6].copy()
   if self.head and h>0:
    lo=np.searchsorted(offsets,max(0,h-eps));hi=np.searchsorted(offsets,min(self.max_h,h+eps))
    if offsets[hi]>offsets[lo] and flags[lo]==flags[hi]:
     correction=positions-states[:,:3]
     v+=(correction[hi]-correction[lo])/(offsets[hi]-offsets[lo])
   p=self.R@positions[i]+self.shift;v=self.R@v
   rows.append([float(t+h),*p.tolist(),*v.tolist(),bool(flags[i])])
  self.cache={'seq':self.seq,'last_capture_t_s':t,'model_version':self.selection['candidate'],'max_horizon_s':self.max_h,'valid_until':rows[-1][0] if rows else t,'domain_valid':bool(rows),'next_bounce_t_s':next_bounce,'height_contact_t_s':snap.get('height_contact_t_s'),'hold_active':bool(snap.get('hold_active',False)),'resume_t_s':snap.get('resume_t_s'),'future_covariance':None,'grid':rows,'forecast_compute_ms':1000*(time.perf_counter()-start)}
  return self.cache

def service_model_bundle(args):
 if getattr(args,'bundle',None):
  if args.transform or args.synthetic_world or getattr(args,'project_root',None):raise ValueError('bundle already declares resources and transform; do not override them')
  bundle=ModelBundle.load(args.bundle)
 else:bundle=ModelBundle.from_legacy(args.selection,transform_path=args.transform,synthetic_world=args.synthetic_world,root=getattr(args,'project_root',None) or ROOT)
 horizon=getattr(args,'forecast_max_horizon_s',None)
 return bundle if horizon is None else bundle.with_forecast_horizon(horizon)

def serve(args):
 bundle=service_model_bundle(args)
 predictor=ControllerPredictor(bundle=bundle);path=args.socket
 if getattr(args,'resolved_manifest',None):
  args.resolved_manifest.parent.mkdir(parents=True,exist_ok=True)
  with args.resolved_manifest.open('x',encoding='utf-8') as stream:stream.write(json.dumps(predictor.resolved_manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError('socket already exists; refusing to replace another server')
 server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(path));os.chmod(path,0o600);server.listen(1)
 print(json.dumps({'ready':str(path),'selection':predictor.selection['candidate'],'synthetic_frame':bundle.settings['frame']['mode']=='synthetic_world','bundle_fingerprint':bundle.fingerprint}),flush=True)
 try:
  while True:
   conn,_=server.accept();conn.settimeout(30)
   with conn,conn.makefile('rwb') as stream:
    while True:
     try:raw=stream.readline(2000001)
     except (TimeoutError,ConnectionResetError):break
     if not raw:break
     try:
      if len(raw)>2000000:raise ValueError('oversized request')
      req=json.loads(raw)
      if req.get('schema')!=SCHEMA:raise ValueError('schema mismatch')
      op=req.get('op');started=time.perf_counter()
      if op=='reset':r=predictor.reset()
      elif op=='update':r=predictor.update(req['seq'],req['t_s'],req['xyz'])
      elif op=='forecast':r=predictor.forecast(req['seq'])
      else:raise ValueError('unknown operation')
      reply={'schema':SCHEMA,'ok':True,**r}
      if args.telemetry:
       with args.telemetry.open('a') as f:f.write(json.dumps({'op':op,'elapsed_ms':1000*(time.perf_counter()-started)})+'\n')
     except (ValueError,KeyError,TypeError,FloatingPointError) as e:reply={'schema':SCHEMA,'ok':False,'error':str(e)}
     try:stream.write((json.dumps(reply,allow_nan=False)+'\n').encode());stream.flush()
     except (BrokenPipeError,ConnectionResetError):break
 finally:server.close();path.unlink(missing_ok=True)

def main():
 p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True)
 source.add_argument('--selection',type=Path);source.add_argument('--bundle',type=Path)
 p.add_argument('--socket',type=Path,required=True);p.add_argument('--transform',type=Path);p.add_argument('--telemetry',type=Path);p.add_argument('--synthetic-world',action='store_true')
 p.add_argument('--project-root',type=Path,help='legacy selection资源根；默认当前源码项目，bundle模式禁止覆盖')
 p.add_argument('--forecast-max-horizon-s',type=float,help='显式物理预测域实验0.8--2.0s，残差域仍不超过0.7s')
 p.add_argument('--resolved-manifest',type=Path,help='write resolved inputs/config to a new file; refuses overwrite')
 args=p.parse_args()
 if args.bundle and (args.transform or args.synthetic_world or args.project_root):p.error('--bundle already declares its resources/frame; omit --project-root/--transform/--synthetic-world')
 if args.selection and not (args.transform or args.synthetic_world):p.error('--selection requires --transform or --synthetic-world')
 try:serve(args)
 except (OSError,ValueError,KeyError,TypeError) as error:p.exit(2,f'predictor service error: {error}\n')
if __name__=='__main__':main()
