"""Causality, domain, frame and actual C++/Python IPC checks on Ubuntu."""
import json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
import numpy as np
from tennis_calibration.controller_service import ControllerPredictor,ROOT
from tennis_calibration.controller_contract import residual_domain_gain
from tennis_calibration.model_bundle import ModelBundle
from tennis_calibration.online_data import load_dataset

class ControllerTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.selection=json.loads((ROOT/'artifacts/tennis_controller/20261014/selected.json').read_text())
  cls.trace=load_dataset(ROOT/'artifacts/new_ubuntu/tennis_calibration')[0]
  cls.frame={'R_world_from_model':np.eye(3).tolist(),'translation_world_from_model_m':[0,0,0]}
 def make(self):return ControllerPredictor(self.selection,self.frame)
 def seed(self,p,n=80):
  for i,(t,x) in enumerate(zip(self.trace['raw_times'][:n],self.trace['raw_xyz'][:n])):p.update(i+1,float(t),x)
 def test_duplicate_and_nonfinite_rejected_without_advancing(self):
  p=self.make();p.update(1,0,[0,0,1]);self.assertRaises(ValueError,p.update,2,0,[1,1,1]);self.assertEqual(p.seq,1)
  self.assertRaises(ValueError,p.update,2,.1,[np.nan,0,1]);self.assertEqual(len(p.times),1)
 def test_forecast_read_only_bounded_and_versioned(self):
  p=self.make();self.seed(p);state=p.tracker.state.copy();stamp=p.tracker.last_t;r=p.forecast(p.seq)
  np.testing.assert_array_equal(state,p.tracker.state);self.assertEqual(stamp,p.tracker.last_t)
  self.assertLessEqual(r['valid_until']-stamp,1.5+1e-12);self.assertIs(r,p.forecast(p.seq));self.assertRaises(ValueError,p.forecast,p.seq-1)
  self.assertTrue(np.isfinite(np.asarray(r['grid'],float)).all())
  p.reset();self.assertRaises(ValueError,p.forecast,0)
 def test_batch_residual_matches_scalar_after_resume(self):
  from tennis_calibration.residual_head import causal_features,apply_residual,residual_gate
  p=self.make();self.seed(p,len(self.trace['raw_times'])-20)
  # Explicit causal resumed snapshot; isolate the output correction calculation.
  snap=p.snapshot;snap.update(bounced=True,hold_active=False,resume_t_s=p.times[-1]-.2,height_contact_t_s=p.times[-1]-.4)
  hs=np.array([0,.1,.2,.5,.6,.7,1.2]);raw=np.tile(np.array([1.,2.,1.]),(len(hs),1));common=causal_features(snap,raw[0],0,p.times,p.xyz)
  features=np.tile(common,(len(hs),1));features[:,0]=hs;features[:,1]=hs**2;features[:,9:12]=raw-np.asarray(snap['state'])[:3]
  batch=raw+(residual_gate(snap)*residual_domain_gain(hs))[:,None]*p.head.predict_bounded(features,hs)
  scalar=np.array([r+residual_domain_gain(h)*(apply_residual(p.head,causal_features(snap,r,h,p.times,p.xyz),snap,h,r)-r) for h,r in zip(hs,raw)])
  np.testing.assert_allclose(batch,scalar,atol=1e-12)
 def test_second_ground_domain_returns_empty_forecast_not_service_failure(self):
  p=self.make();self.seed(p)
  # A downward state below the floor after first impact is outside this model.
  p.tracker._forecast_with_flags=lambda ts:(np.tile([0,0,p.bank.height-.02,0,0,-1.],(len(ts),1)),np.ones(len(ts),bool))
  r=p.forecast(p.seq);self.assertFalse(r['domain_valid']);self.assertEqual(r['grid'],[])
 def test_frame_rotation_translation_roundtrip(self):
  a=self.make();angle=.4;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
  shift=np.array([2.,-1.,.1]);b=ControllerPredictor(self.selection,{'R_world_from_model':R.tolist(),'translation_world_from_model_m':shift.tolist()})
  for i,(t,x) in enumerate(zip(self.trace['raw_times'][:80],self.trace['raw_xyz'][:80])):a.update(i+1,float(t),x);b.update(i+1,float(t),R@x+shift)
  x=np.asarray(a.forecast(a.seq)['grid'],float);y=np.asarray(b.forecast(b.seq)['grid'],float)
  np.testing.assert_allclose(y[:,1:4],x[:,1:4]@R.T+shift,atol=1e-7);np.testing.assert_allclose(y[:,4:7],x[:,4:7]@R.T,atol=1e-6)
 def test_residual_domain_is_not_full_physics_domain(self):
  np.testing.assert_allclose(residual_domain_gain([.1,.5,.6,.7,1.5]),[1,1,.5,0,0],atol=1e-14)
 def test_bad_transform_rejected(self):
  self.assertRaises(ValueError,ControllerPredictor,self.selection,{'R_world_from_model':(-np.eye(3)).tolist(),'translation_world_from_model_m':[0,0,0]})
 def test_bundle_roundtrip_preserves_state_forecast_and_cache(self):
  legacy=self.make()
  bundle=ModelBundle.from_legacy(self.selection,self.frame,root=ROOT)
  with tempfile.TemporaryDirectory() as folder:
   path=bundle.write(Path(folder)/'bundle.json')
   deployed=ControllerPredictor(bundle=ModelBundle.load(path))
   self.seed(legacy);self.seed(deployed)
   np.testing.assert_array_equal(legacy.tracker.state,deployed.tracker.state)
   a=legacy.forecast(legacy.seq);b=deployed.forecast(deployed.seq)
   np.testing.assert_array_equal(a['grid'],b['grid'])
   for key in a:
    if key not in ('grid','forecast_compute_ms'):self.assertEqual(a[key],b[key])
   self.assertIs(b,deployed.forecast(deployed.seq))
   self.assertEqual(deployed.bundle_fingerprint,bundle.fingerprint)
 def test_independent_physics_horizons_share_observer_and_common_forecast_prefix(self):
  bundle=ModelBundle.from_legacy(self.selection,self.frame,root=ROOT)
  baseline=ControllerPredictor(bundle=bundle);self.seed(baseline)
  original=baseline.forecast(baseline.seq)
  forecasts={}
  for horizon in (.8,1.,1.2,1.5,1.8,2.):
   predictor=ControllerPredictor(bundle=bundle,forecast_max_horizon_s=horizon);self.seed(predictor)
   np.testing.assert_array_equal(predictor.tracker.state,baseline.tracker.state)
   result=predictor.forecast(predictor.seq);forecasts[horizon]=result
   self.assertEqual(result['max_horizon_s'],horizon)
   self.assertLessEqual(result['valid_until'],predictor.times[-1]+horizon+1e-12)
   self.assertEqual(result['domain_valid'],bool(result['grid']))
   if result['grid']:self.assertEqual(result['valid_until'],result['grid'][-1][0])
   self.assertEqual(predictor.settings['residual'],baseline.settings['residual'])
  # No truncation-derived caches: every forecast above was computed separately.
  longest=np.asarray(forecasts[2.]['grid'],float)
  for horizon,result in forecasts.items():
   grid=np.asarray(result['grid'],float)
   if len(grid):np.testing.assert_allclose(grid,longest[:len(grid)],rtol=0,atol=1e-10)
  np.testing.assert_array_equal(original['grid'],forecasts[1.5]['grid'])
  for key in original:
   if key not in ('grid','forecast_compute_ms'):self.assertEqual(original[key],forecasts[1.5][key])
 @unittest.skipUnless(os.environ.get('TEAM_PREDICTOR_PROBE'),'requires built team C++ probe')
 def test_actual_cpp_consumes_same_snapshot_and_rejects_overhorizon(self):
  # Identity transform deliberately isolates transport from synthetic frame map.
  (ROOT/'.tmp').mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=ROOT/'.tmp',prefix='ipc-') as folder:
   folder=Path(folder);sock=folder/'p.sock';frame=folder/'frame.json';frame.write_text(json.dumps(self.frame))
   cmd=[sys.executable,'-m','tennis_calibration.controller_service','--selection',str(ROOT/'artifacts/tennis_controller/20261014/selected.json'),'--socket',str(sock),'--transform',str(frame)]
   process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
   try:
    for _ in range(100):
     if sock.exists():break
     if process.poll() is not None:self.fail(process.stderr.read())
     time.sleep(.02)
    p=self.make();self.seed(p);raw=''.join(','.join(map(str,[t,*x]))+'\n' for t,x in zip(self.trace['raw_times'][:80],self.trace['raw_xyz'][:80]))
    result=subprocess.run([os.environ['TEAM_PREDICTOR_PROBE'],str(sock)],input=raw,text=True,capture_output=True,timeout=15);self.assertEqual(result.returncode,0,result.stderr)
    out=json.loads(result.stdout);grid=np.asarray(p.forecast(p.seq)['grid'],float)
    for s in out['samples']:
     target=out['t_s']+s['horizon_s']
     if s['valid']:
      index=np.argmin(abs(grid[:,0]-target));np.testing.assert_allclose(s['p'],grid[index,1:4],atol=1e-8);np.testing.assert_allclose(s['v'],grid[index,4:7],atol=1e-8)
     if s['horizon_s']>1.5:self.assertFalse(s['valid'])
   finally:process.terminate();process.wait(timeout=5);process.stderr.close()
if __name__=='__main__':unittest.main()
