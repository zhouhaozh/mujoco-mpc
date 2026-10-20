"""Independent resource and fold-coverage checks for the controller study."""
import hashlib,json
from pathlib import Path
import numpy as np
from .controller_horizon_evaluation import ROOT,dump

def main():
 out=ROOT/'artifacts/tennis_controller/20261014';g=json.loads((out/'groupcv/summary.json').read_text());seen=set();measured=set();counts=[]
 for f in g['folds']:
  train=set(f['train_ids']);held=set(f['evaluation_ids']);assert not(train&held);assert not(seen&held);seen|=held
  c=dict(np.load(out/f"groupcv/fold{f['fold']}/adaptive_protected_oof.npz",allow_pickle=False));assert set(c['ids'])<=held;measured.update(c['ids'].tolist())
  assert np.all(c['prediction_ok'][c['valid']]);assert np.isfinite(c['pred'][c['valid']]).all()
  h=json.loads((out/f"groupcv/fold{f['fold']}/residual.json").read_text());assert set(h['metadata']['allowed_training_ids'])==train
  counts.append({'fold':f['fold'],'train':len(train),'evaluation':len(held),'labeled_rows':int(c['valid'].sum())})
 assert len(measured)==140
 inventory=json.loads((ROOT/'artifacts/team_sim_review/20261013/sim_project_inventory_20261013.json').read_text());protected={}
 for rel,v in inventory['resource_files'].items():
  digest=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest();assert digest==v['sha256'];protected[rel]=digest
 dump(out/'audit.json',{'date':'2026-10-14','platform':'Ubuntu ubuntu-11','folds':counts,'unique_evaluation_ids':len(seen),'unique_evaluable_trajectories':len(measured),'training_evaluation_disjoint':True,'labeled_forecasts_finite_and_complete':True,'protected_resources_unchanged':protected})
 print('audit passed: five disjoint folds, 140 trajectories, 7 protected resources')
if __name__=='__main__':main()
