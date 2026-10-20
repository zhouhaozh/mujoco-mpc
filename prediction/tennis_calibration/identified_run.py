"""Ubuntu constrained physics identification and historical regression replay."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .flight import _load_clean, _resolve_clean
from .online_data import (load_dataset, causal_anchors, future_labels,
                          aggregate_predictions, aggregate_landings)
from .online_run import write_csv, timing_summary
from .identified_dynamics import (ASSUMPTIONS, fit_shared_drag_magnus,
                                    fit_conditional_impact, impact_map)
from .identified_filter import (IdentifiedFilter, DEFAULT_CONFIG,
                                  forecast_batch, landing_batch)
from .identified_evaluation import (paired_comparison, read_prediction_rows,
                                      assert_fit_uses_training_only)

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [.1,.2,.5]


def plain(value):
    if isinstance(value,np.ndarray): return plain(value.tolist())
    if isinstance(value,np.generic): return plain(value.item())
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')


def calibrate(calibration,out,*,reuse_aero=False):
    split=json.loads((calibration/'split.json').read_text())
    annotations=json.loads((calibration/'clean/annotations.json').read_text())
    segments, identities, excluded=[],[],[]
    for row in annotations:
        if row['trajectory_id'] not in split['train']: continue
        start,end=row.get('pre_start_t_s'),row.get('pre_end_t_s')
        if start is None or end is None:
            excluded.append({'id':row['trajectory_id'],'reason':'no_pre_segment'});continue
        t,p=_load_clean(_resolve_clean(calibration/'clean',row['cleaned_file']))
        keep=(t>=start)&(t<=end)
        if keep.sum()<24 or np.ptp(t[keep])<.14:
            excluded.append({'id':row['trajectory_id'],'reason':'insufficient_pre_segment'});continue
        segments.append((t[keep],p[keep]));identities.append(row['trajectory_id'])
    training_audit=assert_fit_uses_training_only(identities,split)
    if reuse_aero:
        fit=json.loads((out/'aero_fit.json').read_text())
        if fit['trajectory_ids']!=identities:
            raise ValueError('Reused aero fit training identities changed')
        fit['initial_states']=np.asarray(fit['initial_states'])
        print('IDENTIFIED_AERO_REUSED '+json.dumps({'training':len(identities),'k_drag':fit['k_drag_per_m']}),flush=True)
    else:
        print('IDENTIFIED_AERO_START '+json.dumps({'training':len(segments),'starts':[.01,.02,.05]}),flush=True)
        fit=fit_shared_drag_magnus(segments)
        fit.update(trajectory_ids=identities,excluded=excluded,training_audit=training_audit)
        dump(out/'aero_fit.json',fit)
    if not fit['converged']:
        raise RuntimeError('Aero fitting did not converge; inspect preserved aero_fit.json')
    print('IDENTIFIED_AERO_DONE '+json.dumps({k:fit[k] for k in ('k_drag_per_m','drag_C_D_conditional','converged','nfev')}),flush=True)
    old=json.loads((calibration/'bounce/summary.json').read_text())
    oldp=old['parameters'];height=oldp['effective_contact_center_height_m']
    effective={**ASSUMPTIONS,'k_drag':fit['k_drag_per_m'],'e_n':oldp['e_n'],
               'alpha_t':oldp['alpha_t'],'impact_mode':'effective'}
    state_map={identity:state for identity,state in zip(identities,fit['initial_states'])}
    impacts=[dict(row,b_minus_per_s=state_map[row['trajectory_id']][6:9])
             for row in old['impact_estimates']
             if row['used_for_fit'] and row['trajectory_id'] in state_map]
    impact_audit=assert_fit_uses_training_only([r['trajectory_id'] for r in impacts],split)
    impact_fits={}
    for name,speed_law in [('stick_slip',False),('stick_slip_speed',True)]:
        fitted=fit_conditional_impact(impacts,effective,speed_law=speed_law)
        fitted.update(trajectory_ids=[r['trajectory_id'] for r in impacts],training_audit=impact_audit)
        incoming=np.array([np.r_[np.zeros(3),r['v_minus_m_s'],r['b_minus_per_s']] for r in impacts])
        target=np.array([r['v_plus_m_s'] for r in impacts])
        profile=[]
        for mu in np.linspace(0,1,41):
            prediction=impact_map(incoming,{**fitted['parameters'],'mu':float(mu)})[:,3:6]
            profile.append({'mu':float(mu),'outgoing_velocity_rmse_m_s':float(np.sqrt(np.mean(np.sum((prediction-target)**2,axis=1))))})
        fitted['mu_profile_with_other_parameters_fixed']=profile
        fitted['training_outgoing_velocity_rmse_m_s']=float(np.sqrt(np.mean(np.sum((impact_map(incoming,fitted['parameters'])[:,3:6]-target)**2,axis=1))))
        impact_fits[name]=fitted
        print('IDENTIFIED_IMPACT_DONE '+json.dumps({'name':name,'parameters':fitted['parameters'],'converged':fitted['converged']}),flush=True)
    dump(out/'impact_fit.json',impact_fits)
    base_old=json.loads((calibration/'flight/shared_parameters.json').read_text())['k_drag_per_m']
    result={'effective':effective,'repaired6':{**effective,'k_drag':base_old},
            'contact_height':height,'assumptions':ASSUMPTIONS,
            'test_role':'previously exposed reused regression holdout; no fresh blind test'}
    for key,value in impact_fits.items():
        if not value['converged']: raise RuntimeError('Conditional impact fit did not converge: '+key)
        result[key]=value['parameters']
    dump(out/'calibrated_parameters.json',result)
    return result


def replay(traces,parameters,height,config,augmented,name):
    cached,times,diagnostics={},[],Counter()
    for n,trace in enumerate(traces):
        tracker=IdentifiedFilter(parameters,height,config=config,augmented=augmented)
        anchors={a['current_index']:a for a in causal_anchors(trace)}
        records=[]
        for index,(stamp,xyz) in enumerate(zip(trace['raw_times'],trace['raw_xyz'])):
            snap=tracker.update(float(stamp),xyz)
            diagnostics[snap['reason']]+=1
            times.append({'ms':snap['diagnostics']['update_latency_ms'],'initialization':snap['reason']=='initialized'})
            if index in anchors:
                diagnostics['anchors']+=1
                if not snap['initialized']: diagnostics['uninitialized_anchors']+=1
                records.append({'t_s':float(stamp),'snapshot':snap})
        cached[trace['id']]=records
        if (n+1)%10==0 or n+1==len(traces): print(f'IDENTIFIED_REPLAY {name} {n+1}/{len(traces)}',flush=True)
    return cached,times,dict(diagnostics)


def score(traces,cached,parameters,height,config,name):
    pr,lr=[],[]
    for trace in traces:
        records=cached[trace['id']]
        usable=[r for r in records if r['snapshot']['initialized']]
        pp,land=None,None
        indices={id(r):i for i,r in enumerate(usable)}
        if usable:
            states=np.array([r['snapshot']['state'] for r in usable])
            flags=np.array([r['snapshot']['bounced'] for r in usable])
            pp=forecast_batch(states,flags,HORIZONS,parameters,height,config)
            land=landing_batch(states,flags,parameters,height,config)
        for r in records:
            stamp=r['t_s'];j=indices.get(id(r))
            identity={'trajectory_id':trace['id'],'split':trace['split'],'groupid':trace['groupid'],
                      'capture_t_s':stamp,'anchor_t_s':stamp,'scenario_delay_s':0.,'model':name}
            for k,label in enumerate(future_labels(trace,stamp,HORIZONS)):
                prediction=None if j is None else pp[j,k].tolist()
                pr.append(dict(identity,**label,predicted_xyz=prediction,
                               prediction_reason='not_initialized' if j is None else ''))
            row=dict(identity,target_time_s=trace['bounce_reference_s'],
                     target_xy=None if trace['bounce_reference_xy'] is None else trace['bounce_reference_xy'].tolist(),
                     predicted_time_s=None,predicted_xy=None,prediction_reason='not_initialized')
            if j is not None:
                row['prediction_reason']=str(land['reason'][j])
                if land['valid'][j] and land['time_to_contact'][j]>0:
                    row.update(predicted_time_s=stamp+float(land['time_to_contact'][j]),
                               predicted_xy=land['position'][j,:2].tolist(),prediction_reason='')
            lr.append(row)
    return pr,lr


def macro(rows):
    result=aggregate_predictions(rows)
    value=result['trajectory_macro']['mean_per_trajectory_rmse_m']
    if result['prediction_coverage_given_label'] != 1:
        # Never select an incomplete model by an error score omitting failures.
        return float('inf')
    return float(value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',type=Path,default=ROOT/'artifacts/tennis_calibration/20261008_calibration')
    parser.add_argument('--baseline',type=Path,default=ROOT/'artifacts/tennis_online/20261008_final')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_identified/20261009_identified')
    parser.add_argument('--stage',choices=['calibration','impact','replay','all'],default='all')
    parser.add_argument('--training-prior',action='store_true',help='Derive online b prior scale from training fits, then select on validation only')
    args=parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Formal identification/replay must use Ubuntu project .venv-tennis')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    source_sha={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT/'tennis_calibration').glob('identified*.py'))}
    dump(out/'execution.json',{'hostname':platform.node(),'platform':platform.platform(),
          'python':sys.version,'python_prefix':sys.prefix,'created_utc':datetime.now(timezone.utc).isoformat(),
          'source_sha256_at_start':source_sha,'assumptions':ASSUMPTIONS})
    if args.stage in ('calibration','impact','all'):
        parameters=calibrate(args.calibration,out,reuse_aero=args.stage=='impact')
    else:
        parameters=json.loads((out/'calibrated_parameters.json').read_text())
    if args.stage in ('calibration','impact'): return
    traces=load_dataset(args.calibration)
    val=[t for t in traces if t['split']=='validation']
    baseline_pr=[r for r in read_prediction_rows(args.baseline/'predictions.csv') if r['scenario_delay_s']==0]
    baseline_lr=[r for r in read_prediction_rows(args.baseline/'landings.csv') if r['scenario_delay_s']==0]
    candidate_specs=[('repaired6','repaired6',False,.12,.6)]
    if args.training_prior:
        training_fit=json.loads((out/'aero_fit.json').read_text())
        split=json.loads((args.calibration/'split.json').read_text())
        assert_fit_uses_training_only(training_fit['trajectory_ids'],split)
        b=np.asarray(training_fit['initial_states'])[:,6:9]
        sigma=max(.02,float(np.sqrt(np.mean(np.sum(b*b,axis=1))/3)))
        drives=[max(.003,.1*sigma),max(.003,.3*sigma)]
        prior={'source':'training_aero_fit_only','mean':'zero','scale_formula':'max(.02, sqrt(mean(norm(b)^2)/3))',
               'initial_sigma_per_s':sigma,'drive_sigmas_per_s_sqrt_s':drives,
               'n_training_trajectories':len(b),'b_norm_p50_per_s':float(np.median(np.linalg.norm(b,axis=1))),
               'b_norm_p95_per_s':float(np.quantile(np.linalg.norm(b,axis=1),.95)),
               'test_used':False}
        dump(out/'training_prior.json',prior)
        candidate_specs += [(f'magnus_effective_prior_s{s}_q{q}','effective',True,drive,sigma*factor)
                             for s,factor in enumerate((1.,2.),1) for q,drive in enumerate(drives,1)]
        candidate_specs += [(f'magnus_{law}_prior',law,True,drives[0],sigma)
                             for law in ('stick_slip','stick_slip_speed')]
    else:
        candidate_specs += [(f'magnus_effective_{sigma:g}','effective',True,sigma,.6) for sigma in (.03,.12,.4)]
        candidate_specs += [(f'magnus_{law}_{sigma:g}',law,True,sigma,.6)
                             for law in ('stick_slip','stick_slip_speed') for sigma in (.03,.12)]
    validations,caches,timer,scored=[],{}, {},{}
    height=parameters['contact_height']
    for name,law,augmented,drive,initial in candidate_specs:
        cfg=dict(DEFAULT_CONFIG,magnus_drive_sigma=drive,initial_magnus_sigma=initial)
        cache,times,diag=replay(val,parameters[law],height,cfg,augmented,name+'/validation')
        pr,lr=score(val,cache,parameters[law],height,cfg,name)
        item={'name':name,'law':law,'augmented':augmented,'config':cfg,'parameters':parameters[law],
              'macro_rmse_m':macro(pr),'prediction':aggregate_predictions(pr),
              'landing':aggregate_landings(lr),'diagnostics':diag}
        validations.append(item);caches[name]=cache;timer[name]=times;scored[name]=(pr,lr)
        dump(out/'validation_selection.json',validations)
        print('IDENTIFIED_VALIDATION '+json.dumps({'name':name,'macro_rmse_m':item['macro_rmse_m']}),flush=True)
    selected=[validations[0]]
    for law in ('effective','stick_slip','stick_slip_speed'):
        pool=[v for v in validations if v['law']==law and v['augmented']]
        selected.append(min(pool,key=lambda x:x['macro_rmse_m']))
    winner=min(selected,key=lambda x:x['macro_rmse_m'])
    deployment={'schema':'TENNIS_IDENTIFIED/1','selected_name':winner['name'],
       'parameters':winner['parameters'],'contact_height':height,'filter':winner['config'],
       'augmented':winner['augmented'],'selection_source':'validation_only_frozen_before_test_replay',
       'models_for_regression':[s['name'] for s in selected],'assumptions':ASSUMPTIONS,
       'observations':'raw finite monotonic tXYZ; no offline keep or impact labels',
       'test_role':'historical exposed test groups reused as regression comparison'}
    dump(out/'selected_model.json',deployment)
    frozen_sha=hashlib.sha256((out/'selected_model.json').read_bytes()).hexdigest()
    print('IDENTIFIED_MODEL_FROZEN '+frozen_sha,flush=True)
    all_pr,all_lr=list(baseline_pr),list(baseline_lr)
    timing={}
    others=[t for t in traces if t['split']!='validation']
    for item in selected:
        name=item['name']
        cache,times,diag=replay(others,item['parameters'],height,item['config'],item['augmented'],name+'/train_test')
        cache.update(caches[name])
        pr,lr=score(traces,cache,item['parameters'],height,item['config'],name)
        all_pr.extend(pr);all_lr.extend(lr)
        timing[name]={'update':timing_summary(timer[name]+times),'diagnostics_train_test':diag}
    metrics={}
    for split in ('train','validation','test'):
        metrics[split]={}
        for name in sorted({r['model'] for r in all_pr}):
            pr=[r for r in all_pr if r['split']==split and r['model']==name]
            lr=[r for r in all_lr if r['split']==split and r['model']==name]
            metrics[split][name]={'prediction':aggregate_predictions(pr),'landing':aggregate_landings(lr)}
    paired=paired_comparison(all_pr,all_lr)
    write_csv(out/'predictions.csv',all_pr);write_csv(out/'landings.csv',all_lr)
    dump(out/'paired_comparison.json',paired)
    report={'schema':'TENNIS_IDENTIFIED_RUN/1','completed':True,'hostname':platform.node(),
        'created_utc':datetime.now(timezone.utc).isoformat(),'metrics':metrics,'timing':timing,
        'deployment':deployment,'model_sha256_frozen_before_test':frozen_sha,
        'baseline_sha256':{f:hashlib.sha256((args.baseline/f).read_bytes()).hexdigest()
                           for f in ('config.json','predictions.csv','landings.csv')},
        'evaluation_role':'historical holdout already exposed; regression, not fresh blind validation',
        'reference':'same offline provisional cleaned positions and firstcontact labels as old run',
        'source_sha256_at_start':source_sha,'assumptions':ASSUMPTIONS,
        'limitations':['t/XYZ units and center/ground frame are inherited assumptions',
                       'b_eff is effective lift, not measured RPM',
                       'mu and spin conversion conditional on fixed lift slope and thin-shell inertia',
                       'firstbounce only; covariance not a calibrated confidence interval']}
    dump(out/'summary.json',report)
    from .identified_plots import make_identified_plots
    make_identified_plots(out,all_pr,all_lr,report)
    print('IDENTIFIED_REPORT='+str(out/'summary.json'),flush=True)


if __name__=='__main__': main()
