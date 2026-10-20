"""Ubuntu-only frozen precontact -> postcontact learning and evaluation.

References define evaluation cuts/targets. Predictors receive only the raw
prefix, fixed training parameters, and their own predicted contact state.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .online_data import load_dataset, future_labels, error_summary
from .online_run import write_csv
from .online_filter import OnlineFilter
from .identified_filter import IdentifiedFilter
from .identified_dynamics import step
from .identified_run import plain, dump
from .collision_prefix import estimate_prefix, predict_contact
from .collision_forecast import hybrid_forecast
from .collision_mapping import (pre_features, baseline_velocity, velocity_targets,
    trajectory_weights, fit_model, predict_model, predict_velocity,
    post_lift_targets, predict_post_lift, save_model)

ROOT=Path(__file__).resolve().parents[1]
GAPS=[0.,.005,.01,.02,.04,.08]
HORIZONS=[.02,.05,.1,.2,.5]
# 0.005 is included explicitly; 0.04 roughly spans the largest observed gap.
WINDOWS={'full':0.,'recent':.4}


def identity(trace,gap):
    return trace['id']+'|'+format(gap,'.6f')


def fingerprint(times,xyz):
    sha=hashlib.sha256()
    sha.update(np.ascontiguousarray(times).tobytes());sha.update(np.ascontiguousarray(xyz).tobytes())
    return sha.hexdigest()


def fit_case(trace,gap,parameters,height,window,*,audit_only=False):
    from .collision_prefix import DEFAULT_CONFIG as prefix_defaults
    reference=trace['bounce_reference_s']
    cutoff=reference-gap
    # Slice here AND inside the estimator, ensuring post data are never passed.
    mask=trace['raw_times']<=cutoff
    times,xyz=trace['raw_times'][mask],trace['raw_xyz'][mask]
    context={'schema':'PREFIX_CACHE_CONTEXT/2','parameters':plain(parameters),
       'contact_height_m':float(height),'window_s':float(window),
       'prefix_defaults':plain(prefix_defaults),
       'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (ROOT/'tennis_calibration/collision_prefix.py',
                    ROOT/'tennis_calibration/identified_dynamics.py',
                    ROOT/'tennis_calibration/flight.py')}}
    context_sha=hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':'),
                                        ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    record={'trajectory_id':trace['id'],'split':trace['split'],'groupid':trace['groupid'],
       'planned_gap_s':gap,'cut_t_s':cutoff,'input_count':len(times),
       'last_input_t_s':None if not len(times) else float(times[-1]),
       'input_sha256':fingerprint(times,xyz),'context_sha256':context_sha,'context_payload':context,
       'input_role':'raw prefix only; no keep flags, reference contact or post observations passed'}
    if audit_only:
        return plain(record)
    started=perf_counter()
    fit=estimate_prefix(times,xyz,cutoff,parameters,height,{'window_s':window})
    record.update(fit=fit,fit_elapsed_ms=(perf_counter()-started)*1000)
    return plain(record)


def cache_cases(traces,parameters,height,out,requests):
    path=out/'prefix_cache.json'
    cache=json.loads(path.read_text()) if path.exists() else {'full':{},'recent':{}}
    done=0;total=sum(len(gaps) for mode,trace,gaps in requests)
    for mode,trace,gaps in requests:
        for gap in gaps:
            key=identity(trace,gap)
            if key not in cache[mode]:
                cache[mode][key]=fit_case(trace,gap,parameters,height,WINDOWS[mode])
            else:
                expected=fit_case(trace,gap,parameters,height,WINDOWS[mode],audit_only=True)
                stored=cache[mode][key]
                if 'context_sha256' not in stored:
                    raise RuntimeError('Prefix cache lacks context_sha256: use a new output directory '
                                       'or perform explicit verified migration; silent reuse is prohibited')
                if stored['context_sha256']!=expected['context_sha256']:
                    raise RuntimeError('Prefix cache context mismatch (parameters, height, window, '
                                       'defaults or source changed): use a new output directory')
                if stored.get('input_sha256')!=expected['input_sha256']:
                    raise RuntimeError('Prefix cache raw-prefix fingerprint mismatch: '
                                       'use a new output directory')
            done+=1
            if done%10==0:
                dump(path,cache)
                print(f'COLLISION_PREFIX {done}/{total} {mode}',flush=True)
    dump(path,cache)
    return cache


def prepare(calibration,selected,out):
    traces=load_dataset(calibration)
    eligible=[t for t in traces if t['bounce_reference_s'] is not None]
    excluded=[{'id':t['id'],'reason':'no_firstcontact_reference_for_cut_or_targets'}
              for t in traces if t['bounce_reference_s'] is None]
    old=json.loads((calibration/'bounce/summary.json').read_text())
    impacts={r['trajectory_id']:r for r in old['impact_estimates']}
    training=[t for t in eligible if t['split']=='train' and impacts.get(t['id'],{}).get('used_for_fit')]
    validation=[t for t in eligible if t['split']=='validation']
    parameters=selected['parameters'];height=selected['contact_height']
    requests=[(mode,t,[0.]) for mode in WINDOWS for t in training]
    requests += [(mode,t,GAPS) for mode in WINDOWS for t in validation]
    cache=cache_cases(eligible,parameters,height,out,requests)
    heights={};audit={}
    for mode in WINDOWS:
        values=[];ids=[]
        for t in training:
            fit=cache[mode][identity(t,0.)]['fit']
            if fit['converged']:
                at=step(np.array(fit['state']),t['bounce_reference_s']-fit['capture_t_s'],parameters['k_drag'])
                values.append(float(at[2]));ids.append(t['id'])
        if len(values)<20: raise RuntimeError('Too few training prefixes for event-height calibration')
        heights[mode]=float(np.median(values))
        audit[mode]={'fit_trajectory_ids':ids,'n_training':len(ids),'effective_height_m':heights[mode],
                     'mad_height_m':float(np.median(abs(np.asarray(values)-heights[mode]))),
                     'definition':'median pre-only fitted z at offline training reference event; not true ground'}
    dump(out/'height_calibration.json',audit)
    dump(out/'dataset_audit.json',{'n_trajectories':len(traces),'eligible':len(eligible),'excluded':excluded,
       'training_reference_ids':[t['id'] for t in training],
       'split_counts':dict(Counter(t['split'] for t in eligible)),
       'gaps_s':GAPS,'post_horizons_s':HORIZONS,
       'old_test_role':'previously exposed historical groups, reused regression only'})
    return traces,eligible,training,validation,impacts,cache,heights


def case_pack(traces,cache,mode,gaps,parameters,height):
    records=[]
    for trace in traces:
        for gap in gaps:
            base=cache[mode][identity(trace,gap)]
            fit=base['fit']
            contact=predict_contact(fit,parameters,height)
            records.append({'trace':trace,'gap':gap,'base':base,'fit':fit,'contact':contact})
    return records


def score_pack(pack,parameters,height,name,v_model=None,b_policy='carry',lift_model=None):
    rows,events=[],[]
    active=[c for c in pack if c['contact']['valid']]
    forecasts={};event_predictions={}
    if active:
        vm=np.array([c['contact']['v_minus'] for c in active])
        bm=np.array([c['contact']['b_minus'] for c in active])
        vp=baseline_velocity(vm,parameters['e_n'],parameters['alpha_t']) if v_model is None else predict_velocity(v_model,vm,bm,parameters['e_n'],parameters['alpha_t'])
        bp=bm.copy() if b_policy=='carry' else np.zeros_like(bm) if b_policy=='zero' else .5*bm
        if lift_model is not None: bp=predict_post_lift(lift_model,vm,bm)
        plus=np.array([c['contact']['state_minus'] for c in active]);plus[:,3:6]=vp;plus[:,6:9]=bp
        capture=np.array([c['fit']['state'] for c in active]);ct=np.array([c['fit']['capture_t_s'] for c in active])
        eventt=np.array([c['contact']['contact_t_s'] for c in active])
        target=np.array([[c['trace']['bounce_reference_s']+h for h in HORIZONS] for c in active])
        prediction=hybrid_forecast(capture,ct,np.array([c['contact']['state_minus'] for c in active]),eventt,plus,target,parameters)
        for i,c in enumerate(active):
            key=identity(c['trace'],c['gap'])
            forecasts[key]=prediction[i]
            event_predictions[key]={'predicted_v_minus':vm[i].tolist(),'predicted_v_plus':vp[i].tolist(),
               'predicted_pb':c['contact']['position'],'predicted_tb':float(eventt[i]),
               'predicted_b_plus':bp[i].tolist(),
               'normal_rebound_positive':bool(vp[i,2]>0)}
    for c in pack:
        trace,gap=c['trace'],c['gap'];key=identity(trace,gap)
        cap=c['fit']['capture_t_s'];actual_gap=None if cap is None else trace['bounce_reference_s']-cap
        base={'trajectory_id':trace['id'],'groupid':trace['groupid'],'split':trace['split'],'model':name,
              'gap_s':gap,'planned_gap_s':gap,'actual_gap_s':actual_gap,'capture_t_s':cap,
              'input_count':c['base']['input_count'],'input_end_t_s':c['base']['last_input_t_s']}
        for j,label in enumerate(future_labels(trace,trace['bounce_reference_s'],HORIZONS)):
            pred=forecasts.get(key)
            rows.append(dict(base,**label,predicted_xyz=None if pred is None else pred[j,:3].tolist(),
                  prediction_reason=c['contact']['reason'] if pred is None else '',
                  velocity_predicted=None if pred is None else pred[j,3:6].tolist()))
        ref=trace.get('impact_reference',{})
        event=dict(base,tb_ref=trace['bounce_reference_s'],pb_ref=ref.get('p_b_m'),
             v_minus_ref=ref.get('v_minus_m_s'),v_plus_ref=ref.get('v_plus_m_s'),
             predicted_v_minus=None,predicted_v_plus=None,predicted_pb=None,predicted_tb=None,
             prediction_reason=c['contact']['reason'])
        event.update(event_predictions.get(key,{}));events.append(event)
    return plain(rows),plain(events)


def selection_score(rows):
    labeled=[r for r in rows if r['valid']]
    failures=sum(r['predicted_xyz'] is None for r in labeled)
    grouped={}
    for r in labeled:
        if r['predicted_xyz'] is not None:
            error=np.linalg.norm(np.asarray(r['predicted_xyz'])-r['target_xyz'])
            grouped.setdefault(r['trajectory_id'],[]).append(error*error)
    macro=float(np.mean([np.sqrt(np.mean(v)) for v in grouped.values()])) if grouped else np.inf
    return (failures,macro)


def learning_arrays(pack,impacts,parameters):
    active=[c for c in pack if c['contact']['valid'] and impacts.get(c['trace']['id'],{}).get('used_for_fit')]
    vm=np.array([c['contact']['v_minus'] for c in active]);bm=np.array([c['contact']['b_minus'] for c in active])
    target=np.array([impacts[c['trace']['id']]['v_plus_m_s'] for c in active])
    X,Y=velocity_targets(vm,bm,target,parameters['e_n'],parameters['alpha_t'])
    weights=trajectory_weights([c['trace']['id'] for c in active])
    return active,X,Y,weights,vm,bm


def run_experiment(calibration,selected,out,prepared):
    traces,eligible,training,validation,impacts,cache,heights=prepared
    for t in eligible: t['impact_reference']=impacts.get(t['id'],{})
    parameters=selected['parameters'];old_height=selected['contact_height']
    # First choose the prefix/event estimator on validation, with fixed contact law.
    upstream=[]
    for mode in WINDOWS:
        for height_name,height in [('legacy',old_height),('trained',heights[mode])]:
            pack=case_pack(validation,cache,mode,GAPS,parameters,height)
            rows,events=score_pack(pack,parameters,height,'upstream_'+mode+'_'+height_name)
            score=selection_score(rows)
            upstream.append({'mode':mode,'height_name':height_name,'height':height,
                             'failed_labeled':score[0],'macro_rmse_m':score[1]})
    choice=min(upstream,key=lambda x:(x['failed_labeled'],x['macro_rmse_m']))
    dump(out/'upstream_selection.json',{'source':'validation only; frozen prefix and constant contact baseline',
                                       'candidates':upstream,'selected':choice})
    print('COLLISION_UPSTREAM '+json.dumps(choice),flush=True)
    mode,height=choice['mode'],choice['height']
    # Complete training prefixes. Test arrays are loaded, but not fitted or used for selection.
    requests=[(mode,t,GAPS) for t in training]
    cache=cache_cases(eligible,parameters,old_height,out,requests)
    train_pack=case_pack(training,cache,mode,GAPS,parameters,height)
    val_pack=case_pack(validation,cache,mode,GAPS,parameters,height)
    active,X,Y,w,vm,bm=learning_arrays(train_pack,impacts,parameters)
    configurations=[{'kind':'ridge','degree':1,'ridge':.1},
                    {'kind':'ridge','degree':2,'ridge':1.},
                    {'kind':'gp','length_scale':1.,'noise':.3},
                    {'kind':'gp','length_scale':1.,'noise':.6},
                    {'kind':'gp','length_scale':2.,'noise':.3},
                    {'kind':'gp','length_scale':2.,'noise':.6}]
    models={};training_reports={};specs=[];validation_rows=[];validation_events=[]
    for policy in ('carry','zero','shrink'):
        specs.append({'name':'fixed_'+policy,'velocity_model':None,'b_policy':policy,'lift_model':None})
    for i,cfg in enumerate(configurations):
        key='velocity_'+str(i)
        model,report=fit_model(X,Y,w,cfg);models[key]=model;training_reports[key]=report
        save_model(model,out/'models'/key)
        for policy in ('carry','zero','shrink'):
            specs.append({'name':key+'_'+policy,'velocity_model':key,'b_policy':policy,'lift_model':None})
    labels=json.loads((ROOT/'artifacts/tennis_collision/20261009_post_labels_with_covariance.json').read_text())
    post={r['trajectory_id']:r for r in labels['records']}
    label_list=[post.get(c['trace']['id'],{}) for c in active]
    lx,ly,sigma=post_lift_targets(vm,bm,label_list)
    reliable=(sigma<=.08)
    reliable_independent_events=len(set(c['trace']['id'] for c,yes in zip(active,reliable) if yes))
    if reliable_independent_events>=20:
        for i,cfg in enumerate(({'kind':'ridge','degree':1,'ridge':1.},
                                {'kind':'gp','length_scale':2.,'noise':.6})):
            key='lift_'+str(i)
            model,report=fit_model(lx[reliable],ly[reliable],trajectory_weights([c['trace']['id'] for c,yes in zip(active,reliable) if yes]),cfg)
            models[key]=model;training_reports[key]=report;save_model(model,out/'models'/key)
            # Evaluate paired with the fixed physical velocity, then best residual velocity.
            specs.append({'name':'fixed_'+key,'velocity_model':None,'b_policy':'carry','lift_model':key})
            for vel in ('velocity_0','velocity_3','velocity_5'):
                specs.append({'name':vel+'_'+key,'velocity_model':vel,'b_policy':'carry','lift_model':key})
    scores=[]
    for spec in specs:
        rows,events=score_pack(val_pack,parameters,height,spec['name'],models.get(spec['velocity_model']),spec['b_policy'],models.get(spec['lift_model']))
        score=selection_score(rows);scores.append(dict(spec,failed_labeled=score[0],macro_rmse_m=score[1]))
        validation_rows.extend(rows);validation_events.extend(events)
        print('COLLISION_VALIDATION '+json.dumps({'name':spec['name'],'macro_rmse_m':score[1],'failed':score[0]}),flush=True)
    best=min(scores,key=lambda x:(x['failed_labeled'],x['macro_rmse_m']))
    deployment={'schema':'FROZEN_PRECONTACT/1','selected_model':best,'upstream':choice,
       'parameters':parameters,'prefix_config':{'window_s':WINDOWS[mode]},
       'selection':'training-fit only + validation selection; frozen before test prefix evaluation',
       'input_features':'predicted incident tangent/normal speed and b_eff local components only',
       'prohibited_features':['actual missing duration','reference impact time','offline incident velocity','post observations'],
       'label_source':'offline joint outgoing-velocity reference; post lift label only if conditional sigma_e2<=.08',
       'true_spin_output':False}
    dump(out/'selected_collision.json',deployment)
    frozen=hashlib.sha256((out/'selected_collision.json').read_bytes()).hexdigest()
    dump(out/'mapping_training.json',{'velocity_reports':training_reports,
       'n_training_variants':len(active),'n_independent_training_events':len(set(c['trace']['id'] for c in active)),
       'n_lift_reliable_variants':int(reliable.sum()),
       'n_lift_reliable_independent_events':reliable_independent_events,
       'weight_source':'inverse prefix-variant count per event'})
    dump(out/'mapping_validation.json',scores)
    print('COLLISION_FROZEN_BEFORE_TEST '+frozen,flush=True)
    test=[t for t in eligible if t['split']=='test']
    cache=cache_cases(eligible,parameters,old_height,out,[(mode,t,GAPS) for t in test])
    # Keep a representative comparison set, all selected on validation.
    retained=[next(s for s in scores if s['name']=='fixed_carry')]
    for kind in ('fixed_','velocity_'):
        pool=[s for s in scores if s['name'].startswith(kind)]
        candidate=min(pool,key=lambda x:(x['failed_labeled'],x['macro_rmse_m']))
        if candidate['name'] not in [s['name'] for s in retained]: retained.append(candidate)
    for tag in ('velocity_0','velocity_1','velocity_2','velocity_3','velocity_4','velocity_5'):
        pool=[s for s in scores if s['name'].startswith(tag)]
        candidate=min(pool,key=lambda x:(x['failed_labeled'],x['macro_rmse_m']))
        if candidate['name'] not in [s['name'] for s in retained]: retained.append(candidate)
    all_rows=[];all_events=[]
    for group in (training,validation,test):
        pack=case_pack(group,cache,mode,GAPS,parameters,height)
        for spec in retained:
            rows,events=score_pack(pack,parameters,height,spec['name'],models.get(spec['velocity_model']),spec['b_policy'],models.get(spec['lift_model']))
            all_rows.extend(rows);all_events.extend(events)
    write_csv(out/'post_predictions.csv',all_rows);write_csv(out/'event_predictions.csv',all_events)
    from .collision_metrics import aggregate_collision_metrics
    metrics=aggregate_collision_metrics(all_rows,all_events)
    report={'schema':'FROZEN_PRECONTACT_EVALUATION/1','completed':True,'hostname':platform.node(),
       'created_utc':datetime.now(timezone.utc).isoformat(),'selected_model':best,'upstream':choice,
       'metrics':metrics,'model_sha256_frozen_before_test':frozen,
       'conditions':{'gaps_s':GAPS,'horizons_s_after_reference_contact':HORIZONS,
                     'prediction_input':'raw pre-only prefixes; never post observations',
                     'evaluation':'targets on the same absolute reference-contact clock'},
       'test_role':'historical exposed test groups reused for regression; new groups required for blind verification',
       'limitations':['units/center/ground still assumptions','instantaneous first-impact model',
                      'vp target is offline reconstructed reference','full3D post b not identifiable from current labels']}
    dump(out/'summary.json',report)
    from .collision_plots import make_collision_plots
    make_collision_plots(out,all_rows,all_events,report)
    print('COLLISION_REPORT='+str(out/'summary.json'),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibration',type=Path,default=ROOT/'artifacts/tennis_calibration/20261008_calibration')
    parser.add_argument('--selected',type=Path,default=ROOT/'artifacts/tennis_identified/20261009_training_prior/selected_model.json')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_collision/20261009_frozen_pre')
    parser.add_argument('--stage',choices=['prepare','experiment','all'],default='all')
    args=parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Formal collision experiment must run in Ubuntu project .venv-tennis')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    selected=json.loads(args.selected.read_text())
    dump(out/'execution.json',{'hostname':platform.node(),'python':sys.version,'python_prefix':sys.prefix,
       'source_sha256_at_start':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/'tennis_calibration').glob('collision*.py')}})
    prepared=prepare(args.calibration,selected,out)
    if args.stage=='prepare': return
    run_experiment(args.calibration,selected,out,prepared)


if __name__=='__main__': main()
