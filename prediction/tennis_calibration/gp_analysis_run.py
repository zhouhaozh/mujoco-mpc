"""Ubuntu analysis of method 6: all precontact data, frozen vs state updates.

The trained GP checkpoint is immutable. Reference contact chooses the test cut
and evaluation clock only; incoming/contact states are independently predicted.
Every finite chronological precontact observation participates in the fit.
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

from .collision_prefix import estimate_prefix, predict_contact, DEFAULT_CONFIG
from .identified_run import dump, plain
from .online_data import load_dataset, future_labels
from .online_run import write_csv
from .player_bounce import BounceModelBank

ROOT = Path(__file__).resolve().parents[1]
TAU = np.round(np.arange(.01, 1.0001, .01), 8)
HORIZONS = (.1, .2, .5)
ANCHOR_STRIDE = .02


def fingerprint(times, xyz):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(times).tobytes())
    h.update(np.ascontiguousarray(xyz).tobytes())
    return h.hexdigest()


def all_pre_fit(trace, bank):
    reference = float(trace['bounce_reference_s'])
    before = trace['raw_times'] < reference
    times, xyz = trace['raw_times'][before], trace['raw_xyz'][before]
    config = {**DEFAULT_CONFIG, 'window_s': 0.,
              'representative_points': max(DEFAULT_CONFIG['minimum_samples'], len(times))}
    started = perf_counter()
    fit = estimate_prefix(times, xyz, reference, bank.parameters, bank.height, config)
    contact = predict_contact(fit, bank.parameters, bank.height)
    if fit['converged'] and fit['diagnostics']['sampled_rows'] != len(times):
        raise AssertionError('All precontact observations must be used; no representative reduction')
    return {'valid': bool(contact['valid']), 'reason': contact['reason'],
            'fit': fit, 'contact': contact, 'cut_t_s': reference,
            'raw_prefix_times': times, 'raw_prefix_xyz': xyz,
            'input_sha256': fingerprint(times, xyz), 'prefix_config': config,
            'all_precontact_observations_used': True, 'uses_post_contact_observations': False,
            'elapsed_ms': (perf_counter()-started)*1000}


def prepare(traces, bank, out):
    cache_path = out/'all_pre_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    context = {'parameters': bank.parameters, 'height': bank.height,
               'prefix_defaults': DEFAULT_CONFIG, 'window_s': 0., 'all_rows': True,
               'source_sha256': {name: hashlib.sha256((ROOT/'tennis_calibration'/name).read_bytes()).hexdigest()
                                 for name in ('collision_prefix.py','identified_dynamics.py','flight.py')}}
    context_sha = hashlib.sha256(json.dumps(plain(context),sort_keys=True).encode()).hexdigest()
    for index,t in enumerate(traces):
        key = t['id']
        before = t['raw_times'] < t['bounce_reference_s']
        digest = fingerprint(t['raw_times'][before], t['raw_xyz'][before])
        if key in cache:
            item = cache[key]
            if item.get('context_sha256') != context_sha or item['input_sha256'] != digest:
                raise RuntimeError('Full-prefix cache context/input changed; use a new output directory')
        else:
            item = plain(all_pre_fit(t, bank))
            item['context_sha256'] = context_sha
            cache[key] = item
            dump(cache_path,cache)
        print(f'GP_ALL_PRE {index+1}/{len(traces)} N={len(item["raw_prefix_times"])} '
              f'valid={item["valid"]} fit_ms={item["elapsed_ms"]:.1f}',flush=True)
    return cache, context


def safe_prediction(bank, prepared, targets):
    if not prepared['valid']:
        return None, prepared['reason']
    try:
        return bank.predict(prepared, 'gp', targets), ''
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return None, type(error).__name__+': '+str(error)


def row(trace, protocol, capture, label, prediction=None, reason='', **extras):
    return plain({'trajectory_id':trace['id'],'split':trace['split'],'protocol':protocol,
        'capture_t_s':capture,'relative_capture_s':capture-trace['bounce_reference_s'],
        'reference_contact_t_s':trace['bounce_reference_s'],
        'tau_s':label['target_t_s']-trace['bounce_reference_s'], **label,
        'predicted_xyz':None if prediction is None else prediction[:3],
        'predicted_v':None if prediction is None else prediction[3:6],
        'prediction_reason':reason, **extras})


def run_analysis(traces, cache, bank, selected, out):
    from .gp_online import GPOnlinePredictor
    frozen_rows, online_rows, target_rows, snapshots = [], [], [], []
    timings = []
    example = None
    for index,trace in enumerate(traces):
        prepared = cache[trace['id']]
        fit = prepared['fit']
        capture = fit.get('capture_t_s')
        if capture is None:
            capture = float(trace['raw_times'][0])
        tb = trace['bounce_reference_s']
        targets = tb+TAU
        frozen, reason = safe_prediction(bank,prepared,targets)
        for j,label in enumerate(future_labels(trace,tb,TAU)):
            frozen_rows.append(row(trace,'frozen',capture,label,
                                   None if frozen is None else frozen[j],reason))
        if not prepared['valid']:
            # Retain requested rows for every would-be online anchor too.
            due = capture
            for stamp in trace['raw_times']:
                if stamp+1e-12 < due or stamp > float(trace.get('eval_scope_end_s') or trace['raw_times'][-1]):
                    continue
                for label in future_labels(trace,float(stamp),HORIZONS):
                    online_rows.append(row(trace,'online',float(stamp),label,None,reason,
                        frozen_xyz=None,frozen_reason=reason,raw_rows_received=int(np.sum(trace['raw_times']<=stamp))))
                for h in HORIZONS:
                    if tb+h >= stamp+.02:
                        label = future_labels(trace,tb,[h])[0]
                        target_rows.append(row(trace,'fixed_target',float(stamp),label,None,reason,
                                              frozen_xyz=None,frozen_reason=reason))
                due = float(stamp)+ANCHOR_STRIDE
            continue
        tracker = GPOnlinePredictor(bank, config=selected['filter'])
        tracker.seed_from_prefix(fit, prepared['raw_prefix_times'], prepared['raw_prefix_xyz'])
        count = len(prepared['raw_prefix_times'])
        anchors = []
        due = capture+ANCHOR_STRIDE
        scope_end = float(trace.get('eval_scope_end_s') or trace['raw_times'][-1])
        anchors.append((capture,tracker.snapshot(),count))
        for stamp,xyz in zip(trace['raw_times'],trace['raw_xyz']):
            if stamp <= capture:
                continue
            if stamp > scope_end:
                break
            started=perf_counter()
            snapshot=tracker.update(float(stamp),xyz)
            timings.append((perf_counter()-started)*1000)
            count += 1
            if stamp+1e-12 >= due:
                anchors.append((float(stamp),plain(snapshot),count))
                # Capture now; later target prediction will be from a copy of
                # the corresponding posterior, not the final replay state.
                due += max(1,int(np.floor((stamp-due)/ANCHOR_STRIDE))+1)*ANCHOR_STRIDE
        # Independent projector objects restore each saved posterior. No later
        # measurement can affect an earlier prediction in this offline replay.
        for stamp,snapshot,count in anchors:
            projector=GPOnlinePredictor(bank, config=selected['filter'])
            projector.restore_snapshot(snapshot)
            started=perf_counter()
            requested=stamp+np.asarray(HORIZONS)
            try:
                forecast=projector.predict_target_times(requested);forecast_reason=''
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                forecast=None;forecast_reason=type(error).__name__+': '+str(error)
            forecast_ms=(perf_counter()-started)*1000
            matched,matched_reason=safe_prediction(bank,prepared,requested)
            for j,label in enumerate(future_labels(trace,stamp,HORIZONS)):
                online_rows.append(row(trace,'online',stamp,label,
                    None if forecast is None else forecast[j],forecast_reason,
                    frozen_xyz=None if matched is None else matched[j,:3],frozen_reason=matched_reason,
                    raw_rows_received=count,forecast_ms=forecast_ms))
            fixed_h=[h for h in HORIZONS if tb+h >= stamp+.02]
            if fixed_h:
                fixed_times=tb+np.asarray(fixed_h)
                try:
                    refinement=projector.predict_target_times(fixed_times);refinement_reason=''
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                    refinement=None;refinement_reason=type(error).__name__+': '+str(error)
                fixed_original,fixed_reason=safe_prediction(bank,prepared,fixed_times)
                for j,label in enumerate(future_labels(trace,tb,fixed_h)):
                    target_rows.append(row(trace,'fixed_target',stamp,label,
                        None if refinement is None else refinement[j],refinement_reason,
                        frozen_xyz=None if fixed_original is None else fixed_original[j,:3],frozen_reason=fixed_reason,
                        actual_prediction_horizon_s=label['target_t_s']-stamp,raw_rows_received=count))
            snapshots.append({'trajectory_id':trace['id'],'capture_t_s':stamp,
                'relative_capture_s':stamp-tb,'raw_rows_received':count,
                'latest_input_t_s':stamp,'snapshot':plain(snapshot)})
        if example is None:
            example={'trajectory_id':trace['id'],'reference_tb':tb,
                'scope_end_s':trace.get('eval_scope_end_s'),
                'reference_events':trace.get('bounce_event_times_s',[]),
                'raw_times':trace['raw_times'],'raw_xyz':trace['raw_xyz'],
                'reference_times':trace['eval_times'],'reference_xyz':trace['eval_xyz'],
                'frozen_times':targets,'frozen_xyz':frozen[:,:3], 'forecasts':[]}
            wanted=[0.,.1,.2,.3,.4]
            for desired in wanted:
                stamp,snapshot,_=min(anchors,key=lambda a:abs(a[0]-tb-desired))
                projector=GPOnlinePredictor(bank,config=selected['filter'])
                projector.restore_snapshot(snapshot)
                display_times=np.linspace(stamp,min(stamp+.5,scope_end),51)
                if display_times[-1]>display_times[0]:
                    states=projector.predict_target_times(display_times)
                    example['forecasts'].append({'capture_t_s':stamp,'times':display_times,'xyz':states[:,:3]})
        print(f'GP_ONLINE {index+1}/{len(traces)} anchors={len(anchors)}',flush=True)
    write_csv(out/'frozen_predictions.csv',frozen_rows)
    write_csv(out/'online_predictions.csv',online_rows)
    write_csv(out/'fixed_target_predictions.csv',target_rows)
    dump(out/'online_snapshots.json',snapshots)
    dump(out/'example.json',example)
    return frozen_rows,online_rows,target_rows,example,timings


def run_fullstream(traces, cache, bank, selected, out):
    """Additional online view before and after contact, without batch seeding.

    This curve has a different initialization from the primary matched-seed
    comparison. Frozen forecasts are unavailable before their observation cut.
    """
    from .gp_online import GPOnlinePredictor
    rows,targets=[],[]
    for index,trace in enumerate(traces):
        tracker=GPOnlinePredictor(bank,config=selected['filter'])
        prepared=cache[trace['id']]
        freeze_t=float(prepared['fit']['capture_t_s'])
        tb=trace['bounce_reference_s']
        due=float(trace['raw_times'][0])+selected['filter']['init_window_s']
        scope_end=float(trace.get('eval_scope_end_s') or trace['raw_times'][-1])
        count=0
        for stamp,xyz in zip(trace['raw_times'],trace['raw_xyz']):
            if stamp>scope_end:break
            count+=1
            snapshot=tracker.update(float(stamp),xyz)
            if stamp+1e-12<due:continue
            due+=max(1,int(np.floor((stamp-due)/ANCHOR_STRIDE))+1)*ANCHOR_STRIDE
            requested=stamp+np.asarray(HORIZONS)
            prediction=None;reason='not_initialized'
            if snapshot['initialized']:
                try:
                    prediction=tracker.predict_target_times(requested);reason=''
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                    reason=type(error).__name__+': '+str(error)
            if stamp>=freeze_t:
                frozen,frozen_reason=safe_prediction(bank,prepared,requested)
            else:
                frozen=None;frozen_reason='frozen_forecast_not_available_before_full_precontact_cut'
            for j,label in enumerate(future_labels(trace,float(stamp),HORIZONS)):
                rows.append(row(trace,'online_fullstream',float(stamp),label,
                    None if prediction is None else prediction[j],reason,
                    frozen_xyz=None if frozen is None else frozen[j,:3],frozen_reason=frozen_reason,
                    raw_rows_received=count,initialization='causal_stream_from_first_observation'))
            fixed_h=[h for h in HORIZONS if tb+h>=stamp+.02]
            if fixed_h:
                requested=tb+np.asarray(fixed_h)
                prediction=None;reason='not_initialized'
                if snapshot['initialized']:
                    try:
                        prediction=tracker.predict_target_times(requested);reason=''
                    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
                        reason=type(error).__name__+': '+str(error)
                if stamp>=freeze_t:
                    frozen,frozen_reason=safe_prediction(bank,prepared,requested)
                else:
                    frozen=None;frozen_reason='frozen_forecast_not_available_before_full_precontact_cut'
                for j,label in enumerate(future_labels(trace,tb,fixed_h)):
                    targets.append(row(trace,'target_fullstream',float(stamp),label,
                        None if prediction is None else prediction[j],reason,
                        frozen_xyz=None if frozen is None else frozen[j,:3],frozen_reason=frozen_reason,
                        raw_rows_received=count,actual_prediction_horizon_s=label['target_t_s']-stamp,
                        initialization='causal_stream_from_first_observation'))
        print(f'GP_FULLSTREAM {index+1}/{len(traces)} raw={count}',flush=True)
    write_csv(out/'fullstream_predictions.csv',rows)
    write_csv(out/'fullstream_fixed_target_predictions.csv',targets)
    return rows,targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',type=Path,default=ROOT/'artifacts/new_ubuntu/tennis_calibration')
    p.add_argument('--selected',type=Path,default=ROOT/'artifacts/tennis_identified/20261009_training_prior/selected_model.json')
    p.add_argument('--collision',type=Path,default=ROOT/'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_gp_analysis/20261010_full_pre_gp')
    p.add_argument('--stage',choices=['prepare','all','finish'],default='all')
    args=p.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Formal analysis must run on Ubuntu in the project .venv-tennis')
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    selected=json.loads(args.selected.read_text())
    bank=BounceModelBank(args.collision)
    checkpoints=[args.collision,args.collision.parent/'models/velocity_3']
    immutable={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}
    all_traces=load_dataset(args.calibration)
    traces=[t for t in all_traces if t['split']=='test' and t['bounce_reference_s'] is not None]
    dump(out/'execution.json',{'hostname':platform.node(),'python':sys.version,'python_prefix':sys.prefix,
        'created_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_hashes':immutable,
        'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (ROOT/'tennis_calibration').glob('gp_*.py')}})
    cache,context=prepare(traces,bank,out)
    if args.stage=='prepare':return
    if args.stage=='finish':
        from .collision_metrics import read_collision_rows
        rows=read_collision_rows(out/'frozen_predictions.csv')
        online=read_collision_rows(out/'online_predictions.csv')
        target=read_collision_rows(out/'fixed_target_predictions.csv')
        example=json.loads((out/'example.json').read_text());timings=[]
    else:
        rows,online,target,example,timings=run_analysis(traces,cache,bank,selected,out)
    stream,stream_targets=run_fullstream(traces,cache,bank,selected,out)
    if immutable!={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints}:
        raise AssertionError('The trained GP checkpoint must remain unchanged')
    summary={'schema':'GP_METHOD6_ALL_PRE/1','completed':True,'hostname':platform.node(),
        'created_utc':datetime.now(timezone.utc).isoformat(),'n_test_trajectories':len(traces),
        'excluded_test_ids':[t['id'] for t in all_traces if t['split']=='test' and t['bounce_reference_s'] is None],
        'input_protocol':'all finite chronological raw observations strictly before reference first contact; no subsampling',
        'frozen_protocol':'independently predicted contact and GP outgoing state fixed; no further observations',
        'online_protocol':'same full-prefix posterior seed; GP-aware EKF receives subsequent chronological raw observations',
        'fullstream_protocol':'additional causal online tracking from the first observation, including precontact updates; initialization differs from matched-seed primary',
        'horizons_s':HORIZONS,'frozen_tau_s':TAU,'anchor_stride_s':ANCHOR_STRIDE,
        'prefix_context':context,'checkpoint_hashes_unchanged':immutable,
        'checkpoint_training_history':'original frozen GP trained using recent 0.4s estimates; this full-prefix evaluation changes upstream input distribution, without retraining',
        'test_role':'previously exposed historical regression, not new blind data',
        'failures':dict(Counter(c['reason'] for c in cache.values() if not c['valid'])),
        'update_latency_ms':{'n':len(timings),'p50':float(np.quantile(timings,.5)) if timings else None,
                             'p95':float(np.quantile(timings,.95)) if timings else None},
        'limitations':['only XYZ observations are measured; effective b and velocity are latent',
                       'automatic event reference must be reviewed','first impact only',
                       'local covariance/floors are approximate, not calibrated confidence intervals']}
    dump(out/'summary.json',summary)
    from .gp_analysis_plots import make_gp_analysis_plots
    plot_result=make_gp_analysis_plots(out,rows,online,target,plain(summary),plain(example),
                                    fullstream_rows=stream,fulltarget_rows=stream_targets)
    summary['plot_result']=plot_result
    dump(out/'summary.json',summary)
    print('GP_REPORT='+str(out/'summary.json'),flush=True)


if __name__=='__main__':main()
