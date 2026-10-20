"""Isolated Ubuntu observer ablations with causal residual-feature caches.

All observers keep the existing height-triggered 100ms hold. Ranking uses only
the unchanged common 20ms anchors. A separate all-received-frame H500 diagnostic
covers the first 50ms after each observer's actual resumption; these additional
rows never change the ranking weights. Reference labels are read after replay.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .group_fold_evaluation import calibration_hashes, checked_model
from .height_forecast import forecast_posteriors
from .height_policy_run import FAILURES, phase, row_error
from .hold_sweep import aggregate as anchor_aggregate, digest, expected_resume
from .hybrid_hold_evaluation import describe
from .identified_run import dump, plain
from .measurement_noise_probe import probe_measurement_noise
from .online_data import causal_anchors, future_labels, load_dataset
from .player_bounce import BounceModelBank
from .residual_head import FEATURE_NAMES, causal_features, residual_gate
from .robust_observer import EXPERIMENTS, FIXED_SETTINGS, RobustHeightPredictor

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (.1, .2, .5)
DEFAULT_CALIBRATION = ROOT/'artifacts/new_ubuntu/tennis_calibration'
DEFAULT_CHECKPOINT = ROOT/'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json'
DEFAULT_SELECTED = ROOT/'artifacts/tennis_identified/20261009_training_prior/selected_model.json'
DEFAULT_BASELINE = ROOT/'artifacts/tennis_height_policy/20261010_height_hold100_v2/comparison_summary.json'


def worker_init(checkpoint, selected, noise_sigma):
    global BANK, SELECTED, NOISE_SIGMA
    BANK = BounceModelBank(checkpoint)
    SELECTED = selected
    NOISE_SIGMA = np.asarray(noise_sigma, float)


def run_case(trace, experiment, *, model_id='gp', horizons=None):
    """Replay raw data once; labels never enter the observer or feature builder."""
    horizons = HORIZONS if horizons is None else tuple(horizons)
    tracker = RobustHeightPredictor(BANK, model_id=model_id, config=SELECTED['filter'],
                                    experiment=experiment, noise_sigma_xyz=NOISE_SIGMA)
    times, points = trace['raw_times'], trace['raw_xyz']
    scope = trace.get('eval_scope_end_s')
    keep = times <= (times[-1] if scope is None else scope)
    times, points = times[keep], points[keep]
    anchors = {r['current_index'] for r in causal_anchors(
        {'raw_times': times}, initialization_s=0., stride_s=.02)}
    records, states, flags, captures, snapshots = [], [], [], [], []
    frame_reasons, diagnostics_counts = Counter(), Counter()
    failure = ''; held = 0; timings = []; minimum_covariance_eigenvalue = None
    previous_hold = False; resume_velocity_change = None; old_state = None
    for index, (stamp, position) in enumerate(zip(times, points)):
        if not failure:
            try:
                snapshot = tracker.update(float(stamp), position)
            except FAILURES as exc:
                failure = type(exc).__name__+': '+str(exc)
        if failure:
            snapshot = tracker.snapshot()
        frame_reasons[snapshot['reason'] if not failure else 'tracking_failure'] += 1
        diag = snapshot['diagnostics']
        for name in ('smoothing_applied', 'recovery_gain_protected', 'observed_bounce_confirmed_this_frame'):
            if diag.get(name): diagnostics_counts[name] += 1
        if diag.get('update_latency_ms') is not None: timings.append(diag['update_latency_ms'])
        if snapshot['hold_active'] and not failure:
            held += 1
            if snapshot['accepted'] or snapshot['reason'] != 'height_policy_prediction_only':
                raise AssertionError('observation assimilation during hold')
        if (previous_hold and not snapshot['hold_active'] and not failure
                and resume_velocity_change is None and old_state is not None):
            resume_velocity_change = float(np.linalg.norm(snapshot['state'][3:6]-old_state[3:6]))
        previous_hold = snapshot['hold_active']
        old_state = None if snapshot['state'] is None else snapshot['state'].copy()
        resume = snapshot.get('resume_t_s')
        dense = bool(resume is not None and resume-1e-12 <= stamp < resume+.05-1e-12)
        if index not in anchors and not dense:
            continue
        initialized = bool(snapshot['initialized'] and not failure)
        record = {'trajectory_id': trace['id'], 'split': trace['split'],
            'groupid': trace['groupid'], 'candidate': experiment, 'hold_ms': 100.,
            'capture_t_s': float(stamp), 'elapsed_capture_s': float(stamp-times[0]),
            'relative_capture_s': float(stamp-trace['bounce_reference_s']),
            'twenty_ms_anchor': index in anchors, 'dense_recovery_frame': dense,
            'hold_active': bool(snapshot['hold_active']),
            'height_contact_t_s': snapshot['height_contact_t_s'],
            'height_discovery_t_s': snapshot['height_discovery_t_s'],
            'resume_t_s': resume, 'initialized': initialized,
            'prediction_reason': failure or ('' if initialized else 'not_initialized'),
            'raw_index': index, 'original_split': trace.get('original_split', trace['split']),
            'fold': trace.get('fold')}
        records.append(record)
        if initialized:
            states.append(snapshot['state'].copy()); flags.append(snapshot['bounced'])
            captures.append(float(stamp)); snapshots.append(snapshot)
            covariance = np.asarray(snapshot['covariance'])
            if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-10):
                raise AssertionError('invalid observer covariance')
            eig = float(np.min(np.linalg.eigvalsh(covariance)))
            minimum_covariance_eigenvalue = eig if minimum_covariance_eigenvalue is None else min(eig, minimum_covariance_eigenvalue)
            if eig < -1e-9: raise AssertionError('observer covariance lost positive semidefiniteness')
    forecast = None; forecast_failure = ''; forecast_reasons = ['']*len(states)
    if states:
        def project(indices):
            value = forecast_posteriors(np.asarray(states)[indices], np.asarray(flags)[indices],
                np.asarray(captures)[indices], horizons=horizons, bank=BANK, model_id=model_id,
                max_step_s=tracker.config['max_step_s'])
            if not np.isfinite(value).all(): raise FloatingPointError('nonfinite forecast')
            return value
        try:
            forecast = project(slice(None))
        except FAILURES as exc:
            forecast_failure = type(exc).__name__+': '+str(exc)
            forecast = np.full((len(states),len(horizons),9),np.nan)
            for index in range(len(states)):
                try: forecast[index] = project(slice(index,index+1))[0]
                except FAILURES as error: forecast_reasons[index] = type(error).__name__+': '+str(error)
    rows = []; features = []; cursor = 0
    for record in records:
        issue = forecast_reasons[cursor] if record['initialized'] else ''
        predicted = forecast[cursor] if record['initialized'] and forecast is not None and not issue else None
        at_capture = snapshots[cursor] if record['initialized'] else None
        cursor += int(record['initialized'])
        labels = future_labels(trace, record['capture_t_s'], horizons)
        for hindex, label in enumerate(labels):
            if not record['twenty_ms_anchor'] and label['horizon_s'] != .5: continue
            prediction = None if predicted is None else predicted[hindex,:3]
            gate = 0. if at_capture is None else residual_gate(at_capture)
            feature = np.full(len(FEATURE_NAMES),np.nan)
            if prediction is not None:
                # History is sliced at capture even though batch forecasting is
                # performed after replay. Never pass final tracker metadata.
                end = record['raw_index']+1
                feature = causal_features(at_capture,prediction,label['horizon_s'],times[:end],points[:end])
            row = {**record,**label,'predicted_xyz':None if prediction is None else prediction.tolist(),
                   'prediction_reason':record['prediction_reason'] or issue,'residual_gate':gate}
            rows.append(row); features.append(feature)
    contact, discovery, resume = (snapshot.get(name) for name in
        ('height_contact_t_s','height_discovery_t_s','resume_t_s'))
    expected = expected_resume(times,contact,discovery,100.)
    if not failure and not ((resume is None and expected is None) or
                           (resume is not None and expected is not None and abs(resume-expected)<=1e-12)):
        raise AssertionError('resume is not first received frame after causal hold')
    event = {'trajectory_id':trace['id'],'split':trace['split'],'groupid':trace['groupid'],
        'candidate':experiment,'hold_ms':100.,'height_contact_t_s':contact,
        'height_discovery_t_s':discovery,'resume_t_s':resume,
        'diagnostic_resume_or_event_t_s':resume,'expected_resume_or_event_t_s':expected,
        'reference_contact_t_s':trace['bounce_reference_s'],'held_frames':held,'received_frames':len(times),
        'frame_reasons':dict(frame_reasons),'diagnostics_counts':dict(diagnostics_counts),
        'initialization_t_s':snapshot['initialization_t_s'],'height_event_status':snapshot['height_event_status'],
        'tracking_failure':failure,'forecast_failure':forecast_failure,
        'contact_minus_reference_s':None if contact is None else contact-trace['bounce_reference_s'],
        'discovery_minus_reference_s':None if discovery is None else discovery-trace['bounce_reference_s'],
        'resume_minus_reference_s':None if resume is None else resume-trace['bounce_reference_s'],
        'confirmation_minus_reference_s':None if tracker._bounce_confirmation_t is None else tracker._bounce_confirmation_t-trace['bounce_reference_s'],
        'resume_rounding_delay_s':None if resume is None else resume-max(discovery,contact+.1),
        'first_resume_velocity_change_including_propagation_m_s':resume_velocity_change,
        'minimum_sampled_covariance_eigenvalue':minimum_covariance_eigenvalue,
        'update_latency_ms_p50':float(np.median(timings)) if timings else None,
        'update_latency_ms_p95':float(np.quantile(timings,.95)) if timings else None,
        'original_split':trace.get('original_split',trace['split']),'fold':trace.get('fold')}
    return rows,event,np.asarray(features,dtype=float).reshape(-1,len(FEATURE_NAMES))


def summarize(rows, events):
    """Common-anchor metrics plus explicitly separate dense diagnostics."""
    anchor_rows = [r for r in rows if r['twenty_ms_anchor']]
    result = anchor_aggregate(anchor_rows,events)
    result.pop('frozen_model_group_resampling',None)
    keys = defaultdict(set); counts = defaultdict(Counter)
    first200 = defaultdict(lambda:defaultdict(list)); dense = defaultdict(lambda:defaultdict(list))
    dense_counts = defaultdict(Counter); axes = defaultdict(lambda:defaultdict(list))
    for event in events: keys[(event['split'],event['candidate'])]
    for row in rows:
        s,c,h = row['split'],row['candidate'],row['horizon_s']
        error = row_error(row)
        if row.get('dense_recovery_frame',False) and h == .5:
            count = dense_counts[(s,c)]
            count['requested'] += 1
            count['labeled' if row['valid'] else 'missing_label'] += 1
            if row['valid']: count['successful' if error is not None else 'failed_labeled'] += 1
            if error is not None: dense[(s,c)][row['trajectory_id']].append(error)
        if not row['twenty_ms_anchor'] or row['elapsed_capture_s'] < .1-1e-12 or phase(row['relative_capture_s']) is None:
            continue
        count = counts[(s,c)]; count['requested'] += 1
        if not row['valid']: count['missing_label'] += 1
        else: count['labeled'] += 1; count['successful' if error is not None else 'failed_labeled'] += 1
        if error is None: continue
        keys[(s,c)].add((row['trajectory_id'],row['capture_t_s'],h,row['target_t_s']))
        if h == .5 and 0 <= row['relative_capture_s'] < .2:
            first200[(s,c)][row['trajectory_id']].append(error)
        delta = np.asarray(row['predicted_xyz'])-np.asarray(row['target_xyz'])
        axes[(s,c,phase(row['relative_capture_s']),h)][row['trajectory_id']].append(delta)
    coverage = []
    for s in sorted({s for s,c in keys}):
        names = sorted(c for split,c in keys if split == s)
        baseline = 'baseline' if 'baseline' in names else names[0]
        expected = keys[(s,baseline)]
        for c in names:
            count = counts[(s,c)]; actual = keys[(s,c)]
            coverage.append({'split':s,'candidate':c,**dict(count),'successful_keys':len(actual),
                'missing_vs_reference':len(expected-actual),'additional_vs_reference':len(actual-expected),
                'reference_candidate':baseline,'equal_to_reference':actual == expected,
                'prediction_coverage_given_label':count['successful']/count['labeled'] if count['labeled'] else None,
                'tracking_failures':sum(bool(e['tracking_failure']) for e in events if (e['split'],e['candidate'])==(s,c))})
    result['successful_key_coverage'] = coverage
    result['absolute_labeled_coverage'] = coverage
    result['first200ms_h500_metrics'] = [{'split':s,'candidate':c,'horizon_s':.5,
        'capture_window':'reference [0,200ms)',**describe(v)} for (s,c),v in sorted(first200.items())]
    dense_metrics = []
    for (s,c),count in sorted(dense_counts.items()):
        groups = dense.get((s,c),{})
        peaks = [max(v) for v in groups.values()]
        dense_metrics.append({'split':s,'candidate':c,'horizon_s':.5,**describe(groups),**dict(count),
            'trajectory_window_max_p95_m':float(np.quantile(peaks,.95)) if peaks else None,
            'trajectory_window_max_mean_m':float(np.mean(peaks)) if peaks else None,
            'trajectory_window_max_max_m':float(np.max(peaks)) if peaks else None,
            'sampling':'every received raw frame in [own actual resume, resume+50ms); H500; separate from20ms ranking'})
    result['dense_resume_first50ms_metrics'] = dense_metrics
    result['axis_metrics'] = []
    for (s,c,p,h),groups in sorted(axes.items()):
        means = [np.asarray(v).mean(axis=0) for v in groups.values()]
        mse = [np.square(v).mean(axis=0) for v in groups.values()]
        absolute = [describe({identity:np.abs(np.asarray(v)[:,axis]) for identity,v in groups.items()})
                    for axis in range(3)]
        result['axis_metrics'].append({'split':s,'candidate':c,'capture_phase':p,'horizon_s':h,
            'N':len(groups),'signed_mean_xyz_m':np.mean(means,axis=0).tolist(),
            'rmse_xyz_m':np.sqrt(np.mean(mse,axis=0)).tolist(),
            'absolute_mean_xyz_m':[v['mean_m'] for v in absolute],
            'absolute_p95_xyz_m':[v['p95_frame_m'] for v in absolute],
            'absolute_p99_xyz_m':[v['p99_frame_m'] for v in absolute]})
    result['axis_phase_metrics'] = result['axis_metrics']
    timing = []
    for s,c in sorted(keys):
        matching = [e for e in events if (e['split'],e['candidate']) == (s,c)]
        cell = {'split':s,'candidate':c,'N':len(matching)}
        for name in ('contact_minus_reference_s','discovery_minus_reference_s','resume_minus_reference_s','confirmation_minus_reference_s'):
            values = np.asarray([e[name] for e in matching if e.get(name) is not None])
            cell[name] = {'N':len(values),'missing':len(matching)-len(values),
                'mean_ms':float(values.mean()*1000) if len(values) else None,
                'abs_p95_ms':float(np.quantile(np.abs(values),.95)*1000) if len(values) else None,
                'min_ms':float(values.min()*1000) if len(values) else None,
                'max_ms':float(values.max()*1000) if len(values) else None,
                'max_positive_ms':float(max(0.,values.max())*1000) if len(values) else None}
        timing.append(cell)
    result['event_timing_metrics'] = timing
    coverage_map = {(v['split'],v['candidate']):v for v in coverage}
    for rank in result['ranking']:
        cover = coverage_map[(rank['split'],rank['candidate'])]
        rank['eligible_for_ranking'] = bool(cover['equal_to_reference'] and
            cover.get('failed_labeled',0)==0 and cover['tracking_failures']==0)
    result['dense_rows_excluded_from_ranking'] = True
    return result


def cache_arrays(rows, features):
    """Plain numeric/unicode NPZ, no pickle; row order equals CSV candidate subset."""
    vector = lambda key: np.asarray([r.get(key) if r.get(key) is not None else [np.nan]*3 for r in rows],float).reshape(-1,3)
    result = {'X':np.asarray(features,float).reshape(-1,len(FEATURE_NAMES)),
        'pred':vector('predicted_xyz'),'ref':vector('target_xyz'),
        'H':np.asarray([r['horizon_s'] for r in rows],float),
        'gate':np.asarray([r['residual_gate'] for r in rows],float),
        'ids':np.asarray([str(r['trajectory_id']) for r in rows]),
        'anchor':np.asarray([r['twenty_ms_anchor'] for r in rows],bool),
        'dense':np.asarray([r['dense_recovery_frame'] for r in rows],bool),
        'valid':np.asarray([r['valid'] for r in rows],bool),
        'initialized':np.asarray([r['initialized'] for r in rows],bool),
        'prediction_ok':np.asarray([r['predicted_xyz'] is not None and not r['prediction_reason'] for r in rows],bool)}
    for key in ('capture_t_s','elapsed_capture_s','relative_capture_s','target_t_s'):
        result[key] = np.asarray([r[key] for r in rows],float)
    for key in ('split','groupid','candidate'):
        result[key] = np.asarray([str(r[key]) for r in rows])
    return result


def fit_noise_proxy(traces, training_ids):
    """Training-only frozen discrepancy scale, not a sensor noise estimate."""
    allowed = set(training_ids)
    selected = [dict(t,split='train') for t in traces if t['id'] in allowed]
    if {t['id'] for t in selected} != allowed: raise ValueError('missing noise training identity')
    probe = probe_measurement_noise(selected,windows_s=(.08,))
    cell = probe['by_split']['train']['80ms']
    sigma = np.asarray(cell['summary']['trajectory_equal_mean_signed_axis_mad_sigma_proxy_m'],float)
    if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('no positive finite training-only noise proxy')
    return {'schema':'TRAIN_ONLY_NOISE_PROXY/1','sigma_xyz_m':sigma.tolist(),
        'allowed_training_ids':sorted(allowed),
        'observed_training_ids':sorted(v['trajectory_id'] for v in cell['trajectories'] if v['statistics'] is not None),
        'training_input_count':len(selected),'proxy_summary':cell['summary'],
        'recipe':probe['configuration'],'interpretation':'causal extrapolation discrepancy scale; not device precision',
        'offline_event_use':'training-only exclusion of contact neighborhoods when fitting fixed scales; never runtime phase input'}


def check_baseline(summary, path, *, complete, fold=False):
    if path is None or not path.exists() or not complete or not any(v['candidate']=='baseline' for v in summary['ranking']):
        return {'status':'not_checked_partial_missing_or_no_baseline','path':None if path is None else str(path)}
    source = json.loads(path.read_text())
    expected_name = 'gp_hold100ms' if fold else 'gp:hold100'
    expected = {(v['split'],v.get('phase',v.get('capture_phase')),v['horizon_s']):v
                for v in source['phase_metrics'] if v['candidate']==expected_name
                and v['horizon_s'] in HORIZONS}
    requested_splits = {v['split'] for v in summary['absolute_labeled_coverage'] if v['candidate']=='baseline'}
    comparable_splits = requested_splits & {key[0] for key in expected}
    expected_keys = {key for key in expected if key[0] in comparable_splits}
    actual_keys = {(v['split'],v['capture_phase'],v['horizon_s']) for v in summary['phase_metrics']
                   if v['candidate']=='baseline' and v['split'] in comparable_splits}
    if actual_keys != expected_keys:
        raise AssertionError('baseline phase/horizon coverage changed')
    n = 0
    for cell in summary['phase_metrics']:
        if cell['candidate'] != 'baseline': continue
        key = (cell['split'],cell['capture_phase'],cell['horizon_s'])
        if key not in expected: continue
        old = expected[key]
        np.testing.assert_allclose(cell['rmse_m'],old['rmse_m'],atol=1e-10,rtol=1e-10)
        if cell['N'] != old.get('N',old.get('successful_trajectories')): raise AssertionError('baseline trajectory coverage changed')
        n += 1
    ranks = {v['split']:v for v in source.get('ranking',source.get('candidate_ranking',[])) if v['candidate']==expected_name}
    actual_rank_splits = {v['split'] for v in summary['ranking'] if v['candidate']=='baseline'
                          and v['split'] in comparable_splits}
    if actual_rank_splits != comparable_splits & set(ranks):
        raise AssertionError('baseline ranking split coverage changed')
    rank_n = 0
    for row in summary['ranking']:
        if row['candidate']=='baseline' and row['split'] in ranks:
            np.testing.assert_allclose(row['balanced_future_rmse_m'],ranks[row['split']]['balanced_future_rmse_m'],atol=1e-10,rtol=1e-10)
            rank_n += 1
    return {'status':'passed' if n and rank_n else 'not_applicable_training_partition',
            'phase_cells':n,'ranking_cells':rank_n,'path':str(path),'sha256':digest(path)}


def execute(args):
    if args.output.exists() and any(args.output.iterdir()): raise RuntimeError('preserve prior outputs; choose an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    traces = load_dataset(args.calibration)
    if args.fold_dir:
        if args.split: raise ValueError('--split cannot accompany --fold-dir')
        plan = json.loads((args.fold_dir/'fold_plan.json').read_text())
        _,plan,_,checkpoint,selected,model_sha = checked_model(args.fold_dir.parent,args.calibration,plan['fold'])
        training_ids = plan['train_ids']; selected_ids = set(plan[args.partition+'_ids'])
        selected_traces = [dict(t,original_split=t['split'],split='train' if args.partition=='train' else 'oof',fold=plan['fold'])
                           for t in traces if t['id'] in selected_ids]
        selected_path = args.fold_dir/'selected_model.json'
        baseline = args.baseline_summary or ROOT/f'artifacts/tennis_hold_sweep/20261013_groupcv_eval/fold{plan["fold"]}/summary.json'
    else:
        if args.partition != 'evaluation': raise ValueError('--partition train requires --fold-dir; otherwise use --split train')
        plan = None; selected = json.loads(args.selected.read_text()); checkpoint=args.checkpoint; selected_path=args.selected
        training_ids = sorted(t['id'] for t in traces if t['split']=='train')
        selected_traces = [t for t in traces if t['split'] in (args.split or ['validation','test'])]
        bank = BounceModelBank(checkpoint)
        protected = [checkpoint,selected_path,*[checkpoint.parent/'models'/name for name in bank.checkpoint_names.values()]]
        model_sha = {str(p):digest(p) for p in protected}
        baseline = args.baseline_summary or DEFAULT_BASELINE
    allowed = set(training_ids)
    if args.noise_proxy:
        proxy=json.loads(args.noise_proxy.read_text())
        if set(proxy['allowed_training_ids']) != allowed or set(proxy['observed_training_ids'])-allowed:
            raise ValueError('noise proxy was not fitted on this training partition')
    else: proxy=fit_noise_proxy(traces,training_ids)
    dump(args.output/'noise_proxy.json',proxy)
    excluded = [{'id':t['id'],'reason':'no_original_first_contact_reference'} for t in selected_traces if t['bounce_reference_s'] is None]
    selected_traces = [t for t in selected_traces if t['bounce_reference_s'] is not None]
    if args.limit: selected_traces=selected_traces[:args.limit]
    if not selected_traces: raise ValueError('no evaluable trajectories')
    labels_sha=calibration_hashes(args.calibration)
    started=perf_counter()
    execution={'schema':'ROBUST_HEIGHT_OBSERVER_EXPERIMENT/1','created_utc':datetime.now(timezone.utc).isoformat(),
        'hostname':platform.node(),'python_prefix':sys.prefix,'experiments':args.experiments,
        'n_trajectories':len(selected_traces),'splits':dict(Counter(t['split'] for t in selected_traces)),
        'training_ids':training_ids,'evaluation_ids':[t['id'] for t in selected_traces],
        'excluded':excluded,'fold_plan':plan,'partition':args.partition if plan else None,
        'checkpoint':str(checkpoint),'selected_config':str(selected_path),'model_hashes':model_sha,
        'calibration_hashes':labels_sha,'noise_proxy':proxy,'observer_fixed_settings':FIXED_SETTINGS,
        'source_hashes':{name:digest(ROOT/'tennis_calibration'/name) for name in
            ('robust_optimization.py','robust_observer.py','residual_head.py','height_policy.py','identified_filter.py',
             'height_forecast.py','online_data.py','hold_sweep.py')},
        'feature_names':list(FEATURE_NAMES),'csv_encoding':'every cell JSON encoded',
        'scoring':'common20ms anchors; 3 future horizons; phase/horizon/trajectory balanced ranking',
        'dense_diagnostic':'H500 at every received frame in own actual resume+50ms window; not ranking',
        'hold_feature_rule':'features may describe cached raw history; residual_gate strictly zero throughout hold and first50ms after resume'}
    dump(args.output/'execution.json',execution)
    rows=[];events=[];cache_rows=defaultdict(list);cache_features=defaultdict(list);writer=None
    with gzip.open(args.output/'predictions.csv.gz','wt',newline='') as stream:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,
            initargs=(checkpoint,selected,proxy['sigma_xyz_m'])) as pool:
            pending=[pool.submit(run_case,t,name) for t in selected_traces for name in args.experiments]
            for index,future in enumerate(as_completed(pending),1):
                batch,event,x=future.result(); rows.extend(batch);events.append(event)
                name=event['candidate'];cache_rows[name].extend(batch);cache_features[name].append(x)
                if writer is None and batch: writer=csv.DictWriter(stream,fieldnames=list(batch[0]));writer.writeheader()
                for row in batch:
                    writer.writerow({k:json.dumps(plain(v),ensure_ascii=False,allow_nan=False) for k,v in row.items()})
                if index%10==0 or index==len(pending):
                    print('ROBUST_REPLAY '+json.dumps({'completed':index,'total':len(pending),'elapsed_s':perf_counter()-started}),flush=True)
    events.sort(key=lambda e:(e['split'],e['trajectory_id'],e['candidate']))
    dump(args.output/'events.json',events)
    files={}
    for name in args.experiments:
        path=args.output/f'cache_{name}.npz'
        np.savez_compressed(path,**cache_arrays(cache_rows[name],np.concatenate(cache_features[name],axis=0)))
        files[name]={'path':str(path),'sha256':digest(path),'rows':len(cache_rows[name])}
    if labels_sha != calibration_hashes(args.calibration): raise RuntimeError('scoring labels changed')
    if args.fold_dir:
        if model_sha != {name:digest(args.fold_dir/name) for name in model_sha}: raise RuntimeError('fold model changed')
    elif model_sha != {name:digest(Path(name)) for name in model_sha}: raise RuntimeError('source model changed')
    result=summarize(rows,events)
    result['baseline_check']=check_baseline(result,baseline,complete=not args.limit,fold=bool(args.fold_dir))
    result.update(**execution,completed=True,elapsed_s=perf_counter()-started,events=events,
        feature_caches=files,protected_labels_and_models_unchanged=True,
        predictions_sha256=digest(args.output/'predictions.csv.gz'),
        limitations=['historical exposed groups; not new blind data',
            'noise proxy includes model and observation discrepancy; not device precision',
            'features are causal; future label arrays are targets only',
            'dense recovery windows are relative to each observer and differ in absolute time',
            'model and player defaults unchanged'])
    dump(args.output/'summary.json',result)
    print(json.dumps({'out':str(args.output),'ranking':result['ranking'],'baseline_check':result['baseline_check']},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',nargs='+',choices=('train','validation','test'))
    parser.add_argument('--fold-dir',type=Path)
    parser.add_argument('--partition',choices=('train','evaluation'),default='evaluation')
    parser.add_argument('--experiments',nargs='+',choices=tuple(EXPERIMENTS),default=list(EXPERIMENTS))
    parser.add_argument('--output','--out',dest='output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--calibration',type=Path,default=DEFAULT_CALIBRATION)
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--selected',type=Path,default=DEFAULT_SELECTED)
    parser.add_argument('--noise-proxy',type=Path)
    parser.add_argument('--baseline-summary',type=Path)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('run numerical experiments on Ubuntu in project .venv-tennis')
    if args.workers<1 or not args.experiments or len(set(args.experiments))!=len(args.experiments):
        raise ValueError('positive workers and distinct experiments required')
    if args.limit is not None and args.limit<1: raise ValueError('limit must be positive')
    execute(args)


if __name__=='__main__': main()
