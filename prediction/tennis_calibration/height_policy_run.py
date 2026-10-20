"""Ubuntu causal full-stream comparison of immutable height/hold combinations.

Only received observations enter the tracker. Reference events and clean labels
are exclusively the replay scorer. Validation selects candidates before test
scores are inspected. All policies share capture/target clocks and labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .height_policy import HeightPolicyPredictor
from .height_forecast import forecast_posteriors
from .identified_run import dump, plain
from .online_data import load_dataset, future_labels
from .player_bounce import BounceModelBank

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (0., .1, .2, .5)
POLICIES = ('continuous', 'height_online', 'hold100', 'permanent_hold', 'init_hold100')
CANDIDATES = [(m,p) for p in POLICIES for m in BounceModelBank.model_ids] + [('gp','freeze_after_init')]
PHASES = ('precontact', 'first100ms', 'recovery100_200ms', 'late200_500ms')
FAILURES = (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError)


def _worker_init(checkpoint, selected):
    global BANK, SELECTED
    BANK = BounceModelBank(checkpoint)
    SELECTED = selected


def schedule(times, xyz, near_height):
    """Common causal anchors: 20ms slots plus a received near-height window.

    Extra dense observations are selected from input height alone, identically
    for every candidate, never from an annotated event or forecast accuracy.
    The near-height input window is sampling only, not the contact detector.
    """
    due = float(times[0]); near_start = None
    result = []
    for i,(stamp,point) in enumerate(zip(times,xyz)):
        base = stamp + 1e-12 >= due
        if base:
            due += max(1,int(np.floor((stamp-due+1e-12)/.02))+1)*.02
        if near_start is None and point[2] <= near_height:
            near_start = float(stamp)
        dense = near_start is not None and stamp <= near_start+.35+1e-12
        if base or dense:
            result.append((i,base))
    return dict(result)


def _run_case(trace, model, policy):
    tracker = HeightPolicyPredictor(BANK,model,policy,config=SELECTED['filter'],hold_s=.1)
    times,xyz = trace['raw_times'],trace['raw_xyz']
    end = float(trace.get('eval_scope_end_s') or times[-1])
    mask = times <= end
    times,xyz = times[mask],xyz[mask]
    anchors = schedule(times,xyz,BANK.height+SELECTED['filter']['bounce_observation_height_m'])
    records, posterior, flags, captures = [], [], [], []
    frame_reasons = Counter(); tracking_failure = ''; first_resume_delta = None
    timings = []
    for i,(stamp,point) in enumerate(zip(times,xyz)):
        previous_hold = tracker._skip_observation_at(tracker.last_t) if tracker.state is not None else False
        old_state = None if tracker.state is None else tracker.state.copy()
        started = perf_counter()
        if not tracking_failure:
            try:
                snapshot = tracker.update(float(stamp),point)
            except FAILURES as exc:
                tracking_failure = type(exc).__name__+': '+str(exc)
        timings.append((perf_counter()-started)*1000)
        if tracking_failure:
            snapshot = tracker.snapshot()
        frame_reasons[snapshot['reason'] if not tracking_failure else 'tracking_failure'] += 1
        if (previous_hold and not snapshot['hold_active'] and snapshot['resume_t_s'] is not None
                and first_resume_delta is None and old_state is not None and not tracking_failure):
            first_resume_delta = float(np.linalg.norm(snapshot['state'][3:6]-old_state[3:6]))
        if i not in anchors:
            continue
        initialized = snapshot['initialized'] and not tracking_failure
        record = {'trajectory_id':trace['id'],'split':trace['split'],
            'candidate':model+':'+policy,'model':model,'policy':policy,
            'capture_t_s':float(stamp),'elapsed_capture_s':float(stamp-times[0]),
            'relative_capture_s':float(stamp-trace['bounce_reference_s']),
            'reference_contact_t_s':trace['bounce_reference_s'],
            'twenty_ms_anchor':anchors[i], 'hold_active':snapshot['hold_active'],
            'height_contact_t_s':snapshot['height_contact_t_s'],'resume_t_s':snapshot['resume_t_s'],
            'initialized':bool(initialized),
            'prediction_reason':tracking_failure or ('' if initialized else 'not_initialized')}
        records.append(record)
        if initialized:
            posterior.append(snapshot['state'].copy()); flags.append(snapshot['bounced']); captures.append(float(stamp))
    forecast = None; forecast_failure = ''; forecast_reasons=['']*len(posterior)
    if posterior:
        try:
            forecast = forecast_posteriors(np.asarray(posterior),np.asarray(flags),np.asarray(captures),
                horizons=HORIZONS,bank=BANK,model_id=model,max_step_s=tracker.config['max_step_s'])
            if not np.isfinite(forecast).all():
                raise FloatingPointError('nonfinite batched forecast')
        except FAILURES as exc:
            forecast_failure = type(exc).__name__+': '+str(exc)
            # A numerically invalid later posterior must never erase an earlier
            # causal forecast. Retry each anchor independently on batch failure.
            forecast=np.full((len(posterior),len(HORIZONS),9),np.nan)
            for n in range(len(posterior)):
                try:
                    individual=forecast_posteriors(np.asarray(posterior[n:n+1]),np.asarray(flags[n:n+1]),np.asarray(captures[n:n+1]),
                        horizons=HORIZONS,bank=BANK,model_id=model,max_step_s=tracker.config['max_step_s'])[0]
                    if not np.isfinite(individual).all():raise FloatingPointError('nonfinite individual forecast')
                    forecast[n]=individual
                except FAILURES as error:forecast_reasons[n]=type(error).__name__+': '+str(error)
    rows=[]; k=0
    for record in records:
        anchor_reason=forecast_reasons[k] if record['initialized'] else ''
        prediction = forecast[k] if record['initialized'] and forecast is not None and not anchor_reason else None
        if record['initialized']: k += 1
        for j,label in enumerate(future_labels(trace,record['capture_t_s'],HORIZONS)):
            rows.append({**record,**label,
                'predicted_xyz':None if prediction is None else prediction[j,:3].tolist(),
                'prediction_reason':record['prediction_reason'] or anchor_reason})
    event = {key:snapshot[key] for key in ('height_contact_t_s','height_discovery_t_s','resume_t_s',
             'height_triggered','initialization_t_s','height_event_status','bounce_confirmed')}
    event.update(trajectory_id=trace['id'],split=trace['split'],candidate=model+':'+policy,model=model,policy=policy,
        reference_contact_t_s=trace['bounce_reference_s'], first_input_t_s=float(times[0]),
        tracking_failure=tracking_failure,forecast_failure=forecast_failure,
        frame_reasons=dict(frame_reasons),first_resume_velocity_change_including_propagation_m_s=first_resume_delta,
        update_latency_ms_p50=float(np.median(timings)),update_latency_ms_p95=float(np.quantile(timings,.95)))
    return rows,event


def phase(tau):
    if tau < 0: return 'precontact'
    if tau < .1: return 'first100ms'
    if tau < .2: return 'recovery100_200ms'
    if tau <= .5+1e-12: return 'late200_500ms'
    return None


def row_error(row):
    if not row['valid'] or row.get('prediction_reason') or row.get('predicted_xyz') is None:
        return None
    p,y=np.asarray(row['predicted_xyz']),np.asarray(row['target_xyz'])
    return float(np.linalg.norm(p-y)) if p.shape==y.shape==(3,) and np.isfinite(p).all() else None


def reduce_case(rows):
    """One collision contributes one RMS per cell; density is never weight."""
    groups=defaultdict(list); ranks=defaultdict(list)
    for r in rows:
        if r['twenty_ms_anchor']:
            groups[('elapsed20',r['horizon_s'],round(round(r['elapsed_capture_s']/.02)*.02,8))].append(r)
            groups[('contact20',r['horizon_s'],round(round(r['relative_capture_s']/.02)*.02,8))].append(r)
        if r.get('dense_sampling',True) and -.08 <= r['relative_capture_s'] <= .25:
            groups[('contact_dense2p5',r['horizon_s'],round(round(r['relative_capture_s']/.0025)*.0025,8))].append(r)
        band=phase(r['relative_capture_s'])
        if r['twenty_ms_anchor'] and r['elapsed_capture_s'] >= .10-1e-12 and band:
            ranks[(band,r['horizon_s'])].append(r)
    cells=[]
    for (axis,h,t),members in groups.items():
        errors=[v for r in members if (v:=row_error(r)) is not None]
        cells.append({'axis':axis,'horizon_s':h,'time_bin_s':t,'labeled':any(r['valid'] for r in members),
            'successful':bool(errors),'mse_m2':float(np.mean(np.square(errors))) if errors else None,
            'failed_labeled_rows':sum(r['valid'] and row_error(r) is None for r in members),
            'missing_label_rows':sum(not r['valid'] for r in members)})
    bands=[]
    for (band,h),members in ranks.items():
        labeled=[r for r in members if r['valid']]
        errors=[v for r in labeled if (v:=row_error(r)) is not None]
        bands.append({'phase':band,'horizon_s':h,'labeled_rows':len(labeled),
            'failed_labeled_rows':len(labeled)-len(errors),
            'mse_m2':float(np.mean(np.square(errors))) if errors else None,
            'max_error_m':max(errors) if errors else None})
    return cells,bands


class Accumulator:
    def __init__(self):
        self.curves=defaultdict(list);self.bands=defaultdict(list);self.events=[];self.cases=[]

    def add(self,rows,event):
        cells,bands=reduce_case(rows)
        identity=event['trajectory_id'];split=event['split'];candidate=event['candidate']
        for c in cells: self.curves[(split,candidate,c['axis'],c['horizon_s'],c['time_bin_s'])].append(c)
        for b in bands: self.bands[(split,candidate,b['phase'],b['horizon_s'])].append(b)
        future=[b for b in bands if b['horizon_s']>0 and b['mse_m2'] is not None]
        phase_means=[float(np.mean([b['mse_m2'] for b in future if b['phase']==p]))
                     for p in PHASES if any(b['phase']==p for b in future)]
        self.cases.append({'split':split,'candidate':candidate,'trajectory_id':identity,
            'balanced_future_mse_m2':float(np.mean(phase_means)) if phase_means else None,
            'failed_future_labeled_rows':sum(b['failed_labeled_rows'] for b in bands if b['horizon_s']>0)})
        self.events.append(event)

    def summary(self):
        curves=[];bands=[];ranking=[]
        for (split,candidate,axis,h,t),v in sorted(self.curves.items()):
            mse=np.asarray([c['mse_m2'] for c in v if c['mse_m2'] is not None])
            curves.append({'split':split,'candidate':candidate,'axis':axis,'horizon_s':h,'time_bin_s':t,
                'requested_trajectories':len(v),'labeled_trajectories':sum(c['labeled'] for c in v),
                'successful_trajectories':len(mse),'failed_labeled_rows':sum(c['failed_labeled_rows'] for c in v),
                'missing_label_rows':sum(c['missing_label_rows'] for c in v),
                'rmse_m':float(np.sqrt(np.mean(mse))) if len(mse) else None,
                'p95_m':float(np.quantile(np.sqrt(mse),.95)) if len(mse) else None})
        for (split,candidate,band,h),v in sorted(self.bands.items()):
            mse=np.asarray([c['mse_m2'] for c in v if c['mse_m2'] is not None])
            bands.append({'split':split,'candidate':candidate,'phase':band,'horizon_s':h,
                'requested_trajectories':len(v),'successful_trajectories':len(mse),
                'labeled_rows':sum(c['labeled_rows'] for c in v),
                'failed_labeled_rows':sum(c['failed_labeled_rows'] for c in v),
                'rmse_m':float(np.sqrt(np.mean(mse))) if len(mse) else None,
                'p95_m':float(np.quantile(np.sqrt(mse),.95)) if len(mse) else None,
                'max_error_m':max((c['max_error_m'] for c in v if c['max_error_m'] is not None),default=None)})
        keys=sorted({(c['split'],c['candidate']) for c in self.cases})
        for split,candidate in keys:
            cases=[c for c in self.cases if c['split']==split and c['candidate']==candidate]
            events=[e for e in self.events if e['split']==split and e['candidate']==candidate]
            mean=np.asarray([c['balanced_future_mse_m2'] for c in cases if c['balanced_future_mse_m2'] is not None])
            policy=events[0]['policy'];needs_height=policy in ('height_online','hold100','permanent_hold','init_hold100')
            contact_errors=[e['height_contact_t_s']-e['reference_contact_t_s'] for e in events if e.get('height_contact_t_s') is not None]
            height_misses=sum(not e.get('height_triggered',False) for e in events) if needs_height else 0
            ranking.append({'split':split,'candidate':candidate,'policy':policy,'model':events[0]['model'],
                'independent_trajectories':len(cases),'scored_trajectories':len(mean),
                'failed_future_labeled_rows':sum(c['failed_future_labeled_rows'] for c in cases),
                'required_height_event_misses':height_misses,
                'detected_height_events':len(contact_errors),
                'height_event_status_counts':dict(Counter(e.get('height_event_status','legacy_baseline') for e in events)),
                'height_time_error_median_ms':float(np.median(contact_errors)*1000) if contact_errors else None,
                'height_time_abs_error_p95_ms':float(np.quantile(np.abs(contact_errors),.95)*1000) if contact_errors else None,
                'actual_resume_trajectories':sum(e.get('resume_t_s') is not None for e in events),
                'balanced_future_rmse_m':float(np.sqrt(np.mean(mean))) if len(mean) else None,
                'per_trajectory_balanced_rms_p95_m':float(np.quantile(np.sqrt(mean),.95)) if len(mean) else None})
        key=lambda r:(r['failed_future_labeled_rows'],r['required_height_event_misses'],r['balanced_future_rmse_m'] if r['balanced_future_rmse_m'] is not None else float('inf'))
        val=sorted((r for r in ranking if r['split']=='validation'),key=key)
        selected={p:min((r for r in val if r['policy']==p),key=key)['candidate'] for p in {r['policy'] for r in val}}
        height=sorted((r for r in val if r['policy'] in ('height_online','hold100','permanent_hold','init_hold100')),key=key)
        selection= {'overall':val[0]['candidate'],'height_policy':height[0]['candidate'] if height else None,
                'per_policy':selected,'ordering':'future prediction failure rows, required height-event misses, equal-collision balanced future RMSE',
                'scoring':'capture>=first+.100s and capture<=reference contact+.500s; four phases equally per collision, horizons100/200/500 equally over available labels; current-state h0 excluded'} if val else {'unavailable':'no validation cases; test is never used to select'}
        return {'curves':curves,'phase_metrics':bands,'candidate_ranking':ranking,'events':self.events,
            'validation_only_selection':selection,
            'test_role':'historical exposed regression; selection uses validation only; not a fresh blind test'}


def append_csv(stream,rows,writer):
    if writer is None:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(plain(v),ensure_ascii=False) if isinstance(v,(list,dict,np.ndarray)) else v for k,v in row.items()})
    return writer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_height_policy/20261010_height_hold100_v2')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--splits',nargs='+',choices=['validation','test'],default=['validation','test'])
    p.add_argument('--limit',type=int)
    p.add_argument('--skip-legacy',action='store_true')
    args=p.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Run only on Ubuntu in project .venv-tennis')
    if (args.out/'summary.json').exists():raise RuntimeError('Preserve completed output; use a new directory')
    args.out.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json'
    selected_path=ROOT/'artifacts/tennis_identified/20261009_training_prior/selected_model.json'
    selected=json.loads(selected_path.read_text())
    bank=BounceModelBank(checkpoint)
    protected=[checkpoint,selected_path,*[checkpoint.parent/'models'/n for n in bank.checkpoint_names.values()]]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    traces=[t for t in load_dataset(ROOT/'artifacts/new_ubuntu/tennis_calibration') if t['split'] in args.splits and t['bounce_reference_s'] is not None]
    if args.limit:traces=traces[:args.limit]
    acc=Accumulator();started=perf_counter();writer=None
    dump(args.out/'execution.json',{'hostname':platform.node(),'python_prefix':sys.prefix,'created_utc':datetime.now(timezone.utc).isoformat(),
        'cases':len(traces)*len(CANDIDATES),'splits':dict(Counter(t['split'] for t in traces)),
        'candidates':[m+':'+p for m,p in CANDIDATES],'checkpoint_hashes':hashes,'hold_s':.1,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tennis_calibration').glob('height_*.py')}})
    with (args.out/'predictions.csv').open('w',newline='') as stream:
        with ProcessPoolExecutor(max_workers=args.jobs,initializer=_worker_init,initargs=(checkpoint,selected)) as pool:
            pending={pool.submit(_run_case,t,m,p):(t['id'],m,p) for t in traces for m,p in CANDIDATES}
            for index,future in enumerate(as_completed(pending),1):
                rows,event=future.result();acc.add(rows,event);writer=append_csv(stream,rows,writer)
                if index%10==0 or index==len(pending):
                    print(f'HEIGHT_CASE {index}/{len(pending)} elapsed_s={perf_counter()-started:.1f}',flush=True)
    if not args.skip_legacy:
        path=args.out/'legacy_predictions.csv'
        if path.exists():
            with path.open(newline='') as stream:legacy_rows=list(csv.DictReader(stream))
            identities={t['id'] for t in traces}
            legacy_rows=[r for r in legacy_rows if r['trajectory_id'] in identities]
            for r in legacy_rows:
                for key in ('capture_t_s','elapsed_capture_s','relative_capture_s','reference_contact_t_s','horizon_s','target_t_s'):r[key]=float(r[key])
                r['valid']=r['valid'].lower() in ('true','1')
                for key in ('predicted_xyz','target_xyz'):r[key]=json.loads(r[key]) if r[key] else None
            if {r['trajectory_id'] for r in legacy_rows}!=identities:raise RuntimeError('legacy cached baseline lacks requested identities')
            if any(r['predicted_xyz'] is not None and r['prediction_reason'] for r in legacy_rows):raise RuntimeError('legacy output inference note incorrectly appears as prediction failure')
        else:
            from .height_legacy_baseline import run_legacy_baselines
            legacy_rows=run_legacy_baselines(traces,args.out)
        grouped=defaultdict(list)
        for r in legacy_rows:grouped[(r['trajectory_id'],r['candidate'])].append(r)
        for (identity,candidate),rows in grouped.items():
            for r in rows:r.update(twenty_ms_anchor=True,initialized=not bool(r['prediction_reason']),dense_sampling=False)
            acc.add(rows,{'trajectory_id':identity,'split':rows[0]['split'],'candidate':candidate,
                         'policy':rows[0]['policy'],'model':rows[0]['model']})
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    summary=acc.summary()
    summary.update(hostname=platform.node(),created_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=perf_counter()-started,checkpoint_hashes_unchanged=hashes,
        horizons_s=HORIZONS,anchor_stride_s=.02,dense_bin_s=.0025,
        dense_capture_rule='every received raw point during first input-height-near-plane window (h+bounce_observation_height), next350ms; identical raw-based schedule for every candidate',
        cold_start_rule='receive from first point; original >=8points/80ms initialization; no backfilling',
        hold_rule='height crossing before measurement fusion; only dynamics/covariance propagation; received history retained for causal resume; no observation replay',
        coordinate_rule='effective calibrated z plane, propagated descending state; reference contact only scorer',
        uncertainty_rule='local EKF approximation, no full saltation correction, GP variance not injected',
        limitations=['historical exposed test, two inferred acquisition groups per validation/test',
                     'current h0 is posterior-vs-measurement tracking error, not independent truth or future accuracy',
                     'different initialization and inference contract from full-precontact batch analysis',
                     'learned weights immutable; no deep ResNet trained or evaluated'])
    dump(args.out/'summary.json',summary)
    print('HEIGHT_SUMMARY='+str(args.out/'summary.json'),flush=True)


if __name__=='__main__':main()
