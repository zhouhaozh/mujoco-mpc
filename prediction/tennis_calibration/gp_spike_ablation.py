"""Immutable collision-map ablation and causal early-contact diagnostics.

Formal computation runs only in Ubuntu's project environment. Every seeded
candidate uses identical full-precontact mean/covariance, raw observations,
EKF parameters, event height and flight dynamics. Only the impact map changes.
The naturally initialized existing IdentifiedFilter is a separate supplement.
Nothing here trains a model, selects hyperparameters, or replaces defaults.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .gp_online import GPOnlinePredictor
from .identified_filter import IdentifiedFilter
from .identified_run import dump, plain
from .online_data import load_dataset, future_labels
from .online_run import write_csv
from .player_bounce import BounceModelBank

ROOT=Path(__file__).resolve().parents[1]
HORIZONS=(.1,.2,.5)


class DiagnosticMixin:
    """Inspect the existing filter without changing any state transitions."""
    def _impact_transition(self,state):
        result=super()._impact_transition(state)
        self._last_impact_jacobian={
            'velocity_velocity_norm':float(np.linalg.norm(result[3:6,3:6],ord=2)),
            'velocity_b_norm':float(np.linalg.norm(result[3:6,6:9],ord=2)) if result.shape[1]>=9 else 0.}
        return result

    def _advance(self,duration):
        before=bool(self.bounced)
        super()._advance(duration)
        self._new_impact=bool(self.bounced and not before)
        self._after_advance_state=self.state.copy()
        self._after_advance_covariance=self.covariance.copy()

    def _confirm_bounce(self):
        before=self.state.copy()
        result=super()._confirm_bounce()
        self._recovery_state_delta=self.state-before
        self._before_position_update_state=self.state.copy()
        self._before_position_update_covariance=self.covariance.copy()
        return result

    def frame_diagnostics(self,position,snapshot):
        if not hasattr(self,'_before_position_update_state'):
            return {}
        state=self._before_position_update_state
        covariance=self._before_position_update_covariance
        innovation=np.asarray(position)-state[:3]
        noise=np.diag(self.config['measurement_sigma_xyz']**2)
        inflation=snapshot['diagnostics'].get('measurement_noise_inflation',1.)
        gain=np.linalg.solve(covariance[:3,:3]+noise*inflation,covariance[:3,:]).T
        delta=snapshot['state']-state
        ordinary=snapshot['reason']=='accepted'
        if ordinary:
            np.testing.assert_allclose(delta,gain@innovation,rtol=2e-9,atol=2e-11)
        return plain({'new_predicted_impact':getattr(self,'_new_impact',False),
            'impact_jacobian':getattr(self,'_last_impact_jacobian',None),
            'prior_sigmas_p_v_b':np.sqrt(np.maximum(0,np.diag(covariance))),
            'position_innovation_m':innovation,
            'ordinary_position_update_velocity_gain_per_s':float(np.linalg.norm(gain[3:6,:],ord=2)) if ordinary else None,
            'ordinary_position_update_velocity_delta_m_s':delta[3:6] if ordinary else None,
            'recovery_velocity_delta_m_s':self._recovery_state_delta[3:6]})


class CollisionMapPredictor(DiagnosticMixin,GPOnlinePredictor):
    def __init__(self,bank,model_id,config):
        self.model_id=model_id
        super().__init__(bank,config=config)

    def _gp_impact(self,incoming):
        if np.linalg.norm(incoming[3:5])<1e-6:
            return self.bank.apply_state(incoming,'fixed')
        return self.bank.apply_state(incoming,self.model_id)


class OriginalOnlinePredictor(DiagnosticMixin,IdentifiedFilter):
    """Unmodified original selected online mean and initialization, diagnostic only."""


def input_hash(times,xyz):
    digest=hashlib.sha256()
    digest.update(np.ascontiguousarray(times).tobytes())
    digest.update(np.ascontiguousarray(xyz).tobytes())
    return digest.hexdigest()


def error(row,field='predicted_xyz'):
    reason='prediction_reason' if field=='predicted_xyz' else 'frozen_reason'
    if not row['valid'] or row.get(reason) or row.get(field) is None or row.get('target_xyz') is None:
        return None
    prediction,target=np.asarray(row[field],dtype=float),np.asarray(row['target_xyz'],dtype=float)
    if prediction.shape!=(3,) or target.shape!=(3,) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        return None
    return float(np.linalg.norm(prediction-target))


def curve(rows,*,width_s=.02,anchors_only=True):
    groups=defaultdict(list)
    for row in rows:
        if anchors_only and not row['twenty_ms_anchor']:
            continue
        key=(row['model'],row['horizon_s'],round(round(row['relative_capture_s']/width_s)*width_s,8))
        groups[key].append(row)
    output=[]
    for key,members in sorted(groups.items()):
        requested={r['trajectory_id'] for r in members}
        labeled={r['trajectory_id'] for r in members if r['valid']}
        values=defaultdict(list)
        for row in members:
            value=error(row)
            if value is not None:
                values[row['trajectory_id']].append(value**2)
        mse=np.asarray([np.mean(v) for v in values.values()])
        output.append({'model':key[0],'horizon_s':key[1],'capture_bin_s':key[2],
            'requested_trajectories':len(requested),'labeled_trajectories':len(labeled),
            'successful_trajectories':len(values),
            'requested_rows':len(members),'labeled_rows':sum(r['valid'] for r in members),
            'failed_labeled_rows':sum(r['valid'] and error(r) is None for r in members),
            'failures_or_missing_labels':dict(Counter((r.get('prediction_reason') or 'nonfinite_prediction')
                  if r['valid'] else r.get('reason','missing_label') for r in members if error(r) is None)),
            'prediction_coverage_given_label':len(values)/len(labeled) if labeled else None,
            'rmse_m':float(np.sqrt(np.mean(mse))) if len(mse) else None,
            'p95_m':float(np.quantile(np.sqrt(mse),.95)) if len(mse) else None})
    return output


def peak_statistics(rows):
    groups=defaultdict(list)
    for row in rows:
        if -1e-12<=row['relative_capture_s']<=.08+1e-12:
            groups[(row['model'],row['horizon_s'])].append(row)
    reports=[]
    for key,members in sorted(groups.items()):
        by_event=defaultdict(list)
        requested={r['trajectory_id'] for r in members}
        labeled_counts=Counter(r['trajectory_id'] for r in members if r['valid'])
        for row in members:
            value=error(row)
            if value is not None:
                by_event[row['trajectory_id']].append((row,value))
        incomplete=[identity for identity,count in labeled_counts.items() if len(by_event[identity])!=count]
        by_event={identity:values for identity,values in by_event.items()
                  if values and identity not in incomplete}
        peaks=[max(values,key=lambda x:x[1]) for values in by_event.values()]
        rms=np.asarray([np.mean([value**2 for _,value in values]) for values in by_event.values()])
        peak_values=np.asarray([value for _,value in peaks])
        reports.append({'model':key[0],'horizon_s':key[1],'requested_trajectories':len(requested),
            'labeled_trajectories':len(labeled_counts),'independent_trajectories':len(by_event),
            'incomplete_prediction_trajectory_ids':incomplete,
            'missing_label_rows':sum(not r['valid'] for r in members),
            'failed_labeled_rows':sum(r['valid'] and error(r) is None for r in members),
            'definition':'per-event maximum over available measurement labels; scores exclude events with failed labeled predictions, explicitly counted',
            'band_rmse_m':float(np.sqrt(np.mean(rms))) if len(rms) else None,
            'trajectory_peak_p50_m':float(np.quantile(peak_values,.5)) if len(peak_values) else None,
            'trajectory_peak_p95_m':float(np.quantile(peak_values,.95)) if len(peak_values) else None,
            'max_error_m':float(np.max(peak_values)) if len(peak_values) else None,
            'worst_events':[{'trajectory_id':row['trajectory_id'],'capture_relative_s':row['relative_capture_s'],
                             'error_m':value} for row,value in sorted(peaks,key=lambda x:x[1],reverse=True)[:5]]})
    return reports


def diagnose_frames(frames):
    grouped=defaultdict(list)
    for frame in frames:
        grouped[(frame['model'],frame['trajectory_id'])].append(frame)
    events=[]
    for (model,identity),samples in sorted(grouped.items()):
        failed=[s for s in samples if s.get('tracking_failure_reason')]
        available=[s for s in samples if not s.get('tracking_failure_reason')]
        early=[s for s in available if 0<=s['relative_capture_s']<=.03]
        spike=[s for s in available if .01<=s['relative_capture_s']<=.03]
        confirmed=[s for s in available if s['bounce_confirmed']]
        corrections=[(s['relative_capture_s'],np.linalg.norm(s['frame_diagnostics']['ordinary_position_update_velocity_delta_m_s']))
                     for s in early if s['frame_diagnostics'].get('ordinary_position_update_velocity_delta_m_s') is not None]
        events.append({'model':model,'trajectory_id':identity,
            'tracking_failed_frames':len(failed),
            'early_reasons':dict(Counter(s['reason'] for s in early)),
            'spike_frames':len(spike),'spike_frames_unconfirmed':sum(not s['bounce_confirmed'] for s in spike),
            'bounce_confirmed_relative_s':confirmed[0]['relative_capture_s'] if confirmed else None,
            'early_largest_position_update_velocity_correction_m_s':float(max((x[1] for x in corrections),default=0.)),
            'early_largest_velocity_correction_capture_s':max(corrections,key=lambda x:x[1])[0] if corrections else None,
            'near_contact_wait_reasons':dict(Counter(s['diagnostics'].get('bounce_recovery_wait','') for s in early))})
    return events


NUMERICAL_FAILURES=(ValueError,FloatingPointError,np.linalg.LinAlgError)


def failure_reason(stage,exc):
    return stage+': '+type(exc).__name__+': '+str(exc)


def validated_forecast(forecast,count):
    result=np.asarray(forecast,dtype=float)
    if result.ndim!=2 or result.shape[0]!=count or result.shape[1] not in (6,9) or not np.isfinite(result).all():
        raise ValueError('forecast must contain finite H x 6/9 states')
    return result


def safe_forecast(tracker,requested):
    try:
        if isinstance(tracker,GPOnlinePredictor):
            forecast=tracker.predict_target_times(requested)
        else:
            forecast=np.asarray([tracker.predict(float(t))['state'] for t in requested])
        return validated_forecast(forecast,len(requested)),''
    except NUMERICAL_FAILURES as exc:
        return None,failure_reason('forecast',exc)


def validate_cache_item(trace,prepared):
    prefix_t=np.asarray(prepared['raw_prefix_times'],dtype=float)
    prefix_p=np.asarray(prepared['raw_prefix_xyz'],dtype=float)
    before=trace['raw_times']<trace['bounce_reference_s']
    np.testing.assert_array_equal(prefix_t,trace['raw_times'][before])
    np.testing.assert_array_equal(prefix_p,trace['raw_xyz'][before])
    if input_hash(prefix_t,prefix_p)!=prepared['input_sha256']:
        raise RuntimeError('cached input hash differs from the actual full prefix')
    fit=prepared['fit']
    if not prepared['valid'] or not fit['converged'] or fit['capture_t_s']!=prefix_t[-1]:
        raise RuntimeError('a converged full-prefix state at its final observation is required')
    if prepared['prefix_config']['window_s']!=0 or fit['diagnostics']['sampled_rows']!=len(prefix_t):
        raise RuntimeError('cached fit must use every full-prefix observation without subsampling')
    return prefix_t,prefix_p


def paired_rows(rows,models):
    """Retain identical requests but mask scores outside common successes."""
    groups=defaultdict(dict)
    for row in rows:
        if row['model'] in models:
            key=(row['trajectory_id'],row['capture_t_s'],row['target_t_s'])
            if row['model'] in groups[key]:
                raise RuntimeError('duplicate model/observation/target row')
            groups[key][row['model']]=row
    result=[]
    for members in groups.values():
        if set(members)!=set(models):
            raise RuntimeError('collision-map comparisons must request the same clock grid')
        successful=all(error(row) is not None for row in members.values())
        for row in members.values():
            if not successful:
                row={**row,'predicted_xyz':None,'prediction_reason':row.get('prediction_reason') or 'outside_common_success_subset'}
            result.append(row)
    return result


def replay(traces,cache,bank,selected,models,out,*,include_original_online=True):
    rows,frames,seed_checks=[],[],[]
    for n,trace in enumerate(traces):
        prepared=cache[trace['id']]
        prefix_t,prefix_p=validate_cache_item(trace,prepared)
        capture=float(prepared['fit']['capture_t_s'])
        first_seed=None
        seed_failures={}
        candidates=[*models,*(['original_online'] if include_original_online else [])]
        for model in candidates:
            tracking_failure=''
            snapshot=None
            try:
                if model=='original_online':
                    tracker=OriginalOnlinePredictor(selected['parameters'],selected['contact_height'],
                        config=selected['filter'],augmented=bool(selected.get('augmented',True)))
                    for stamp,position in zip(prefix_t,prefix_p):
                        snapshot=tracker.update(float(stamp),position)
                else:
                    tracker=CollisionMapPredictor(bank,model,selected['filter'])
                    snapshot=tracker.seed_from_prefix(prepared['fit'],prefix_t,prefix_p)
            except NUMERICAL_FAILURES as exc:
                tracking_failure=failure_reason('seed_or_initialization',exc)
                seed_failures[model]=tracking_failure
            if model!='original_online' and not tracking_failure:
                if first_seed is None:
                    first_seed=(snapshot['state'].copy(),snapshot['covariance'].copy())
                else:
                    np.testing.assert_array_equal(snapshot['state'],first_seed[0])
                    np.testing.assert_array_equal(snapshot['covariance'],first_seed[1])
            frozen_model='fixed' if model=='original_online' else model
            due=capture+.02
            schedule=[(capture,None,True)]
            for stamp,position in zip(trace['raw_times'],trace['raw_xyz']):
                if stamp<=capture:continue
                if stamp>trace['bounce_reference_s']+.12:break
                anchor=stamp+1e-12>=due
                if anchor:
                    due+=max(1,int(np.floor((stamp-due)/.02))+1)*.02
                schedule.append((float(stamp),position,anchor))
            # Frozen control is evaluated at exactly the same absolute target
            # timestamps, sorted once without shifting its predicted event.
            clock=np.asarray([stamp+h for stamp,_,_ in schedule for h in HORIZONS])
            order=np.argsort(clock,kind='stable')
            frozen=None;frozen_reason=''
            try:
                frozen_sorted=validated_forecast(bank.predict(prepared,frozen_model,clock[order]),len(clock))
                frozen=np.empty_like(frozen_sorted);frozen[order]=frozen_sorted
                frozen=frozen.reshape(len(schedule),len(HORIZONS),9)
            except NUMERICAL_FAILURES as exc:
                frozen_reason=failure_reason('frozen_forecast',exc)
            for k,(stamp,position,anchor) in enumerate(schedule):
                if position is not None and not tracking_failure:
                    try:
                        snapshot=tracker.update(stamp,position)
                    except NUMERICAL_FAILURES as exc:
                        # An update may have partially changed its tracker.
                        # Never resume it or invent later posterior states.
                        tracking_failure=failure_reason('update',exc)
                if position is not None:
                    fd={};diagnostic_reason=''
                    if not tracking_failure:
                        try:
                            fd=tracker.frame_diagnostics(position,snapshot)
                        except NUMERICAL_FAILURES as exc:
                            diagnostic_reason=failure_reason('read_only_diagnostic',exc)
                    frames.append(plain({'trajectory_id':trace['id'],'model':model,'capture_t_s':stamp,
                        'relative_capture_s':stamp-trace['bounce_reference_s'],
                        'state':snapshot['state'] if not tracking_failure else None,
                        'covariance':snapshot['covariance'] if not tracking_failure else None,
                        'bounced':snapshot['bounced'] if not tracking_failure else False,
                        'bounce_confirmed':snapshot['bounce_confirmed'] if not tracking_failure else False,
                        'reason':snapshot['reason'] if not tracking_failure else 'tracking_failure',
                        'diagnostics':snapshot['diagnostics'] if not tracking_failure else {},
                        'frame_diagnostics':fd,'diagnostic_failure_reason':diagnostic_reason,
                        'tracking_failure_reason':tracking_failure}))
                requested=stamp+np.asarray(HORIZONS)
                if tracking_failure:
                    forecast,forecast_reason=None,tracking_failure
                else:
                    forecast,forecast_reason=safe_forecast(tracker,requested)
                for j,label in enumerate(future_labels(trace,stamp,HORIZONS)):
                    rows.append(plain({'trajectory_id':trace['id'],'model':model,'capture_t_s':stamp,
                        'relative_capture_s':stamp-trace['bounce_reference_s'],
                        'reference_contact_t_s':trace['bounce_reference_s'], 'twenty_ms_anchor':anchor,
                        **label,'predicted_xyz':forecast[j,:3] if forecast is not None else None,
                        'prediction_reason':forecast_reason,
                        'frozen_xyz':frozen[k,j,:3] if frozen is not None else None,'frozen_reason':frozen_reason}))
            print(f'SPIKE_ABLATION {n+1}/{len(traces)} model={model} frames={len(schedule)}',flush=True)
        seed_checks.append({'trajectory_id':trace['id'],
            'identical_seed_mean_and_covariance_for':[m for m in models if m not in seed_failures],
            'seed_failures':seed_failures})
    write_csv(out/'predictions.csv',rows)
    dump(out/'frames.json',frames)
    return rows,frames,seed_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'artifacts/tennis_gp_analysis/20261010_full_pre_gp/all_pre_cache.json')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_gp_spike/20261010_collision_map_ablation')
    parser.add_argument('--models',nargs='+',choices=BounceModelBank.model_ids,default=['fixed','gp'])
    parser.add_argument('--skip-original-online',action='store_true',
                        help='run only the fair seeded collision-map candidates; omit the natural-initialization supplement')
    args=parser.parse_args()
    if len(set(args.models))!=len(args.models):
        raise ValueError('each collision map may be requested only once')
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Formal numerical diagnostics must run on Ubuntu in project .venv-tennis')
    out=args.out
    if (out/'summary.json').exists():
        raise RuntimeError('completed diagnostics must be preserved; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json'
    bank=BounceModelBank(checkpoint)
    selected_path=ROOT/'artifacts/tennis_identified/20261009_training_prior/selected_model.json'
    selected=json.loads(selected_path.read_text())
    resources=[checkpoint,selected_path,*[checkpoint.parent/'models'/name for name in bank.checkpoint_names.values()]]
    immutable={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in resources}
    execution=json.loads((args.cache.parent/'execution.json').read_text())
    for filename,expected in execution['checkpoint_hashes'].items():
        if filename in immutable and immutable[filename]!=expected:
            raise RuntimeError('checkpoint changed since the full-precontact analysis')
    previous_summary=json.loads((args.cache.parent/'summary.json').read_text())
    context=previous_summary['prefix_context']
    if context['parameters']!=bank.parameters or context['height']!=bank.height:
        raise RuntimeError('cached upstream parameters differ from the current frozen checkpoint')
    for filename,expected in context['source_sha256'].items():
        if hashlib.sha256((ROOT/'tennis_calibration'/filename).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('cached upstream source changed: '+filename)
    traces=[t for t in load_dataset(ROOT/'artifacts/new_ubuntu/tennis_calibration')
            if t['split']=='test' and t['bounce_reference_s'] is not None]
    cache=json.loads(args.cache.read_text())
    assert set(cache)=={t['id'] for t in traces}
    rows,frames,seed_checks=replay(traces,cache,bank,selected,args.models,out,
                                 include_original_online=not args.skip_original_online)
    common=paired_rows(rows,args.models)
    assert immutable=={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in resources}
    dump(out/'summary.json',{'hostname':platform.node(),'created_utc':datetime.now(timezone.utc).isoformat(),
        'models':args.models,'independent_test_collisions':len(traces),'checkpoint_hashes_unchanged':immutable,
        'primary_protocol':'same full-precontact state and covariance, same raw observations, same EKF/flight parameters; only map and its linked Jacobian differ',
        'supplement_protocol':'original_online uses old natural approximately 80ms initialization and no GP impact hook; it is not a map-only ablation' if not args.skip_original_online else 'original_online supplement omitted by explicit run option',
        'frame_band_s':[0,.12],'peak_band_s':[0,.08], 'horizons_s':HORIZONS,
        'mean_covariance_seed_checks':seed_checks,'twenty_ms_curves':curve(rows),
        'twenty_ms_common_success_curves':curve(common),
        'dense_2p5ms_curves':curve(rows,width_s=.0025,anchors_only=False),
        'per_trajectory_peak_statistics':peak_statistics(rows),
        'per_trajectory_common_success_peak_statistics':peak_statistics(common),
        'prediction_failure_rows':dict(Counter(r['prediction_reason'] for r in rows if r['prediction_reason'])),
        'diagnostic_failure_frames':dict(Counter(f['diagnostic_failure_reason'] for f in frames if f['diagnostic_failure_reason'])),
        'frame_diagnostic_events':diagnose_frames(frames),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test_role':'previously exposed historical regression; no tuning or training',
        'limitations':['event time covariance uses existing reset Jacobian approximation, not a full saltation matrix',
                       'velocities and effective lift are inferred, not directly measured',
                       'changing an impact map also changes its Jacobian, as required by the same EKF rule']})
    print('SPIKE_SUMMARY='+str(out/'summary.json'),flush=True)


if __name__=='__main__':main()
