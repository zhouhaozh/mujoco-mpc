"""Independent Ubuntu CSV audit of full-stream height observation policies.

Reconstructs labels, common capture grid and trajectory/phase weighting without
importing the evaluated runner's reducers or selection logic. Predictions and
immutable checkpoints are read-only; the sole output is an audit JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .online_data import future_labels, load_dataset

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUT=ROOT/'artifacts/tennis_height_policy/20261010_height_hold100'
PHASES=('precontact','first100ms','recovery100_200ms','late200_500ms')
HEIGHT_POLICIES={'height_online','hold100','permanent_hold','init_hold100'}


def file_digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):value.update(chunk)
    return value.hexdigest()


def get_phase(tau):
    if tau < 0: return PHASES[0]
    if tau < .1: return PHASES[1]
    if tau < .2: return PHASES[2]
    return PHASES[3] if tau <= .5+1e-12 else None


def boolean(value):
    return str(value).lower() in ('true','1')


def vector(value):
    return json.loads(value) if value else None


def distribution(values):
    a=np.asarray(values,float)
    return {'n':len(a),'p50_m':float(np.median(a)) if len(a) else None,
            'p95_m':float(np.quantile(a,.95)) if len(a) else None,
            'max_m':float(np.max(a)) if len(a) else None}


def audit(out):
    out=Path(out)
    summary=json.loads((out/'summary.json').read_text())
    execution=json.loads((out/'execution.json').read_text())
    legacy=json.loads((out/'legacy_baseline_summary.json').read_text())
    traces={t['id']:t for t in load_dataset(ROOT/'artifacts/new_ubuntu/tennis_calibration')
            if t['split'] in ('validation','test') and t['bounce_reference_s'] is not None}
    events={(e['trajectory_id'],e['candidate']):e for e in summary['events']}
    curves=defaultdict(list); bands=defaultdict(list); cases=[]
    counts=Counter(); label_cache={}; case_keys=set(); peaks=defaultdict(list)
    expected_anchors={}; expected_dense={}
    collision=json.loads((ROOT/'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json').read_text())
    filter_cfg=json.loads((ROOT/'artifacts/tennis_identified/20261009_training_prior/selected_model.json').read_text())['filter']
    for identity,t in traces.items():
        raw=t['raw_times']; scope=t.get('eval_scope_end_s')
        scope=float(raw[-1]) if scope is None else float(scope)
        raw=raw[raw<=scope]
        slots=raw[0]+np.arange(int(np.floor((raw[-1]-raw[0])/.02))+1)*.02
        indices=np.searchsorted(raw,slots-1e-12)
        expected_anchors[identity]={float(raw[i]) for i in np.unique(indices) if i<len(raw)}
        near_height=collision['upstream']['height']+filter_cfg['bounce_observation_height_m']
        zz=t['raw_xyz'][t['raw_times']<=scope,2]
        near=np.flatnonzero(zz<=near_height)
        dense=set() if not len(near) else {float(stamp) for stamp in raw
                                             if raw[near[0]]<=stamp<=raw[near[0]]+.35+1e-12}
        expected_dense[identity]=expected_anchors[identity]|dense

    def flush(members):
        if not members: return
        first=members[0]; identity=first['trajectory_id'];candidate=first['candidate']
        split=first['split']; event=events[(identity,candidate)]
        key=(identity,candidate)
        if key in case_keys: raise AssertionError('duplicated/noncontiguous candidate case')
        case_keys.add(key)
        legacy_case=candidate.startswith('legacy_')
        anchors={r['capture_t_s'] for r in members if r['twenty_ms_anchor']}
        all_anchors={r['capture_t_s'] for r in members}
        assert anchors==expected_anchors[identity], (identity,candidate,'base grid')
        assert all_anchors==(expected_anchors[identity] if legacy_case else expected_dense[identity]), (identity,candidate,'dense grid')
        counts['common_grid_cases']+=1
        assert len(members)==4*len(all_anchors)
        cell=defaultdict(list); phase_cells=defaultdict(list)
        contact_values={r['height_contact_t_s'] for r in members
                        if r['height_contact_t_s'] is not None}
        if not legacy_case:
            assert len(contact_values)<=1
            observed=next(iter(contact_values),None)
            assert observed==event.get('height_contact_t_s')
            assert bool(contact_values)==bool(event.get('height_triggered'))
            counts['event_csv_consistency_cases']+=1
        for r in members:
            h=r['horizon_s'];stamp=r['capture_t_s'];tau=r['relative_capture_s']
            assert abs(r['target_t_s']-stamp-h)<=1e-12
            assert abs(r['reference_contact_t_s']-traces[identity]['bounce_reference_s'])<=1e-12
            assert abs(r['elapsed_capture_s']-(stamp-traces[identity]['raw_times'][0]))<=1e-12
            assert abs(tau-(stamp-r['reference_contact_t_s']))<=1e-12
            cache_key=(identity,stamp,h)
            if cache_key not in label_cache:
                label_cache[cache_key]=future_labels(traces[identity],stamp,[h])[0]
            label=label_cache[cache_key]
            assert r['valid']==label['valid']
            assert r['reason']==label['reason']
            assert r['target_xyz']==label['target_xyz']
            counts['label_clock_rows']+=1
            if r['elapsed_capture_s']<.08-1e-12:
                assert r['predicted_xyz'] is None and r['prediction_reason']=='not_initialized'
                counts['warmup_unavailable_rows']+=1
            if r['predicted_xyz'] is not None:
                assert not r['prediction_reason']
            error=None
            if r['valid'] and r['predicted_xyz'] is not None and not r['prediction_reason']:
                predicted=np.asarray(r['predicted_xyz'],float)
                target=np.asarray(r['target_xyz'],float)
                assert predicted.shape==target.shape==(3,) and np.isfinite(predicted).all()
                error=float(np.linalg.norm(predicted-target))
            r['error']=error
            if r['twenty_ms_anchor']:
                cell[('elapsed20',h,round(round(r['elapsed_capture_s']/.02)*.02,8))].append(r)
                cell[('contact20',h,round(round(tau/.02)*.02,8))].append(r)
            if not legacy_case and -.08<=tau<=.25:
                cell[('contact_dense2p5',h,round(round(tau/.0025)*.0025,8))].append(r)
            phase=get_phase(tau)
            if r['twenty_ms_anchor'] and r['elapsed_capture_s']>=.1-1e-12 and phase:
                phase_cells[(phase,h)].append(r)
        for (axis,h,stamp),rr in cell.items():
            errors=[r['error'] for r in rr if r['error'] is not None]
            curves[(split,candidate,axis,h,stamp)].append({
                'mse':float(np.mean(np.square(errors))) if errors else None,
                'labeled':any(r['valid'] for r in rr),
                'fail':sum(r['valid'] and r['error'] is None for r in rr),
                'missing':sum(not r['valid'] for r in rr)})
        mse_by_phase=defaultdict(list);fail=0
        for (phase,h),rr in phase_cells.items():
            errors=[r['error'] for r in rr if r['error'] is not None]
            mse=float(np.mean(np.square(errors))) if errors else None
            failure=sum(r['valid'] and r['error'] is None for r in rr)
            bands[(split,candidate,phase,h)].append({
                'mse':mse,'labeled':sum(r['valid'] for r in rr),'fail':failure,
                'max':max(errors) if errors else None})
            if h>0:
                fail+=failure
                if mse is not None:mse_by_phase[phase].append(mse)
        phase_means=[float(np.mean(mse_by_phase[p])) for p in PHASES if mse_by_phase[p]]
        cases.append({'split':split,'candidate':candidate,'policy':event['policy'],
            'mse':float(np.mean(phase_means)) if phase_means else None,'fail':fail})
        if not legacy_case and event['policy'] in ('height_online','hold100','init_hold100'):
            available=[r for r in members if r['horizon_s']==.5 and r['error'] is not None]
            for name,selected in (
                ('contact_0_100ms',[r['error'] for r in available if 0<=r['relative_capture_s']<.1]),
                ('contact_100_200ms',[r['error'] for r in available if .1<=r['relative_capture_s']<.2]),
                ('resume_0_50ms',[r['error'] for r in available if event.get('resume_t_s') is not None
                                 and 0<=r['capture_t_s']-event['resume_t_s']<=.05])):
                if selected:peaks[(split,candidate,name)].append(max(selected))

    for filename in ('predictions.csv','legacy_predictions.csv'):
        current=None;members=[]
        legacy_groups=defaultdict(list) if filename=='legacy_predictions.csv' else None
        with (out/filename).open(newline='') as stream:
            for r in csv.DictReader(stream):
                for name in ('capture_t_s','elapsed_capture_s','relative_capture_s','reference_contact_t_s','horizon_s','target_t_s'):
                    r[name]=float(r[name])
                for name in ('valid','twenty_ms_anchor'):r[name]=boolean(r.get(name,'True'))
                for name in ('predicted_xyz','target_xyz'):r[name]=vector(r[name])
                r['height_contact_t_s']=float(r['height_contact_t_s']) if r.get('height_contact_t_s') else None
                key=(r['trajectory_id'],r['candidate'])
                if legacy_groups is not None:
                    # The small legacy file interleaves two heads by anchor;
                    # the large main CSV stays streaming one case at a time.
                    legacy_groups[key].append(r)
                    continue
                if current is not None and current!=key:flush(members);members=[]
                current=key;members.append(r)
        if legacy_groups is not None:
            for members in legacy_groups.values():flush(members)
        else:flush(members)
    assert case_keys==set(events)

    def compare(a,b):
        if a is None or b is None:assert a is b
        else:assert abs(a-b)<=2e-12*max(1.,abs(a),abs(b)), (a,b)
        counts['numeric_cells_compared']+=1

    for row in summary['curves']:
        rr=curves[(row['split'],row['candidate'],row['axis'],row['horizon_s'],row['time_bin_s'])]
        mse=np.array([r['mse'] for r in rr if r['mse'] is not None])
        for key,value in (('requested_trajectories',len(rr)),('labeled_trajectories',sum(r['labeled'] for r in rr)),
                          ('successful_trajectories',len(mse)),('failed_labeled_rows',sum(r['fail'] for r in rr)),
                          ('missing_label_rows',sum(r['missing'] for r in rr))):assert row[key]==value
        compare(row['rmse_m'],float(np.sqrt(np.mean(mse))) if len(mse) else None)
        compare(row['p95_m'],float(np.quantile(np.sqrt(mse),.95)) if len(mse) else None)
        counts['curve_cells']+=1
    for row in summary['phase_metrics']:
        rr=bands[(row['split'],row['candidate'],row['phase'],row['horizon_s'])]
        mse=np.array([r['mse'] for r in rr if r['mse'] is not None])
        for key,value in (('requested_trajectories',len(rr)),('successful_trajectories',len(mse)),
                          ('labeled_rows',sum(r['labeled'] for r in rr)),('failed_labeled_rows',sum(r['fail'] for r in rr))):assert row[key]==value
        compare(row['rmse_m'],float(np.sqrt(np.mean(mse))) if len(mse) else None)
        compare(row['p95_m'],float(np.quantile(np.sqrt(mse),.95)) if len(mse) else None)
        compare(row['max_error_m'],max((r['max'] for r in rr if r['max'] is not None),default=None))
        counts['phase_cells']+=1
    independent_rank=[]
    for row in summary['candidate_ranking']:
        rr=[c for c in cases if c['split']==row['split'] and c['candidate']==row['candidate']]
        ee=[e for e in events.values() if e['split']==row['split'] and e['candidate']==row['candidate']]
        mse=np.array([c['mse'] for c in rr if c['mse'] is not None])
        failure=sum(c['fail'] for c in rr)
        misses=sum(not e.get('height_triggered',False) for e in ee) if row['policy'] in HEIGHT_POLICIES else 0
        assert row['scored_trajectories']==len(mse) and row['independent_trajectories']==len(rr)
        assert row['failed_future_labeled_rows']==failure and row['required_height_event_misses']==misses
        compare(row['balanced_future_rmse_m'],float(np.sqrt(np.mean(mse))) if len(mse) else None)
        compare(row['per_trajectory_balanced_rms_p95_m'],float(np.quantile(np.sqrt(mse),.95)) if len(mse) else None)
        error=[e['height_contact_t_s']-e['reference_contact_t_s'] for e in ee if e.get('height_contact_t_s') is not None]
        assert row['detected_height_events']==len(error)
        compare(row['height_time_error_median_ms'],float(np.median(error)*1000) if error else None)
        compare(row['height_time_abs_error_p95_ms'],float(np.quantile(np.abs(error),.95)*1000) if error else None)
        independent_rank.append({'candidate':row['candidate'],'split':row['split'],'policy':row['policy'],
                                 'fail':failure,'miss':misses,'rmse':float(np.sqrt(np.mean(mse))) if len(mse) else float('inf')})
    key=lambda r:(r['fail'],r['miss'],r['rmse'],r['candidate'])
    validation=sorted((r for r in independent_rank if r['split']=='validation'),key=key)
    chosen=summary['validation_only_selection']
    assert chosen['overall']==validation[0]['candidate']
    height=sorted((r for r in validation if r['policy'] in HEIGHT_POLICIES),key=key)
    assert chosen['height_policy']==height[0]['candidate']
    for policy,candidate in chosen['per_policy'].items():
        assert candidate==min((r for r in validation if r['policy']==policy),key=key)['candidate']
    checkpoint_hashes=execution['checkpoint_hashes']
    assert checkpoint_hashes=={p:file_digest(p) for p in checkpoint_hashes}
    legacy_hashes=legacy['checkpoint_sha256_before']
    assert legacy_hashes==legacy['checkpoint_sha256_after']
    legacy_dir=ROOT/'artifacts/tennis_online/20261008_final'
    assert legacy_hashes=={p:file_digest(legacy_dir/p) for p in legacy_hashes}
    audit_result={'platform':platform.platform(),'hostname':platform.node(),
        'completed_utc':datetime.now(timezone.utc).isoformat(),'all_checks_passed':True,
        'counts':dict(counts),'cases':len(case_keys),'distinct_offline_labels_checked':len(label_cache),
        'selection_independently_recomputed':chosen,
        'weighting':'within trajectory: available horizons equally within each phase; phases equally; then trajectories equally; h0 excluded',
        'original_and_new_checkpoints_unchanged':True,
        'dense_legacy_comparison_excluded':True,
        'peak_migration_diagnostic':[{'split':split,'candidate':candidate,'window':window,
                                     'max_error_per_trajectory':distribution(values)}
                                    for (split,candidate,window),values in sorted(peaks.items())],
        'interpretation':['h0 is posterior tracking error against measurement labels; positive horizons are future prediction errors',
            'height events audited against saved CSV metadata, not independently re-inferred physical contacts',
            'peak diagnostic uses each trajectory maximum among available dense/20ms forecasts at h500; no legacy dense peak comparison',
            'historical test already exposed and acquisition groups inferred; this is regression, not new blind generalization'],
        'inputs_sha256':{name:file_digest(out/name)
                         for name in ('summary.json','execution.json','predictions.csv','legacy_predictions.csv')},
        'auditor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'independent_height_audit.json').write_text(json.dumps(audit_result,indent=2)+'\n')
    print(json.dumps({'all_checks_passed':True,'counts':dict(counts),'selection':chosen,'cases':len(case_keys)},indent=2))
    return audit_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        parser.error('run independent numerical audit in Ubuntu project .venv-tennis')
    audit(args.out)


if __name__=='__main__':main()
