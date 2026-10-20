"""Isolated Ubuntu GP height-event hold-duration ablation.

Every raw observation advances the same frozen observer. Only causal 20 ms
anchors are forecast/scored, independently of the hold and offline labels.
No selected checkpoint or deployed predictor is modified by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .gp_analysis_plots import _style
from .height_forecast import forecast_posteriors
from .height_policy import HeightPolicyPredictor
from .height_policy_run import FAILURES, phase, row_error
from .hybrid_hold_evaluation import describe
from .identified_run import plain
from .online_data import load_dataset, future_labels, causal_anchors
from .player_bounce import BounceModelBank

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (.1, .2, .5)
HOLDS_MS = (0, 20, 40, 60, 80, 100)
HOLD_COLORS = {0: '#78828B', 20: '#CE4946', 40: '#D98B20',
               60: '#23999D', 80: '#2879B3', 100: '#7148A1'}


def candidate(hold_ms):
    return f'gp_hold{hold_ms:g}ms'


def make_tracker(bank, config, hold_ms):
    if not np.isfinite(hold_ms) or hold_ms < 0:
        raise ValueError('hold duration must be finite and nonnegative')
    # height_online is the matched zero-hold arm: unlike continuous, it does
    # not confirm/reset a bounce before the propagated height event.
    return HeightPolicyPredictor(bank, 'gp', 'height_online' if hold_ms == 0 else 'hold100',
                                 config=config, hold_s=hold_ms / 1000 if hold_ms else .1)


def expected_resume(times, contact, discovery, hold_ms):
    if contact is None or discovery is None:
        return None
    threshold = max(discovery, contact + hold_ms / 1000)
    index = np.searchsorted(times, threshold - 1e-12)
    return float(times[index]) if index < len(times) else None


def _worker_init(checkpoint, selected):
    global BANK, SELECTED
    BANK, SELECTED = BounceModelBank(checkpoint), selected


def run_case(trace, hold_ms):
    tracker = make_tracker(BANK, SELECTED['filter'], hold_ms)
    times, xyz = trace['raw_times'], trace['raw_xyz']
    scope = trace.get('eval_scope_end_s')
    mask = times <= (times[-1] if scope is None else scope)
    times, xyz = times[mask], xyz[mask]
    anchors = {a['current_index'] for a in causal_anchors(
        {'raw_times': times}, initialization_s=0., stride_s=.02)}
    records, states, bounced, captures = [], [], [], []
    reasons = Counter(); failure = ''; held_frames = 0
    for index, (stamp, point) in enumerate(zip(times, xyz)):
        if not failure:
            try:
                snapshot = tracker.update(float(stamp), point)
            except FAILURES as exc:
                failure = type(exc).__name__ + ': ' + str(exc)
        if failure:
            snapshot = tracker.snapshot()
        reasons[snapshot['reason'] if not failure else 'tracking_failure'] += 1
        if snapshot['hold_active'] and not failure:
            held_frames += 1
            assert not snapshot['accepted']
            assert snapshot['reason'] == 'height_policy_prediction_only'
        if index not in anchors:
            continue
        initialized = snapshot['initialized'] and not failure
        records.append({'trajectory_id': trace['id'], 'split': trace['split'],
            'groupid': trace['groupid'], 'candidate': candidate(hold_ms), 'hold_ms': hold_ms,
            'capture_t_s': float(stamp), 'elapsed_capture_s': float(stamp-times[0]),
            'relative_capture_s': float(stamp-trace['bounce_reference_s']),
            'twenty_ms_anchor': True, 'hold_active': snapshot['hold_active'],
            'height_contact_t_s': snapshot['height_contact_t_s'],
            'initialized': bool(initialized),
            'prediction_reason': failure or ('' if initialized else 'not_initialized')})
        if initialized:
            states.append(snapshot['state'].copy()); bounced.append(snapshot['bounced']); captures.append(float(stamp))
    forecast = None; forecast_failure = ''; forecast_reasons = [''] * len(states)
    if states:
        def project(indices):
            result = forecast_posteriors(np.asarray(states)[indices], np.asarray(bounced)[indices],
                np.asarray(captures)[indices], horizons=HORIZONS, bank=BANK, model_id='gp',
                max_step_s=tracker.config['max_step_s'])
            if not np.isfinite(result).all():
                raise FloatingPointError('nonfinite forecast')
            return result
        try:
            forecast = project(slice(None))
        except FAILURES as exc:
            forecast_failure = type(exc).__name__ + ': ' + str(exc)
            forecast = np.full((len(states), len(HORIZONS), 9), np.nan)
            # A failed later state must never erase prior causal forecasts.
            for index in range(len(states)):
                try:
                    forecast[index] = project(slice(index, index+1))[0]
                except FAILURES as error:
                    forecast_reasons[index] = type(error).__name__ + ': ' + str(error)
    rows = []; cursor = 0
    for record in records:
        issue = forecast_reasons[cursor] if record['initialized'] else ''
        predictions = forecast[cursor] if record['initialized'] and forecast is not None and not issue else None
        cursor += int(record['initialized'])
        for index, label in enumerate(future_labels(trace, record['capture_t_s'], HORIZONS)):
            rows.append({**record, **label,
                'predicted_xyz': None if predictions is None else predictions[index, :3].tolist(),
                'prediction_reason': record['prediction_reason'] or issue})
    contact, discovery = snapshot['height_contact_t_s'], snapshot['height_discovery_t_s']
    # For 0 ms there is no resumption; the event-frame clock permits matched
    # post-event diagnostics without pretending an update was ever suppressed.
    resume = discovery if hold_ms == 0 else snapshot['resume_t_s']
    expected = expected_resume(times, contact, discovery, hold_ms)
    if not failure:
        assert (resume is None and expected is None) or (resume is not None and expected is not None
            and abs(resume-expected)<=1e-12), (trace['id'], hold_ms, resume, expected)
    event = {'trajectory_id': trace['id'], 'split': trace['split'], 'groupid': trace['groupid'],
        'candidate': candidate(hold_ms), 'hold_ms': hold_ms, 'height_contact_t_s': contact,
        'height_discovery_t_s': discovery, 'resume_t_s': snapshot['resume_t_s'],
        'diagnostic_resume_or_event_t_s': resume, 'expected_resume_or_event_t_s': expected,
        'reference_contact_t_s': trace['bounce_reference_s'], 'held_frames': held_frames,
        'received_frames': len(times), 'frame_reasons': dict(reasons),
        'initialization_t_s': snapshot['initialization_t_s'],
        'height_event_status': snapshot['height_event_status'],
        'tracking_failure': failure, 'forecast_failure': forecast_failure,
        'contact_minus_reference_s': None if contact is None else contact-trace['bounce_reference_s'],
        'resume_minus_reference_s': None if resume is None else resume-trace['bounce_reference_s'],
        'resume_rounding_delay_s': None if resume is None else resume-max(discovery, contact+hold_ms/1000)}
    return rows, event


def aggregate(rows, events):
    groups = defaultdict(lambda: defaultdict(list)); curves = defaultdict(lambda: defaultdict(list))
    grouped = defaultdict(lambda: defaultdict(list)); recovery = defaultdict(lambda: defaultdict(list))
    group_first200 = defaultdict(lambda: defaultdict(list))
    stage_cells = defaultdict(lambda: defaultdict(list)); reasons = Counter(); keys = defaultdict(set)
    event_map = {(e['trajectory_id'], e['candidate']): e for e in events}
    for event in events:
        keys[(event['split'],event['candidate'])]  # Preserve all-failure arms in coverage.
    for row in rows:
        if row['elapsed_capture_s'] < .1-1e-12:
            continue
        band = phase(row['relative_capture_s'])
        if band is None:
            continue
        split, name, h = row['split'], row['candidate'], row['horizon_s']
        identity, group = row['trajectory_id'], row['groupid']
        value = row_error(row)
        if value is None:
            reasons[(split, name, row.get('reason') or row['prediction_reason'] or 'invalid_prediction')] += 1
            continue
        keys[(split, name)].add((identity, row['capture_t_s'], h, row['target_t_s']))
        groups[(split, name, band, h)][identity].append(value)
        grouped[(split, group, name, band, h)][identity].append(value)
        if h == .5 and 0 <= row['relative_capture_s'] < .2:
            group_first200[(split, group, name)][identity].append(value)
        stage_cells[(split, group, name, identity)][(band, h)].append(value)
        stamp = round(round(row['relative_capture_s'] / .02) * .02, 8)
        curves[(split, name, h, stamp)][identity].append(value)
        event = event_map[(identity, name)]
        resume = event['diagnostic_resume_or_event_t_s']
        if resume is not None and resume-1e-12 <= row['capture_t_s'] < resume+.05-1e-12:
            recovery[(split, name, h)][identity].append(value)
    metrics = [{'split':s,'candidate':c,'capture_phase':p,'horizon_s':h,**describe(v)}
               for (s,c,p,h),v in sorted(groups.items())]
    curve_metrics = [{'split':s,'candidate':c,'horizon_s':h,'time_bin_s':t,**describe(v)}
                     for (s,c,h,t),v in sorted(curves.items())]
    group_metrics = [{'split':s,'groupid':g,'candidate':c,'capture_phase':p,'horizon_s':h,**describe(v)}
                     for (s,g,c,p,h),v in sorted(grouped.items())]
    group_first200_metrics = [{'split':s,'groupid':g,'candidate':c,'horizon_s':.5,**describe(v)}
                             for (s,g,c),v in sorted(group_first200.items())]
    case_scores = []
    for (s,g,c,identity),cells in sorted(stage_cells.items()):
        bands = defaultdict(list)
        for (band,h),v in cells.items():
            bands[band].append(float(np.mean(np.square(v))))
        case_scores.append({'split':s,'groupid':g,'candidate':c,'trajectory_id':identity,
            'balanced_future_mse_m2':float(np.mean([np.mean(v) for v in bands.values()])),
            'phase_horizon_mse_m2':{f'{band}|{h:g}':float(np.mean(np.square(v))) for (band,h),v in cells.items()}})
    rankings = []
    for split,name in sorted({(v['split'],v['candidate']) for v in case_scores}):
        cells = [v for v in case_scores if (v['split'],v['candidate'])==(split,name)]
        rankings.append({'split':split,'candidate':name,'N':len(cells),
            'balanced_future_rmse_m':float(np.sqrt(np.mean([v['balanced_future_mse_m2'] for v in cells])))})
    group_rankings = []
    for split,group,name in sorted({(v['split'],v['groupid'],v['candidate']) for v in case_scores}):
        cells=[v for v in case_scores if (v['split'],v['groupid'],v['candidate'])==(split,group,name)]
        group_rankings.append({'split':split,'groupid':group,'candidate':name,'N':len(cells),
            'balanced_future_rmse_m':float(np.sqrt(np.mean([v['balanced_future_mse_m2'] for v in cells])))})
    recovery_metrics = []
    for (s,c,h),values in sorted(recovery.items()):
        peaks=np.asarray([max(v) for v in values.values()])
        recovery_metrics.append({'split':s,'candidate':c,'horizon_s':h,**describe(values),
            'trajectory_window_max_p95_m':float(np.quantile(peaks,.95)),
            'trajectory_window_max_mean_m':float(peaks.mean()),
            'sampling':'common 20ms anchors in [actual resume, resume+50ms); 0ms uses height discovery'})
    coverage=[]
    for split in sorted({s for s,c in keys}):
        names=sorted(c for s,c in keys if s==split)
        reference=keys[(split,candidate(100))] if candidate(100) in names else keys[(split,names[0])]
        for name in names:
            k=keys[(split,name)]
            coverage.append({'split':split,'candidate':name,'successful_keys':len(k),
                'missing_vs_reference':len(reference-k),'additional_vs_reference':len(k-reference),
                'equal_to_reference':k==reference})
    return {'phase_metrics':metrics,'curves':curve_metrics,'group_phase_metrics':group_metrics,
            'ranking':rankings,'group_ranking':group_rankings,'trajectory_scores':case_scores,
            'group_first200ms_h500_metrics':group_first200_metrics,
            'frozen_model_group_resampling':leave_one_group_selection(case_scores),
            'resume_first50ms_metrics':recovery_metrics,'successful_key_coverage':coverage,
            'coverage_reasons':[{'split':s,'candidate':c,'reason':r,'n':n} for (s,c,r),n in sorted(reasons.items())]}


def leave_one_group_selection(case_scores):
    """Select hold on other held-out groups, then score the left-out group.

    This never refits a model and therefore is deliberately not called CV.
    Frozen training observations are excluded even when --splits includes them.
    The historically exposed test groups remain exploratory robustness data.
    """
    cases=[v for v in case_scores if v['split'] in ('validation','test')]
    group_ids=sorted({v['groupid'] for v in cases});result=[]
    if len(group_ids)<2:return result
    names=sorted({v['candidate'] for v in cases})
    for left_out in group_ids:
        for weighting in ('equal_acquisition_groups','equal_trajectories'):
            scores={}
            for name in names:
                fit=[v for v in cases if v['groupid']!=left_out and v['candidate']==name]
                if weighting=='equal_acquisition_groups':
                    group_mse=[np.mean([v['balanced_future_mse_m2'] for v in fit if v['groupid']==g])
                               for g in group_ids if g!=left_out]
                    scores[name]=float(np.mean(group_mse))
                else:scores[name]=float(np.mean([v['balanced_future_mse_m2'] for v in fit]))
            chosen=min(scores,key=lambda name:(scores[name],name))
            evaluate=[v for v in cases if v['groupid']==left_out and v['candidate']==chosen]
            baseline=[v for v in cases if v['groupid']==left_out and v['candidate']==candidate(100)]
            result.append({'left_out_group':left_out,'selection_weighting':weighting,
                'selection_groups':[g for g in group_ids if g!=left_out],
                'selected_candidate':chosen,'selection_mse_m2_by_candidate':scores,
                'left_out_N':len(evaluate),
                'left_out_rmse_m':float(np.sqrt(np.mean([v['balanced_future_mse_m2'] for v in evaluate]))),
                'left_out_hold100_rmse_m':float(np.sqrt(np.mean([v['balanced_future_mse_m2'] for v in baseline]))) if baseline else None,
                'semantics':'Frozen-model hold selection resampling on original held-out groups; no refit or blind CV.'})
    return result


def verify_baseline(summary, baseline, complete):
    if not baseline or not complete or not any(v['candidate']==candidate(100) for v in summary['ranking']):
        return {'status':'not_checked_partial_or_custom_run'}
    source=json.loads(baseline.read_text());checked=0
    expected={(v['split'],v['phase'],v['horizon_s']):v for v in source['phase_metrics'] if v['candidate']=='gp:hold100'}
    for cell in summary['phase_metrics']:
        if cell['candidate']!=candidate(100):continue
        old=expected[(cell['split'],cell['capture_phase'],cell['horizon_s'])]
        np.testing.assert_allclose(cell['rmse_m'],old['rmse_m'],atol=1e-10,rtol=1e-10)
        assert cell['N']==old['successful_trajectories'];checked+=1
    ranks={r['split']:r for r in source['candidate_ranking'] if r['candidate']=='gp:hold100'}
    for item in summary['ranking']:
        if item['candidate']==candidate(100):
            np.testing.assert_allclose(item['balanced_future_rmse_m'],ranks[item['split']]['balanced_future_rmse_m'],atol=1e-10,rtol=1e-10)
    return {'status':'passed','phase_cells':checked,'path':str(baseline),'sha256':digest(baseline)}


def render(summary,out,split,holds):
    _style();colors=[HOLD_COLORS.get(h,'#4A5967') for h in holds]
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,gridspec_kw={'height_ratios':[3,3,1]})
    fig.subplots_adjust(top=.83,bottom=.12,hspace=.26,wspace=.22)
    name={'validation':'原验证组','test':'历史测试组',
          'oof':'分组交叉验证留出预测','train':'原训练组诊断'}.get(split,split)
    fig.suptitle(f'GP 停更时长：误差均值与 P95（{name}）',fontsize=17,y=.98)
    fig.text(.5,.938,'高度触发；0 ms 仍等待同一高度事件确认反弹。横轴按离线参考触地对齐。',ha='center',fontsize=10)
    handles=[]
    for col,h in enumerate((.1,.5)):
        maxima=[0.,0.];cover_max=0
        for hold,color in zip(holds,colors):
            c=candidate(hold)
            cells=sorted([v for v in summary['curves'] if v['split']==split and v['candidate']==c and
                v['horizon_s']==h and -.2-1e-12<=v['time_bin_s']<=.5+1e-12],key=lambda v:v['time_bin_s'])
            if not cells:continue
            x=np.asarray([v['time_bin_s']*1000 for v in cells])
            for row,field in enumerate(('mean_m','p95_frame_m')):
                y=np.asarray([v[field]*100 for v in cells]);maxima[row]=max(maxima[row],float(y.max()))
                line,=axes[row,col].plot(x,y,color=color,label=f'{hold:g} ms',lw=2)
                np.testing.assert_array_equal(line.get_ydata(),y)
                if row==0 and col==0:handles.append(line)
            counts=[v['N'] for v in cells];cover_max=max(cover_max,max(counts))
            axes[2,col].step(x,counts,where='mid',color=color,lw=1)
        for row in (0,1):
            axes[row,col].set_ylim(0,max(.01,maxima[row]*1.12));assert axes[row,col].get_ylim()[1]>=maxima[row]
            axes[row,col].set_ylabel('距离均值（cm）' if row==0 else '帧距离 P95（cm）')
        axes[0,col].set_title(f'预测未来 {h*1000:g} ms')
        axes[2,col].set_ylim(0,cover_max+2);axes[2,col].set_ylabel('轨迹 N')
        axes[2,col].set_xlabel('相对参考触地时间（ms）')
        for ax in axes[:,col]:
            ax.axvline(0,color='#9DA5AD',ls=':',lw=1);ax.set_xlim(-200,500)
            ax.set_xticks([-200,-100,0,100,200,300,400,500])
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.91),ncol=len(handles),frameon=False)
    fig.text(.07,.05,'每20 ms共同捕获点；逐轨迹等权；不平滑、不截峰。短停更和长停更的实际恢复时刻不同。',fontsize=10)
    fig.text(.07,.021,'P95是帧距离加权分位；同设备测量作参考，非独立真值。N减少会影响尾部曲线解释。',fontsize=10)
    files=[]
    for suffix in ('png','pdf'):
        path=out/f'hold_sweep_mean_p95_{split}.{suffix}';fig.savefig(path,dpi=180,bbox_inches='tight');files.append(str(path))
    plt.close(fig);return files


def render_simple(summary,out):
    """Two comparison columns, with unchanged 100 ms purple semantics."""
    splits=('validation','test');holds=(40,80,100)
    available={(v['split'],v['candidate']) for v in summary['curves']}
    if not all((s,candidate(h)) in available for s in splits for h in holds):
        return []
    _style()
    fig,axes=plt.subplots(3,2,figsize=(13,9.4),sharex=True,
                          gridspec_kw={'height_ratios':[3,3,1]})
    fig.subplots_adjust(top=.82,bottom=.14,hspace=.25,wspace=.19)
    fig.suptitle('暂停多久再恢复 GP？只看未来 500 ms 的位置误差',fontsize=17,y=.985)
    fig.text(.5,.94,'紫线仍是原来的 100 ms；蓝线 80 ms；橙线 40 ms。恢复后继续融合观测，GP 碰撞参数固定。',
             ha='center',fontsize=10.7)
    handles=[];maxima=[0.,0.]
    for col,split in enumerate(splits):
        cover_max=0
        for hold in holds:
            cells=sorted([v for v in summary['curves'] if v['split']==split and
                v['candidate']==candidate(hold) and v['horizon_s']==.5 and
                -.2-1e-12<=v['time_bin_s']<=.5+1e-12],key=lambda v:v['time_bin_s'])
            x=np.asarray([v['time_bin_s']*1000 for v in cells])
            for row,field in enumerate(('mean_m','p95_frame_m')):
                y=np.asarray([v[field]*100 for v in cells]);maxima[row]=max(maxima[row],float(y.max()))
                line,=axes[row,col].plot(x,y,color=HOLD_COLORS[hold],lw=2.4,
                    label=f'停更 {hold} ms 后恢复 GP',zorder=4 if hold==100 else 3)
                np.testing.assert_array_equal(line.get_ydata(),y)
                if col==0 and row==0:handles.append(line)
            counts=[v['N'] for v in cells];cover_max=max(cover_max,max(counts))
            axes[2,col].step(x,counts,where='mid',color=HOLD_COLORS[hold],lw=1.3)
        split_name={'validation':'原验证组','test':'历史测试组',
                    'oof':'分组交叉验证留出预测','train':'原训练组诊断'}.get(split,split)
        axes[0,col].set_title(f'{split_name}：最多 {cover_max} 条轨迹',fontsize=13)
        axes[0,col].set_ylabel('距离误差均值（cm）')
        axes[1,col].set_ylabel('帧距离误差 P95（cm）')
        axes[2,col].set_ylabel('评分轨迹 N');axes[2,col].set_ylim(0,cover_max+2)
        axes[2,col].set_xlabel('收到观测的时刻，相对参考触地（ms）')
        for ax in axes[:,col]:
            ax.axvline(0,color='#8F99A2',ls=':',lw=1.2)
            ax.set_xlim(-200,500);ax.set_xticks([-200,-100,0,100,200,300,400,500])
    # Shared scales make columns comparable. Compute from every displayed line
    # before setting either limit, so a later-added candidate cannot be clipped.
    for row in (0,1):
        for col in (0,1):
            axes[row,col].set_ylim(0,max(.01,maxima[row]*1.12))
            assert axes[row,col].get_ylim()[1]>=maxima[row]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.899),ncol=3,frameon=False)
    fig.text(.065,.073,'越低越准。每20 ms共同捕获点；逐轨迹等权；不平滑、不截峰。P95描述误差尾部。',fontsize=10)
    fig.text(.065,.045,'触地由估计高度检测；实际恢复发生在各条轨迹的检测触地＋停更时长之后首个观测帧。',fontsize=10)
    fig.text(.065,.017,'横轴零点仅用于评分对齐。尾部可用未来参考减少，底排 N 同时给出；参考仍含采集设备误差。',fontsize=10)
    files=[]
    for suffix in ('png','pdf'):
        path=out/f'hold_sweep_simple_h500.{suffix}';fig.savefig(path,dpi=180,bbox_inches='tight');files.append(str(path))
    plt.close(fig)
    return files


def render_saved(out):
    source=out/'summary.json';before=digest(source)
    summary=json.loads(source.read_text())
    if summary.get('schema')!='GP_HEIGHT_HOLD_SWEEP/1':
        raise ValueError('render-only requires a completed GP_HEIGHT_HOLD_SWEEP/1 summary')
    holds=summary['holds_ms'];splits=sorted({v['split'] for v in summary['curves']})
    files=sum([render(summary,out,s,holds) for s in splits],[])+render_simple(summary,out)
    # Unit conversion only: retain saved means and quantiles without scoring or
    # interpolating observations again. These are convenient report source cells.
    table=[{k:v for k,v in cell.items() if k in ('split','candidate','capture_phase','horizon_s','N','n_frames')} |
           {f'{k[:-1]}cm':cell[k]*100 for k in ('mean_m','rmse_m','p95_frame_m','p99_frame_m')}
           for cell in summary['phase_metrics']]
    table_path=out/'phase_mean_p95_cm.json'
    table_path.write_text(json.dumps(table,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    files.append(str(table_path))
    assert digest(source)==before
    manifest={'schema':'GP_HOLD_SWEEP_RENDER/1','created_utc':datetime.now(timezone.utc).isoformat(),
        'hostname':platform.node(),'python_prefix':sys.prefix,'source_summary':str(source),
        'source_summary_sha256_unchanged':before,'source_metrics_recomputed':False,
        'colors':HOLD_COLORS,'simple_holds_ms':[40,80,100],
        'all_displayed_peaks_within_axes':True,'files':files}
    (out/'render_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(args):
    if (args.out/'summary.json').exists():
        raise RuntimeError('Preserve completed output; choose a new directory')
    args.out.mkdir(parents=True,exist_ok=True)
    started=perf_counter();selected=json.loads(args.selected.read_text());bank=BounceModelBank(args.checkpoint)
    protected=[args.selected,args.checkpoint,*[args.checkpoint.parent/'models'/n for n in bank.checkpoint_names.values()],
        args.calibration/'split.json',args.calibration/'clean/annotations.json',args.calibration/'clean/summary.json',
        args.calibration/'bounce/summary.json',*sorted((args.calibration/'clean').rglob('*.csv'))]
    hashes={str(p):digest(p) for p in protected}
    traces=[t for t in load_dataset(args.calibration) if t['split'] in args.splits and t['bounce_reference_s'] is not None]
    if args.trajectory_ids:
        ids=json.loads(args.trajectory_ids.read_text())
        if not isinstance(ids,list) or not all(isinstance(v,str) for v in ids):
            raise ValueError('--trajectory-ids must contain a JSON list of trajectory ids')
        traces=[t for t in traces if t['id'] in set(ids)]
        if set(ids)!={t['id'] for t in traces}:
            raise ValueError('requested ids missing, excluded by split, or without evaluable bounce')
    if args.groups:
        traces=[t for t in traces if str(t['groupid']) in args.groups]
    if not traces:raise ValueError('no evaluable traces selected')
    if args.limit:traces=traces[:args.limit]
    execution={'created_utc':datetime.now(timezone.utc).isoformat(),'hostname':platform.node(),
        'python_prefix':sys.prefix,'holds_ms':args.holds_ms,'splits':dict(Counter(t['split'] for t in traces)),
        'tasks':len(traces)*len(args.holds_ms),'source_hashes':{str(p.relative_to(ROOT)):digest(p)
            for p in (ROOT/'tennis_calibration').glob('*.py')},'protected_hashes':hashes}
    (args.out/'execution.json').write_text(json.dumps(execution,indent=2,ensure_ascii=False)+'\n')
    rows=[];events=[];writer=None
    with gzip.open(args.out/'predictions.csv.gz','wt',newline='') as stream:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=_worker_init,initargs=(args.checkpoint,selected)) as pool:
            pending=[pool.submit(run_case,t,h) for t in traces for h in args.holds_ms]
            for i,future in enumerate(as_completed(pending),1):
                batch,event=future.result();rows.extend(batch);events.append(event)
                if writer is None:
                    writer=csv.DictWriter(stream,fieldnames=list(batch[0]));writer.writeheader()
                for r in batch:
                    writer.writerow({k:json.dumps(plain(v),ensure_ascii=False) if isinstance(v,(list,dict,np.ndarray)) else v for k,v in r.items()})
                if i%10==0 or i==len(pending):
                    print(f'HOLD_SWEEP {i}/{len(pending)} elapsed_s={perf_counter()-started:.1f}',flush=True)
    summary=aggregate(rows,events)
    summary['baseline_check']=verify_baseline(summary,args.baseline,not (args.limit or args.trajectory_ids or args.groups))
    assert hashes=={str(p):digest(p) for p in protected}
    # All arms must localize the same first event, since they are numerically
    # identical before it; any difference indicates an implementation defect.
    by_trace=defaultdict(list)
    for e in events:by_trace[e['trajectory_id']].append(e)
    for identity,items in by_trace.items():
        for key in ('height_contact_t_s','height_discovery_t_s','initialization_t_s'):
            assert len({e[key] for e in items})==1,(identity,key)
    eligible={v['candidate'] for v in summary['successful_key_coverage'] if v['split']=='validation' and v['equal_to_reference']}
    selection=[r for r in summary['ranking'] if r['split']=='validation' and r['candidate'] in eligible]
    chosen=min(selection,key=lambda r:r['balanced_future_rmse_m'])['candidate'] if selection else None
    summary.update(schema='GP_HEIGHT_HOLD_SWEEP/1',**execution,elapsed_s=perf_counter()-started,
        events=sorted(events,key=lambda e:(e['split'],e['trajectory_id'],e['hold_ms'])),
        horizons_s=HORIZONS,anchor_stride_s=.02,protected_hashes_unchanged=True,
        selected_on_validation=chosen,common_first_height_events=True,
        semantics='0ms=height_online; positive=hold100 with requested duration; all raw frames consumed; no old-frame replay',
        recovery_diagnostic='first 50ms after actual causal resume, common 20ms anchors; 0ms after event discovery, no hold',
        score_definition='trajectory equal weight; within-trajectory phase/horizon balanced MSE then across-trajectory RMSE',
        p95_definition='step inverse frame ECDF, each trajectory has total weight 1/N',
        limitations=['Frozen original model; re-partitioning original training trajectories would leak.',
            'Group summaries assess held-out acquisition groups, not refit grouped cross validation.',
            'Historical exposed test is regression, not blind validation.',
            'Same-device labels are not independent truth; frames within trajectories are correlated.',
            'Resume peak diagnostic is sampled every20ms, not the all-frame maximum.',
            'No deployed default changed; no new physical or collision model was trained.'])
    summary['files']=sum([render(summary,args.out,s,args.holds_ms) for s in sorted({t['split'] for t in traces})],[])+render_simple(summary,args.out)
    (args.out/'summary.json').write_text(json.dumps(plain(summary),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'ranking':summary['ranking'],'selected_on_validation':chosen,
        'baseline_check':summary['baseline_check'],'summary':str(args.out/'summary.json')},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_hold_sweep/20261013_hold0_100')
    parser.add_argument('--render-only',action='store_true',help='Render saved summary in --out without model replay or metric changes')
    parser.add_argument('--calibration',type=Path,default=ROOT/'artifacts/new_ubuntu/tennis_calibration')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json')
    parser.add_argument('--selected',type=Path,default=ROOT/'artifacts/tennis_identified/20261009_training_prior/selected_model.json')
    parser.add_argument('--baseline',type=Path,default=ROOT/'artifacts/tennis_height_policy/20261010_height_hold100_v2/comparison_summary.json')
    parser.add_argument('--skip-baseline-check',action='store_true')
    parser.add_argument('--holds-ms',nargs='+',type=float,default=list(HOLDS_MS))
    parser.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['validation','test'])
    parser.add_argument('--trajectory-ids',type=Path,help='JSON list of exact evaluation ids, e.g. a refit fold')
    parser.add_argument('--groups',nargs='+',help='Keep these acquisition group ids')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Run numerical experiments on Ubuntu in project .venv-tennis')
    if args.render_only:
        render_saved(args.out)
        return
    if args.workers<1 or any(not np.isfinite(h) or h<0 for h in args.holds_ms) or len(set(args.holds_ms))!=len(args.holds_ms):
        raise ValueError('positive worker count and unique finite nonnegative holds required')
    args.holds_ms=sorted(args.holds_ms)
    if args.skip_baseline_check:args.baseline=None
    execute(args)


if __name__=='__main__':main()
