"""Ubuntu-only causal replay of a GP hold followed by the frozen legacy MLP.

The master event schedule is replayed from an immutable earlier GP run. An
event is exposed only at its original discovery time; reference contacts are
never switching inputs. The strict candidate holds both observers, while the
background-output candidate deliberately leaves the old observer continuous.
No parameters or deployed defaults are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
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
from .height_legacy_baseline import forecast_legacy, update_filter_only, checkpoint_hashes, LEGACY_RUN
from .legacy_hold_adapter import create_held_legacy
from .online_data import load_dataset, future_labels, causal_anchors
from .online_run import write_csv
from .residual_distribution import digest, quantile, capture_phase, PHASES

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (.1,.2,.5)
GP = 'gp_hold100'
STRICT = 'gp_hold100_then_held_legacy_mlp'
BACKGROUND = 'gp_hold100_then_continuous_background_legacy_mlp'
LEGACY = 'legacy_mlp_continuous'
NAMES = {GP:'GP：停更后仍用 GP', STRICT:'GP：停更后切 EKF＋MLP（两者停更）',
         BACKGROUND:'GP：切后台持续更新的 EKF＋MLP', LEGACY:'原 EKF＋MLP：始终更新'}


def as_row(row):
    item = dict(row)
    for key in ('capture_t_s','target_t_s','horizon_s','elapsed_capture_s','relative_capture_s'):
        item[key] = float(row[key])
    for key in ('predicted_xyz','target_xyz'):
        item[key] = json.loads(row[key]) if row[key] else None
    item['valid'] = str(row['valid']).lower() in ('true','1')
    if 'hold_active' in row: item['hold_active']=str(row['hold_active']).lower() in ('true','1')
    return item


def row_key(row):
    return row['trajectory_id'], row['capture_t_s'], row['horizon_s'], row['target_t_s']


def load_sources(result):
    summary = json.loads((result/'comparison_summary.json').read_text())
    events = {e['trajectory_id']:e for e in summary['events'] if e['candidate']=='gp:hold100'}
    cases = defaultdict(dict); old = {}
    for filename in ('predictions.csv','legacy_predictions.csv'):
        with (result/filename).open(newline='') as stream:
            for raw in csv.DictReader(stream):
                candidate = raw['candidate']
                if filename=='predictions.csv':
                    if candidate!='gp:hold100' or raw['twenty_ms_anchor'].lower() not in ('true','1'): continue
                elif candidate!='legacy_ekf6_mlp': continue
                if float(raw['horizon_s']) not in HORIZONS: continue
                row = as_row(raw);key = row_key(row)
                if filename=='predictions.csv': cases[row['trajectory_id']][key]=row
                else: old[key]=row
    return summary, events, dict(cases), old


def replay_strict(task):
    trace, event, source_rows = task
    identity = trace['id']; predictor = create_held_legacy(LEGACY_RUN)
    discovery = event['height_discovery_t_s'];resume = event['resume_t_s']
    if discovery is None or resume is None: raise ValueError('master event unavailable')
    assert discovery <= resume and event['height_contact_t_s']+.1 <= resume+1e-12
    times,xyz = trace['raw_times'],trace['raw_xyz']
    expected_resume=float(times[np.searchsorted(times,max(discovery,event['height_contact_t_s']+.1)-1e-12)])
    assert abs(resume-expected_resume)<=1e-12
    scope = trace.get('eval_scope_end_s');scope = float(times[-1]) if scope is None else float(scope)
    anchors = {a['current_index'] for a in causal_anchors(trace,initialization_s=0.,stride_s=.02)}
    received=held=assimilated=0; reasons=Counter(); forecasts={}; boundary=None
    for index,(stamp,point) in enumerate(zip(times,xyz)):
        stamp=float(stamp)
        if stamp>scope+1e-12: break
        # The saved master event is unknown to this observer before discovery.
        skip = stamp >= discovery-1e-12 and stamp < resume-1e-12
        predictor.filter.prediction_only=skip
        snapshot=update_filter_only(predictor,stamp,point)
        received+=1;held+=int(skip);assimilated+=int(not skip)
        reasons[snapshot['reason']]+=1
        assert snapshot['t_s']==stamp
        if skip: assert not snapshot['accepted'] and snapshot['reason']=='legacy_height_policy_prediction_only'
        if index in anchors:
            for h in HORIZONS:
                assert bool(source_rows[(identity,stamp,h,stamp+h)]['hold_active'])==skip
        if index not in anchors or stamp < resume-1e-12: continue
        predictions,prediction_reasons=forecast_legacy(predictor,snapshot)
        labels=future_labels(trace,stamp,(0.,)+HORIZONS)
        for j,h in enumerate(HORIZONS,start=1):
            key=(identity,stamp,h,stamp+h)
            assert key in source_rows
            source=source_rows[key];label=labels[j]
            assert source['valid']==label['valid'] and source['target_xyz']==label['target_xyz']
            output=predictions['legacy_ekf6_mlp'][j]
            item={**source,'candidate':STRICT,'output_source':'held_legacy_mlp',
                  'predicted_xyz':None if output is None else np.asarray(output).tolist(),
                  'prediction_reason':prediction_reasons['legacy_ekf6_mlp'][j] if output is None else '',
                  'master_resume_t_s':resume,'master_discovery_t_s':discovery}
            forecasts[key]=item
        if boundary is None:
            boundary={'first_output_capture_t_s':stamp,'delay_from_resume_s':stamp-resume,
                      'candidate_forecast_jump_same_target_m':{
                          str(h):float(np.linalg.norm(np.asarray(forecasts[(identity,stamp,h,stamp+h)]['predicted_xyz'])-
                                      np.asarray(source_rows[(identity,stamp,h,stamp+h)]['predicted_xyz'])))
                          for h in HORIZONS}}
    rows=[]
    for key,source in source_rows.items():
        if source['capture_t_s']>=resume-1e-12: rows.append(forecasts[key])
        else: rows.append({**source,'candidate':STRICT,'output_source':'gp_hold100',
                           'master_resume_t_s':resume,'master_discovery_t_s':discovery})
    return rows,{'trajectory_id':identity,'split':trace['split'],'received_frames':received,
                 'held_frames':held,'ordinary_update_frames':assimilated,'reasons':dict(reasons),
                 'height_discovery_t_s':discovery,'resume_t_s':resume,'switch_boundary':boundary}


def error(row):
    if not row['valid'] or row['prediction_reason'] or row['predicted_xyz'] is None: return None
    p,y=np.asarray(row['predicted_xyz']),np.asarray(row['target_xyz'])
    assert p.shape==y.shape==(3,) and np.isfinite(p).all() and np.isfinite(y).all()
    return float(np.linalg.norm(p-y))


def describe(groups):
    arrays=[np.asarray(v,float) for v in groups.values() if len(v)]
    if not arrays: return {'N':0,'n_frames':0,'mean_m':None,'rmse_m':None,'p95_frame_m':None}
    x=np.concatenate(arrays);w=np.concatenate([np.full(len(v),1/len(arrays)/len(v)) for v in arrays])
    means=np.array([v.mean() for v in arrays]);mses=np.array([np.mean(v*v) for v in arrays])
    mean=float(np.dot(x,w));mse=float(np.dot(x*x,w))
    np.testing.assert_allclose(mean,means.mean(),atol=1e-12)
    np.testing.assert_allclose(mse,mses.mean(),atol=1e-12)
    assert mean<=np.sqrt(mse)+1e-12
    return {'N':len(arrays),'n_frames':len(x),'mean_m':mean,'rmse_m':float(np.sqrt(mse)),
            'p50_frame_m':quantile(x,w,.5),'p95_frame_m':quantile(x,w,.95),
            'p99_frame_m':quantile(x,w,.99),'max_m':float(x.max())}


def aggregate(rows):
    phases=defaultdict(lambda:defaultdict(list));curves=defaultdict(lambda:defaultdict(list))
    reasons=Counter();successful_keys=defaultdict(set)
    for row in rows:
        if row['elapsed_capture_s']<.1-1e-12: continue
        band=capture_phase(row['relative_capture_s'])
        if band is None: continue
        split,candidate,h=row['split'],row['candidate'],row['horizon_s'];v=error(row)
        if v is None:
            reasons[(split,candidate,row.get('reason') or row['prediction_reason'])]+=1
            continue
        successful_keys[(split,candidate)].add(row_key(row))
        phases[(split,candidate,band,h)][row['trajectory_id']].append(v)
        stamp=round(round(row['relative_capture_s']/.02)*.02,8)
        curves[(split,candidate,h,stamp)][row['trajectory_id']].append(v)
    for split in ('validation','test'):
        assert all(successful_keys[(split,c)]==successful_keys[(split,GP)] for c in (STRICT,BACKGROUND,LEGACY))
    metrics=[{'split':s,'candidate':c,'capture_phase':p,'horizon_s':h,**describe(v)}
             for (s,c,p,h),v in sorted(phases.items())]
    curve=[{'split':s,'candidate':c,'horizon_s':h,'time_bin_s':t,**describe(v)}
           for (s,c,h,t),v in sorted(curves.items())]
    ranking=[]
    for split in ('validation','test'):
        for candidate in (GP,STRICT,BACKGROUND,LEGACY):
            trajectory_cells=defaultdict(lambda:defaultdict(list))
            for (s,c,p,h),groups in phases.items():
                if (s,c)!=(split,candidate): continue
                for identity,values in groups.items():trajectory_cells[identity][p].append(float(np.mean(np.square(values))))
            balanced=[np.mean([np.mean(v) for v in bands.values()]) for bands in trajectory_cells.values()]
            ranking.append({'split':split,'candidate':candidate,'N':len(balanced),
                            'balanced_future_rmse_m':float(np.sqrt(np.mean(balanced)))})
    return metrics,curve,ranking,[{'split':s,'candidate':c,'reason':reason,'n':n} for (s,c,reason),n in reasons.items()]


def render(curves,out,split):
    _style();colors={GP:'#7148A1',STRICT:'#167DA5'}
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,
                          gridspec_kw={'height_ratios':[3,3,1]})
    fig.subplots_adjust(top=.81,bottom=.12,hspace=.27,wspace=.21)
    split_name='验证组' if split=='validation' else '历史测试组'
    fig.suptitle(f'首100 ms都停更：之后继续 GP，还是切 EKF＋MLP？（{split_name}）',fontsize=17,y=.98)
    fig.text(.5,.925,'横轴是收到观测的时刻；曲线越低越准。0为离线参考触地，只用于评分对齐。',ha='center',fontsize=11)
    handles=[]
    for col,h in enumerate((.1,.5)):
        covers=None;maximum=[0.,0.]
        for candidate in (GP,STRICT):
            cells=sorted((r for r in curves if r['split']==split and r['candidate']==candidate and
                          r['horizon_s']==h and -.2-1e-12<=r['time_bin_s']<=.5+1e-12),key=lambda r:r['time_bin_s'])
            x=np.array([r['time_bin_s']*1000 for r in cells]);n=np.array([r['N'] for r in cells])
            if covers is None: covers=(x,n)
            else: np.testing.assert_array_equal(x,covers[0]);np.testing.assert_array_equal(n,covers[1])
            for row,field in ((0,'mean_m'),(1,'p95_frame_m')):
                y=np.array([r[field]*100 for r in cells])
                line,=axes[row,col].plot(x,y,color=colors[candidate],lw=2.3,label=NAMES[candidate])
                np.testing.assert_array_equal(line.get_ydata(),y)
                maximum[row]=max(maximum[row],float(np.max(y)))
                if row==0 and col==0:handles.append(line)
                axes[row,col].set_ylabel('距离误差均值（cm）' if row==0 else '帧距离误差 P95（cm）')
            axes[0,col].set_title(f'预测再过 {int(h*1000)} ms 的球位置',fontsize=13)
        for row in (0,1):
            axes[row,col].set_ylim(0,maximum[row]*1.12)
            assert axes[row,col].get_ylim()[1]>=maximum[row]
        axes[2,col].step(covers[0],covers[1],where='mid',color='#65717D',lw=1.8)
        axes[2,col].set_ylim(0,37 if split=='test' else 32);axes[2,col].set_ylabel('评分轨迹 N')
        axes[2,col].set_xlabel('相对参考触地时间（ms）')
        for ax in axes[:,col]:
            ax.set_xlim(-200,500);ax.set_xticks([-200,-100,0,100,200,300,400,500])
            ax.axvline(0,color='#74818A',ls=':',lw=1);ax.axvline(100,color='#9FA9B0',ls='--',lw=1)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.878),ncol=2,frameon=False)
    fig.text(.07,.048,'停更 = 不融合观测，仍推进预测并缓存已收到历史。两条曲线用相同评分点，不平滑、不截去峰。',fontsize=10,color='#606B74')
    fig.text(.07,.018,'100 ms虚线是参照；实际切换由已保存的因果 GP 高度事件决定。尾部N变少，不能据此判断突然变准。',fontsize=10,color='#606B74')
    paths=[]
    for suffix in ('png','pdf'):
        path=out/f'hybrid_mean_p95_{split}.{suffix}';fig.savefig(path,dpi=180,bbox_inches='tight');paths.append(str(path))
    plt.close(fig);return paths


def run(result,out,workers):
    started=perf_counter();out.mkdir(parents=True,exist_ok=True)
    legacy_hashes=checkpoint_hashes(LEGACY_RUN)
    protected={str(result/n):digest(result/n) for n in ('predictions.csv','legacy_predictions.csv','comparison_summary.json')}
    calibration=ROOT/'artifacts/new_ubuntu/tennis_calibration'
    input_paths=list((calibration/'clean').rglob('*.csv'))+[
        calibration/'split.json',calibration/'clean/annotations.json',
        calibration/'clean/summary.json',calibration/'bounce/summary.json']
    protected.update({str(path):digest(path) for path in input_paths})
    summary,events,cases,legacy=load_sources(result)
    for p,v in summary['checkpoint_hashes_unchanged'].items():assert digest(Path(p))==v;protected[p]=v
    traces=load_dataset(calibration)
    selected=[t for t in traces if t['id'] in cases]
    tasks=[(t,events[t['id']],cases[t['id']]) for t in selected]
    rows=[];diagnostics=[]
    for identity,sources in cases.items():
        event=events[identity]
        for key,source in sources.items():
            assert key in legacy and legacy[key]['target_xyz']==source['target_xyz'] and legacy[key]['valid']==source['valid']
            rows.append({**source,'candidate':GP})
            rows.append({**legacy[key],'candidate':LEGACY})
            switched=source['capture_t_s']>=event['resume_t_s']-1e-12
            rows.append({**(legacy[key] if switched else source),'candidate':BACKGROUND,
                         'output_source':'continuous_background_legacy_mlp' if switched else 'gp_hold100'})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(replay_strict,task) for task in tasks]
        for index,future in enumerate(as_completed(futures),start=1):
            rr,diag=future.result();rows.extend(rr);diagnostics.append(diag)
            print(f'HYBRID_REPLAY {index}/{len(tasks)} {diag["split"]}',flush=True)
    metrics,curves,ranking,coverage=aggregate(rows)
    # Reproduce immutable original phase RMSE, not only the new mean metric.
    reference={(r['split'],r['candidate'],r['phase'],r['horizon_s']):r for r in summary['phase_metrics']}
    checked=0
    for cell in metrics:
        if cell['candidate'] not in (GP,LEGACY):continue
        name='gp:hold100' if cell['candidate']==GP else 'legacy_ekf6_mlp'
        ref=reference[(cell['split'],name,cell['capture_phase'],cell['horizon_s'])]
        np.testing.assert_allclose(cell['rmse_m'],ref['rmse_m'],rtol=1e-12,atol=1e-12)
        assert cell['N']==ref['successful_trajectories'];checked+=1
    assert checked==48
    original_ranking={(r['split'],r['candidate']):r for r in summary['candidate_ranking']}
    for entry in ranking:
        if entry['candidate'] in (GP,LEGACY):
            name='gp:hold100' if entry['candidate']==GP else 'legacy_ekf6_mlp'
            np.testing.assert_allclose(entry['balanced_future_rmse_m'],
                original_ranking[(entry['split'],name)]['balanced_future_rmse_m'],rtol=1e-12,atol=1e-12)
    assert legacy_hashes==checkpoint_hashes(LEGACY_RUN)
    assert all(digest(Path(p))==v for p,v in protected.items())
    rows.sort(key=lambda r:(r['split'],r['trajectory_id'],r['capture_t_s'],r['horizon_s'],r['candidate']))
    write_csv(out/'hybrid_predictions.csv',rows)
    files=render(curves,out,'validation')+render(curves,out,'test')
    validation=[r for r in ranking if r['split']=='validation' and r['candidate'] in (GP,STRICT)]
    chosen=min(validation,key=lambda r:r['balanced_future_rmse_m'])['candidate']
    report={'schema':'GP_LEGACY_HOLD_COMPARISON/1','created_utc':datetime.now(timezone.utc).isoformat(),
            'hostname':platform.node(),'python_prefix':sys.prefix,'elapsed_s':perf_counter()-started,
            'source_run':str(result),'source_hashes_unchanged':protected,'legacy_checkpoint_unchanged':legacy_hashes,
            'master_clock':'immutable causal GP height discovery/resume schedule; event hidden until original discovery',
            'strict_hold':'both observers stop assimilation; physics/covariance propagate; received history retained; no old-frame replay',
            'state_transfer':'none; legacy maintains its own independent 6D state from first frame',
            'background_candidate':'legacy always assimilates; only displayed output switches; does not meet strict hold',
            'mean_definition':'mean of Euclidean distances within each trajectory, then equal mean over trajectories',
            'p95_definition':'inverse weighted frame ECDF: each trajectory has equal total mass',
            'bin_s':.02,'horizons_s':HORIZONS,'common_success_keys_all_four_candidates':True,
            'verified_original_phase_cells':checked,'verified_original_rankings':4,
            'master_resume_and_per_anchor_hold_checked':True,'validation_selected_strict_candidate':chosen,
            'ranking':ranking,'phase_metrics':metrics,'curves':curves,'coverage_reasons':coverage,
            'strict_replay_diagnostics':diagnostics,'files':files,
            'limitations':['Historical exposed test is regression, not new blind data.',
                           'Strict hold changes old MLP feature distribution; unchanged weights were evaluated.',
                           'Reference is same-device retained measurement, not independent ground truth.',
                           'Saved GP causal control schedule is replayed; no live hardware deployment or timing benchmark.',
                           'No global optimality claim; only these two strictly compliant candidates compared.']}
    (out/'hybrid_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'files':files,'ranking':ranking,'validation_selected_strict_candidate':chosen,
                      'verified_original_phase_cells':checked},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,default=ROOT/'artifacts/tennis_height_policy/20261010_height_hold100_v2')
    parser.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_hybrid/20261013_strict_hold')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Evaluate on Ubuntu in project .venv-tennis')
    if args.workers<1:raise ValueError('workers must be positive')
    run(args.result,args.out,args.workers)


if __name__=='__main__':main()
