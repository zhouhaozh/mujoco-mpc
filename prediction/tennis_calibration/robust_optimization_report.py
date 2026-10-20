"""Selection and scientific figures for isolated causal observer experiments.

Execute statistics/rendering in the Ubuntu project environment. Validation
selection is saved before exposing historical test or group-fold results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .gp_analysis_plots import _style
from .identified_run import dump

LABELS = {
    'baseline': '原 GP＋100 ms 停更',
    'huber_only': '仅 Huber 降权',
    'adaptive_huber': '方向／质量自适应＋Huber',
    'protected_only': '仅恢复速度保护',
    'adaptive_protected': '自适应＋Huber＋速度保护',
    'smooth_only': '仅因果平滑',
    'protected_smooth': '速度保护＋因果平滑',
    'adaptive_protected_smooth': '自适应＋速度保护＋平滑',
}
COLORS = {'baseline': '#7A5797', 'protected_only': '#D98431', 'adaptive_huber':'#127FAB',
          'huber_only': '#D98431', 'adaptive_huber': '#D05C59',
          'adaptive_protected': '#659237', 'smooth_only': '#438F85',
          'protected_smooth': '#222E46', 'adaptive_protected_smooth': '#99842F'}
STAGE2 = ('baseline', 'huber_only', 'adaptive_huber', 'protected_only', 'adaptive_protected')
SELECTION_RULE = {
    'phase2_parent_candidates': STAGE2,
    'objective': 'minimum validation trajectory/stage balanced future RMSE',
    'constraints': {
        'all_labeled_predictions_succeed': True,
        'first200ms_h500_frame_p95_max_baseline_ratio': 1.10,
        'dense_resume50ms_h500_trajectory_max_p95_max_baseline_ratio': 1.10,
        'contact_absolute_error_p95_max_extra_s': .005,
        'discovery_absolute_error_p95_max_extra_s': .005,
        'no_additional_missed_height_events': True,
    },
    'residual_kinds': ['ridge', 'mlp'],
    'residual_parents': 'baseline and validation-selected phase2 parent',
    'hold_s': .100,
    'new_blind_test': False,
}


def lookup(summary, key, candidate, split='validation'):
    found = [r for r in summary[key] if r['candidate'] == candidate
             and r.get('split', split) == split]
    if len(found) != 1:
        raise ValueError(f'expected one {key} entry for {split}/{candidate}: {len(found)}')
    return found[0]


def label(name):
    if name in LABELS:
        return LABELS[name]
    if '__residual_' in name:
        parent, head=name.split('__residual_',1)
        return LABELS.get(parent,parent)+'＋'+head.upper()
    for marker in ('+', ':'):
        if marker in name:
            parent, head = name.split(marker, 1)
            return LABELS.get(parent, parent) + '＋' + head.upper()
    return name


def choose(summary, candidates, split='validation'):
    """Fixed tail/detection constraints, followed by validation RMSE only."""
    base_early=lookup(summary,'first200ms_h500_metrics','baseline',split)
    base_dense=lookup(summary,'dense_resume_first50ms_metrics','baseline',split)
    base_time=lookup(summary,'event_timing_metrics','baseline',split)
    decisions=[]
    for name in candidates:
        if not any(r['candidate']==name and r.get('split',split)==split for r in summary['ranking']):
            decisions.append({'candidate':name,'eligible':False,'reasons':['no valid ranking'],
                              'balanced_future_rmse_m':None})
            continue
        rank=lookup(summary,'ranking',name,split)
        coverage=lookup(summary,'absolute_labeled_coverage',name,split)
        early=lookup(summary,'first200ms_h500_metrics',name,split)
        dense=lookup(summary,'dense_resume_first50ms_metrics',name,split)
        timing=lookup(summary,'event_timing_metrics',name,split)
        reasons=[]
        if coverage.get('failed_labeled',0) or coverage['tracking_failures'] or not coverage['equal_to_reference']:
            reasons.append('incomplete matched label coverage')
        if early['p95_frame_m'] is None or early['p95_frame_m'] > base_early['p95_frame_m']*1.1+1e-12:
            reasons.append('first200ms H500 frame P95 exceeds110% baseline')
        if dense['trajectory_window_max_p95_m'] is None or dense['trajectory_window_max_p95_m'] > base_dense['trajectory_window_max_p95_m']*1.1+1e-12:
            reasons.append('dense resume50ms trajectory peak P95 exceeds110% baseline')
        for key in ('contact_minus_reference_s','discovery_minus_reference_s'):
            if timing[key]['missing'] > base_time[key]['missing']:
                reasons.append('additional missing '+key)
            if timing[key]['abs_p95_ms'] is None or timing[key]['abs_p95_ms'] > base_time[key]['abs_p95_ms']+5+1e-12:
                reasons.append('absolute timing P95 adds more than5ms: '+key)
        decisions.append({'candidate':name,'eligible':not reasons,'reasons':reasons,
                          'balanced_future_rmse_m':rank['balanced_future_rmse_m']})
    good=[r for r in decisions if r['eligible']]
    return {'split':split,'selected':min(good,key=lambda r:(r['balanced_future_rmse_m'],r['candidate']))['candidate'],
            'decisions':decisions, 'rule':SELECTION_RULE,
            'selected_utc':datetime.now(timezone.utc).isoformat()}


def render_curves(summary, out, candidates, split='oof', suffix='comparison'):
    font = _style()
    curves = summary['curves']
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1]})
    metrics = [('mean_m', '距离均值（cm）'), ('p95_frame_m', '帧 P95（cm）'), ('N', '轨迹数 N')]
    for j, horizon in enumerate((.1, .5)):
        for name in candidates:
            cells = sorted([r for r in curves if r['split'] == split and r['candidate'] == name
                            and abs(r['horizon_s'] - horizon) < 1e-12], key=lambda r:r['time_bin_s'])
            color = COLORS.get(name.split('__residual_')[0])
            for i, (key, ylabel) in enumerate(metrics):
                x = [r['time_bin_s']*1000 for r in cells]
                y = [r[key]*(1 if key == 'N' else 100) for r in cells]
                axes[i,j].plot(x, y, label=label(name), color=color, lw=1.65)
                axes[i,j].axvline(0, color='#444444', lw=.8)
                axes[i,j].axvspan(0, 100, color='#888888', alpha=.07)
                axes[i,j].set_ylabel(ylabel)
        for i, _ in enumerate(metrics):
            ymax=max((float(np.max(line.get_ydata())) for line in axes[i,j].lines
                      if len(line.get_ydata()) and np.isfinite(line.get_ydata()).all()),default=1.)
            axes[i,j].set_ylim(0,max(ymax*1.07,1.))
            assert all(np.nanmax(line.get_ydata()) <= axes[i,j].get_ylim()[1]
                       for line in axes[i,j].lines if len(line.get_ydata()))
        axes[0,j].set_title(f'始终向未来 {horizon*1000:.0f} ms 预测')
        axes[-1,j].set_xlabel('当前观测时刻 − 离线参考触地（ms）')
    handles, names = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .005))
    split_label={'oof':'五折留出','validation':'原验证组','test':'历史测试组'}.get(split,split)
    fig.suptitle(f'{split_label}：20 ms 共同锚点的均值与 P95（曲线未经平滑）', y=.99)
    fig.text(.5, .09, '灰区标记参考触地后 100 ms，实际恢复按模型高度触发；逐原始帧的极端峰另图展示。', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.965))
    paths=[]
    for extension in ('png','pdf'):
        path = out/f'{suffix}_{split}.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight'); paths.append(str(path))
    plt.close(fig)
    return {'font':font, 'files':paths, 'candidates':candidates, 'split':split,
            'smoothing':False, 'truncated_plotted_values':False,
            'sampling':'20ms anchors; narrow raw-frame recovery spikes may not appear here'}


def render_spike(run, out):
    from .group_fold_evaluation import read_json_csv
    from .height_policy_run import row_error
    font=_style()
    identity='all_trajectories/trajectory_024_gid470591_471794.txt'
    names=['baseline','adaptive_huber','adaptive_protected']
    rows=[r for r in read_json_csv(run/'predictions.csv.gz') if r['trajectory_id']==identity
          and r['candidate'] in names and r['horizon_s']==.5
          and .075 <= r['relative_capture_s'] <= .215 and row_error(r) is not None]
    fig,ax=plt.subplots(figsize=(11,5))
    maximum=0.; details=[]
    for name in names:
        values=sorted([r for r in rows if r['candidate']==name],key=lambda r:r['capture_t_s'])
        x=np.asarray([r['relative_capture_s']*1000 for r in values]); y=np.asarray([row_error(r)*100 for r in values])
        ax.plot(x,y,'o-',label=label(name),color=COLORS.get(name),ms=3,lw=1.7)
        maximum=max(maximum,float(y.max()))
        peak=int(np.argmax(y))
        ax.annotate(f'{y[peak]:.1f} cm',xy=(x[peak],y[peak]),xytext=(x[peak]+14,y[peak]+5),
                    arrowprops={'arrowstyle':'-','color':COLORS.get(name)},color=COLORS.get(name))
        details.append({'candidate':name,'peak_cm':float(y.max()),'reference_relative_ms':float(x[peak])})
    ax.axvspan(100,150,color='#AAAAAA',alpha=.1,label='恢复初期：逐原始帧诊断')
    ax.set(xlabel='当前观测时刻 − 参考触地（ms）',ylabel='未来 500 ms 预测距离误差（cm）',
           ylim=(0,maximum*1.15),title='极端样本：P95 变好，仍可能留下单条尖峰；速度保护限制这次过修正')
    ax.legend(frameon=False,loc='upper right')
    fig.text(.5,.01,'原验证组 trajectory_024_gid470591_471794；恢复首 50 ms 逐帧，其余为 20 ms 锚点；未平滑曲线。',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    paths=[]
    for ext in ('png','pdf'):
        path=out/f'extreme_recovery_case.{ext}'; fig.savefig(path,dpi=180,bbox_inches='tight'); paths.append(str(path))
    plt.close(fig)
    return {'font':font,'files':paths,'peak_details':details,'trajectory_id':identity,'smoothing':False}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--write-protocol', type=Path)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--split',default='oof')
    parser.add_argument('--candidates',nargs='+',default=['baseline','protected_only','smooth_only'])
    parser.add_argument('--suffix',default='comparison')
    parser.add_argument('--select',choices=('stage2','all'))
    parser.add_argument('--spike-run',type=Path)
    args=parser.parse_args()
    if args.write_protocol:
        if args.write_protocol.exists(): raise ValueError('preserve frozen protocol')
        dump(args.write_protocol, {'created_utc':datetime.now(timezone.utc).isoformat(), **SELECTION_RULE})
    if args.spike_run:
        args.out.mkdir(parents=True,exist_ok=True)
        dump(args.out/'render_extreme_case.json',render_spike(args.spike_run,args.out))
    if args.summary:
        args.out.mkdir(parents=True,exist_ok=True)
        summary=json.loads(args.summary.read_text())
        if args.select:
            names=STAGE2 if args.select=='stage2' else sorted({r['candidate'] for r in summary['ranking'] if r['split']=='validation'})
            choice=choose(summary,names)
            target=args.out/f'selection_{args.select}.json'
            if target.exists(): raise ValueError('preserve frozen selection')
            dump(target,choice); print(json.dumps(choice,ensure_ascii=False))
            return
        result=render_curves(summary,args.out,args.candidates,args.split,args.suffix)
        dump(args.out/f'render_{args.suffix}_{args.split}.json',result)


if __name__=='__main__':
    main()
