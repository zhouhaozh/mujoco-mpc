"""Read-only Ubuntu diagnosis of saved forecast residual distributions.

Raw observations, labels, predictions and checkpoints remain unchanged. Each
trajectory has equal mass in each capture-phase/horizon empirical distribution.
This is descriptive analysis of historical data, not new model selection.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
from scipy.stats import wasserstein_distance

from .gp_analysis_plots import _style
from .measurement_noise_probe import run_noise_probe

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ('fixed:continuous', 'ridge1:continuous', 'ridge2:continuous',
              'gp:continuous', 'gp:hold100', 'legacy_ekf6', 'legacy_ekf6_mlp')
PHASES = ('precontact', 'first100ms', 'recovery100_200ms', 'late200_500ms')
PHASE_NAMES = ('触地前', '触地后 0–100 ms', '触地后 100–200 ms', '触地后 200–500 ms')
DISPLAY = (('gp:continuous', 'GP：持续修正', '#25699D'),
           ('gp:hold100', 'GP：停更100 ms后恢复', '#7148A1'),
           ('legacy_ekf6_mlp', '原 EKF＋MLP', '#A45136'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def capture_phase(tau):
    if tau < 0: return PHASES[0]
    if tau < .1: return PHASES[1]
    if tau < .2: return PHASES[2]
    return PHASES[3] if tau <= .5 + 1e-12 else None


def quantile(x, w, q):
    """Left-continuous inverse weighted ECDF; no interpolation convention."""
    order = np.argsort(x, kind='stable')
    xx, ww = np.asarray(x)[order], np.asarray(w)[order]
    cdf = np.cumsum(ww) / np.sum(ww)
    return float(xx[min(int(np.searchsorted(cdf, q, side='left')), len(xx)-1)])


def pack(groups):
    arrays = [np.asarray([row['residual'] for row in rows], float)
              for rows in groups.values() if rows]
    if not arrays: return None
    residual = np.concatenate(arrays)
    weights = np.concatenate([np.full(len(a), 1. / len(arrays) / len(a)) for a in arrays])
    distance = np.linalg.norm(residual, axis=1)
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12)
    return arrays, residual, distance, weights


def summarize(groups):
    data = pack(groups)
    if data is None: return None
    arrays, r, d, w = data
    mean = np.sum(r*w[:, None], axis=0)
    centered = r-mean
    covariance = (centered*w[:, None]).T @ centered
    axes = []
    for axis in range(3):
        x = r[:, axis]; m = quantile(x, w, .5)
        var = float(covariance[axis, axis])
        std = np.sqrt(var)
        axes.append({'axis': 'XYZ'[axis], 'mean_m': float(mean[axis]),
                     'median_m': m, 'p05_m': quantile(x, w, .05),
                     'p95_m': quantile(x, w, .95),
                     'mad_m': quantile(np.abs(x-m), w, .5), 'std_m': float(std),
                     'excess_kurtosis_descriptive': float(np.sum(w*(x-mean[axis])**4)/var**2-3) if var > 1e-18 else None,
                     'mass_outside_mean_3std': float(w[np.abs(x-mean[axis]) > 3*std].sum()) if std > 1e-9 else None})
    rms = np.asarray([np.sqrt(np.mean(np.sum(a*a, axis=1))) for a in arrays])
    maxima = np.asarray([np.linalg.norm(a, axis=1).max() for a in arrays])
    mse = float(np.sum(w*d*d))
    np.testing.assert_allclose(mse, np.mean(rms*rms), atol=1e-12)
    # Fraction of total squared error in the largest 5% of weighted mass.
    order = np.argsort(d)[::-1]; remaining = .05; tail = 0.
    for index in order:
        mass = min(remaining, w[index]); tail += mass*d[index]**2; remaining -= mass
        if remaining <= 1e-15: break
    return {'trajectories': len(arrays), 'frames': len(r), 'rmse_m': float(np.sqrt(mse)),
            'axis_signed_residual': axes, 'covariance_m2': covariance.tolist(),
            'distance_ecdf_quantiles_m': {str(q): quantile(d, w, q) for q in (.5, .9, .95, .99)},
            'distance_max_m': float(d.max()),
            'distance_exceedance_mass': {str(v): float(w[d > v].sum()) for v in (.1, .2, .5)},
            'largest_5percent_mass_squared_error_share': float(tail/mse) if mse else 0.,
            'trajectory_rms_quantiles_m': {str(q): float(np.quantile(rms, q)) for q in (.5, .95)},
            'trajectory_max_quantiles_m': {str(q): float(np.quantile(maxima, q)) for q in (.5, .95)}}


def read_rows(result):
    cells = defaultdict(lambda: defaultdict(list)); coverage = defaultdict(Counter)
    scheduled = defaultdict(set); labeled = defaultdict(set); counts = Counter()
    paired = defaultdict(dict); labels = {}
    for filename in ('predictions.csv', 'legacy_predictions.csv'):
        with (result/filename).open(newline='') as stream:
            for row in csv.DictReader(stream):
                counts['read_rows'] += 1
                candidate = row['candidate']
                if candidate not in CANDIDATES: continue
                if row.get('twenty_ms_anchor', 'True').lower() not in ('true', '1'): continue
                horizon = float(row['horizon_s']); tau = float(row['relative_capture_s'])
                band = capture_phase(tau)
                if horizon <= 0 or band is None or float(row['elapsed_capture_s']) < .1-1e-12: continue
                key = (row['split'], candidate, band, horizon); identity = row['trajectory_id']
                scheduled[key].add(identity); coverage[key]['scheduled_frames'] += 1
                if row['valid'].lower() not in ('true', '1'):
                    coverage[key]['missing_label:'+row['reason']] += 1
                    continue
                labeled[key].add(identity); coverage[key]['labeled_frames'] += 1
                if row['prediction_reason'] or not row['predicted_xyz']:
                    coverage[key]['prediction_failure:'+row['prediction_reason']] += 1
                    continue
                p = np.asarray(json.loads(row['predicted_xyz']), float)
                y = np.asarray(json.loads(row['target_xyz']), float)
                assert p.shape == y.shape == (3,) and np.isfinite(p).all() and np.isfinite(y).all()
                capture = float(row['capture_t_s']); target = float(row['target_t_s'])
                assert abs(target-capture-horizon) <= 1e-12
                pair_key = (identity, capture, horizon, target)
                if pair_key in labels: np.testing.assert_array_equal(labels[pair_key], y)
                else: labels[pair_key] = y
                assert pair_key not in paired[candidate]
                item = {'residual': p-y, 'relation': row['phase'], 'key': pair_key,
                        'split': row['split'], 'band': band, 'horizon': horizon}
                cells[key][identity].append(item); paired[candidate][pair_key] = item
                coverage[key]['scored_frames'] += 1; counts['selected_scored_rows'] += 1
    return cells, coverage, scheduled, labeled, paired, counts


def pair_comparison(paired):
    comparisons = []
    pairs = [('gp:continuous', 'gp:hold100'), ('gp:hold100', 'legacy_ekf6_mlp'),
             ('gp:continuous', 'fixed:continuous'), ('gp:continuous', 'ridge2:continuous'),
             ('legacy_ekf6', 'legacy_ekf6_mlp')]
    for first, second in pairs:
        shared = paired[first].keys() & paired[second].keys()
        groups = defaultdict(lambda: (defaultdict(list), defaultdict(list)))
        for key in shared:
            a, b = paired[first][key], paired[second][key]
            assert (a['split'], a['band'], a['horizon']) == (b['split'], b['band'], b['horizon'])
            x, y = groups[(a['split'], a['band'], a['horizon'])]
            x[key[0]].append(a); y[key[0]].append(b)
        for (split, band, h), (a, b) in sorted(groups.items()):
            a = dict(sorted(a.items())); b = dict(sorted(b.items()))
            assert list(a) == list(b)
            pa, pb = pack(a), pack(b)
            ra = np.array([np.sqrt(np.mean([np.sum(r['residual']**2) for r in rows])) for rows in a.values()])
            rb = np.array([np.sqrt(np.mean([np.sum(r['residual']**2) for r in rows])) for rows in b.values()])
            rho = float(np.corrcoef(ra, rb)[0, 1]) if len(ra)>2 and min(np.std(ra), np.std(rb))>1e-12 else None
            comparisons.append({'first': first, 'second': second, 'split': split, 'capture_phase': band,
                'horizon_s': h, 'paired_trajectories': len(a), 'paired_frames': len(pa[2]),
                'first_rmse_m': float(np.sqrt(np.mean(ra*ra))), 'second_rmse_m': float(np.sqrt(np.mean(rb*rb))),
                'distance_ecdf_wasserstein_m': float(wasserstein_distance(pa[2], pb[2], pa[3], pb[3])),
                'trajectory_rms_pearson_descriptive': rho,
                'second_lower_trajectory_rms_fraction': float(np.mean(rb < ra)),
                'both_success_only': True})
    return comparisons


def render(cells, out):
    font = _style(); paths = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    fig.subplots_adjust(top=.80, bottom=.13, hspace=.37, wspace=.15)
    fig.suptitle('未来 500 ms 预测：大误差出现得多不多？', fontsize=19, y=.97)
    fig.text(.5, .92, '35 条历史测试轨迹｜曲线越靠左，同等比例的预测误差越小', ha='center', fontsize=11)
    handles = []
    for index, (ax, band, title) in enumerate(zip(axes.ravel(), PHASES, PHASE_NAMES)):
        common_keys = None; trajectory_n = None
        for candidate, name, color in DISPLAY:
            groups = cells[('test', candidate, band, .5)]
            keys = {row['key'] for rows in groups.values() for row in rows}
            if common_keys is None: common_keys = keys; trajectory_n = len(groups)
            else: assert keys == common_keys
            arrays, r, d, w = pack(groups)
            order = np.argsort(d); x = d[order]*100; y = np.cumsum(w[order])*100
            line, = ax.step(x, y, where='post', color=color, lw=2, label=name)
            np.testing.assert_array_equal(line.get_xdata(), x)
            np.testing.assert_allclose(line.get_ydata()[-1], 100., atol=1e-9)
            if index == 0: handles.append(line)
        ax.set_xscale('log'); ax.set_ylim(0, 101); ax.set_yticks([0, 50, 90, 100])
        ax.xaxis.set_major_locator(FixedLocator([.1,1,10,100,1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.set_title(f'{title}（共同轨迹 N={trajectory_n}）', fontsize=13)
        ax.set_xlabel('三维距离误差（厘米，对数刻度）')
        ax.set_ylabel('误差不超过横轴值的比例（%）')
        ax.axhline(95, color='#828B92', ls=':', lw=1)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.875), ncol=3, frameon=False)
    fig.text(.08,.035,'同阶段三曲线共用评分点；轨迹等权。评分参考来自同设备测量，不能据此分离设备与模型误差。',fontsize=10,color='#59636D')
    for suffix in ('png','pdf'):
        path=out/('residual_distance_cdf.'+suffix);fig.savefig(path,dpi=180,bbox_inches='tight');paths.append(str(path))
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), sharey=True)
    fig.subplots_adjust(top=.76,bottom=.22,wspace=.20)
    fig.suptitle('推荐 GP 组合：误差是否偏向某个方向？',fontsize=17,y=1.01)
    fig.text(.5,.90,'点 = 平均偏差｜竖线 = 带符号误差 P05–P95｜正值表示预测坐标偏大',ha='center',fontsize=10)
    for ax,h in zip(axes,(.1,.5)):
        for j,(axis,color) in enumerate(zip('XYZ',('#25699D','#4B8D76','#B56048'))):
            values=[summarize(cells[('test','gp:hold100',band,h)])['axis_signed_residual'][j] for band in PHASES]
            x=np.arange(4)+(j-1)*.20
            mean=np.array([r['mean_m'] for r in values])*100
            low=np.array([r['p05_m'] for r in values])*100;high=np.array([r['p95_m'] for r in values])*100
            ax.vlines(x,low,high,color=color,lw=2);ax.scatter(x,mean,color=color,label=axis,s=30,zorder=3)
        ax.set_title(f'预测再过 {int(h*1000)} ms 的位置')
        ax.set_xticks(range(4),['触地前','0–100 ms','100–200 ms','200–500 ms'],rotation=15)
        ax.set_ylabel('有符号坐标误差（厘米）');ax.axhline(0,color='#727B84',lw=1)
    axes[0].legend(ncol=3,frameon=False,loc='upper left')
    fig.text(.08,.045,'轨迹等权分布；触地阶段按收到观测的时刻划分。不同预测时距与不同阶段分别统计。',fontsize=10,color='#59636D')
    for suffix in ('png','pdf'):
        path=out/('residual_axis_bias.'+suffix);fig.savefig(path,dpi=180,bbox_inches='tight');paths.append(str(path))
    plt.close(fig)
    return font, paths


def run(result, out, calibration):
    result, out = Path(result), Path(out);out.mkdir(parents=True,exist_ok=True)
    summary_path=result/'comparison_summary.json';summary=json.loads(summary_path.read_text())
    sources={result/'predictions.csv':None,result/'legacy_predictions.csv':None,summary_path:None}
    sources.update({Path(p):v for p,v in summary['checkpoint_hashes_unchanged'].items()})
    sources.update({calibration/'split.json':None,calibration/'clean/annotations.json':None})
    sources.update({p:None for p in (calibration/'clean').rglob('*.csv')})
    sources.update({p:None for p in (calibration/'bounce/summary.json',calibration/'clean/summary.json') if p.exists()})
    protected={str(p):digest(p) for p in sources}
    for p,v in sources.items():
        if v: assert protected[str(p)] == v
    cells,coverage,scheduled,labeled,paired,counts=read_rows(result)
    phase_reference={(r['split'],r['candidate'],r['phase'],r['horizon_s']):r for r in summary['phase_metrics']}
    records=[];verified=0
    for key in sorted(scheduled):
        split,candidate,band,h=key;metric=summarize(cells[key]);reference=phase_reference[key]
        np.testing.assert_allclose(metric['rmse_m'],reference['rmse_m'],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(metric['trajectory_rms_quantiles_m']['0.95'],reference['p95_m'],rtol=1e-12,atol=1e-12)
        assert metric['trajectories']==reference['successful_trajectories']==len(labeled[key])
        assert coverage[key]['labeled_frames']==reference['labeled_rows']
        assert not any(k.startswith('prediction_failure:') for k in coverage[key])
        verified+=1
        relations={}
        for relation in ('pre_bounce','cross_bounce','post_bounce'):
            subset={identity:[row for row in rows if row['relation']==relation] for identity,rows in cells[key].items()}
            value=summarize(subset)
            if value: relations[relation]=value
        records.append({'split':split,'candidate':candidate,'capture_phase':band,'horizon_s':h,
                        'requested_trajectories':len(scheduled[key]),'labeled_trajectories':len(labeled[key]),
                        'coverage':dict(coverage[key]),**metric,'target_relation_subsets':relations})
    font,paths=render(cells,out)
    noise=run_noise_probe(calibration)
    assert all(digest(Path(p))==v for p,v in protected.items())
    report={'schema':'TENNIS_RESIDUAL_DISTRIBUTION/1','created_utc':datetime.now(timezone.utc).isoformat(),
            'hostname':platform.node(),'python_prefix':sys.prefix,'source_run':str(result),
            'selection':'common20ms anchors; elapsed>=100ms; capture<=reference contact+500ms; H100/200/500 only',
            'signed_error_definition':'predicted_xyz - retained_measurement_reference_xyz',
            'ecdf_weighting':'each trajectory weight 1/N; its frames each weight 1/(N*n_j), separately per phase/horizon',
            'quantile_convention':'inverse step ECDF for frames; numpy linear quantile for trajectory RMS/max',
            'p95_labels_distinct':True,'cdf_display_success_keys_identical':True,'historical_test_not_new_blind_test':True,
            'original_sources_unchanged':protected,'counts':dict(counts),'verified_original_phase_cells':verified,
            'font':font,'files':paths,'cells':records,'paired_comparisons':pair_comparison(paired),
            'measurement_noise_proxy':noise,
            'limitations':['Reference is same-device retained measurements and legal interpolation, not independent ground truth.',
                           'Within-trajectory windows are correlated; no iid normality test or significance claim.',
                           'Each validation/test split has only two inferred collection groups.',
                           'Descriptive tails and correlations cannot uniquely identify sensor versus observer/model causes.',
                           'No smoothing, retraining, selection, or default model/filter modification.']}
    (out/'residual_distribution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return {'files':paths,'verified_original_phase_cells':verified,'source_hashes_unchanged':True,'counts':dict(counts)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,default=ROOT/'artifacts/tennis_height_policy/20261010_height_hold100_v2')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_residual/20261013_distribution')
    p.add_argument('--calibration-dir',type=Path,default=ROOT/'artifacts/new_ubuntu/tennis_calibration')
    args=p.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Run on Ubuntu in project .venv-tennis')
    print(json.dumps(run(args.result,args.out,args.calibration_dir),ensure_ascii=False))


if __name__=='__main__':main()
