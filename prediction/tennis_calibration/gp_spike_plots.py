"""Read-only Ubuntu figure for the four-map early-contact online ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

from .gp_analysis_plots import _style

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('fixed', 'ridge1', 'ridge2', 'gp')
COLORS = {'fixed': '#616C76', 'ridge1': '#4C8D6C', 'ridge2': '#7A5797', 'gp': '#167DAB'}
LABELS = {'fixed': '固定物理映射', 'ridge1': '线性 Ridge 修正',
          'ridge2': '二次 Ridge 修正', 'gp': 'GP 修正'}
STYLES = {'fixed': ':', 'ridge1': '--', 'ridge2': '-.', 'gp': '-'}


def _finite(value):
    return np.nan if value is None else float(value)


def _curves(summary, field):
    curves = {}
    for model in MODELS:
        cells = sorted((cell for cell in summary[field] if cell['model'] == model and
                        abs(float(cell['horizon_s'])-.5) <= 1e-12 and
                        -1e-12 <= float(cell['capture_bin_s']) <= .08+1e-12),
                       key=lambda cell: cell['capture_bin_s'])
        if not cells:
            raise ValueError(f'missing early +500-ms curve: {field}/{model}')
        curves[model] = cells
    keys = [cell['capture_bin_s'] for cell in curves['gp']]
    counts = ('requested_trajectories', 'labeled_trajectories', 'successful_trajectories')
    for model in MODELS:
        if [cell['capture_bin_s'] for cell in curves[model]] != keys:
            raise ValueError('The four maps must have the same requested clock grid')
        if any(tuple(cell[name] for name in counts) != tuple(ref[name] for name in counts)
               for cell, ref in zip(curves[model], curves['gp'])):
            raise ValueError('Cannot show common coverage: the four-map cohort counts differ')
    return curves


def make_spike_comparison(summary, output):
    """Render supplied RMSE cells only; do not fit, replay, or rewrite metrics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font = _style()
    protocols = (
        ('twenty_ms_curves', '原协议：约 20 ms 捕获锚点 / 20 ms 时间桶', .02),
        ('dense_2p5ms_curves', '逐观测帧：2.5 ms 时间桶', .0025),
    )
    curves = [_curves(summary, field) for field, _, _ in protocols]
    finite_errors = [100*_finite(cell['rmse_m']) for group in curves
                     for model in MODELS for cell in group[model] if cell['rmse_m'] is not None]
    if not finite_errors:
        raise ValueError('No successful labeled predictions to display')
    upper = max(finite_errors)*1.18
    fig, axes = plt.subplots(3, 2, figsize=(14.8, 9.4), sharex='col', sharey='row',
                             gridspec_kw={'height_ratios': [4.3, 1.1, 1.1]})
    peaks = {}
    for column, (group, spec) in enumerate(zip(curves, protocols)):
        field, title, width = spec
        ax = axes[0, column]
        for model in MODELS:
            cells = group[model]
            x = np.asarray([cell['capture_bin_s']*1000 for cell in cells])
            y = np.asarray([100*_finite(cell['rmse_m']) for cell in cells])
            ax.plot(x, y, color=COLORS[model], ls=STYLES[model], lw=2.4 if model == 'gp' else 1.8,
                    marker='o', markersize=4.2 if width == .02 else 2.2, label=LABELS[model])
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('未来 500 ms 三维位置 RMSE（cm）')
        ax.set_ylim(0, upper)
        ax.set_xlim(0, 80)
        ax.legend(loc='upper right', fontsize=9)
        gp_cells = group['gp']
        gp_peak = max((cell for cell in gp_cells if cell['rmse_m'] is not None), key=lambda cell: cell['rmse_m'])
        peaks[field] = {'rmse_cm': 100*gp_peak['rmse_m'],
                        'capture_bin_ms': 1000*gp_peak['capture_bin_s'],
                        'successful_trajectories': gp_peak['successful_trajectories']}
        ax.annotate(f"GP 桶峰：{100*gp_peak['rmse_m']:.2f} cm\n中心 {1000*gp_peak['capture_bin_s']:g} ms；N = {gp_peak['successful_trajectories']}",
                    xy=(1000*gp_peak['capture_bin_s'], 100*gp_peak['rmse_m']),
                    xytext=(.025, .965), textcoords='axes fraction', va='top', color=COLORS['gp'],
                    fontsize=10, arrowprops={'arrowstyle': '->', 'color': COLORS['gp'], 'lw': 1.2},
                    bbox={'boxstyle': 'round,pad=.3', 'fc': 'white', 'ec': '#D6DDE1', 'alpha': .92})
        x = np.asarray([cell['capture_bin_s']*1000 for cell in gp_cells])
        requested = np.asarray([cell['requested_trajectories'] for cell in gp_cells], dtype=float)
        labeled = np.asarray([cell['labeled_trajectories'] for cell in gp_cells], dtype=float)
        successful = np.asarray([cell['successful_trajectories'] for cell in gp_cells], dtype=float)
        axes[1, column].step(x, requested, where='mid', color='#BAC1C6', lw=1.6, label='请求 N')
        axes[1, column].step(x, labeled, where='mid', color='#253441', lw=1.7, label='有参考标签 N')
        axes[1, column].step(x, successful, where='mid', color='#4C8D6C', lw=1.5, ls='--', label='成功预测 N')
        axes[1, column].set_ylabel('独立轨迹 N')
        axes[1, column].set_ylim(0, float(summary['independent_test_collisions'])+3)
        axes[1, column].legend(loc='lower right', ncol=3, fontsize=8)
        coverage = np.divide(100*labeled, requested, out=np.full_like(labeled, np.nan), where=requested > 0)
        axes[2, column].step(x, coverage, where='mid', color='#253441', lw=1.7, label='参考标签 N / 请求 N')
        axes[2, column].set_ylim(0, 104)
        axes[2, column].set_ylabel('标签覆盖（%）')
        axes[2, column].set_xlabel('观测捕获时间桶中心 − 参考首次触地时刻（ms）')
        axes[2, column].set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
        axes[2, column].text(.025,.12,f'最近中心分桶：±{width*500:g} ms',transform=axes[2,column].transAxes,
                            fontsize=9,color='#616C76')
    fig.suptitle('触地早期在线峰值对比：同一观测时刻向前预测 500 ms', fontsize=17, y=.975)
    fig.text(.035, .915, '35 条历史测试碰撞；相同全前缀状态/协方差、观测与 EKF，仅改变反弹映射及其雅可比；四模型的 N 在各桶完全相同。',
             fontsize=9, color='#616C76')
    fig.text(.025,.018,
        '每条轨迹在桶内先求三维位置误差均方，再对独立轨迹等权平均并开方；多帧不会增加该轨迹权重。目标绝对时间为实际 capture + 0.5 s。\n'
        '左、右是不同锚点/时间桶统计：20 ms 桶峰不等于密集 2.5 ms 桶峰。右图仍是离散观测桶峰，非连续时间真峰，也不是单条轨迹最大误差。\n'
        '标签不跨 12 ms 采样空洞或碰撞；失败与缺标签均保留在覆盖中。历史数据回归；不训练模型，不改变任何原始指标。',
        fontsize=8.5, color='#616C76', va='bottom')
    fig.subplots_adjust(top=.87, bottom=.16, left=.07, right=.98, hspace=.13, wspace=.13)
    fig.canvas.draw()
    # Independent display audit: all supplied curve cells and complete peaks
    # remain visible, using the specified four nonorange model colors.
    for column, group in enumerate(curves):
        axis = axes[0,column]
        assert len(axis.lines) == 4
        for line, model in zip(axis.lines, MODELS):
            expected_x = np.asarray([1000*cell['capture_bin_s'] for cell in group[model]])
            expected_y = np.asarray([100*_finite(cell['rmse_m']) for cell in group[model]])
            assert np.allclose(line.get_xdata(), expected_x, equal_nan=True)
            assert np.allclose(line.get_ydata(), expected_y, equal_nan=True)
            assert to_rgba(line.get_color()) == to_rgba(COLORS[model])
            finite = expected_y[np.isfinite(expected_y)]
            if len(finite):
                assert axis.get_ylim()[1] >= float(np.max(finite))
        assert np.allclose(axis.get_xlim(), axes[1,column].get_xlim())
        assert np.allclose(axis.get_xlim(), axes[2,column].get_xlim())
    stem='collision_map_early_peak_500ms'
    paths=[]
    for extension in ('png','pdf'):
        path=output/(stem+'.'+extension)
        fig.savefig(path,dpi=210,bbox_inches='tight',facecolor='white')
        paths.append(str(path))
    plt.close(fig)
    return {'paths':paths,'font':font,'GP_bucket_peaks':peaks,
            'display_audit':'all saved RMSE cells and full peaks visible; shared coverage verified; four nonorange colors'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'artifacts/tennis_gp_spike/20261010_collision_map_ablation/summary.json')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if platform.system() != 'Linux' or Path(sys.prefix).resolve() != (ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Formal numerical display and plotting must run in Ubuntu .venv-tennis')
    summary=json.loads(args.summary.read_text())
    protected={str(args.summary):hashlib.sha256(args.summary.read_bytes()).hexdigest()}
    for name,expected in summary['checkpoint_hashes_unchanged'].items():
        path=Path(name)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Frozen checkpoint differs from the completed ablation: '+name)
        protected[name]=actual
    result=make_spike_comparison(summary,args.out or args.summary.parent/'plots')
    assert protected == {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in protected}
    print('SPIKE_PLOTS_OK '+json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
