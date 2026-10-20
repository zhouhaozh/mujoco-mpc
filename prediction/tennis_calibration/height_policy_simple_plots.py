"""Simplified display of immutable historical height-policy metrics on Ubuntu."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .gp_analysis_plots import _style

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES=(('gp:continuous','一直用新观测修正','#25699D','-'),
            ('gp:hold100','触地后停更100 ms，再修正（推荐）','#7148A1','-'),
            ('gp:init_hold100','初始化后停更，到触地+100 ms再修正','#9B3E62','--'))
HORIZONS=(.1,.5)
WINDOW=(-.2,.4)


def fingerprint(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def render(result,out):
    manifest_path=result/'curve_files_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    protected={str(manifest_path):fingerprint(manifest_path)}
    cells={}
    for candidate,_,_,_ in CANDIDATES:
        spec=next(s for s in manifest['files'] if s['split']=='test' and s['candidate']==candidate)
        source=result/spec['path']
        if not source.resolve().is_relative_to(result.resolve()):raise ValueError('invalid metric path')
        digest=fingerprint(source)
        assert digest==spec['sha256']
        data=json.loads(source.read_text())
        assert data['candidate']==candidate and data['split']=='test' and len(data['cells'])==spec['cells']
        protected[str(source)]=digest;cells[candidate]=data['cells']
    metadata_path=result/'comparison_summary.json'
    summary=json.loads(metadata_path.read_text())
    protected[str(metadata_path)]=fingerprint(metadata_path)
    for path,digest in summary['checkpoint_hashes_unchanged'].items():
        assert fingerprint(Path(path))==digest
        protected[path]=digest
    font=_style()
    plt.rcParams.update({'font.size':12,'axes.labelsize':12,'axes.titlesize':15,'legend.fontsize':11})
    fig,axes=plt.subplots(2,2,figsize=(14,8.4),sharex='col',
        gridspec_kw={'height_ratios':[3.4,1],'hspace':.12,'wspace':.20})
    fig.subplots_adjust(left=.085,right=.97,top=.78,bottom=.18)
    fig.suptitle('触地附近：三种更新策略，谁预测得更准？',fontsize=20,y=.975)
    fig.text(.5,.925,'同一 GP 碰撞模型｜35 条历史测试轨迹｜线越低，未来位置预测越准',ha='center',fontsize=12,color='#58646E')
    handles=[];panels=[]
    for col,h in enumerate(HORIZONS):
        ax,nax=axes[:,col];counts=None;times=None;maximum=0.;series=[]
        for candidate,label,color,style in CANDIDATES:
            rows=sorted((r for r in cells[candidate] if r['axis']=='contact20'
                and r['horizon_s']==h and WINDOW[0]-1e-10<=r['time_bin_s']<=WINDOW[1]+1e-10),key=lambda r:r['time_bin_s'])
            x=np.asarray([r['time_bin_s']*1000 for r in rows])
            y=np.asarray([r['rmse_m']*100 if r['rmse_m'] is not None else np.nan for r in rows])
            n=np.asarray([r['successful_trajectories'] for r in rows])
            coverage=np.asarray([(r['requested_trajectories'],r['labeled_trajectories'],r['successful_trajectories']) for r in rows])
            if counts is None:counts=coverage;times=x
            else:
                np.testing.assert_array_equal(x,times)
                np.testing.assert_array_equal(coverage,counts)
            line,=ax.plot(x,y,color=color,ls=style,lw=2.8 if candidate=='gp:hold100' else 2.1,label=label)
            np.testing.assert_array_equal(line.get_xdata(),x)
            np.testing.assert_array_equal(line.get_ydata(),y)
            if col==0:handles.append(line)
            maximum=max(maximum,float(np.nanmax(y)))
            series.append({'candidate':candidate,'points':[{'capture_relative_ms':float(t),'rmse_cm':float(v) if np.isfinite(v) else None,'scored_N':int(k)} for t,v,k in zip(x,y,n)]})
        ax.set_title(f'预测再过 {int(h*1000)} ms 的球位置',pad=14)
        ax.set_ylabel('三维位置综合误差（厘米）')
        ax.set_ylim(0,maximum*1.12)
        assert ax.get_ylim()[1]>=maximum
        ax.grid(alpha=.18)
        nax.step(times,counts[:,2],where='mid',color='#606D78',lw=2)
        nax.set_ylim(0,38);nax.set_yticks([0,10,20,35])
        nax.set_ylabel('评分轨迹数 N')
        nax.set_xlabel('相对参考触地的时间（毫秒）：负数在触地前，正数在触地后',fontsize=10)
        nax.grid(alpha=.18)
        for axis in (ax,nax):
            axis.set_xlim(WINDOW[0]*1000,WINDOW[1]*1000)
            axis.set_xticks([-200,-100,0,100,200,300,400])
            axis.axvline(0,color='#687581',lw=1.25,ls=':')
            axis.axvline(100,color='#A5ADB5',lw=1.0,ls='--')
        ax.text(.06,.965,'触地前',transform=ax.transAxes,va='top',color='#606D78',fontsize=10)
        ax.text(.72,.965,'触地后',transform=ax.transAxes,va='top',color='#606D78',fontsize=10)
        panels.append({'horizon_s':h,'ylim_cm':list(ax.get_ylim()),'all_peaks_in_window_visible':True,'series':series})
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.875),ncol=3,frameon=False,
               handlelength=2.5,columnspacing=1.3)
    fig.text(.085,.098,'0 = 参考触地；100 ms 虚线只是时间参照。实际停更/恢复由每条球的估计高度触地决定。',fontsize=11,color='#58646E')
    fig.text(.085,.055,'下排 N 是可评分轨迹条数。记录结束后 N 会减少，尾部误差变低不等于预测突然变准。',fontsize=11,color='#58646E')
    fig.text(.085,.018,'仅聚焦触地前 200 ms 至触地后 400 ms；使用原 20 ms 分桶数据，不平滑、不重算误差、不修改模型。',fontsize=10,color='#58646E')
    out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for suffix in ('png','pdf'):
        path=out/('height_policy_simple_comparison.'+suffix)
        fig.savefig(path,dpi=190,bbox_inches='tight',facecolor='white');paths.append(str(path))
    plt.close(fig)
    assert all(fingerprint(Path(p))==v for p,v in protected.items())
    report={'schema':'HEIGHT_POLICY_SIMPLE_DISPLAY/1','created_utc':datetime.now(timezone.utc).isoformat(),
        'hostname':platform.node(),'python_prefix':sys.prefix,'font':font,'historical_data_directory':str(result),
        'window_ms':[-200,400],'candidates':[c[0] for c in CANDIDATES],
        'same_GP_for_all_curves':True,'only_future_horizons_s':HORIZONS,
        'metric_definition':'original equal-trajectory three-dimensional position RMSE',
        'original_metrics_and_weights_unchanged':protected,'display_data_exactly_matches_source':True,
        'same_capture_clock_and_coverage':True,'smoothing':False,'cropped_y_peak':False,
        'reference_100ms_marker_is_actual_resume':False,'panels':panels,'files':paths}
    (out/'simple_display_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,default=ROOT/'artifacts/tennis_height_policy/20261010_height_hold100_v2')
    p.add_argument('--out',type=Path,default=ROOT/'artifacts/tennis_height_policy/20261013_simple_reading')
    args=p.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Render on Ubuntu in project .venv-tennis')
    print(json.dumps({'files':render(args.result,args.out)},ensure_ascii=False))


if __name__=='__main__':main()
