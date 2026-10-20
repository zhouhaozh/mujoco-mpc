"""Scientific diagnostics for raw observations, automatic labels and held-outs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def make_plots(root):
    root=Path(root);out=root/'plots';out.mkdir(exist_ok=True)
    rows=json.loads((root/'clean/annotations.json').read_text())
    split=json.loads((root/'split.json').read_text())
    group={identity:name for name in ('train','validation','test') for identity in split[name]}
    # Every trajectory is represented for human label audit, not only good fits.
    for page in range((len(rows)+15)//16):
        fig,axes=plt.subplots(4,4,figsize=(16,12),constrained_layout=True)
        for ax,row in zip(axes.ravel(),rows[page*16:(page+1)*16]):
            with (root/'clean'/row['cleaned_file']).open() as stream:data=list(csv.DictReader(stream))
            t=np.array([float(r['t_s']) for r in data]);z=np.array([float(r['z_m']) for r in data])
            keep=np.array([r['keep'].lower()=='true' for r in data])
            ax.plot(t,z,color='#bbbbbb',linewidth=.7,label='raw')
            ax.scatter(t[keep],z[keep],s=2,color='#1976b8',label='retained')
            ax.scatter(t[~keep],z[~keep],s=8,color='#d55244',label='excluded')
            if row.get('first_bounce_t_s') is not None:
                ax.axvline(row['first_bounce_t_s'],color='black',linewidth=.8)
                ax.axvspan(row['bounce_interval_start_s'],row['bounce_interval_end_s'],alpha=.2,color='#ffba31')
            ax.set_title(Path(row['trajectory_id']).stem.replace('trajectory_',''),fontsize=7)
            ax.set_xlabel('time (s)',fontsize=7);ax.set_ylabel('measured z (m)',fontsize=7)
            ax.text(.02,.96,group[row['trajectory_id']],transform=ax.transAxes,va='top',fontsize=7)
        for ax in axes.ravel()[len(rows[page*16:(page+1)*16]):]:ax.set_visible(False)
        fig.suptitle('Automatic first-bounce labels: all trajectories; units and ground unconfirmed')
        fig.savefig(out/f'annotation_review_{page+1:02d}.png',dpi=120);plt.close(fig)
    if (root/'flight/flight_metrics.json').exists():
        metrics=json.loads((root/'flight/flight_metrics.json').read_text())
        fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
        groups=['train','validation','test'];x=np.arange(3)
        for dx,key,name,color in [(-.18,'ballistic_forecast','Gravity only','#bcbcbc'),(.18,'drag_forecast','Shared quadratic drag','#1976b8')]:
            ax.bar(x+dx,[metrics['metrics'][g][key]['rmse_m'] for g in groups],width=.35,label=name,color=color)
        ax.set_xticks(x,groups);ax.set_ylabel('3D forecast RMSE (m)');ax.set_title('Forecast after first 80 ms raw observation window')
        ax.legend();fig.savefig(out/'flight_forecast_errors.png',dpi=150);plt.close(fig)
        traces=json.loads((root/'flight/trajectory_fits.json').read_text())['trajectories']
        val=[r for r in traces if r['split']=='test' and r.get('prediction_csv')]
        val.sort(key=lambda r:r['drag_forecast']['position_error']['rmse_m'])
        selected=[val[i] for i in sorted(set([0,len(val)//2,len(val)-1]))] if val else []
        fig,axes=plt.subplots(len(selected),2,figsize=(11,3.3*max(1,len(selected))),squeeze=False,constrained_layout=True)
        for axrow,row in zip(axes,selected):
            data=np.genfromtxt(root/'flight'/row['prediction_csv'],delimiter=',',names=True)
            for ax,coord in zip(axrow,['x','z']):
                ax.plot(data['t_s'],data[f'measured_{coord}_m'],color='#444444',linewidth=1,label='retained observations')
                ax.plot(data['t_s'],data[f'forecast_{coord}_m'],color='#1976b8',label='forecast')
                ax.axvline(row['drag_forecast']['init_end_t_s'],color='#d55244',linestyle='--',label='initialization ends')
                ax.set_ylabel(f'{coord} (m)');ax.set_xlabel('time (s)');ax.legend(fontsize=7)
                ax.set_title(Path(row['trajectory_id']).stem,fontsize=9)
        if selected:fig.savefig(out/'heldout_flight_examples.png',dpi=150)
        plt.close(fig)
    if (root/'bounce/summary.json').exists():
        report=json.loads((root/'bounce/summary.json').read_text());records=report['impact_estimates']
        fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        keys=['e_n_observed','alpha_t_observed'];colors={'train':'#1976b8','validation':'#e8a228','test':'#d55244'}
        for group_name in colors:
            r=[r for r in records if r['split']==group_name]
            speed=[-r['v_minus_m_s'][2] for r in r]
            for ax,key in zip(axes[:2],keys):ax.scatter(speed,[r[key] for r in r],s=13,label=group_name,color=colors[group_name],alpha=.7)
            axes[2].hist([r['p_b_m'][2] for r in r],bins=15,alpha=.4,label=group_name,color=colors[group_name])
        for ax,key,param in zip(axes[:2],keys,['e_n','alpha_t']):
            ax.axhline(report['parameters'][param],color='black',linestyle='--');ax.set_xlabel('incident normal speed (m/s)');ax.set_ylabel(key);ax.legend(fontsize=8)
        axes[2].axvline(report['parameters']['effective_contact_center_height_m'],color='black',linestyle='--')
        axes[2].set_xlabel('effective measured contact height (m)');axes[2].set_ylabel('trajectory count');axes[2].legend(fontsize=8)
        fig.savefig(out/'bounce_parameters.png',dpi=150);plt.close(fig)
        full=json.loads((root/'bounce/full_predictions.json').read_text())
        tests=sorted([r for r in full if r['split']=='test' and r['available']],key=lambda r:r['postimpact_position_error']['rmse_m'])
        selected=[tests[i] for i in sorted(set([0,len(tests)//2,len(tests)-1]))] if tests else []
        fig,axes=plt.subplots(len(selected),2,figsize=(11,3.3*max(1,len(selected))),squeeze=False,constrained_layout=True)
        for axrow,row in zip(axes,selected):
            t=np.asarray(row['postimpact_time_s']);actual=np.asarray(row['measured_position_m']);prediction=np.asarray(row['predicted_position_m'])
            for ax,j in zip(axrow,[0,2]):
                ax.plot(t,actual[:,j],color='#444444',label='observations');ax.plot(t,prediction[:,j],color='#1976b8',label='causal forecast')
                ax.set_xlabel('time (s)');ax.set_ylabel(['x','y','z'][j]+' (m)');ax.set_title(Path(row['trajectory_id']).stem,fontsize=9);ax.legend(fontsize=8)
        if selected:fig.savefig(out/'heldout_first_bounce_examples.png',dpi=150)
        plt.close(fig)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(make_plots(parser.parse_args().root))
