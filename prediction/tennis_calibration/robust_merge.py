"""Combine observer/residual or disjoint OOF outputs, with exact paired audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .group_fold_evaluation import read_json_csv
from .hold_sweep import digest
from .identified_run import dump
from .robust_optimization import summarize


def merge(paths, out):
    if (out/'summary.json').exists():
        raise ValueError('preserve completed combined result')
    rows={}; events={}; sources=[]; duplicate_rows=0; folds=[]
    for path in paths:
        summary=json.loads((path/'summary.json').read_text())
        sources.append({'path':str(path),'summary_sha256':digest(path/'summary.json'),
                        'predictions_sha256':digest(path/'predictions.csv.gz')})
        if summary.get('fold_plan'):
            folds.extend(dict(r,fold=summary['fold_plan']['fold']) for r in summary['ranking'])
        for row in read_json_csv(path/'predictions.csv.gz'):
            key=(row['split'],row['trajectory_id'],row['candidate'],row['capture_t_s'],row['horizon_s'])
            if key in rows:
                previous=rows[key]
                for field in ('predicted_xyz','target_xyz','target_t_s','valid','prediction_reason',
                              'twenty_ms_anchor','dense_recovery_frame'):
                    if previous[field] != row[field]: raise ValueError('overlapping source disagreement: '+field)
                duplicate_rows+=1
            else: rows[key]=row
        for event in json.loads((path/'events.json').read_text()):
            key=(event['split'],event['trajectory_id'],event['candidate'])
            if key in events:
                for field in ('height_contact_t_s','height_discovery_t_s','resume_t_s','tracking_failure'):
                    if events[key][field]!=event[field]: raise ValueError('duplicate event differs')
            else: events[key]=event
    result=summarize(list(rows.values()),list(events.values()))
    result.update(completed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node(),python_prefix=sys.prefix,sources=sources,events=list(events.values()),
        duplicate_parent_rows_verified=duplicate_rows,rows=len(rows),fold_ranking=folds)
    # Independently recompute the trajectory/stage/horizon balanced score from
    # raw distances, without calling the production aggregation helper.
    cells=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in rows.values():
        if not row['twenty_ms_anchor'] or row['elapsed_capture_s'] < .1-1e-12: continue
        tau=row['relative_capture_s']
        stage=0 if tau<0 else 1 if tau<.1 else 2 if tau<.2 else 3 if tau<=.5+1e-12 else None
        if stage is None or not row['valid'] or row.get('prediction_reason') or row['predicted_xyz'] is None: continue
        squared=sum((x-y)**2 for x,y in zip(row['predicted_xyz'],row['target_xyz']))
        cells[(row['split'],row['candidate'])][row['trajectory_id']][(stage,row['horizon_s'])].append(squared)
    checks=[]
    for row in result['ranking']:
        trajectories=[]
        for identity, entries in cells[(row['split'],row['candidate'])].items():
            phases=defaultdict(list)
            for (stage,h),values in entries.items(): phases[stage].append(sum(values)/len(values))
            trajectories.append(sum(sum(v)/len(v) for v in phases.values())/len(phases))
        score=(sum(trajectories)/len(trajectories))**.5
        np.testing.assert_allclose(score,row['balanced_future_rmse_m'],atol=1e-12,rtol=1e-12)
        if len(trajectories)!=row['N']: raise ValueError('independent trajectory count mismatch')
        checks.append({'split':row['split'],'candidate':row['candidate'],'independent_rmse_m':score,'N':len(trajectories)})
    result['independent_ranking_audit']=checks
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'summary.json',result)
    dump(out/'events.json',list(events.values()))
    print(json.dumps({'out':str(out),'rows':len(rows),'duplicate_parent_rows_verified':duplicate_rows,
                      'ranking':result['ranking']},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(root/'.venv-tennis').resolve():
        raise RuntimeError('statistics must run in Ubuntu project .venv-tennis')
    merge(args.runs,args.out)


if __name__=='__main__': main()
