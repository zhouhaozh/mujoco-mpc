"""Ubuntu-only data audit, flight calibration, and effective first bounce."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

from .clean import clean_archive
from .flight import run_flight
from .bounce import run_bounce

ROOT=Path(__file__).resolve().parents[1]


def frozen_split(annotations, archive_sha):
    group_map={r['trajectory_id']:int(r['session_id'].rsplit('_',1)[1])-1 for r in annotations}
    groups=sorted(set(group_map.values()))
    if groups!=list(range(11)):
        raise ValueError('Expected audited eleven acquisition groups; inspect grouping before splitting')
    counts=[sum(g==i for g in group_map.values()) for i in groups]
    if counts!=[12,15,11,19,21,22,30,7,2,1,3]:
        raise ValueError(f'Acquisition groups differ from the frozen audit: {counts}')
    assignments={g:('validation' if g in (2,3) else 'test' if g in (1,4) else 'train') for g in groups}
    out={group:sorted(i for i,g in group_map.items() if assignments[g]==group) for group in ('train','validation','test')}
    out.update({'source_archive_sha256':archive_sha,'seed':20261008,'trajectory_to_group':group_map,
                'group_assignment':assignments,'group_counts':counts,
                'grouping_is_inferred':True,
                'rule':'GID order, large acquisition gaps and trajectory numbering resets; metadata unconfirmed',
                'frozen_before_fit':True,'selection_uses_model_errors':False})
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'data/tennis/raw/all_trajectories.zip')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--clean-only',action='store_true')
    args=p.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        p.error('Use Ubuntu and the project .venv-tennis/bin/python')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%fZ')
    out=(args.output_dir or ROOT/'artifacts/tennis_calibration'/stamp).resolve()
    out.mkdir(parents=True,exist_ok=False)
    paths=list((ROOT/'tennis_calibration').glob('*.py'))
    assumptions={'time_unit':'seconds assumed','position_scale':.001,'position_unit':'millimetres assumed',
                 'vertical_axis':'positive z assumed','measurement_point':'ball center unconfirmed',
                 'ground_height':'unconfirmed','spin':'not measured','surface':'not provided',
                 'gravity_m_s2':[0.,0.,-9.81],'model_scope':'effective airborne and first ground bounce, no MuJoCo deployment'}
    report={'schema':'TENNIS_CALIBRATION/1','created_utc':datetime.now(timezone.utc).isoformat(),
            'hostname':platform.node(),'platform':platform.platform(),'python':sys.version,
            'python_prefix':sys.prefix,'archive':str(args.archive.resolve()),
            'assumptions':assumptions,'completed':False,
            'source_sha256':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}}
    (out/'parameters.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        print('TENNIS_CLEAN_START',flush=True)
        clean=clean_archive(args.archive,out/'clean')
        annotations=json.loads((out/'clean/annotations.json').read_text())
        split=frozen_split(annotations,clean['archive_sha256'])
        (out/'split.json').write_text(json.dumps(split,indent=2)+'\n')
        report['clean']=clean;report['split_counts']={g:len(split[g]) for g in ('train','validation','test')}
        print('TENNIS_CLEAN_DONE '+json.dumps({k:clean[k] for k in ('n_raw','n_keep','n_bounce_candidates','n_incomplete')}),flush=True)
        if not args.clean_only:
            print('TENNIS_FLIGHT_START',flush=True)
            flight=run_flight(out/'clean',out/'clean/annotations.csv',out/'split.json',out/'flight')
            report['flight']={'calibration_status':flight['calibration_status'],
                              'shared_parameters':flight['shared_parameters'],'metrics':flight['metrics'],
                              'trajectories_used':flight['trajectories_used'],'excluded':flight['excluded']}
            print('TENNIS_FLIGHT_DONE '+json.dumps(flight['shared_parameters']),flush=True)
            if not flight['shared_parameters']['converged']:
                raise RuntimeError('Shared flight optimization did not converge; inspect results before bounce calibration')
            print('TENNIS_BOUNCE_START',flush=True)
            bounce=run_bounce(out/'clean',out/'clean/annotations.json',out/'split.json',out/'flight',out/'bounce')
            report['bounce']={'parameters':bounce['parameters'],'metrics':bounce['metrics'],
                              'causal_full_flight_bounce_forecast':bounce['causal_full_flight_bounce_forecast']}
            print('TENNIS_BOUNCE_DONE '+json.dumps(bounce['parameters']),flush=True)
            from .plots import make_plots
            report['plots_dir']=str(make_plots(out))
        report['completed']=True
        report['status']='completed_effective_calibration_under_unconfirmed_metadata'
    except Exception as error:
        report['error']=str(error);report['status']='incomplete'
        raise
    finally:
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print('TENNIS_CALIBRATION_REPORT='+str(out/'summary.json'),flush=True)


if __name__=='__main__':main()
