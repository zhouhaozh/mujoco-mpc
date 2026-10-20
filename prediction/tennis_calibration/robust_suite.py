"""Run historical test and five fixed group folds after freezing validation selection."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(args):
    out=args.out.resolve()
    selection=json.loads((out/'selection_all.json').read_text())
    parent=json.loads((out/'selection_stage2.json').read_text())['selected']
    parents=list(dict.fromkeys(('baseline',parent)))
    if selection['split']!='validation': raise ValueError('validation selection must precede evaluation')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    def command(module,arguments,logname):
        log=out/(logname+'.log')
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-m','tennis_calibration.'+module,*map(str,arguments)],
                                  cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{logname} failed; see {log}')
        print(json.dumps({'completed':logname}),flush=True)
    command('robust_optimization',['--split','test','--noise-proxy',out/'validation_v2/noise_proxy.json',
        '--output',out/'test','--workers',str(args.workers)],'test')
    command('residual_evaluation',['--train-run',out/'train','--eval-run',out/'test',
        '--train-ids',out/'train_ids.json','--parents',*parents,'--eval-splits','test','--out',out/'residual_test'],'residual_test')
    command('robust_merge',['--runs',out/'test',out/'residual_test','--out',out/'combined_test'],'combined_test')
    def fold(number):
        model=ROOT/f'artifacts/tennis_hold_sweep/20261013_groupcv/fold{number}'
        dest=out/f'fold{number}'
        command('robust_optimization',['--fold-dir',model,'--partition','evaluation','--output',dest/'evaluation',
            '--workers',args.workers],f'fold{number}_evaluation')
        command('robust_optimization',['--fold-dir',model,'--partition','train','--experiments',*parents,
            '--noise-proxy',dest/'evaluation/noise_proxy.json','--output',dest/'train','--workers',args.workers],f'fold{number}_train')
        command('residual_evaluation',['--train-run',dest/'train','--eval-run',dest/'evaluation',
            '--train-ids',model/'fold_plan.json','--parents',*parents,'--eval-splits','oof','--out',dest/'residual'],f'fold{number}_residual')
        command('robust_merge',['--runs',dest/'evaluation',dest/'residual','--out',dest/'combined'],f'fold{number}_combined')
        return number
    with ThreadPoolExecutor(max_workers=args.parallel_folds) as pool:
        for completed in as_completed([pool.submit(fold,number) for number in range(5)]): completed.result()
    runs=[out/f'fold{i}'/part for i in range(5) for part in ('evaluation','residual')]
    command('robust_merge',['--runs',*runs,'--out',out/'combined_oof'],'combined_oof')
    print(json.dumps({'completed':'whole_suite','validation_selected':selection['selected']}),flush=True)


def extra(args):
    """Explicit follow-up: protected parent residuals; exposed-data exploration."""
    out=args.out.resolve(); parent='adaptive_protected'
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    plan=out/'extra_protocol.json'
    if plan.exists(): raise ValueError('preserve supplemental protocol')
    from datetime import datetime, timezone
    plan.write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
        'parent':parent,'kinds':['ridge','mlp'],'hyperparameters':'unchanged original fixed settings',
        'reason':'follow-up to observed original validation and OOF extreme peak',
        'evaluation_status':'exploratory on previously inspected groups, not independent selection estimate'},indent=2))
    def command(module,arguments,name):
        log=out/(name+'.log')
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-m','tennis_calibration.'+module,*map(str,arguments)],
                cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'{name} failed: {log}')
        print(json.dumps({'completed':name}),flush=True)
    command('robust_optimization',['--split','train','--experiments',parent,
        '--noise-proxy',out/'validation_v2/noise_proxy.json','--output',out/'extra_train','--workers',args.workers],'extra_train')
    for split in ('validation','test'):
        source=out/('validation_v2' if split=='validation' else 'test')
        command('residual_evaluation',['--train-run',out/'extra_train','--eval-run',source,
            '--train-ids',out/'train_ids.json','--parents',parent,'--eval-splits',split,'--out',out/f'extra_residual_{split}'],f'extra_residual_{split}')
        if split=='validation':
            score=json.loads((out/'extra_residual_validation/summary.json').read_text())
            selected=min(score['ranking'],key=lambda r:r['balanced_future_rmse_m'])['candidate']
            (out/'extra_validation_selection.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),
                'selected':selected,'scope':'fixed protected parent head comparison; descriptive exploratory follow-up',
                'ranking':score['ranking']},indent=2))
        command('robust_merge',['--runs',source,out/f'residual_{split}',out/f'extra_residual_{split}',
            '--out',out/f'final_{split}'],f'final_{split}')
    def fold(number):
        model=ROOT/f'artifacts/tennis_hold_sweep/20261013_groupcv/fold{number}'
        dest=out/f'fold{number}'
        command('robust_optimization',['--fold-dir',model,'--partition','train','--experiments',parent,
            '--noise-proxy',dest/'evaluation/noise_proxy.json','--output',dest/'extra_train','--workers',args.workers],f'fold{number}_extra_train')
        command('residual_evaluation',['--train-run',dest/'extra_train','--eval-run',dest/'evaluation',
            '--train-ids',model/'fold_plan.json','--parents',parent,'--eval-splits','oof','--out',dest/'extra_residual'],f'fold{number}_extra_residual')
        command('robust_merge',['--runs',dest/'evaluation',dest/'residual',dest/'extra_residual','--out',dest/'final'],f'fold{number}_final')
    with ThreadPoolExecutor(max_workers=args.parallel_folds) as pool:
        for finished in as_completed([pool.submit(fold,i) for i in range(5)]): finished.result()
    command('robust_merge',['--runs',*[out/f'fold{i}'/p for i in range(5) for p in ('evaluation','residual','extra_residual')],
        '--out',out/'final_oof'],'final_oof')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--parallel-folds',type=int,default=3)
    parser.add_argument('--extra-protected-residual',action='store_true')
    args=parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('run in Ubuntu project .venv-tennis')
    (extra if args.extra_protected_residual else run)(args)


if __name__=='__main__': main()
