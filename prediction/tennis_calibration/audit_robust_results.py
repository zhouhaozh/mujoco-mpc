"""Independent completed-suite audit; no replay, fitting, or source mutation.

Checks the original validation/test and five group folds directly from saved
rows, caches, training manifests and current protected resources. Run in the
Ubuntu project environment after robust_suite finishes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

from .group_fold_evaluation import read_json_csv
from .online_data import causal_anchors, future_labels, load_dataset

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (.1,.2,.5)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): hasher.update(chunk)
    return hasher.hexdigest()


def require(value,message):
    if not value: raise AssertionError(message)


def stage(value):
    return 0 if value<0 else 1 if value<.1 else 2 if value<.2 else 3 if value<=.5+1e-12 else None


def row_key(row):
    return (str(row['trajectory_id']),float(row['capture_t_s']),float(row['horizon_s']))


def scoring_row(row):
    return row['twenty_ms_anchor'] and row['elapsed_capture_s']>=.1-1e-12 and stage(row['relative_capture_s']) is not None


def expected_targets(trace):
    """Label-only census built independently of any observer output."""
    times=trace['raw_times']; end=trace.get('eval_scope_end_s')
    times=times[times<=(times[-1] if end is None else end)]
    result={}
    for anchor in causal_anchors({'raw_times':times},initialization_s=0.,stride_s=.02):
        stamp=anchor['t_s']; relative=stamp-trace['bounce_reference_s']
        if stamp-times[0]<.1-1e-12 or stage(relative) is None: continue
        for label in future_labels(trace,stamp,HORIZONS):
            result[(trace['id'],stamp,label['horizon_s'])]=label
    return result


def independent_scores(rows):
    cells=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for row in rows:
        if not scoring_row(row) or not row['valid']: continue
        require(not row['prediction_reason'] and row['predicted_xyz'] is not None,'valid scoring target has no prediction')
        p,y=np.asarray(row['predicted_xyz']),np.asarray(row['target_xyz'])
        require(p.shape==y.shape==(3,) and np.isfinite(p).all() and np.isfinite(y).all(),'nonfinite scoring vectors')
        cells[(row['split'],row['candidate'])][row['trajectory_id']][
            (stage(row['relative_capture_s']),row['horizon_s'])].append(float(np.dot(p-y,p-y)))
    scores={}
    for key,trajectories in cells.items():
        means=[]
        for identity,entries in trajectories.items():
            phases=defaultdict(list)
            for (p,h),values in entries.items(): phases[p].append(sum(values)/len(values))
            means.append(sum(sum(v)/len(v) for v in phases.values())/len(phases))
        scores[key]=(float(np.sqrt(sum(means)/len(means))),len(means))
    return scores


class Auditor:
    def __init__(self,root,calibration,models_root):
        self.root,self.calibration,self.models_root=root,calibration,models_root
        self.dataset=load_dataset(calibration)
        self.traces={t['id']:t for t in self.dataset}
        self.expected={t['id']:expected_targets(t) for t in self.dataset if t['bounce_reference_s'] is not None}
        self.protected={}; self.observer_results=[];self.residual_results=[];self.baselines=[]
        self.seen_fold_evaluation=Counter()
        self.result_scores={}

    def protect(self,path,sha):
        path=Path(path).resolve()
        if path in self.protected: require(self.protected[path]==sha,'inconsistent protected hash: '+str(path))
        self.protected[path]=sha

    def record_scores(self,scores,fold,partition):
        for (split,candidate),value in scores.items():
            key=(fold,partition,split,candidate)
            if key in self.result_scores:
                np.testing.assert_allclose(self.result_scores[key],value,atol=1e-12,rtol=1e-12)
            self.result_scores[key]=value

    def expected_ids(self,fold,partition):
        if fold is None:
            train={t['id'] for t in self.dataset if t['split']=='train'}
            selected={t['id'] for t in self.dataset if t['split']==partition}
        else:
            plan=read(self.models_root/f'fold{fold}/fold_plan.json')
            train=set(plan['train_ids']); held=set(plan['evaluation_ids'])
            require(not train & held and train | held == set(self.traces),'invalid group-fold partition')
            training_groups={self.traces[i]['groupid'] for i in train}
            held_groups={self.traces[i]['groupid'] for i in held}
            require(not training_groups & held_groups,'acquisition group crosses train/evaluation')
            selected=train if partition=='train' else held
        return train,selected

    def check_rows(self,rows,events,selected_ids,candidates):
        expected={key:label for identity in selected_ids for key,label in self.expected[identity].items()}
        grouped=defaultdict(dict);dense=defaultdict(set)
        event_map={(e['trajectory_id'],e['candidate']):e for e in events}
        require(len(event_map)==len(events),'duplicate event identity')
        require(set(event_map)=={(i,c) for i in selected_ids for c in candidates},'event coverage differs from all requested candidates')
        for row in rows:
            key=row_key(row); candidate=row['candidate']
            require(candidate in candidates and row['trajectory_id'] in selected_ids,'unexpected row identity')
            event=event_map[(row['trajectory_id'],candidate)]
            stamp=row['capture_t_s'];resume=event['resume_t_s']
            is_dense=resume is not None and resume-1e-12<=stamp<resume+.05-1e-12
            require(row['dense_recovery_frame']==is_dense,'dense-frame flag does not match actual recovery')
            if is_dense and row['horizon_s']==.5: dense[(row['trajectory_id'],candidate)].add(stamp)
            if not scoring_row(row): continue
            require(key not in grouped[candidate],'duplicate scoring key')
            grouped[candidate][key]=row
            require(key in expected,'unexpected scoring target key')
            label=expected[key]
            for name in ('valid','reason','target_t_s','target_xyz'):
                require(row[name]==label[name],'original label altered: '+name)
            if label['valid']:
                require(row['predicted_xyz'] is not None and not row['prediction_reason'],'failed labeled target')
        for candidate in candidates:
            require(set(grouped[candidate])==set(expected),'incomplete common-anchor requested coverage '+candidate)
        dense_requests=0
        for (identity,candidate),event in event_map.items():
            require(not event['tracking_failure'] and not event['forecast_failure'],'recorded tracking/forecast failure')
            resume=event['resume_t_s']; trace=self.traces[identity]
            times=trace['raw_times'];end=trace.get('eval_scope_end_s')
            expected_dense=set() if resume is None else {float(t) for t in times if
                resume-1e-12<=t<resume+.05-1e-12 and (end is None or t<=end)}
            require(dense[(identity,candidate)]==expected_dense,'dense diagnostic missed raw frames')
            dense_requests+=len(expected_dense)
        return {'candidates':len(candidates),'trajectories':len(selected_ids),
            'requested_common_targets_per_candidate':len(expected),
            'labeled_common_targets_per_candidate':sum(label['valid'] for label in expected.values()),
            'dense_h500_requested_all_candidates':dense_requests,'all_labeled_predictions_present':True,
            'original_targets_unchanged':True,'all_raw_recovery_frames_present':True}

    def observer(self,path,fold=None,partition='validation'):
        summary=read(path/'summary.json');execution=read(path/'execution.json');events=read(path/'events.json')
        require(summary.get('completed'),'observer replay incomplete '+str(path))
        train,partition_ids=self.expected_ids(fold,partition)
        selected={i for i in partition_ids if i in self.expected}
        require(set(summary['training_ids'])==train,'observer training IDs incorrect')
        require(set(summary['evaluation_ids'])==selected,'observer selected IDs incorrect')
        require({e['id'] for e in summary['excluded']}==partition_ids-selected,'unreported unscorable identity')
        if partition!='train':require(not train & selected,'observer train/evaluation leakage')
        proxy=read(path/'noise_proxy.json')
        require(proxy==summary['noise_proxy']==execution['noise_proxy'],'noise proxy metadata differs')
        require(set(proxy['allowed_training_ids'])==train,'noise proxy normalization uses wrong fold')
        require(set(proxy['observed_training_ids'])<=train,'noise proxy uses held-out data')
        require(bool(proxy['observed_training_ids']),'empty noise training scope')
        rows=read_json_csv(path/'predictions.csv.gz')
        require(digest(path/'predictions.csv.gz')==summary['predictions_sha256'],'prediction CSV modified')
        candidates=set(summary['experiments'])
        census=self.check_rows(rows,events,selected,candidates)
        scores=independent_scores(rows)
        self.record_scores(scores,fold,partition)
        require(set(scores)=={(r['split'],r['candidate']) for r in summary['ranking']},'ranking keys lost a candidate')
        for rank in summary['ranking']:
            value,count=scores[(rank['split'],rank['candidate'])]
            np.testing.assert_allclose(value,rank['balanced_future_rmse_m'],atol=1e-12,rtol=1e-12)
            require(count==rank['N'] and rank['eligible_for_ranking'],'invalid observer ranking count/eligibility')
        for name,sha in summary['calibration_hashes'].items():self.protect(self.calibration/name,sha)
        checkpoint=Path(summary['checkpoint'])
        for name,sha in summary['model_hashes'].items():
            model=Path(name);self.protect(model if model.is_absolute() else checkpoint.parent/model,sha)
        for name,sha in summary['source_hashes'].items():self.protect(ROOT/'tennis_calibration'/name,sha)
        cache_counts={}
        for candidate,info in summary['feature_caches'].items():
            file=path/f'cache_{candidate}.npz'
            require(digest(file)==info['sha256'],'feature cache modified')
            source=[r for r in rows if r['candidate']==candidate]
            # NpzFile.__getitem__ decompresses an entire member on every call.
            # Materialize each used array once before iterating over rows.
            with np.load(file,allow_pickle=False) as archive:
                cache={key:archive[key] for key in ('ids','X','capture_t_s','H','gate','anchor','dense')}
            require(len(cache['ids'])==len(source)==info['rows'],'cache row count differs')
            require(cache['X'].shape==(len(source),31),'unexpected feature width')
            for index,row in enumerate(source):
                require((str(cache['ids'][index]),float(cache['capture_t_s'][index]),float(cache['H'][index]))==row_key(row),'cache identity mismatch')
                require(float(cache['gate'][index])==row['residual_gate'],'cache gate differs from replay')
                require(bool(cache['anchor'][index])==row['twenty_ms_anchor'] and bool(cache['dense'][index])==row['dense_recovery_frame'],'cache sampling markers differ')
                if row['hold_active'] or row['dense_recovery_frame']:
                    require(cache['gate'][index]==0.,'residual gate open during hold/first50ms recovery')
            cache_counts[candidate]=len(source)
        baseline_result=None
        if partition!='train':
            require(summary['baseline_check']['status']=='passed','baseline replay not checked')
            historical=read(summary['baseline_check']['path'])
            ranks=historical.get('ranking',historical.get('candidate_ranking'))
            name='gp_hold100ms' if fold is not None else 'gp:hold100'
            split='oof' if fold is not None else partition
            old=next(r for r in ranks if r['candidate']==name and r['split']==split)
            value,count=scores[(split,'baseline')]
            np.testing.assert_allclose(value,old['balanced_future_rmse_m'],atol=1e-10,rtol=1e-10)
            baseline_result={'partition':partition,'fold':fold,'N':count,'replayed_rmse_m':value,
                'historical_rmse_m':old['balanced_future_rmse_m'],'difference_m':value-old['balanced_future_rmse_m']}
            self.baselines.append(baseline_result)
            if fold is not None:self.seen_fold_evaluation.update(selected)
        result={'path':str(path),'fold':fold,'partition':partition,'train_count':len(train),
            'train_evaluation_disjoint':True if partition!='train' else None,
            'training_partition_replay':partition=='train','noise_fitted_only_on_partition_train':True,
            **census,'cache_rows':cache_counts,'independent_score_cells':len(scores),'baseline':baseline_result}
        self.observer_results.append(result)
        return summary

    def residual(self,path,train_path,fold=None,partition='validation'):
        summary=read(path/'summary.json');execution=summary['residual_evaluation']['execution']
        require(execution==read(path/'execution.json'),'residual execution metadata differs from summary')
        evaluation_path=Path(execution['eval_run'])
        for filename,key in (('predictions.csv.gz','evaluation_predictions'),('events.json','evaluation_events')):
            require(digest(evaluation_path/filename)==execution['source_hashes'][key],
                    'residual source observer data changed: '+filename)
        for name,sha in execution['source_hashes'].items():
            if name.endswith('.py'):self.protect(ROOT/'tennis_calibration'/name,sha)
        audits=read(path/'training_audit.json');events=read(path/'events.json');rows=read_json_csv(path/'predictions.csv.gz')
        train,partition_ids=self.expected_ids(fold,partition);selected={i for i in partition_ids if i in self.expected}
        require(set(execution['training_ids'])==train and set(execution['evaluation_ids'])==selected,'residual partition IDs differ')
        require(not train & selected,'residual train/evaluation leakage')
        candidates={r['candidate'] for r in rows}; census=self.check_rows(rows,events,selected,candidates)
        lookup={(r['candidate'],*row_key(r)):r for r in rows}
        require(len(lookup)==len(rows),'duplicate residual rows')
        zero=0;dense=0;paired=0
        for row in rows:
            if '__residual_' not in row['candidate']:continue
            parent=row['residual_parent'];original=lookup[(parent,*row_key(row))]
            for name in ('target_t_s','target_xyz','valid','reason','twenty_ms_anchor','dense_recovery_frame','residual_gate','height_contact_t_s','height_discovery_t_s','resume_t_s'):
                require(row[name]==original[name],'residual altered parent '+name)
            paired+=1
            if row['residual_gate']==0:
                require(row['predicted_xyz']==original['predicted_xyz'],'zero-gate prediction changed')
                require(row['residual_displacement_xyz']==[0.,0.,0.],'zero-gate displacement nonzero')
                zero+=1
            if row['dense_recovery_frame']:
                require(row['residual_gate']==0 and row['predicted_xyz']==original['predicted_xyz'],'ML head changed first50ms recovery prediction')
                dense+=1
        event_map={(e['trajectory_id'],e['candidate']):e for e in events}
        for event in events:
            if '__residual_' not in event['candidate']:continue
            parent=event_map[(event['trajectory_id'],event['residual_parent'])]
            for name in ('height_contact_t_s','height_discovery_t_s','resume_t_s','contact_minus_reference_s','discovery_minus_reference_s','resume_minus_reference_s'):
                require(event[name]==parent[name],'head changed event clock')
        models={};normalizations=0
        for audit in audits:
            require(set(audit['allowed_training_ids'])==train and set(audit['observed_training_ids'])<=train,'head fitted nontraining identities')
            model_path=path/'models'/(audit['candidate']+'.json');model=read(model_path)
            require(digest(model_path)==audit['saved_model_sha256'],'head weights changed')
            require(set(model['metadata']['allowed_training_ids'])==train,'head normalization allowed IDs differ')
            require(set(model['metadata']['observed_training_ids'])==set(audit['observed_training_ids']),'head normalization observed IDs differ')
            parent=audit['parent'];cachefile=train_path/f'cache_{parent}.npz'
            require(digest(cachefile)==execution['source_hashes']['train_cache_'+parent],'residual training cache differs')
            with np.load(cachefile,allow_pickle=False) as archive:
                cache={key:archive[key] for key in ('ids','anchor','gate','valid','prediction_ok',
                    'initialized','elapsed_capture_s','X','pred','ref')}
            ids=cache['ids'].astype(str)
            mask=(np.isin(ids,sorted(train)) & cache['anchor'] & np.isclose(cache['gate'],1.,atol=1e-12,rtol=0.) & cache['valid'] & cache['prediction_ok'] & cache['initialized'] & (cache['elapsed_capture_s']>=.1-1e-12)
                  & np.isfinite(cache['X']).all(axis=1) & np.isfinite(cache['pred']).all(axis=1) & np.isfinite(cache['ref']).all(axis=1))
            x=cache['X'][mask];y=cache['ref'][mask]-cache['pred'][mask];used=ids[mask]
            require(len(x)==audit['eligible_training_rows']==model['metadata']['training_rows'],'head training sample count differs')
            require(set(used)==set(audit['observed_training_ids']),'head actual fitting identity scope differs')
            counts=Counter(used);w=np.asarray([1./len(counts)/counts[i] for i in used])
            mean=np.sum(w[:,None]*x,axis=0);scale=np.sqrt(np.sum(w[:,None]*(x-mean)**2,axis=0));scale[scale<1e-5]=1.
            target_scale=np.maximum(np.sqrt(np.sum(w[:,None]*y*y,axis=0)),.02)
            for actual,expected in ((model['mean'],mean),(model['scale'],scale),(model['target_scale'],target_scale)):
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
            normalizations+=1;models[audit['candidate']]=digest(model_path)
        score=independent_scores(rows)
        self.record_scores(score,fold,partition)
        for rank in summary['ranking']:
            value,count=score[(rank['split'],rank['candidate'])]
            np.testing.assert_allclose(value,rank['balanced_future_rmse_m'],atol=1e-12,rtol=1e-12)
            require(count==rank['N'],'residual independent N differs')
        result={'path':str(path),'fold':fold,'partition':partition,**census,
            'paired_parent_rows':paired,'zero_gate_rows_exactly_unchanged':zero,
            'dense_recovery_rows_exactly_unchanged':dense,'head_models':models,
            'training_only_normalizations_recomputed':normalizations,'train_evaluation_disjoint':True}
        self.residual_results.append(result)
        return result

    def check_final_summary(self,path,fold=None,partition='validation'):
        summary=read(path/'summary.json')
        require(summary.get('completed'),'final combined summary incomplete')
        for source in summary['sources']:
            directory=Path(source['path'])
            require(digest(directory/'summary.json')==source['summary_sha256'],'final merge source summary changed')
            require(digest(directory/'predictions.csv.gz')==source['predictions_sha256'],'final merge source rows changed')
        if partition=='oof':
            by_candidate=defaultdict(list)
            for (f,p,s,c),value in self.result_scores.items():
                if f is not None and p=='evaluation' and s=='oof':by_candidate[c].append((f,*value))
            expected={}
            for candidate,values in by_candidate.items():
                require({v[0] for v in values}==set(range(5)),'final OOF candidate missing fold')
                total=sum(v[2] for v in values)
                expected[('oof',candidate)]=(float(np.sqrt(sum(v[1]**2*v[2] for v in values)/total)),total)
        else:
            expected={(s,c):value for (f,p,s,c),value in self.result_scores.items() if f==fold and p==partition}
        require(set(expected)=={(r['split'],r['candidate']) for r in summary['ranking']},'final combined candidate set differs')
        for rank in summary['ranking']:
            score,n=expected[(rank['split'],rank['candidate'])]
            np.testing.assert_allclose(score,rank['balanced_future_rmse_m'],atol=1e-12,rtol=1e-12)
            require(n==rank['N'],'final combined trajectory count differs')
        return {'path':str(path),'fold':fold,'partition':partition,'ranking_cells':len(expected),
                'source_hashes_unchanged':True,'independently_verified_from_source_rows':True}

    def execute(self,include_extra=False):
        selection=read(self.root/'selection_all.json');stage2=read(self.root/'selection_stage2.json')
        require(selection['split']==stage2['split']=='validation','selection used nonvalidation partition')
        selected_utc=selection['selected_utc']
        test_created=read(self.root/'test/execution.json')['created_utc']
        require(datetime.fromisoformat(selected_utc)<=datetime.fromisoformat(test_created),'test replay preceded saved selection')
        self.observer(self.root/'validation_v2',partition='validation')
        self.observer(self.root/'train',partition='train')
        self.observer(self.root/'test',partition='test')
        validation=self.residual(self.root/'residual_validation',self.root/'train',partition='validation')
        test=self.residual(self.root/'residual_test',self.root/'train',partition='test')
        require(validation['head_models']==test['head_models'],'validation/test head model hashes differ')
        for fold in range(5):
            path=self.root/f'fold{fold}'
            self.observer(path/'evaluation',fold=fold,partition='evaluation')
            self.observer(path/'train',fold=fold,partition='train')
            self.residual(path/'residual',path/'train',fold=fold,partition='evaluation')
            print('ROBUST_AUDIT '+json.dumps({'fold_complete':fold}),flush=True)
        require(set(self.seen_fold_evaluation)==set(self.expected),'OOF misses evaluable trajectories')
        require(all(count==1 for count in self.seen_fold_evaluation.values()),'OOF evaluates a trajectory multiple times')
        require(len(self.baselines)==7,'expected original validation/test and five baseline scores')
        extra_result=None
        if include_extra:
            protocol=read(self.root/'extra_protocol.json')
            require('exploratory' in protocol.get('evaluation_status','').lower(),
                    'supplemental protocol lacks explicit exploratory status')
            require(protocol['parent']=='adaptive_protected' and set(protocol['kinds'])=={'ridge','mlp'},
                    'unexpected supplemental model recipe')
            extra_selection=read(self.root/'extra_validation_selection.json')
            require('exploratory' in extra_selection.get('scope','').lower(),
                    'supplemental selection lacks explicit exploratory status')
            test_metadata=read(self.root/'extra_residual_test/execution.json')
            require(datetime.fromisoformat(extra_selection['created_utc'])<=datetime.fromisoformat(test_metadata['created_utc']),
                    'supplemental test run preceded saved validation comparison')
            self.observer(self.root/'extra_train',partition='train')
            require(read(self.root/'extra_train/noise_proxy.json')==read(self.root/'validation_v2/noise_proxy.json'),
                    'extra original train changed fixed noise proxy')
            extra_validation=self.residual(self.root/'extra_residual_validation',self.root/'extra_train',partition='validation')
            extra_test=self.residual(self.root/'extra_residual_test',self.root/'extra_train',partition='test')
            require(extra_validation['head_models']==extra_test['head_models'],'supplemental validation/test head hashes differ')
            original_settings=read(self.root/'residual_validation/execution.json')['candidate_settings']
            for name in ('extra_residual_validation','extra_residual_test'):
                require(read(self.root/name/'execution.json')['candidate_settings']==original_settings,
                        'supplemental residual optimizer settings changed')
            final_checks=[self.check_final_summary(self.root/'final_validation',partition='validation'),
                          self.check_final_summary(self.root/'final_test',partition='test')]
            for fold in range(5):
                path=self.root/f'fold{fold}'
                self.observer(path/'extra_train',fold=fold,partition='train')
                require(read(path/'extra_train/noise_proxy.json')==read(path/'evaluation/noise_proxy.json'),
                        'extra fold train changed fixed fold noise proxy')
                self.residual(path/'extra_residual',path/'extra_train',fold=fold,partition='evaluation')
                require(read(path/'extra_residual/execution.json')['candidate_settings']==original_settings,
                        'supplemental fold optimizer settings changed')
                final_checks.append(self.check_final_summary(path/'final',fold=fold,partition='evaluation'))
                print('ROBUST_AUDIT '+json.dumps({'extra_fold_complete':fold}),flush=True)
            final_checks.append(self.check_final_summary(self.root/'final_oof',partition='oof'))
            extra_result={'protocol':protocol,'saved_validation_comparison':extra_selection,
                'status':'exploratory follow-up after inspecting historical OOF results; not independent blind validation',
                'additional_observer_training_runs':6,'additional_residual_runs':7,
                'validation_test_head_hashes_identical':extra_validation['head_models'],
                'residual_hyperparameters_unchanged':True,'fixed_training_noise_proxies_unchanged':True,
                'final_summary_audits':final_checks}
        for path,sha in self.protected.items():require(digest(path)==sha,'protected resource changed: '+str(path))
        return {'schema':'INDEPENDENT_ROBUST_SUITE_AUDIT/1','completed':True,'passed':True,
            'created_utc':datetime.now(timezone.utc).isoformat(),'hostname':platform.node(),'python_prefix':sys.prefix,
            'suite':str(self.root),'validation_selected':selection['selected'],'stage2_selected':stage2['selected'],
            'selection_saved_before_test_replay':True,'validation_test_head_hashes_identical':validation['head_models'],
            'baseline_reproductions':self.baselines,'observer_run_audits':self.observer_results,
            'residual_run_audits':self.residual_results,'oof_trajectories_exactly_once':len(self.seen_fold_evaluation),
            'protected_source_model_label_resources_unchanged':len(self.protected),
            'protected_hashes':{str(path):sha for path,sha in self.protected.items()},
            'includes_exploratory_followup':include_extra,'exploratory_followup_audit':extra_result,
            'no_deployment_or_training_performed_by_audit':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--include-extra',action='store_true',help='Also audit the explicitly exploratory protected-residual follow-up')
    parser.add_argument('--calibration',type=Path,default=ROOT/'artifacts/new_ubuntu/tennis_calibration')
    parser.add_argument('--models-root',type=Path,default=ROOT/'artifacts/tennis_hold_sweep/20261013_groupcv')
    args=parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('audit numerical outputs in Ubuntu project .venv-tennis')
    output=args.output or args.suite/('independent_audit_with_extra.json' if args.include_extra else 'independent_audit.json')
    if output.exists():raise FileExistsError('preserve completed audit; choose a new --output')
    result=Auditor(args.suite,args.calibration,args.models_root).execute(include_extra=args.include_extra)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'baseline_checks':len(result['baseline_reproductions']),
        'observer_runs':len(result['observer_run_audits']),'residual_runs':len(result['residual_run_audits']),
        'protected_resources':result['protected_source_model_label_resources_unchanged'],'output':str(output)}),flush=True)


if __name__=='__main__':main()
