"""Effective first-bounce laws and a causal flight-to-bounce validation.

Ground height, spin and friction are not independently observed. Parameters
describe measured center trajectories under the stated coordinate assumptions.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .clean import robust_poly
from .flight import GRAVITY, _load_clean, _resolve_clean, predicted_contact, simulate, error_stats


def acceleration(v, k):
    return GRAVITY-k*np.linalg.norm(v)*v


def impact_estimate(t, p, annotation, k, window_s=.12):
    """Common impact position/time, independent incoming/outgoing velocities.

    Leave a contact guard and fit the two sides separately. The local quadratic
    uses the fitted air law at each impact velocity, not sample derivatives.
    """
    center=float(annotation['first_bounce_t_s'])
    pre_end=min(center-.015,float(annotation['pre_end_t_s']))
    post_start=max(center+.015,float(annotation['post_start_t_s']))
    next_end=float(annotation.get('post_end_t_s') or t[-1])
    left=(t>=center-window_s)&(t<=pre_end)
    right=(t>=post_start)&(t<=min(center+window_s,next_end))
    if min(left.sum(),right.sum())<12 or min(np.ptp(t[left]),np.ptp(t[right]))<.04:
        raise ValueError('insufficient_independent_pre_post_coverage')
    lfits=[robust_poly(t[left],p[left,j],center) for j in range(3)]
    rfits=[robust_poly(t[right],p[right,j],center) for j in range(3)]
    pb=np.array([(a[0][0]+b[0][0])/2 for a,b in zip(lfits,rfits)])
    vm=np.array([v[0][1] for v in lfits]);vp=np.array([v[0][1] for v in rfits])
    sigma=np.clip(np.array([(a[1]+b[1])/2 for a,b in zip(lfits,rfits)]),.004,.05)
    lo=max(center-.020,t[left][-1]+.001)
    hi=min(center+.020,t[right][0]-.001)
    if not lo<center<hi:raise ValueError('event_bracket_invalid')
    def positions(x, times, velocity):
        dt=(times-x[9])[:,None]
        return x[:3]+dt*velocity+.5*dt**2*acceleration(velocity,k)
    def residual(x):
        return np.vstack(((positions(x,t[left],x[3:6])-p[left])/sigma,
                          (positions(x,t[right],x[6:9])-p[right])/sigma)).ravel()
    x0=np.r_[pb,np.clip(vm,-119,119),np.clip(vp,-119,119),center]
    result=least_squares(residual,x0,bounds=(np.r_[[-100]*3,[-120]*6,lo],
                                           np.r_[[100]*3,[120]*6,hi]),
                         loss='soft_l1',f_scale=1.5,max_nfev=120,
                         x_scale='jac',ftol=1e-9,xtol=1e-9,gtol=1e-9)
    x=result.x;before=x[3:6];after=x[6:9]
    errors=np.r_[np.linalg.norm(positions(x,t[left],before)-p[left],axis=1),
                  np.linalg.norm(positions(x,t[right],after)-p[right],axis=1)]
    if before[2]>=-.4 or after[2]<=.4:raise ValueError('not_descending_then_ascending')
    horizontal2=np.dot(before[:2],before[:2])
    if horizontal2<.25:raise ValueError('incident_horizontal_speed_too_small')
    alpha=float(np.dot(after[:2],before[:2])/horizontal2)
    transverse=after[:2]-alpha*before[:2]
    return {'converged':bool(result.success),'nfev':int(result.nfev),
            't_b_s':float(x[9]),'p_b_m':x[:3].tolist(),
            'v_minus_m_s':before.tolist(),'v_plus_m_s':after.tolist(),
            'e_n_observed':float(-after[2]/before[2]),'alpha_t_observed':alpha,
            'tangential_orthogonal_residual_m_s':float(np.linalg.norm(transverse)),
            'position_error':error_stats(errors),'window_s':window_s,
            'time_at_bound':bool(min(x[9]-lo,hi-x[9])<.0002),
            'pre_points':int(left.sum()),'post_points':int(right.sum())}


def predict_velocity(v, e, alpha):
    return np.array([alpha*v[0],alpha*v[1],-e*v[2]])


def fit_law(records):
    before=np.asarray([r['v_minus_m_s'] for r in records])
    after=np.asarray([r['v_plus_m_s'] for r in records])
    if len(before)<5:raise ValueError('need_at_least_five_training_impacts')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('nonfinite impact velocity')
    def residual(x):
        predicted=before*np.array([x[1],x[1],-x[0]])
        return (predicted-after).ravel()
    fit=least_squares(residual,[.8,.8],bounds=([0.,0.],[1.,2.]),loss='soft_l1',
                      f_scale=.20,max_nfev=100,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    return {'e_n':float(fit.x[0]),'alpha_t':float(fit.x[1]),
            'converged':bool(fit.success),'nfev':int(fit.nfev),
            'at_parameter_bound':bool(fit.x[0]<1e-6 or fit.x[0]>1-1e-6 or fit.x[1]<1e-6 or fit.x[1]>2-1e-6)}


def _statistics(values):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    return None if not len(a) else {'count':len(a),'p50':float(np.median(a)),
                                    'p95':float(np.quantile(a,.95)),'max':float(a.max())}


def run_bounce(clean_dir,annotations_path,split_path,flight_dir,out_dir,*,seed=20261008):
    clean_dir,flight_dir,out_dir=map(Path,(clean_dir,flight_dir,out_dir))
    out_dir.mkdir(parents=True,exist_ok=False)
    annotations=json.loads(Path(annotations_path).read_text())
    split=json.loads(Path(split_path).read_text())
    assignments={s:g for g in ('train','validation','test') for s in split[g]}
    k=json.loads((flight_dir/'shared_parameters.json').read_text())['k_drag_per_m']
    flight={r['trajectory_id']:r for r in json.loads((flight_dir/'trajectory_fits.json').read_text())['trajectories']}
    records=[];excluded=[];traces={}
    for row in annotations:
        identity=row['trajectory_id']
        try:
            if row.get('first_bounce_t_s') is None:raise ValueError('no_bounce_annotation')
            t,p=_load_clean(_resolve_clean(clean_dir,row['cleaned_file']))
            traces[identity]=(t,p,row)
            base=impact_estimate(t,p,row,k)
            sensitivity=[]
            for window in (.08,.18):
                try:sensitivity.append(impact_estimate(t,p,row,k,window))
                except ValueError:pass
            e_values=[base['e_n_observed']]+[s['e_n_observed'] for s in sensitivity]
            alpha_values=[base['alpha_t_observed']]+[s['alpha_t_observed'] for s in sensitivity]
            base.update({'trajectory_id':identity,'split':assignments[identity],
                         'annotation_confidence':float(row.get('confidence',0)),
                         'window_e_span':float(np.ptp(e_values)),
                         'window_alpha_span':float(np.ptp(alpha_values)),
                         'sensitivity_windows_successful':len(sensitivity),
                         'sensitivity_windows':sensitivity,
                         'estimate_definition':'offline pre/post reconstruction; not a causal prediction'})
            reasons=[]
            if not base['converged']:reasons.append('local_optimizer_not_converged')
            if not sensitivity:reasons.append('window_sensitivity_not_available')
            if base['position_error']['rmse_m']>.08:reasons.append('position_reconstruction_error_above_80mm')
            if base['time_at_bound']:reasons.append('impact_time_at_bracket_bound')
            if base['window_e_span']>.18:reasons.append('normal_velocity_window_unstable')
            if base['window_alpha_span']>.12:reasons.append('tangential_velocity_window_unstable')
            if not .1<base['e_n_observed']<1.3:reasons.append('observed_normal_ratio_unusual')
            if base['annotation_confidence']<.55:reasons.append('annotation_low_confidence')
            base['quality_reasons']=reasons;records.append(base)
        except (ValueError,KeyError) as error:
            excluded.append({'trajectory_id':identity,'reason':str(error)})
    training=[r for r in records if r['split']=='train' and not r['quality_reasons']]
    if len(training)<5:raise ValueError('fewer_than_five_quality_training_impacts')
    heights=np.array([r['p_b_m'][2] for r in training]);h=float(np.median(heights))
    spread=max(.010,1.4826*float(np.median(abs(heights-h))))
    for r in records:
        if abs(r['p_b_m'][2]-h)>max(.04,4*spread):r['quality_reasons'].append('contact_height_inconsistent_with_training')
        r['used_for_fit']=r['split']=='train' and not r['quality_reasons']
    training=[r for r in records if r['used_for_fit']]
    law=fit_law(training);h=float(np.median([r['p_b_m'][2] for r in training]))
    if not law['converged']:raise RuntimeError('shared_bounce_optimization_did_not_converge')
    rng=np.random.default_rng(seed);bootstrap=[]
    # Resample inferred acquisition groups, preserving within-session dependence.
    session_map=split['trajectory_to_group'];groups=sorted(set(session_map[r['trajectory_id']] for r in training))
    for _ in range(200):
        selected=rng.choice(groups,len(groups),replace=True)
        sample=[r for group in selected for r in training if session_map[r['trajectory_id']]==group]
        try:
            fitted=fit_law(sample)
            if not fitted['converged']:continue
            bootstrap.append([fitted['e_n'],fitted['alpha_t'],np.median([r['p_b_m'][2] for r in sample])])
        except ValueError:pass
    boot=np.asarray(bootstrap)
    intervals={name:np.quantile(boot[:,i],[.025,.975]).tolist() for i,name in enumerate(('e_n','alpha_t','contact_height_m'))} if len(boot)>20 else {}
    metrics={};full=[]
    for group in ('train','validation','test'):
        velocity_errors=[];height_errors=[]
        eligible=[r for r in records if r['split']==group and not r['quality_reasons']]
        for r in eligible:
            before=np.array(r['v_minus_m_s']);actual=np.array(r['v_plus_m_s'])
            estimate=predict_velocity(before,law['e_n'],law['alpha_t'])
            err=float(np.linalg.norm(estimate-actual));velocity_errors.append(err)
            r['predicted_v_plus_given_measured_v_minus_m_s']=estimate.tolist()
            r['conditional_outgoing_velocity_error_m_s']=err
            height_errors.append(abs(r['p_b_m'][2]-h))
        metrics[group]={'impacts':len(eligible),
                        'conditional_velocity_error_m_s':_statistics(velocity_errors),
                        'contact_height_abs_difference_m':_statistics(height_errors),
                        'definition':'Bounce-law test conditional on offline measured incident velocity; not full causal forecast'}
    # Evaluate every held-out trajectory, including difficult/unreliable local
    # velocity labels. The comparison times depend on the observed annotation,
    # never on when the predictor claims that its own bounce happened.
    by_id={r['trajectory_id']:r for r in records}
    for identity in split['validation']+split['test']:
        group=assignments[identity]
        try:
            t,p,annotation=traces[identity]
            f=flight[identity]['drag_forecast']
            initial=np.r_[f['p0_m'],f['v0_m_s']]
            contact=predicted_contact(initial,k,ground_height_m=h-.0335,ball_radius_m=.0335)
            if not contact['available']:raise ValueError(contact['reason'])
            tb=float(f['t0_s']+contact['relative_time_s'])
            pb=np.array(contact['position_m']);vp=predict_velocity(np.array(contact['velocity_m_s']),law['e_n'],law['alpha_t'])
            use=(t>=float(annotation['post_start_t_s']))&(t<=float(annotation['post_end_t_s']))
            if use.sum()<8:raise ValueError('insufficient_observed_postimpact_evaluation_points')
            evaluation_times=t[use];prediction=np.empty((len(evaluation_times),3))
            pre=evaluation_times<tb
            if pre.any():prediction[pre]=simulate(evaluation_times[pre]-f['t0_s'],initial,k)[:,:3]
            if (~pre).any():prediction[~pre]=simulate(evaluation_times[~pre]-tb,np.r_[pb,vp],k)[:,:3]
            err=np.linalg.norm(prediction-p[use],axis=1)
            local=by_id.get(identity)
            reference_time=local['t_b_s'] if local else float(annotation['first_bounce_t_s'])
            reference_xy=np.array(local['p_b_m'][:2]) if local else np.array([
                robust_poly(t[abs(t-reference_time)<.025],p[abs(t-reference_time)<.025,j],reference_time,1)[0][0]
                for j in range(2)])
            full.append({'trajectory_id':identity,'split':group,'available':True,
                         'predicted_bounce_t_s':tb,'reference_bounce_t_s':reference_time,
                         'touchdown_time_error_s':abs(tb-reference_time),
                         'touchdown_xy_error_m':float(np.linalg.norm(pb[:2]-reference_xy)),
                         'reference_quality_reasons':local['quality_reasons'] if local else ['annotation_only_reference'],
                         'postimpact_position_error':error_stats(err),
                         'postimpact_time_s':evaluation_times.tolist(),'measured_position_m':p[use].tolist(),
                         'predicted_position_m':prediction.tolist(),
                         'prediction_definition':'training k/e/alpha/height; initial 80ms raw observations only; fixed observed evaluation horizon'})
        except (ValueError,KeyError,TypeError) as error:
            full.append({'trajectory_id':identity,'split':group,'available':False,'reason':str(error)})
    causal={}
    for group in ('validation','test'):
        available=[r for r in full if r['split']==group and r['available']]
        causal[group]={'available':len(available),'failed':sum(r['split']==group and not r['available'] for r in full),
                       'touchdown_time_error_s':_statistics([r['touchdown_time_error_s'] for r in available]),
                       'touchdown_xy_error_m':_statistics([r['touchdown_xy_error_m'] for r in available]),
                       'postimpact_per_trace_rmse_m':_statistics([r['postimpact_position_error']['rmse_m'] for r in available])}
    parameters={**law,'effective_contact_center_height_m':h,'bootstrap_95_percentile_intervals':intervals,
                'bootstrap_draws':len(boot),'bootstrap_unit':'inferred acquisition group',
                'bootstrap_training_groups':len(groups),
                'training_impacts':len(training),'fit_trajectory_ids':[r['trajectory_id'] for r in training],
                'map':'vplus=[alpha_t*vxminus,alpha_t*vyminus,-e_n*vzminus]',
                'alpha_t_is_friction_coefficient':False,'spin_observed':False,
                'ground_height_m':None,'ground_radius_bias_identifiability':'h_contact=ground+radius+measurement_bias; components unconfirmed',
                'status':'conditional_effective_calibration_pending_coordinate_and_label_confirmation'}
    report={'parameters':parameters,'metrics':metrics,'causal_full_flight_bounce_forecast':causal,
            'impact_estimates':records,'excluded':excluded,
            'input_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (annotations_path,split_path,flight_dir/'shared_parameters.json')},
            'limitations':['Automatic labels are provisional, not manually confirmed.',
                           'Full-segment fits estimate impact reference velocities; they are not forecasting results.',
                           'Unknown spin, wind and surface may limit a single shared e/alpha law.',
                           'Contact height is effective measured center height, not independently calibrated ground.']}
    (out_dir/'parameters.json').write_text(json.dumps(parameters,indent=2,allow_nan=False)+'\n')
    (out_dir/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out_dir/'full_predictions.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
    fields=['trajectory_id','split','t_b_s','e_n_observed','alpha_t_observed','window_e_span','window_alpha_span','used_for_fit','quality_reasons']
    with (out_dir/'impact_estimates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    return report
