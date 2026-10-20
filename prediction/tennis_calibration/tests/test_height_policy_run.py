"""Causal replay, failure isolation and independent-event selection checks."""
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration import height_policy_run as run
from tennis_calibration.identified_dynamics import impact_map
from tennis_calibration.identified_filter import DEFAULT_CONFIG,forecast_states_batch


class Bank:
    parameters={'k_drag':0.,'e_n':.83,'alpha_t':.56,'impact_mode':'effective'}
    height=.03
    def apply_state(self,state,model):return impact_map(state,self.parameters)


def trace():
    times=np.arange(0,.3501,.0025)
    states=forecast_states_batch(np.array([[0.,0.,.3,3.,.1,-1.,0.,0.,0.]]),np.array([False]),times,Bank.parameters,Bank.height)[0]
    return {'id':'synthetic','split':'validation','raw_times':times,'raw_xyz':states[:,:3].copy(),
            'eval_times':times,'eval_xyz':states[:,:3].copy(),'bounce_reference_s':.155,
            'bounce_event_times_s':[.155],'eval_scope_end_s':.35}


def event(candidate,split='validation',identity='one',triggered=True):
    model,policy=candidate.split(':')
    return {'trajectory_id':identity,'split':split,'candidate':candidate,'model':model,'policy':policy,
            'height_triggered':triggered,'height_event_status':'test',
            'height_contact_t_s':1. if triggered else None,'reference_contact_t_s':1.,'resume_t_s':1.1 if triggered else None}


def row(candidate,error,*,split='validation',identity='one',tau=.02,h=.5):
    return {'trajectory_id':identity,'candidate':candidate,'split':split,'horizon_s':h,
            'elapsed_capture_s':.5,'relative_capture_s':tau,'twenty_ms_anchor':True,
            'valid':True,'prediction_reason':'' if error is not None else 'failure',
            'target_xyz':[0.,0.,0.],'predicted_xyz':[error,0.,0.] if error is not None else None}


class HeightReplayTests(unittest.TestCase):
    def setUp(self):
        run.BANK=Bank();run.SELECTED={'filter':dict(DEFAULT_CONFIG)}

    def test_future_observations_and_reference_labels_cannot_change_past_forecasts(self):
        a=trace();b=trace();b['raw_xyz'][b['raw_times']>.25]+=np.array([5.,-3.,0.])
        first,_=run._run_case(a,'gp','hold100')
        future,_=run._run_case(b,'gp','hold100')
        c=trace();c['bounce_reference_s']+=.06;c['bounce_event_times_s']=[c['bounce_reference_s']]
        relabeled,_=run._run_case(c,'gp','hold100')
        def predictions(rows,stop):return {(r['capture_t_s'],r['horizon_s']):r['predicted_xyz'] for r in rows if r['capture_t_s']<=stop}
        self.assertEqual(predictions(first,.2),predictions(future,.2))
        self.assertEqual(predictions(first,.35),predictions(relabeled,.35))
        self.assertTrue(all(r['predicted_xyz'] is None for r in first if r['capture_t_s']<.08-1e-12))

    def test_later_batch_failure_does_not_erase_earlier_forecasts(self):
        def failing(states,bounced,capture_times,**_):
            if len(states)>1 or capture_times[0]>.2:raise ValueError('late invalid posterior')
            return np.repeat(states[:,None,:],4,axis=1)
        with patch.object(run,'forecast_posteriors',side_effect=failing):rows,_=run._run_case(trace(),'gp','hold100')
        early=[r for r in rows if .1<=r['capture_t_s']<=.2]
        late=[r for r in rows if r['capture_t_s']>.2]
        self.assertTrue(all(r['predicted_xyz'] is not None and not r['prediction_reason'] for r in early))
        self.assertTrue(all(r['predicted_xyz'] is None and r['prediction_reason'] for r in late))

    def test_phases_are_equal_weight_even_if_horizon_coverage_differs(self):
        candidate='gp:hold100';acc=run.Accumulator()
        rows=[row(candidate,1.,tau=-.1,h=h) for h in (.1,.2,.5)]
        rows.append(row(candidate,3.,tau=.02,h=.1))
        acc.add(rows,event(candidate))
        self.assertAlmostEqual(acc.cases[0]['balanced_future_mse_m2'],5.)

    def test_selection_uses_validation_only_and_counts_missing_height_events(self):
        acc=run.Accumulator()
        for split in ('validation','test'):
            for candidate,error,triggered in [('fixed:hold100',1. if split=='validation' else 9.,True),
                                               ('gp:hold100',2. if split=='validation' else .1,True),
                                               ('ridge1:hold100',.01,False)]:
                acc.add([row(candidate,error,split=split)],event(candidate,split,triggered=triggered))
        summary=acc.summary()
        self.assertEqual(summary['validation_only_selection']['overall'],'fixed:hold100')
        self.assertEqual(summary['validation_only_selection']['per_policy']['hold100'],'fixed:hold100')

    def test_dense_duplicates_do_not_increase_independent_event_weight(self):
        candidate='gp:hold100';acc=run.Accumulator()
        acc.add([row(candidate,1.)]*10,event(candidate,identity='one'))
        acc.add([row(candidate,3.,identity='two')],event(candidate,identity='two'))
        cell=next(c for c in acc.summary()['curves'] if c['axis']=='contact20')
        self.assertEqual(cell['requested_trajectories'],2)
        self.assertAlmostEqual(cell['rmse_m'],np.sqrt(5.))


if __name__=='__main__':unittest.main()
