"""Invariants for hold duration comparisons; run numerical checks on Ubuntu."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration import hold_sweep
from tennis_calibration.tests.test_height_policy import SyntheticBank, HEIGHT


class HoldSweepTests(unittest.TestCase):
    def seed(self,hold_ms):
        tracker=hold_sweep.make_tracker(SyntheticBank(),None,hold_ms)
        state=np.array([0.,0.,HEIGHT+.01,3.,.2,-2.,0.,0.,0.])
        times=np.array([.06,.08,.1]);xyz=state[:3]+(times-.1)[:,None]*state[3:6]
        tracker.seed_from_prefix({'state':state,'covariance':np.eye(9)*1e-4,
            'capture_t_s':.1,'converged':True},times,xyz)
        return tracker

    def test_zero_is_height_matching_online_not_early_confirmation(self):
        zero=self.seed(0)
        self.assertEqual(zero.policy,'height_online')
        with patch('tennis_calibration.identified_filter.IdentifiedFilter._confirm_bounce',
                   side_effect=AssertionError('cannot confirm before propagated height event')):
            self.assertFalse(zero._confirm_bounce())
        output=zero.update(.11,[.03,.002,.025])
        self.assertTrue(output['height_triggered'])
        self.assertFalse(output['hold_active'])
        self.assertIsNone(output['resume_t_s'])
        self.assertNotEqual(output['reason'],'height_policy_prediction_only')

    def test_durations_share_height_event_and_resume_first_eligible_raw_frame(self):
        times=np.arange(.1025,.2301,.0025)
        trackers=[self.seed(h) for h in hold_sweep.HOLDS_MS]
        for timestamp in times:
            # Identical causal inputs, including intentionally imperfect data.
            point=np.array([3*(timestamp-.1),.2*(timestamp-.1),.04+1.5*(timestamp-.1)])
            for tracker in trackers:tracker.update(float(timestamp),point)
        events=[t.height_contact_t_s for t in trackers]
        np.testing.assert_array_equal(events,np.full(len(events),events[0]))
        for h,t in zip(hold_sweep.HOLDS_MS,trackers):
            if h:
                expected=hold_sweep.expected_resume(times,t.height_contact_t_s,t.height_discovery_t_s,h)
                self.assertAlmostEqual(t.resume_t_s,expected,places=12)
            else:self.assertIsNone(t.resume_t_s)

    def test_short_hold_measurements_do_not_assimilate_until_exact_boundary(self):
        left,right=self.seed(20),self.seed(20)
        left.update(.11,[.03,.002,.025]);right.update(.11,[.13,.102,.025])
        np.testing.assert_array_equal(left.state,right.state)
        np.testing.assert_array_equal(left.covariance,right.covariance)
        # Hold's reference is the localized contact, not discovery frame +20ms.
        boundary=left.height_contact_t_s+.02
        self.assertLess(boundary,.13)
        for t in (.115,.12,boundary-1e-5):
            xyz=left.predict_target_times([t])[0,:3]
            a,b=left.update(t,xyz),right.update(t,xyz+[.03,.02,.01])
            self.assertTrue(a['hold_active'])
            np.testing.assert_array_equal(a['state'],b['state'])
        xyz=left.predict_target_times([boundary])[0,:3]
        now=left.update(boundary,xyz)
        self.assertFalse(now['hold_active']);self.assertAlmostEqual(now['resume_t_s'],boundary)
        self.assertNotEqual(now['reason'],'height_policy_prediction_only')

    def test_expected_resume_handles_irregular_stream_and_truncated_tail(self):
        times=np.array([.1,.117,.131,.16])
        self.assertAlmostEqual(hold_sweep.expected_resume(times,.108,.117,20),.131)
        self.assertAlmostEqual(hold_sweep.expected_resume(times,.108,.117,0),.117)
        self.assertIsNone(hold_sweep.expected_resume(times,.108,.117,100))

    def test_group_selection_has_no_leftout_or_training_influence(self):
        cases=[]
        for group,best in (('A',20),('B',20),('C',100),('D',100)):
            for h in (20,100):
                for i in range(2 if group=='A' else 1):
                    cases.append({'split':'validation','groupid':group,'candidate':hold_sweep.candidate(h),
                        'trajectory_id':f'{group}{i}','balanced_future_mse_m2':1. if h==best else 4.})
        before=hold_sweep.leave_one_group_selection(cases)
        modified=deepcopy(cases)
        for row in modified:
            if row['groupid']=='A':row['balanced_future_mse_m2']*=1e6
        modified.append({'split':'train','groupid':'E','candidate':hold_sweep.candidate(20),
                         'trajectory_id':'E0','balanced_future_mse_m2':1e12})
        after=hold_sweep.leave_one_group_selection(modified)
        for a,b in zip([x for x in before if x['left_out_group']=='A'],[x for x in after if x['left_out_group']=='A']):
            self.assertEqual(a['selection_mse_m2_by_candidate'],b['selection_mse_m2_by_candidate'])
            self.assertEqual(a['selected_candidate'],b['selected_candidate'])
            self.assertAlmostEqual(a['left_out_rmse_m']*1e3,b['left_out_rmse_m'])
        self.assertTrue(all('E' not in x['selection_groups'] for x in after))

    def test_metrics_weight_trajectories_not_frame_counts(self):
        rows=[];events=[]
        for identity,count,error in (('short',1,1.),('long',20,3.)):
            events.append({'trajectory_id':identity,'split':'validation','groupid':'g',
                           'candidate':'gp_hold100ms','diagnostic_resume_or_event_t_s':.1})
            for index in range(count):
                t=.12+index*.001
                rows.append({'trajectory_id':identity,'split':'validation','groupid':'g','candidate':'gp_hold100ms',
                    'capture_t_s':t,'target_t_s':t+.5,'elapsed_capture_s':t,'relative_capture_s':t,
                    'horizon_s':.5,'valid':True,'prediction_reason':'','predicted_xyz':[error,0,0],
                    'target_xyz':[0,0,0]})
        result=hold_sweep.aggregate(rows,events)
        cell=result['phase_metrics'][0]
        self.assertEqual(cell['N'],2);self.assertAlmostEqual(cell['mean_m'],2.)
        self.assertAlmostEqual(cell['rmse_m'],np.sqrt(5.));self.assertEqual(cell['p95_frame_m'],3.)
        self.assertAlmostEqual(result['ranking'][0]['balanced_future_rmse_m'],np.sqrt(5.))
        self.assertTrue(result['successful_key_coverage'][0]['equal_to_reference'])

    def test_illegal_durations_rejected(self):
        for h in (-1,np.nan,np.inf):
            with self.assertRaises(ValueError):hold_sweep.make_tracker(SyntheticBank(),None,h)


if __name__=='__main__':unittest.main()
