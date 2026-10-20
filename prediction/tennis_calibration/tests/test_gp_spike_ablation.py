"""Diagnostic wrappers must not change the measured system under inspection."""
import unittest

import numpy as np

from tennis_calibration.gp_online import GPOnlinePredictor
from tennis_calibration.gp_spike_ablation import CollisionMapPredictor,curve,peak_statistics,paired_rows,safe_forecast
from tennis_calibration.identified_dynamics import impact_map


class SyntheticMapBank:
    parameters={'k_drag':0.,'e_n':.83,'alpha_t':.56,'impact_mode':'effective'}
    height=.03

    def apply_state(self,incoming,model):
        outgoing=impact_map(incoming,self.parameters)
        if model=='gp':
            outgoing[3]+=.8
        elif model!='fixed':
            raise ValueError(model)
        return outgoing


def seed(tracker):
    fit={'converged':True,'capture_t_s':.1,
         'state':np.array([0.,0.,.04,8.,.4,-2.,0.,0.,0.]),'covariance':np.eye(9)*1e-5}
    times=np.array([.06,.08,.1])
    xyz=fit['state'][None,:3]+(times-.1)[:,None]*fit['state'][None,3:6]
    tracker.seed_from_prefix(fit,times,xyz)


class GPSpikeAblationTests(unittest.TestCase):
    def test_instrumentation_preserves_gp_state_covariance_and_predictions(self):
        bank=SyntheticMapBank()
        normal=GPOnlinePredictor(bank)
        instrumented=CollisionMapPredictor(bank,'gp',normal.config)
        seed(normal);seed(instrumented)
        for stamp in (.105,.11,.115,.12,.125):
            position=normal.predict(stamp)['p']+np.array([.001,-.0005,.0003])
            a,b=normal.update(stamp,position),instrumented.update(stamp,position)
            instrumented.frame_diagnostics(position,b)
            np.testing.assert_array_equal(a['state'],b['state'])
            np.testing.assert_array_equal(a['covariance'],b['covariance'])
            np.testing.assert_array_equal(normal.predict_target_times([stamp+.1,stamp+.2]),
                                          instrumented.predict_target_times([stamp+.1,stamp+.2]))

    def test_map_swap_keeps_mean_covariance_and_filter_initialization_identical(self):
        bank=SyntheticMapBank()
        fixed=CollisionMapPredictor(bank,'fixed',{})
        gp=CollisionMapPredictor(bank,'gp',{})
        seed(fixed);seed(gp)
        np.testing.assert_array_equal(fixed.state,gp.state)
        np.testing.assert_array_equal(fixed.covariance,gp.covariance)
        self.assertEqual(set(fixed.config),set(gp.config))
        for key in fixed.config:
            np.testing.assert_array_equal(fixed.config[key],gp.config[key])
        self.assertGreater(np.linalg.norm(fixed.predict(.2)['state']-gp.predict(.2)['state']),.1)

    def test_equal_trajectory_statistics_do_not_treat_dense_rows_as_more_events(self):
        def row(identity,error):
            return {'model':'gp','trajectory_id':identity,'relative_capture_s':.02,
                    'horizon_s':.5,'twenty_ms_anchor':True,'valid':True,
                    'target_xyz':[0.,0.,0.],'predicted_xyz':[error,0.,0.]}
        rows=[row('one',1.),row('two',3.)]
        replicated=[row('one',1.)]*10+[row('two',3.)]
        self.assertEqual(curve(rows)[0]['rmse_m'],curve(replicated)[0]['rmse_m'])
        self.assertEqual(peak_statistics(rows)[0]['independent_trajectories'],2)
        self.assertEqual(peak_statistics(rows)[0]['trajectory_peak_p95_m'],
                         peak_statistics(replicated)[0]['trajectory_peak_p95_m'])

    def test_failed_labeled_predictions_are_counted_and_partial_peaks_not_scored(self):
        def row(identity,predicted,capture=.02):
            return {'model':'gp','trajectory_id':identity,'relative_capture_s':capture,
                    'horizon_s':.5,'twenty_ms_anchor':True,'valid':True,
                    'target_xyz':[0.,0.,0.],'predicted_xyz':predicted,
                    'prediction_reason':'forecast failed' if predicted is None else ''}
        rows=[row('one',[1.,0.,0.]),row('one',None,.021),row('two',[3.,0.,0.])]
        cell=curve(rows)[0]
        self.assertEqual(cell['requested_trajectories'],2)
        self.assertEqual(cell['failed_labeled_rows'],1)
        peak=peak_statistics(rows)[0]
        self.assertEqual(peak['requested_trajectories'],2)
        self.assertEqual(peak['independent_trajectories'],1)
        self.assertEqual(peak['incomplete_prediction_trajectory_ids'],['one'])

    def test_paired_comparison_masks_other_model_failure_without_dropping_requests(self):
        common={'trajectory_id':'one','capture_t_s':1.,'target_t_s':1.5,
                'relative_capture_s':.02,'horizon_s':.5,'twenty_ms_anchor':True,
                'valid':True,'target_xyz':[0.,0.,0.]}
        rows=[dict(common,model='fixed',predicted_xyz=[1.,0.,0.],prediction_reason=''),
              dict(common,model='gp',predicted_xyz=None,prediction_reason='forecast failed')]
        paired=paired_rows(rows,['fixed','gp'])
        self.assertEqual(len(paired),2)
        self.assertTrue(all(r['predicted_xyz'] is None for r in paired))
        self.assertTrue(all(c['requested_trajectories']==1 and c['successful_trajectories']==0 for c in curve(paired)))
        with self.assertRaisesRegex(RuntimeError,'same clock grid'):
            paired_rows(rows[:1],['fixed','gp'])

    def test_nonfinite_forecast_returns_explicit_failure(self):
        class BadProjector:
            def predict(self,_):return {'state':np.full(9,np.nan)}
        forecast,reason=safe_forecast(BadProjector(),[1.])
        self.assertIsNone(forecast)
        self.assertIn('finite H x 6/9',reason)


if __name__=='__main__':unittest.main()
