"""Synthetic leakage, label-coverage and exact-horizon tests for replay data."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tennis_calibration.online_data import (
    aggregate_landings, aggregate_predictions, causal_anchors, future_labels,
    history_features, load_dataset,
)


class OnlineDataTests(unittest.TestCase):
    def test_raw_does_not_consult_offline_keep_or_bounce_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'clean/cleaned').mkdir(parents=True)
            (root/'bounce').mkdir()
            identity = 'trajectory_a'
            with (root/'clean/cleaned/a.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['t_s', 'x_m', 'y_m', 'z_m', 'keep'])
                writer.writeheader()
                for t, x, keep in [(0., 0., True), (.002, 20., False),
                                   (.002, 3., True), (.001, 4., True),
                                   (.004, 5., True), (.006, np.nan, True)]:
                    writer.writerow(dict(t_s=t, x_m=x, y_m=0., z_m=1., keep=keep))
            (root/'clean/annotations.json').write_text(json.dumps([
                dict(trajectory_id=identity, cleaned_file='cleaned/a.csv',
                     first_bounce_t_s=.003, post_end_t_s=.004,
                     events=[dict(time_s=.003)])]))
            (root/'split.json').write_text(json.dumps(dict(
                train=[identity], validation=[], test=[],
                trajectory_to_group={identity: 1})))
            (root/'bounce/summary.json').write_text(json.dumps(dict(impact_estimates=[
                dict(trajectory_id=identity, t_b_s=.0031, p_b_m=[3., 2., .02])
            ])))
            trace = load_dataset(root)[0]
            np.testing.assert_equal(trace['raw_times'], [0., .002, .004])
            np.testing.assert_equal(trace['raw_xyz'][:, 0], [0., 20., 5.])
            np.testing.assert_equal(trace['eval_times'], [0., .004])
            self.assertEqual(trace['bounce_reference_s'], .0031)
            np.testing.assert_equal(trace['bounce_reference_xy'], [3., 2.])
            self.assertEqual(trace['eval_scope_end_s'], .004)

    def test_frozen_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'clean').mkdir()
            (root/'clean/annotations.json').write_text('[]')
            (root/'split.json').write_text(json.dumps(dict(train=['a'], validation=[], test=['a'])))
            with self.assertRaisesRegex(ValueError, 'duplicate split'):
                load_dataset(root)

    def test_anchors_are_prefix_invariant_with_gaps(self):
        times = np.asarray([0., .04, .08, .10, .13, .30, .32])
        complete = causal_anchors({'raw_times': times})
        partial = causal_anchors({'raw_times': times[:5]})
        self.assertEqual(complete[:len(partial)], partial)
        self.assertEqual([r['current_index'] for r in complete], [2, 3, 4, 5, 6])
        self.assertEqual(causal_anchors({'raw_times': times[:2]}), [])

    def test_history_has_no_future_leakage_and_is_translation_invariant(self):
        times = np.arange(101)*.0025
        xyz = np.column_stack([2*times, -times, 1+.2*times])
        index = 40
        history = history_features(times, xyz, index)
        changed = xyz.copy()
        changed[index+1:] = 1e6
        np.testing.assert_equal(history['features'], history_features(times, changed, index)['features'])
        np.testing.assert_allclose(history['features'],
                                   history_features(times, xyz+[3., 4., 5.], index)['features'], atol=1e-12)
        self.assertEqual(len(history['features']), 82)
        self.assertTrue(history['history_complete'])
        np.testing.assert_allclose(history['relative_xyz'][-1], [0., 0., 0.])
        np.testing.assert_allclose(history['relative_xyz'][0], [-.16, .08, -.016])

    def test_history_masks_early_missing_or_long_gap(self):
        times = np.asarray([0., .002, .004, .04])
        xyz = np.column_stack([times, times, times])
        early = history_features(times, xyz, 2)
        self.assertFalse(early['history_complete'])
        self.assertFalse(early['mask'][0])
        np.testing.assert_equal(early['relative_xyz'][0], 0.)
        with_gap = history_features(times, xyz, 3)
        self.assertFalse(with_gap['history_complete'])
        self.assertTrue(np.any(with_gap['bracket_gaps_s'] > .012))

    def test_fixed_future_time_interpolation_and_missingness(self):
        times = np.asarray([0., .005, .01, .025, .03])
        trace = dict(eval_times=times, eval_xyz=np.column_stack([times, 2*times, 3*times]))
        labels = future_labels(trace, 0., horizons=[.007, .02, .04])
        self.assertTrue(labels[0]['valid'])
        self.assertEqual(labels[0]['target_t_s'], .007)
        np.testing.assert_allclose(labels[0]['target_xyz'], [.007, .014, .021])
        self.assertEqual(labels[1]['reason'], 'label_gap')
        self.assertEqual(labels[2]['reason'], 'outside_observed_range')

    def test_offline_labels_cannot_bridge_bounce_or_score_contact_neighborhood(self):
        times = np.asarray([.09, .095, .105, .11])
        trace = dict(eval_times=times, eval_xyz=np.column_stack([times, times, times]),
                     raw_times=times.copy(), bounce_reference_s=.1)
        labels = future_labels(trace, .08, horizons=[.02, .0155, .03])
        self.assertEqual(labels[0]['reason'], 'bounce_neighborhood')
        self.assertEqual(labels[1]['reason'], 'bounce_bracket')
        self.assertEqual(labels[2]['phase'], 'cross_bounce')
        self.assertTrue(labels[2]['valid'])
        np.testing.assert_equal(trace['raw_times'], times)
        trace['eval_scope_end_s'] = .107
        self.assertEqual(future_labels(trace, .08, [.03])[0]['reason'], 'beyond_first_bounce_scope')

    def test_error_metrics_include_failures_and_trajectory_macro(self):
        rows = []
        for i in range(3):
            rows.append(dict(trajectory_id='many', horizon_s=.1, phase='pre_bounce',
                             target_xyz=[0., 0., 0.], predicted_xyz=[1., 0., 0.]))
        rows.append(dict(trajectory_id='one', horizon_s=.2, phase='cross_bounce',
                         target_xyz=[0., 0., 0.], predicted_xyz=[3., 0., 0.]))
        rows.append(dict(trajectory_id='one', horizon_s=.2, phase='cross_bounce',
                         target_xyz=[0., 0., 0.], predicted_xyz=None, prediction_reason='uninitialized'))
        rows.append(dict(trajectory_id='one', horizon_s=.5, phase='post_bounce',
                         target_xyz=None, predicted_xyz=[0., 0., 0.], valid=False, reason='label_gap'))
        report = aggregate_predictions(rows)
        self.assertEqual(report['n_labeled'], 5)
        self.assertEqual(report['n_evaluated'], 4)
        self.assertEqual(report['n_prediction_failures'], 1)
        self.assertEqual(report['n_missing_labels'], 1)
        self.assertEqual(report['prediction_coverage_given_label'], .8)
        self.assertAlmostEqual(report['position_error']['rmse_m'], np.sqrt(3))
        self.assertEqual(report['trajectory_macro']['mean_per_trajectory_rmse_m'], 2.)
        self.assertEqual(report['by_horizon_s']['0.2']['n_prediction_failures'], 1)

    def test_first_landing_excludes_post_bounce_and_scores_exact_time(self):
        rows = [dict(anchor_t_s=.2, target_time_s=.5, predicted_time_s=.52,
                     target_xy=[0., 0.], predicted_xy=[.3, .4]),
                dict(anchor_t_s=.4, target_time_s=.5, predicted_time_s=None,
                     target_xy=[0., 0.], predicted_xy=None),
                dict(anchor_t_s=.51, target_time_s=.5, predicted_time_s=.9,
                     target_xy=[0., 0.], predicted_xy=[8., 9.])]
        report = aggregate_landings(rows)
        self.assertEqual(report['n_requested'], 2)
        self.assertEqual(report['n_post_first_bounce_excluded'], 1)
        self.assertEqual(report['n_evaluated'], 1)
        self.assertEqual(report['landing_xy_error']['rmse_m'], .5)
        self.assertAlmostEqual(report['time_error']['rmse_s'], .02)


if __name__ == '__main__':
    unittest.main()
