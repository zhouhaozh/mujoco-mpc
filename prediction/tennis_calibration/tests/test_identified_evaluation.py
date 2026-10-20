"""Synthetic checks of strict label pairing and fair coverage reporting."""
import unittest

import numpy as np

from tennis_calibration.identified_evaluation import (
    assert_fit_uses_training_only, compare_model_rows, paired_comparison,
)


def position(model, t=.1, predicted=(.1, 0., 0.), **kwargs):
    row = dict(model=model, split='test', trajectory_id='a', capture_t_s=t,
               anchor_t_s=t, horizon_s=.2, scenario_delay_s=0., target_t_s=t+.2,
               target_xyz=[0., 0., 0.], predicted_xyz=predicted,
               phase='pre_bounce', valid=True, reason='', prediction_reason='')
    return dict(row, **kwargs)


def landing(model, t=.1, predicted=(.1, 0.), **kwargs):
    row = dict(model=model, split='test', trajectory_id='a', capture_t_s=t,
               anchor_t_s=t, scenario_delay_s=0., target_time_s=.5,
               predicted_time_s=.52, target_xy=[0., 0.], predicted_xy=predicted,
               prediction_reason='')
    return dict(row, **kwargs)


class IdentifiedEvaluationTests(unittest.TestCase):
    def test_identical_labels_and_exact_common_success_metrics(self):
        rows = [position('old'), position('new', predicted=(.3, .4, 0.))]
        report = compare_model_rows(rows)
        self.assertEqual(report['n_common_keys'], 1)
        self.assertEqual(report['n_common_success'], 1)
        self.assertEqual(report['n_label_mismatches'], 0)
        self.assertAlmostEqual(report['common_success_models']['new']['position_error']['rmse_m'], .5)

    def test_prediction_failure_stays_in_coverage_and_shared_subset(self):
        rows = [position('old'), position('new'), position('old', .2),
                position('new', .2, predicted=None, prediction_reason='uninitialized')]
        report = compare_model_rows(rows)
        self.assertEqual(report['models']['new']['prediction_coverage_given_label'], .5)
        self.assertEqual(report['models']['old']['prediction_coverage_given_label'], 1.)
        self.assertEqual(report['n_common_success'], 1)
        self.assertEqual(report['common_success_models']['old']['n_evaluated'], 1)

    def test_key_missing_is_not_silent_subsetting(self):
        rows = [position('old'), position('new'), position('old', .2)]
        with self.assertRaisesRegex(ValueError, 'different anchor'):
            compare_model_rows(rows)
        report = compare_model_rows(rows, strict_keys=False)
        self.assertEqual(report['missing_keys_by_model']['new'], 1)

    def test_changed_target_or_phase_rejected_even_for_failed_output(self):
        for change in [dict(target_xyz=[0., 0., .001]), dict(phase='cross_bounce'),
                       dict(target_t_s=.31), dict(valid=False), dict(reason='gap')]:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, 'different evaluation labels'):
                    compare_model_rows([position('old'), position('new', predicted=None, **change)])

    def test_duplicate_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            compare_model_rows([position('old'), position('old')])

    def test_unavailable_ground_truth_stays_in_requested_count(self):
        rows = [position(model, target_xyz=None, valid=False, reason='label_gap') for model in ('old', 'new')]
        report = compare_model_rows(rows)
        self.assertEqual(report['n_common_keys'], 1)
        self.assertEqual(report['n_common_success'], 0)
        self.assertEqual(report['models']['old']['n_requested'], 1)
        self.assertEqual(report['models']['old']['n_missing_labels'], 1)

    def test_landings_exclude_after_first_bounce_and_keep_failure(self):
        rows = [landing('old'), landing('new', predicted=(.3, .4)),
                landing('old', .4), landing('new', .4, predicted=None, predicted_time_s=None),
                landing('old', .6), landing('new', .6)]
        report = compare_model_rows(rows, landing=True)
        self.assertEqual(report['n_common_keys'], 3)
        self.assertEqual(report['n_common_success'], 1)
        self.assertEqual(report['models']['new']['prediction_coverage_given_label'], .5)
        self.assertEqual(report['models']['new']['n_post_first_bounce_excluded'], 1)
        self.assertAlmostEqual(report['common_success_models']['new']['landing_xy_error']['rmse_m'], .5)

    def test_landing_reference_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different evaluation labels'):
            compare_model_rows([landing('old'), landing('new', target_time_s=.501)], landing=True)

    def test_report_separates_split_delay_and_calls_test_regression(self):
        rows = [position(model, split=split) for model in ('old', 'new') for split in ('validation', 'test')]
        rows += [position(model, scenario_delay_s=.01, anchor_t_s=.11, target_t_s=.31)
                 for model in ('old', 'new')]
        report = paired_comparison(rows, [])
        self.assertEqual(len(report['scenarios']), 3)
        self.assertIn('historical test already exposed', report['evaluation_role'])

    def test_global_physics_fit_rejects_validation_test_or_unknown_ids(self):
        frozen = dict(train=['train1', 'train2'], validation=['val'], test=['test'])
        self.assertFalse(assert_fit_uses_training_only(['train1'], frozen)['validation_or_test_in_fit'])
        for ids in ([], ['train1', 'val'], ['test'], ['unknown']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                assert_fit_uses_training_only(ids, frozen)


if __name__ == '__main__':
    unittest.main()
