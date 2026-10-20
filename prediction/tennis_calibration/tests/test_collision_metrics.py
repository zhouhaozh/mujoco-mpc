import unittest

from tennis_calibration.collision_metrics import aggregate_collision_metrics, summarize_position_rows, summarize_event_rows


class CollisionMetricsTests(unittest.TestCase):
    def row(self, identity="a", model="x", prediction=(1, 0, 0), horizon=.1):
        return dict(split="test", model=model, trajectory_id=identity, groupid=0, gap_s=0.,
            horizon_s=horizon, target_t_s=1.+horizon, target_xyz=[0., 0., 0.], predicted_xyz=prediction, valid=True)

    def test_macro_does_not_weight_longer_trajectory_more(self):
        rows = [self.row(horizon=.1), self.row(horizon=.2), self.row("b", prediction=(3, 0, 0))]
        score = summarize_position_rows(rows)
        self.assertEqual(score["trajectory_macro"]["mean_per_trajectory_rmse_m"], 2.)
        self.assertAlmostEqual(score["position_error"]["rmse_m"], (11/3)**.5)

    def test_failure_and_missing_label_stay_in_coverage(self):
        rows = [self.row(), self.row("b", prediction=None), {**self.row("c"), "valid": False, "target_xyz": None}]
        score = summarize_position_rows(rows)
        self.assertEqual((score["n_requested"], score["n_labeled"], score["n_evaluated"], score["n_prediction_failures"]), (3, 2, 1, 1))
        self.assertEqual(score["prediction_coverage_given_label"], .5)

    def test_shared_success_excludes_failure_in_both_models(self):
        rows = [self.row(model="x"), self.row(model="y"), self.row("b", model="x"), self.row("b", model="y", prediction=None)]
        result = aggregate_collision_metrics(rows)
        cells = result["comparison"]["common_success_position"]["cells"]
        self.assertEqual([cell["n_evaluated"] for cell in cells], [1, 1])
        self.assertEqual(result["comparison"]["label_grid"]["test"]["n_unique_collision_events"], 2)

    def test_different_label_clocks_rejected(self):
        with self.assertRaisesRegex(ValueError, "Different requested"):
            aggregate_collision_metrics([self.row(model="x"), self.row(model="y", horizon=.2)])

    def test_gap_variants_are_one_event_and_time_is_absolute_error(self):
        row = dict(split="test", model="x", trajectory_id="a", gap_s=0., valid=True,
            v_minus_ref=[1., 0., -1.], v_plus_ref=[.5, 0., 1.], pb_ref=[0., 0., 0.], tb_ref=1.,
            v_minus_pred=[1., 0., -1.], v_plus_pred=[.5, 0., 2.], pb_pred=[3., 4., 2.], tb_pred=.99)
        score = summarize_event_rows([row, {**row, "gap_s": .02}])
        self.assertEqual(score["n_unique_collision_events_requested"], 1)
        self.assertEqual(score["v_plus_error"]["rmse_m_s"], 1.)
        self.assertEqual(score["impact_xy_error"]["rmse_m"], 5.)
        self.assertAlmostEqual(score["impact_time_abs_error"]["rmse_s"], .01)


if __name__ == "__main__":
    unittest.main()
