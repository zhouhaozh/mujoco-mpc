"""Causal ingestion and replay-clock invariants of the live viewer.

All trajectory labels in these fixtures are deliberately separate from the
raw observation stream. Model tests use a small synthetic physical bundle;
no measured tennis data are fitted by these tests.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tennis_calibration.flight import simulate
from tennis_calibration.online_features import FEATURE_SCHEMA
from tennis_calibration.online_predictor import OnlinePredictor
from tennis_calibration.viewer import LiveViewerPredictor, ViewerReplayEngine, plain


def make_bundle(path):
    config = {
        "physical": {"k_drag": 0.02, "e_n": 0.83, "alpha_t": 0.56,
                     "contact_height": 0.03},
        "filter": {}, "augmented": True,
        "position_mlp_enabled": False, "landing_mlp_enabled": False,
        "supported_horizons_s": [0.1, 0.2, 0.5],
        "trained_delays_s": [0.0], "feature_schema": FEATURE_SCHEMA,
    }
    Path(path, "config.json").write_text(json.dumps(config), encoding="utf-8")


def observations():
    times = np.arange(0.0, 0.16 + 1e-9, 0.0025)
    positions = simulate(times, np.array([0.0, 0.0, 3.0, 12.0, 1.0, 1.0]), 0.02)[:, :3]
    # A large measurement anomaly remains a model input, for causal gating.
    positions[48] += np.array([0.7, -0.4, 0.5])
    return times, positions


class ViewerPredictorTests(unittest.TestCase):
    def test_observe_ingestion_matches_public_update_without_predict_work(self):
        with tempfile.TemporaryDirectory() as directory:
            make_bundle(directory)
            live, original = LiveViewerPredictor(directory), OnlinePredictor(directory)
            times, positions = observations()
            with patch.object(live, "predict", side_effect=AssertionError("observe must not forecast")):
                for stamp, xyz in zip(times, positions):
                    live.observe(float(stamp), xyz)
                    original.update(float(stamp), xyz, horizons=[])
            a, b = live.filter.snapshot(), original.filter.snapshot()
            self.assertEqual(a["t_s"], b["t_s"])
            self.assertEqual(a["accepted"], b["accepted"])
            self.assertEqual(a["reason"], b["reason"])
            np.testing.assert_array_equal(a["state"], b["state"])
            np.testing.assert_array_equal(a["covariance"], b["covariance"])
            self.assertEqual(len(live._history), len(original._history))
            for expected, actual in zip(original._history, live._history):
                self.assertEqual(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])

    def test_forecast_without_new_observation_preserves_capture_state(self):
        with tempfile.TemporaryDirectory() as directory:
            make_bundle(directory)
            live = LiveViewerPredictor(directory)
            for stamp, xyz in zip(*observations()):
                live.observe(float(stamp), xyz)
            before = copy.deepcopy(live.filter.snapshot())
            history = [(t, xyz.copy()) for t, xyz in live._history]
            prediction = live.predict(now_t_s=before["t_s"] + 0.03)
            self.assertTrue(prediction["initialized"])
            self.assertAlmostEqual(prediction["diagnostics"]["delay_s"], 0.03)
            after = live.filter.snapshot()
            self.assertEqual(after["t_s"], before["t_s"])
            np.testing.assert_array_equal(after["state"], before["state"])
            np.testing.assert_array_equal(after["covariance"], before["covariance"])
            for expected, actual in zip(history, live._history):
                self.assertEqual(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])

    def test_invalid_timestamp_never_enters_history_and_reset_clears_state(self):
        with tempfile.TemporaryDirectory() as directory:
            make_bundle(directory)
            live = LiveViewerPredictor(directory)
            for stamp, xyz in zip(*observations()):
                live.observe(float(stamp), xyz)
            length, last = len(live._history), live.filter.last_t
            for stamp in (last, last - 0.01, float("nan")):
                live.observe(stamp, [100.0, 100.0, 100.0])
                self.assertEqual(len(live._history), length)
                self.assertEqual(live.filter.last_t, last)
            live.reset()
            self.assertEqual(len(live._history), 0)
            self.assertFalse(live.filter.snapshot()["initialized"])


class FakePredictor:
    """Record exact measurement inputs, with deterministic finite forecasts."""
    def __init__(self, model_dir):
        self.physical = {"k_drag": 0.02, "e_n": 0.83,
                         "alpha_t": 0.56, "contact_height": 0.03}
        self.config = {"filter": {}}
        self.reset()

    def reset(self):
        self.received = []
        self.forecast_stamps = []

    def observe(self, stamp, xyz):
        position = np.array(xyz, dtype=float, copy=True)
        self.received.append((stamp, position))
        state = np.r_[position, [1.0, 0.0, 0.0]]
        self.last_snapshot = {
            "initialized": True, "state": state, "p": position,
            "v": state[3:], "a_res": np.zeros(3), "t_s": stamp,
            "bounced": False, "bounce_confirmed": False,
            "accepted": True, "reason": "accepted",
        }
        return self.last_snapshot

    def predict(self, *, horizons):
        stamp, xyz = self.received[-1]
        self.forecast_stamps.append(stamp)
        return {
            "initialized": True,
            "forecasts": [
                {"horizon_s": h, "target_t_s": stamp+h,
                 "physics_position_xyz_m": xyz + [h, 0.0, 0.0],
                 "position_xyz_m": xyz + [h, 0.0, 0.0],
                 "mlp_used": False, "mlp_reason": "disabled_by_validation_selection"}
                for h in horizons
            ],
            "landing": {"valid": False, "reason": "no_first_contact"},
        }


def trace(identity="test/ball.csv", start=10.0, times=None, eval_keep=None):
    offsets = np.arange(0.0, 0.601, 0.005) if times is None else np.asarray(times)
    positions = np.column_stack((offsets, offsets*0.1, 3.0-offsets))
    keep = np.ones(len(offsets), dtype=bool) if eval_keep is None else np.asarray(eval_keep)
    return {
        "id": identity, "trajectory_id": identity, "split": "test",
        "raw_times": start+offsets, "raw_xyz": positions,
        "eval_times": start+offsets[keep], "eval_xyz": positions[keep].copy(),
        "bounce_reference_s": None, "bounce_reference_xy": None,
        "bounce_event_times_s": [], "eval_scope_end_s": None,
    }


class ViewerReplayEngineTests(unittest.TestCase):
    def engine(self, *traces):
        result = ViewerReplayEngine(list(traces or (trace(),)), "unused",
                                    predictor_factory=FakePredictor)
        result.reset(wall_now=100.0)
        result.loop = False
        return result

    def test_only_due_raw_prefix_is_fed_and_offline_keep_cannot_remove_input(self):
        keep = np.ones(121, dtype=bool)
        keep[1] = False
        data = trace(eval_keep=keep)
        engine = self.engine(data)
        engine.advance(wall_now=100.051)
        expected = data["raw_times"] <= data["raw_times"][0]+0.051+1e-12
        np.testing.assert_allclose([row[0] for row in engine.predictor.received], data["raw_times"][expected])
        np.testing.assert_array_equal(np.stack([row[1] for row in engine.predictor.received]),
                                      data["raw_xyz"][expected])
        self.assertIn(data["raw_times"][1], [row[0] for row in engine.predictor.received])
        self.assertLess(engine.snapshot()["capture_t_s"], data["raw_times"][0]+0.052)

    def test_pause_freezes_and_speed_preserves_capture_time_and_clock_continuity(self):
        engine = self.engine()
        engine.advance(wall_now=100.05)
        engine.set_playing(False, wall_now=100.05)
        frozen = engine.snapshot()
        engine.advance(wall_now=110.0)
        self.assertEqual(engine.snapshot(), frozen)
        engine.set_speed(0.5, wall_now=110.0)
        engine.set_playing(True, wall_now=110.0)
        engine.advance(wall_now=110.1)
        self.assertAlmostEqual(engine.snapshot()["elapsed_s"], 0.10)
        engine.set_speed(2.0, wall_now=110.1)
        self.assertAlmostEqual(engine.cursor(110.1), 0.10)
        engine.advance(wall_now=110.15)
        self.assertAlmostEqual(engine.snapshot()["elapsed_s"], 0.20)
        stamps = np.array([row[0] for row in engine.predictor.received])
        np.testing.assert_allclose(stamps, engine.trace["raw_times"][:engine.index])

    def test_reset_and_track_change_clear_observation_prediction_and_error_histories(self):
        a, b = trace(), trace("test/other.csv", start=20.0)
        engine = self.engine(a, b)
        engine.advance(wall_now=100.22)
        self.assertTrue(engine.observed)
        self.assertTrue(engine.pending or engine.errors)
        engine.reset(wall_now=101.0)
        for records in (engine.observed, engine.estimates, engine.pending, engine.errors,
                        engine.predictor.received, engine.predictor.forecast_stamps):
            self.assertFalse(records)
        self.assertIsNone(engine.prediction)
        engine.advance(wall_now=101.10)
        self.assertTrue(engine.observed)
        engine.load(b["id"], wall_now=102.0)
        self.assertEqual(engine.identity, b["id"])
        self.assertEqual(engine.t0, 20.0)
        self.assertEqual(engine.snapshot()["observation_index"], 0)
        self.assertIsNone(engine.prediction)
        self.assertFalse(engine.errors)
        self.assertFalse(engine.predictor.received)

    def test_predictions_are_decimated_but_all_raw_observations_are_consumed(self):
        engine = self.engine()
        engine.advance(wall_now=100.10)
        stamps = engine.predictor.forecast_stamps
        self.assertGreater(engine.index, len(stamps))
        self.assertEqual(len(stamps), engine.metrics["prediction_count"])
        np.testing.assert_allclose(np.diff(stamps), 0.02, atol=1e-11)
        self.assertEqual(engine.metrics["observations_processed"], engine.index)
        self.assertEqual(engine.index, 21)

    def test_future_data_changes_ui_reference_but_not_received_prefix_or_prediction(self):
        a, b = trace(), trace()
        mask = b["raw_times"] > 10.10+1e-12
        b["raw_xyz"][mask] += [5.0, -3.0, 2.0]
        b["eval_xyz"][mask] += [5.0, -3.0, 2.0]
        x, y = self.engine(a), self.engine(b)
        x.advance(wall_now=100.10)
        y.advance(wall_now=100.10)
        np.testing.assert_array_equal(x.observed, y.observed)
        np.testing.assert_array_equal(x.prediction["physics_xyz_m"], y.prediction["physics_xyz_m"])
        for left, right in zip(x.prediction["mlp_points"], y.prediction["mlp_points"]):
            np.testing.assert_array_equal(left["position_xyz_m"], right["position_xyz_m"])
        self.assertNotEqual(x.snapshot(include_truth=True)["truth"]["xyz_m"],
                            y.snapshot(include_truth=True)["truth"]["xyz_m"])

    def test_errors_remain_pending_until_target_capture_and_tail_missing_is_visible(self):
        engine = self.engine()
        engine.advance(wall_now=100.099)
        self.assertFalse(engine.errors)
        self.assertTrue(engine.pending)
        engine.advance(wall_now=100.10)
        self.assertTrue(engine.errors)
        for row in engine.errors:
            self.assertLessEqual(row["target_elapsed_s"], engine.snapshot()["elapsed_s"]+1e-12)
        engine.advance(wall_now=100.60)
        self.assertEqual(engine.index, engine.count)
        self.assertFalse(engine.playing)
        for point in engine.prediction["mlp_points"]:
            self.assertIsNone(point["target_xyz_m"])
            self.assertEqual(point["label_reason"], "outside_observed_range")
        self.assertFalse(engine.prediction["landing"]["valid"])
        self.assertEqual(engine.prediction["landing"]["reason"], "no_first_contact")

    def test_serialization_does_not_emit_nan_or_hide_failure(self):
        converted = plain({"failure": "RuntimeError: sample failed", "measurement": np.nan,
                           "array": np.array([1.0, np.inf, -np.inf])})
        output = json.dumps(converted, allow_nan=False)
        self.assertIn("RuntimeError: sample failed", output)
        self.assertEqual(converted["array"], [1.0, None, None])


if __name__ == "__main__":
    unittest.main()
