"""Independent batched/scalar equivalence and causal projection checks."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from tennis_calibration.height_forecast import forecast_posteriors
from tennis_calibration.height_policy import HeightPolicyPredictor
from tennis_calibration.identified_dynamics import impact_map


PARAMETERS = {"k_drag": .0204, "e_n": .83, "alpha_t": .56, "impact_mode": "effective"}
HEIGHT = .03
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "artifacts/tennis_collision/20261009_frozen_pre/selected_collision.json"


class SyntheticBank:
    parameters = PARAMETERS
    height = HEIGHT

    def __init__(self):
        self.calls = []

    def apply_state(self, incoming, model_id):
        self.calls.append((model_id, np.asarray(incoming).copy()))
        result = impact_map(incoming, self.parameters)
        if model_id != "fixed":
            result[3] += {"ridge1": .1, "ridge2": .15, "gp": .2}[model_id]
            result[4] -= .05
        return result


def fixtures():
    states = np.array([
        [0., 0., .7, 8., .2, -3., .02, -.03, .01],
        [1., 2., .031, 5., .5, -4., .03, .01, .02],
        [1., 2., .4, 4., .3, 3., -.02, .05, .01],
        [-1., 2., -.01, 3., -.4, -2., 0., .02, .03],
        [2., -1., .4, 0., 0., -2., 0., 0., .01],
        [0., 0., .02, 8., 0., -3., 0., 0., 0.],
    ])
    flags = np.array([False, False, True, True, False, False])
    capture = np.array([.1, .333, 10., -.27, .55, 1.])
    return states, flags, capture


def scalar_forecast(states, flags, capture, horizons, bank, model_id, max_step_s=.004):
    output = []
    for state, flag, timestamp in zip(states, flags, capture):
        projector = HeightPolicyPredictor(bank, model_id, "height_online", config={"max_step_s": max_step_s})
        # Restore only mean/covariance/capture/first-event flags, without history.
        projector.restore_snapshot({"initialized": True, "state": state,
            "covariance": np.eye(9), "t_s": timestamp, "bounced": bool(flag),
            "bounce_confirmed": False})
        before = projector.snapshot()
        output.append(projector.predict_target_times(timestamp + np.asarray(horizons)))
        after = projector.snapshot()
        np.testing.assert_array_equal(before["state"], after["state"])
        np.testing.assert_array_equal(before["covariance"], after["covariance"])
        if before["t_s"] != after["t_s"] or before["bounced"] != after["bounced"]:
            raise AssertionError("projection changed posterior clock/first-impact flags")
    return np.asarray(output)


class HeightForecastTests(unittest.TestCase):
    def test_mixed_posteriors_match_scalar_all_four_maps_and_step_sizes(self):
        states, flags, capture = fixtures()
        for model_id in HeightPolicyPredictor.model_ids:
            for max_step in (.0025, .004, .008):
                with self.subTest(model=model_id, step=max_step):
                    horizons = (0., .1, .2, .5)
                    batch = forecast_posteriors(states, flags, capture, horizons,
                        bank=SyntheticBank(), model_id=model_id, max_step_s=max_step)
                    scalar = scalar_forecast(states, flags, capture, horizons,
                        SyntheticBank(), model_id, max_step)
                    np.testing.assert_allclose(batch[..., :3], scalar[..., :3], atol=1e-8, rtol=0)
                    np.testing.assert_allclose(batch, scalar, atol=1e-7, rtol=0)

    def test_unsorted_duplicate_horizons_zero_horizon_and_input_immutability(self):
        states, flags, capture = fixtures()
        originals = (states.copy(), flags.copy(), capture.copy())
        horizons = np.array([.5, 0., .2, .1, .5, 0.])
        saved = horizons.copy()
        result = forecast_posteriors(states, flags, capture, horizons, bank=SyntheticBank(), model_id="gp")
        np.testing.assert_array_equal(result[:, 1], states)
        np.testing.assert_array_equal(result[:, 5], states)
        np.testing.assert_array_equal(result[:, 0], result[:, 4])
        scalar = scalar_forecast(states, flags, capture, horizons, SyntheticBank(), "gp")
        np.testing.assert_allclose(result, scalar, atol=1e-7, rtol=0)
        for before, after in zip(originals, (states, flags, capture)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(horizons, saved)

    def test_absolute_capture_clocks_are_used_for_scalar_discretization(self):
        states, flags, _ = fixtures()
        capture = np.array([.017, .123456789, 1000., -.4, .7, 0.])
        result = forecast_posteriors(states, flags, capture, (.1, .2, .5), bank=SyntheticBank(), model_id="gp")
        scalar = scalar_forecast(states, flags, capture, (.1, .2, .5), SyntheticBank(), "gp")
        np.testing.assert_allclose(result, scalar, atol=1e-7, rtol=0)

    def test_already_bounced_states_never_apply_second_map(self):
        states = np.array([[0., 0., -.01, 8., .2, -3., .02, -.03, .01]])
        bank = SyntheticBank()
        result = forecast_posteriors(states, np.array([True]), np.array([.1]), (.1, .5), bank=bank, model_id="gp")
        self.assertEqual(bank.calls, [])
        self.assertLess(result[0, 1, 2], states[0, 2])

    def test_vertical_fallback_skips_undefined_gp_features_and_matches_fixed(self):
        state = np.array([[0., 0., HEIGHT, 0., 0., -3., 0., 0., .1]])
        bank = SyntheticBank()
        result = forecast_posteriors(state, np.array([False]), [0.], (0., .01), bank=bank, model_id="gp")
        self.assertEqual(bank.calls, [])
        fixed = forecast_posteriors(state, np.array([False]), [0.], (0., .01), bank=SyntheticBank(), model_id="fixed")
        np.testing.assert_array_equal(result, fixed)
        self.assertGreater(result[0, 1, 5], 0)

    def test_no_future_state_or_event_label_can_modify_saved_prior_forecast(self):
        states, flags, capture = fixtures()
        result = forecast_posteriors(states, flags, capture, bank=SyntheticBank(), model_id="gp")
        saved = result.copy()
        alternate = states.copy()
        alternate[:, 3:6] += 10
        changed = forecast_posteriors(alternate, flags, capture + .2, bank=SyntheticBank(), model_id="gp")
        np.testing.assert_array_equal(result, saved)
        self.assertGreater(np.linalg.norm(result - changed), 1.)
        with self.assertRaises(TypeError):
            forecast_posteriors(states, flags, capture, bank=SyntheticBank(), model_id="gp", reference_contact_t_s=.2)

    def test_empty_inputs_and_invalid_shapes_are_explicit(self):
        self.assertEqual(forecast_posteriors(np.empty((0, 9)), np.zeros(0, bool), [], bank=SyntheticBank(), model_id="gp").shape, (0, 4, 9))
        states, flags, capture = fixtures()
        self.assertEqual(forecast_posteriors(states, flags, capture, [], bank=SyntheticBank(), model_id="gp").shape, (6, 0, 9))
        for args in ((states[:, :6], flags, capture, (0.,)),
                     (states, flags.astype(int), capture, (0.,)),
                     (states, flags, capture[:-1], (0.,)),
                     (states, flags, capture, (-.1,)),
                     (states, flags, capture, (np.nan,))):
            with self.assertRaises(ValueError):
                forecast_posteriors(*args, bank=SyntheticBank(), model_id="gp")

    @unittest.skipUnless(CHECKPOINT.exists(), "real frozen checkpoint absent")
    def test_real_frozen_checkpoints_match_scalar_without_weight_changes(self):
        from tennis_calibration.player_bounce import BounceModelBank
        bank = BounceModelBank(CHECKPOINT)
        frozen = deepcopy(bank.models)
        states, flags, capture = fixtures()
        states[1, 2] = bank.height + .001
        for model_id in bank.model_ids:
            result = forecast_posteriors(states, flags, capture, bank=bank, model_id=model_id)
            scalar = scalar_forecast(states, flags, capture, (0., .1, .2, .5), bank, model_id)
            np.testing.assert_allclose(result[..., :3], scalar[..., :3], atol=1e-8, rtol=0)
            np.testing.assert_allclose(result, scalar, atol=1e-7, rtol=0)
        for model_id in bank.models:
            for key, value in frozen[model_id].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(bank.models[model_id][key], value)


if __name__ == "__main__":
    unittest.main()
