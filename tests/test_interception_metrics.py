"""纯几何协议与两个既有入口的静态姿态适配验收；不推进物理仿真。"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from interception_metrics import (InterceptionProtocol, InterceptionSample,
                                 box_sphere_gap, evaluate_interception)


class GeometryProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = InterceptionProtocol((.2, .01, .3), .05, .01, .1, .1, -.30, .1)

    def sample(self, position=(0, .04, 0), velocity=(0, -1, 0), racket_velocity=(0, 0, 0),
               planned=True, rotation=None):
        return InterceptionSample(position, velocity, (0, 0, 0),
            np.eye(3) if rotation is None else rotation, racket_velocity, planned)

    def evaluate(self, callback, **kwargs):
        parameters = dict(hit_time_s=.5, start_s=0, end_s=1, protocol=self.protocol)
        parameters.update(kwargs)
        return evaluate_interception(callback, **parameters)

    def test_known_intersection_records_minimum_first_time_and_closest(self):
        result = self.evaluate(lambda t: self.sample())
        self.assertTrue(result.contact)
        self.assertTrue(result.approaching)
        self.assertAlmostEqual(result.min_gap_m, -.02)
        self.assertAlmostEqual(result.first_contact_s, .4)
        self.assertEqual(result.checked_steps, 21)
        self.assertEqual(result.planned_steps, 21)
        self.assertEqual(result.closest["ball_m"], [0, .04, 0])

    def test_finite_box_edge_and_corner_are_not_an_infinite_plane(self):
        self.assertEqual(box_sphere_gap((.25, .1875, 0), (.25, .0625, .25), .125), 0)
        self.assertGreater(box_sphere_gap((.25 + .125, .0625 + .125, 0), (.25, .0625, .25), .125), 0)
        result = self.evaluate(lambda t: self.sample(position=(1, .02, 0)))
        self.assertFalse(result.contact)
        self.assertFalse(result.approaching)
        self.assertGreater(result.min_gap_m, .7)

    def test_exact_grazing_is_contact_and_a_small_gap_is_not(self):
        protocol = replace(self.protocol, racket_halfsize_m=(.25, .0625, .25), ball_radius_m=.125)
        hit = self.evaluate(lambda t: self.sample(position=(0, .1875, 0)), protocol=protocol)
        miss = self.evaluate(lambda t: self.sample(position=(0, .1875 + 1e-8, 0)), protocol=protocol)
        self.assertTrue(hit.contact)
        self.assertEqual(hit.min_gap_m, 0)
        self.assertFalse(miss.contact)

    def test_approach_threshold_is_inclusive_and_uses_relative_velocity(self):
        for speed, expected in ((-.30, True), (-.299, False), (.5, False)):
            with self.subTest(speed=speed):
                result = self.evaluate(lambda t: self.sample(velocity=(0, speed, 0)))
                self.assertTrue(result.contact)
                self.assertEqual(result.approaching, expected)
        moving_with_ball = self.evaluate(lambda t: self.sample(racket_velocity=(0, -1, 0)))
        self.assertFalse(moving_with_ball.approaching)
        opposite_side = self.evaluate(lambda t: self.sample(position=(0, -.04, 0), velocity=(0, 1, 0)))
        self.assertTrue(opposite_side.approaching)

    def test_rotated_racket_uses_local_thin_axis_in_world(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        result = self.evaluate(lambda t: self.sample(position=(-.04, 0, 0), velocity=(1, 0, 0), rotation=rotation))
        self.assertTrue(result.approaching)
        np.testing.assert_allclose(result.closest["local_m"], [0, .04, 0])

    def test_time_window_and_recording_end_limit_contacts(self):
        callback = lambda t: self.sample(position=(0, .04 if t >= .8 else 1, 0))
        self.assertFalse(self.evaluate(callback).contact)
        extended = self.evaluate(callback, protocol=replace(self.protocol, window_after_s=.4))
        self.assertTrue(extended.contact)
        self.assertAlmostEqual(extended.first_contact_s, .8)
        self.assertFalse(self.evaluate(callback, end_s=.7, protocol=replace(self.protocol, window_after_s=.4)).contact)
        empty = self.evaluate(callback, start_s=.9, end_s=1)
        self.assertEqual(empty.checked_steps, 0)
        self.assertFalse(empty.contact)

    def test_endpoint_tolerance_differences_are_explicit_and_preserved(self):
        base = replace(self.protocol, interpolation_step_s=.011, window_before_s=.35, window_after_s=.35)
        callback = lambda t: self.sample(position=(0, .04 if t > .85 else 1, 0))
        recorded = self.evaluate(callback, protocol=replace(base, endpoint_step_fraction=.5))
        ablation = self.evaluate(callback, protocol=replace(base, endpoint_step_fraction=.1))
        self.assertTrue(recorded.contact)
        self.assertFalse(ablation.contact)
        self.assertEqual(recorded.checked_steps, ablation.checked_steps + 1)
        clipped = self.evaluate(callback, protocol=replace(base, endpoint_step_fraction=.5), maximum_sample_time_s=.85)
        self.assertFalse(clipped.contact)

    def test_unplanned_missing_hit_and_miss_are_retained_as_failures(self):
        for hit, planned in ((.5, False), (None, True)):
            with self.subTest(hit=hit, planned=planned):
                result = self.evaluate(lambda t: self.sample(planned=planned), hit_time_s=hit)
                self.assertFalse(result.contact)
                self.assertFalse(result.approaching)
                self.assertIsNone(result.min_gap_m)
                self.assertIsNone(result.first_contact_s)
                self.assertGreater(result.checked_steps, 0)
                self.assertEqual(result.planned_steps, 0)

    def test_bad_numeric_data_and_adapter_errors_are_not_silent_misses(self):
        with self.assertRaises(ValueError):
            self.evaluate(lambda t: self.sample(position=(0, np.nan, 0)))
        with self.assertRaises(ValueError):
            replace(self.protocol, interpolation_step_s=0)
        with self.assertRaises(ValueError):
            replace(self.protocol, ball_radius_m=-1)
        def broken(t):
            raise RuntimeError("missing source frame")
        with self.assertRaisesRegex(RuntimeError, "missing source frame"):
            self.evaluate(broken)

    def test_protocol_is_explicit_and_json_serializable(self):
        protocol = json.loads(json.dumps(self.protocol.as_dict(), allow_nan=False))
        self.assertEqual(protocol["racket_halfsize_m"], [.2, .01, .3])
        self.assertEqual(protocol["ball_radius_m"], .05)
        self.assertEqual(protocol["window_before_s"], .1)
        self.assertEqual(protocol["approach_threshold_m_s"], -.30)


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is needed only for existing-entry adapter fixtures")
class ExistingEntryConsistencyTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        import analyze_control_ablation
        import evaluate_measured_interception
        self.mj = mujoco
        self.ablation = analyze_control_ablation
        self.recorded = evaluate_measured_interception
        self.temporary = tempfile.TemporaryDirectory(prefix="interception-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.csv = self.root / "trajectory.csv"
        self.csv.write_text("t_s,x_m,y_m,z_m\n0,0,.5,0\n.5,0,0,0\n1,0,-.5,0\n")
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="racket_face" type="box" size=".2 .01 .3"/>
          <body name="ball"><freejoint name="ball_free"/>
            <geom name="ball_geom" type="sphere" size=".05" mass=".05"/>
          </body></worldbody></mujoco>''')
        self.item = {"csv": str(self.csv), "input_end_s": 1., "bounce_labels": []}
        self.rows = []
        for stamp in (0., .5, 1.):
            qpos = self.model.qpos0.copy()
            qpos[:3] = [0, .5 - stamp, 0]
            self.rows.append({"t": stamp, "qpos": qpos.tolist(), "qvel": [0, -1, 0, 0, 0, 0],
                              "planned": True, "target": [0, 0, 0], "fallback": False})

    def compare(self, rows=None, item=None, hit=.5):
        rows = self.rows if rows is None else rows
        item = self.item if item is None else item
        a = self.ablation.contact(self.model, self.mj.MjData(self.model), rows, item, hit, .01)
        b = self.recorded.recorded_interception(self.model, self.mj.MjData(self.model), rows, item, hit, .01)
        self.assertEqual(a["contact"], b.contact)
        self.assertEqual(a["approaching"], b.approaching)
        if a["min_gap_m"] is None:
            self.assertIsNone(b.min_gap_m)
        else:
            self.assertAlmostEqual(a["min_gap_m"], b.min_gap_m)
        return a, b

    def test_existing_entries_agree_on_known_contact_without_physics_steps(self):
        a, b = self.compare()
        self.assertTrue(a["contact"])
        self.assertTrue(a["approaching"])
        self.assertAlmostEqual(a["first_contact_s"], .44)
        self.assertAlmostEqual(a["first_contact_s"], b.first_contact_s)
        self.assertAlmostEqual(a["min_gap_m"], -.05)

    def test_existing_entries_preserve_second_bounce_scope_and_unplanned_failures(self):
        item = {**self.item, "bounce_labels": [.2, .4]}
        a = self.ablation.contact(self.model, self.mj.MjData(self.model), self.rows, item, .5, .01)
        b = self.recorded.recorded_interception(self.model, self.mj.MjData(self.model), self.rows, item, .5, .01)
        self.assertFalse(a["contact"])
        self.assertFalse(b.contact)
        # 旧消融直接遍历arange(..., end+.1*step)，保留略大于.4的端点；
        # 旧录像另有if now>end:continue，排除该浮点端点，最后检查.39。
        # 球y=.5-t，盒半厚.01、球半径.05，两个旧口径的最小gap分别为.04/.05。
        self.assertAlmostEqual(a["min_gap_m"], (.5 - .4) - .01 - .05)
        self.assertAlmostEqual(b.min_gap_m, (.5 - .39) - .01 - .05)
        self.assertEqual(b.checked_steps, 25)
        rows = [{**row, "planned": False, "target": None} for row in self.rows]
        a, b = self.compare(rows=rows)
        self.assertFalse(a["contact"])
        self.assertEqual(b.planned_steps, 0)
        self.assertGreater(b.checked_steps, 0)

    def test_missing_hit_preserves_original_ablation_failure_shape(self):
        a, b = self.compare(hit=None)
        self.assertEqual(a, {"contact": False, "approaching": False, "min_gap_m": None})
        self.assertFalse(b.contact)
        self.assertGreater(b.checked_steps, 0)

    def test_state_end_difference_between_entries_is_not_silently_unified(self):
        rows = [self.rows[0], {**self.rows[1], "t": .3}]
        a = self.ablation.contact(self.model, self.mj.MjData(self.model), rows, self.item, .5, .01)
        b = self.recorded.recorded_interception(self.model, self.mj.MjData(self.model), rows, self.item, .5, .01)
        self.assertFalse(a["contact"])
        self.assertTrue(b.contact)


if __name__ == "__main__":
    unittest.main()
