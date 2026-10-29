from contextlib import nullcontext
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import view_x4_native_replay as view


class X4NativeViewerTests(unittest.TestCase):
    def setUp(self):
        self.model = view.mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint name="ball_free"/>'
            '<geom type="sphere" size="0.015"/></body></worldbody></mujoco>')
        self.data = view.mujoco.MjData(self.model)
        self.viewer = mock.Mock()
        self.viewer.lock.side_effect = nullcontext
        self.viewer.user_scn = view.mujoco.MjvScene(self.model, maxgeom=16)
        self.frame = {
            't': .7, 'replay_t': .2, 'phase': 'replay', 'executing': False,
            'qpos': self.data.qpos.tolist(), 'qvel': self.data.qvel.tolist(),
            'target': {'valid': True, 't_hit_s': .4, 'p_hit_m': [.5, 0, .8]},
            'predicted_trajectory': [[.2, 1, 0, 2], [.4, .5, 0, .8], [.6, 0, 0, 0]],
        }

    def display(self):
        with mock.patch.object(view.mujoco, 'mj_step', side_effect=AssertionError('physics forbidden')):
            view.display_frame(self.viewer, self.model, self.data, self.frame, 'Test', 'Running', '')

    def test_full_curve_extends_beyond_hit_and_preserves_state(self):
        self.display()
        scene = self.viewer.user_scn
        self.assertEqual(scene.ngeom, 4)  # ball, target, two forecast segments
        for geom in scene.geoms[2:4]:
            self.assertEqual(geom.type, view.mujoco.mjtGeom.mjGEOM_CAPSULE)
            self.assertEqual(geom.category, view.mujoco.mjtCatBit.mjCAT_DECOR)
        np.testing.assert_allclose(scene.geoms[3].pos, [.25, 0, .4])
        np.testing.assert_array_equal(self.data.qpos, self.frame['qpos'])
        np.testing.assert_array_equal(self.data.qvel, self.frame['qvel'])
        self.assertEqual(self.data.time, self.frame['t'])

    def test_forecast_does_not_require_a_valid_height_target(self):
        self.frame['target'] = {'valid': False}
        self.display()
        self.assertEqual(self.viewer.user_scn.ngeom, 3)

    def test_new_forecast_replaces_previous_curve_and_empty_clears_it(self):
        self.display()
        self.frame['predicted_trajectory'] = [[.3, 2, 0, 2], [.7, 1, 0, 0]]
        self.display()
        self.assertEqual(self.viewer.user_scn.ngeom, 3)
        np.testing.assert_allclose(self.viewer.user_scn.geoms[2].pos, [1.5, 0, 1])
        self.frame['predicted_trajectory'] = []
        self.display()
        self.assertEqual(self.viewer.user_scn.ngeom, 2)

    def test_legacy_frames_without_trajectory_still_display(self):
        del self.frame['predicted_trajectory']
        self.display()
        self.assertEqual(self.viewer.user_scn.ngeom, 2)

    def test_settle_hides_forecast_and_final_frame_retains_it(self):
        self.frame['phase'] = 'settle'
        self.display()
        self.assertEqual(self.viewer.user_scn.ngeom, 1)
        self.frame['phase'] = 'stop'
        self.display()
        self.assertEqual(self.viewer.user_scn.ngeom, 3)

    def test_large_forecast_fits_capacity_and_preserves_both_endpoints(self):
        self.viewer.user_scn = view.mujoco.MjvScene(self.model, maxgeom=4)
        self.frame['predicted_trajectory'] = [[i * .001, i * .01, 0, 1] for i in range(2000)]
        self.display()
        scene = self.viewer.user_scn
        self.assertEqual(scene.ngeom, 4)
        first, last = scene.geoms[2], scene.geoms[3]
        start = first.pos - first.size[2] * first.mat.reshape(3, 3)[:, 2]
        end = last.pos + last.size[2] * last.mat.reshape(3, 3)[:, 2]
        np.testing.assert_allclose(start, [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(end, [19.99, 0, 1], atol=1e-6)

    def test_malformed_forecasts_are_rejected(self):
        for points in ([[0, 1, 2]], [[0, float('nan'), 0, 0]],
                       [[1, 0, 0, 1], [0, 1, 0, 0]]):
            with self.subTest(points=points), self.assertRaisesRegex(ValueError, 'invalid predicted trajectory'):
                view.add_predicted_trajectory(self.viewer.user_scn, points)


if __name__ == '__main__':
    unittest.main()
