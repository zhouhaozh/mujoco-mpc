"""Real-engine smoke/invariant checks, run in Ubuntu .venv-tennis."""
import importlib.util
import unittest

import numpy as np

from tennis_calibration.identified_dynamics import rhs
from tennis_calibration.identified_mujoco import StandaloneBall


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'requires project MuJoCo environment')
class StandaloneMujocoTest(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.mj = mujoco
        self.initial = np.array([0., 0., 2., 12., 1., 3., .1, -.08, .06])
        self.parameters = {'k_drag': .025, 'e_n': .8, 'alpha_t': .558, 'impact_mode': 'effective'}

    def test_actual_acceleration_and_freejoint_motion(self):
        with StandaloneBall(self.initial, self.parameters, .014) as sim:
            np.testing.assert_allclose(sim.data.qacc[:3], rhs(self.initial, .025)[3:6], atol=1e-12)
            initial_position = sim.data.qpos[:3].copy()
            sim.advance_to(.1)
            self.assertGreater(np.linalg.norm(sim.data.qpos[:3]-initial_position), 1.)
            np.testing.assert_array_equal(sim.state()[:3], sim.data.qpos[:3])
            self.assertGreater(sim.callback_calls, 3*sim.engine_steps)
            self.assertEqual(sim.data.ncon, 0)

    def test_world_spin_with_nonidentity_orientation(self):
        quaternion = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        with StandaloneBall(self.initial, self.parameters, .014, initial_quaternion=quaternion) as sim:
            velocity = np.empty(6)
            self.mj.mj_objectVelocity(sim.model, sim.data, self.mj.mjtObj.mjOBJ_BODY, sim.ball_id, velocity, 0)
            np.testing.assert_allclose(velocity[:3]*sim.scale, self.initial[6:], atol=1e-12)
            sim.advance_to(.1)
            np.testing.assert_allclose(sim.state()[6:], self.initial[6:], atol=1e-12)

    def test_first_impulse_only_and_native_contact_disabled(self):
        for mode in ('effective', 'stick_slip'):
            with self.subTest(mode=mode):
                p = {**self.parameters, 'impact_mode': mode, 'mu': .25}
                with StandaloneBall(self.initial, p, .014) as sim:
                    sim.advance_to(1.5)
                    self.assertEqual(len(sim.events), 1)
                    event = sim.events[0]
                    self.assertLess(event['incoming_state'][5], 0)
                    self.assertGreater(event['outgoing_state'][5], 0)
                    self.assertEqual(event['native_contact_count'], 0)
                    self.assertLess(abs(event['position_plane_correction_m']), 1e-9)

    def test_callback_restored_after_failure(self):
        old = self.mj.get_mjcb_control()
        def previous(model, data):
            pass
        self.mj.set_mjcb_control(previous)
        try:
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with StandaloneBall(self.initial, self.parameters, .014) as sim:
                    sim.advance_to(.01)
                    raise RuntimeError('intentional')
            self.assertIs(self.mj.get_mjcb_control(), previous)
        finally:
            self.mj.set_mjcb_control(old)

    def test_requires_callback_context_and_monotonic_time(self):
        sim = StandaloneBall(self.initial, self.parameters, .014)
        with self.assertRaises(RuntimeError): sim.advance_to(.01)
        with sim:
            sim.advance_to(.01)
            with self.assertRaises(ValueError): sim.advance_to(0)


if __name__ == '__main__':
    unittest.main()
