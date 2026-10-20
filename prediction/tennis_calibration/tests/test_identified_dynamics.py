"""Synthetic physical invariants; real fits are evaluated on Ubuntu separately."""
import unittest
import numpy as np

from tennis_calibration import identified_dynamics as d
from tennis_calibration.flight import simulate as old_simulate


class DynamicsTest(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1., 2., 1.5, 12., -3., 4., .2, -.1, .3])

    def test_zero_magnus_matches_old_model(self):
        x = self.x.copy(); x[6:] = 0
        times = np.array([0, .003, .057, .2])
        np.testing.assert_allclose(d.simulate(times, x, .02)[:, :6],
                                   old_simulate(times, x[:6], .02), atol=1e-12)

    def test_magnus_perpendicular(self):
        f = d.rhs(self.x, 0)
        magnus = f[3:6]-d.GRAVITY
        self.assertAlmostEqual(float(magnus @ self.x[3:6]), 0, places=12)
        np.testing.assert_array_equal(f[6:], 0)

    def test_rhs_jacobian(self):
        numerical = np.column_stack([(d.rhs(self.x+np.eye(9)[j]*1e-6, .02)
                                      -d.rhs(self.x-np.eye(9)[j]*1e-6, .02))/2e-6
                                     for j in range(9)])
        np.testing.assert_allclose(d.rhs_jacobian(self.x, .02), numerical, atol=1e-8)

    def test_discrete_jacobian(self):
        x, F = d.step_with_jacobian(self.x, .043, .02)
        numerical = np.column_stack([(d.step(self.x+np.eye(9)[j]*1e-6, .043, .02)
                                      -d.step(self.x-np.eye(9)[j]*1e-6, .043, .02))/2e-6
                                     for j in range(9)])
        np.testing.assert_allclose(F, numerical, atol=2e-8)
        np.testing.assert_allclose(x, d.step(self.x, .043, .02), atol=1e-14)

    def test_initial_and_drag_sensitivity(self):
        t = np.array([[0., .013, .043]])
        position, jac = d.simulate_batch_with_jacobian(t, self.x[None], .02)
        for j in range(10):
            if j < 9:
                dx = np.eye(9)[j]*1e-6
                numerical = (d.simulate_batch(t, (self.x+dx)[None], .02)[..., :3]
                              -d.simulate_batch(t, (self.x-dx)[None], .02)[..., :3])/2e-6
            else:
                numerical = (d.simulate_batch(t, self.x[None], .020001)[..., :3]
                              -d.simulate_batch(t, self.x[None], .019999)[..., :3])/2e-6
            np.testing.assert_allclose(jac[..., j], numerical, atol=2e-8)
        np.testing.assert_allclose(position, d.simulate_batch(t, self.x[None], .02)[..., :3])

    def test_batch_different_intervals(self):
        times = np.array([[0., .02, .04], [.002, .007, .05]])
        initial = np.stack((self.x, self.x*1.2))
        result = d.simulate_batch(times, initial, .02)
        for j in range(2):
            np.testing.assert_allclose(result[j], d.simulate(times[j], initial[j], .02), atol=1e-10)

    def test_no_force_at_zero_velocity(self):
        x = np.zeros(9)
        np.testing.assert_allclose(d.rhs(x, .02)[3:6], d.GRAVITY)
        self.assertTrue(np.isfinite(d.rhs_jacobian(x, .02)).all())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): d.step(self.x, .01, -1)
        with self.assertRaises(ValueError): d.simulate([.1, 0], self.x, .02)
        with self.assertRaises(ValueError): d.step(self.x, .01, .02, 0)


class ImpactTest(unittest.TestCase):
    def incoming(self):
        return np.array([0., 0., .014, 10., 0., -5., 0., 0., 0.])

    def test_effective_law(self):
        x = self.incoming(); x[6:] = [.2, .3, .4]
        after = d.impact_map(x, {'e_n': .8, 'alpha_t': .55})
        np.testing.assert_allclose(after[3:6], [5.5, 0., 4.])
        np.testing.assert_array_equal(after[6:], x[6:])

    def test_no_friction_preserves_tangent_and_spin(self):
        x = self.incoming(); x[6:] = [.2, .3, .4]
        after = d.impact_map(x, {'impact_mode': 'stick_slip', 'mu': 0, 'e_n': .8})
        np.testing.assert_allclose(after[3:6], [10., 0., 4.])
        np.testing.assert_allclose(after[6:], x[6:])

    def test_sticking_removes_contact_slip(self):
        x = self.incoming(); x[3] = 1
        p = {'impact_mode': 'stick_slip', 'mu': 1, 'e_n': .8}
        after, diag = d.impact_map(x, p, return_diagnostics=True)
        slip = after[3:6]+np.cross(after[6:]/d.magnus_scale(p), [0., 0., -d.ASSUMPTIONS['radius_m']])
        np.testing.assert_allclose(slip[:2], 0, atol=1e-12)
        self.assertEqual(str(diag['regime']), 'stick')

    def test_slip_coulomb_bound(self):
        p = {'impact_mode': 'stick_slip', 'mu': .01, 'e_n': .8}
        _, diag = d.impact_map(self.incoming(), p, return_diagnostics=True)
        impulse = diag['impulse_N_s']
        self.assertAlmostEqual(float(np.linalg.norm(impulse[:2])), .01*impulse[2], places=12)
        self.assertEqual(str(diag['regime']), 'slip')

    def test_contact_does_not_add_total_energy(self):
        rng = np.random.default_rng(123)
        states = rng.normal(size=(100, 9)); states[:, 3:6] *= 10; states[:, 5] = -abs(states[:, 5])
        states[:, 6:] *= .5
        p = {'impact_mode': 'stick_slip', 'mu': .3, 'e_n': .8}
        after = d.impact_map(states, p)
        m, r = d.ASSUMPTIONS['mass_kg'], d.ASSUMPTIONS['radius_m']
        I, scale = 2/3*m*r*r, d.magnus_scale(p)
        def energy(x):
            return .5*m*np.sum(x[:, 3:6]**2, axis=1)+.5*I*np.sum((x[:, 6:]/scale)**2, axis=1)
        self.assertTrue(np.all(energy(after) <= energy(states)+1e-10))

    def test_ascending_state_unchanged(self):
        x = self.incoming(); x[5] = 2
        for mode in ('effective', 'stick_slip'):
            np.testing.assert_allclose(d.impact_map(x, {'impact_mode': mode}), x)

    def test_synthetic_impact_parameters(self):
        incoming = np.tile(self.incoming(), (20, 1))
        incoming[:, 3] = np.linspace(3, 15, 20); incoming[:, 5] = -np.linspace(2, 8, 20)
        incoming[:, 6] = np.linspace(-.1, .1, 20)
        true = {'impact_mode': 'stick_slip', 'mu': .12, 'e_n': .74}
        outgoing = d.impact_map(incoming, true)
        records = [{'v_minus_m_s': a[3:6], 'b_minus_per_s': a[6:], 'v_plus_m_s': b[3:6]}
                   for a, b in zip(incoming, outgoing)]
        fit = d.fit_conditional_impact(records)
        self.assertTrue(fit['converged'])
        self.assertAlmostEqual(fit['parameters']['e_n'], .74, places=6)
        self.assertAlmostEqual(fit['parameters']['mu'], .12, places=6)

    def test_mu_multistart_escapes_all_stick_platform(self):
        incoming = np.tile(self.incoming(), (20, 1))
        incoming[:, 3] = np.linspace(3, 15, 20)
        incoming[:, 5] = -np.linspace(2, 8, 20)
        true = {'impact_mode': 'stick_slip', 'mu': .12, 'e_n': .74}
        outgoing = d.impact_map(incoming, true)
        records = [{'v_minus_m_s': a[3:6], 'b_minus_per_s': a[6:], 'v_plus_m_s': b[3:6]}
                   for a, b in zip(incoming, outgoing)]
        saturated = d.fit_conditional_impact(records, mu_starts=(.8,))
        self.assertEqual(saturated['training_regimes']['stick'], len(records))
        self.assertEqual(saturated['jacobian_rank'], 1)
        recovered = d.fit_conditional_impact(records)
        self.assertAlmostEqual(recovered['parameters']['mu'], .12, places=6)
        self.assertAlmostEqual(recovered['parameters']['e_n'], .74, places=6)
        self.assertEqual(recovered['jacobian_rank'], 2)
        self.assertLess(recovered['cost'], saturated['cost']*.01)
        costs = [r['cost'] for r in recovered['multi_start'] if r['converged']]
        self.assertEqual(recovered['cost'], min(costs))
        self.assertEqual(len(recovered['multi_start']), 4)


class CalibrationTest(unittest.TestCase):
    def test_synthetic_joint_fit_recovers_shared_drag(self):
        times = np.linspace(0, .35, 28)
        initial = np.array([[0., 0., 2., 12., 1., 3., .2, -.1, .05],
                            [0., 0., 2., 8., -2., 4., -.1, .12, .2]])
        positions = d.simulate_batch(np.tile(times, (2, 1)), initial, .025)[..., :3]
        fit = d.fit_shared_drag_magnus([(times, pos) for pos in positions],
             representative_points=20, starts=(.02,), max_nfev=120)
        self.assertTrue(fit['converged'])
        self.assertAlmostEqual(fit['k_drag_per_m'], .025, places=4)
        self.assertEqual(fit['initial_states'].shape, (2, 9))
        self.assertEqual(fit['b_eff_units'], '1/s; not RPM')
        self.assertEqual(fit['sensitivity_diagnostics']['columns'], 19)

    def test_rank_diagnostic_marks_redundancy(self):
        diag = d.jacobian_diagnostics(np.array([[1., 2.], [2., 4.], [3., 6.]]))
        self.assertEqual(diag['rank'], 1)

    def test_conditional_drag_conversion(self):
        self.assertAlmostEqual(d.drag_coefficient(.02037369660270355), .54442, places=3)


if __name__ == '__main__':
    unittest.main()
