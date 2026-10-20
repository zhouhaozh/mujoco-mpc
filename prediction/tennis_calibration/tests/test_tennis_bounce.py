import unittest
import numpy as np
from tennis_calibration.bounce import impact_estimate,fit_law,predict_velocity,acceleration


class TennisBounceTests(unittest.TestCase):
    def test_joint_position_and_velocity_jump_survive_contact_guard(self):
        k=.020;e=.77;alpha=.83;tb=.6
        t=np.r_[np.linspace(.42,.584,65),np.linspace(.616,.78,65)]
        before=np.array([-17.,-.7,-6.]);after=predict_velocity(before,e,alpha)
        pos=np.array([3.,-.2,.041])
        p=np.empty((len(t),3))
        for i,ti in enumerate(t):
            v=before if ti<tb else after;dt=ti-tb
            p[i]=pos+dt*v+.5*dt*dt*acceleration(v,k)
        annotation={'first_bounce_t_s':tb,'pre_end_t_s':.584,'post_start_t_s':.616,'post_end_t_s':.78}
        r=impact_estimate(t,p,annotation,k)
        self.assertTrue(r['converged'])
        self.assertAlmostEqual(r['t_b_s'],tb,delta=1e-5)
        self.assertAlmostEqual(r['e_n_observed'],e,delta=1e-4)
        self.assertAlmostEqual(r['alpha_t_observed'],alpha,delta=1e-4)

    def test_shared_effective_map_recovers_parameters_and_allows_spin_gain(self):
        records=[]
        for i in range(10):
            v=np.array([-9.-i,.2*i,-4.-.2*i])
            records.append({'v_minus_m_s':v.tolist(),'v_plus_m_s':predict_velocity(v,.74,1.05).tolist()})
        fit=fit_law(records)
        self.assertTrue(fit['converged'])
        self.assertAlmostEqual(fit['e_n'],.74,places=7)
        self.assertAlmostEqual(fit['alpha_t'],1.05,places=7)

    def test_missing_postimpact_observations_do_not_invent_bounce(self):
        t=np.linspace(0,.5,100);p=np.c_[t,0*t,1-t]
        a={'first_bounce_t_s':.5,'pre_end_t_s':.48,'post_start_t_s':.52,'post_end_t_s':.5}
        with self.assertRaisesRegex(ValueError,'coverage'):
            impact_estimate(t,p,a,.02)


if __name__=='__main__':unittest.main()
