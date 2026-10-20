"""Standalone MuJoCo execution of the identified tennis ball model.

MuJoCo advances real freejoint qpos/qvel under gravity and callback air force.
The first ground event uses the same instantaneous impact law as prediction;
native ball/plane contacts are disabled to avoid a second competing law.
The plane height is an effective measured center threshold minus assumed R.
This adapter does not modify or load any robot model or its environment.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .identified_dynamics import ASSUMPTIONS, impact_map, magnus_scale

DOCUMENTATION = {
    'callback': 'https://mujoco.readthedocs.io/en/stable/python.html#callbacks',
    'rk4_control': 'https://mujoco.readthedocs.io/en/stable/programming/simulation.html#simulation-loop',
    'world_velocity': 'https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity',
    'freejoint': 'https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint',
}


def ball_xml(parameters: dict[str, Any], contact_height_m: float, timestep_s=.001):
    p = {**ASSUMPTIONS, **parameters}
    m, radius = float(p['mass_kg']), float(p['radius_m'])
    inertia = float(p['inertia_factor'])*m*radius*radius
    if min(m, radius, inertia, timestep_s) <= 0 or not np.isfinite(contact_height_m):
        raise ValueError('positive ball properties/timestep and finite threshold required')
    return f'''<mujoco model="identified_tennis_standalone">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep_s:.17g}" integrator="RK4" gravity="0 0 -9.81" density="0" viscosity="0"/>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1"/>
    <geom name="effective_ground_visual" type="plane" pos="0 0 {contact_height_m-radius:.17g}"
          size="30 30 .1" contype="0" conaffinity="0" rgba=".2 .35 .25 1"/>
    <body name="tennis_ball" pos="0 0 1">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="{m:.17g}" diaginertia="{inertia:.17g} {inertia:.17g} {inertia:.17g}"/>
      <geom name="ball_visual" type="sphere" size="{radius:.17g}" contype="0" conaffinity="0" rgba=".75 .9 .15 1"/>
    </body>
  </worldbody>
</mujoco>'''


class StandaloneBall:
    """Single first-impact ball simulator; use as a callback-owning context.

    MuJoCo callbacks are global within a process. The previous callback is
    restored on exit; this standalone adapter must not run concurrent scenes.
    Spin in qvel is a *conditional* representation of b_eff, not a measurement.
    """
    def __init__(self, initial_state, parameters, contact_height_m, *, timestep_s=.001,
                 initial_quaternion=None, record_callback_samples=False):
        import mujoco
        self.mujoco = mujoco
        self.parameters = {**ASSUMPTIONS, **dict(parameters)}
        self.parameters.setdefault('impact_mode', 'effective')
        if 'k_drag' not in self.parameters:
            self.parameters['k_drag'] = self.parameters['k_drag_per_m']
        self.k_drag = float(self.parameters['k_drag'])
        if self.k_drag < 0 or not np.isfinite(self.k_drag):
            raise ValueError('nonnegative finite identified drag required')
        self.contact_height = float(contact_height_m)
        self.timestep = float(timestep_s)
        self.xml = ball_xml(self.parameters, self.contact_height, timestep_s)
        # Model compilation itself runs dynamics on temporary data. An
        # installed callback belonging to a different scene cannot safely
        # consume these temporaries; isolate compilation and restore it.
        compilation_callback = mujoco.get_mjcb_control()
        try:
            mujoco.set_mjcb_control(None)
            self.model = mujoco.MjModel.from_xml_string(self.xml)
            self.data = mujoco.MjData(self.model)
        finally:
            mujoco.set_mjcb_control(compilation_callback)
        self.ball_id = self.model.body('tennis_ball').id
        self.mass = float(self.parameters['mass_kg'])
        self.scale = magnus_scale(self.parameters)
        self.bounced = False
        self.events = []
        self.callback_calls = 0
        self.callback_samples = []
        self.record_callback_samples = bool(record_callback_samples)
        self.engine_steps = 0
        self._active = False
        self._old_callback = None
        state = np.asarray(initial_state, dtype=float)
        if state.shape != (9,) or not np.isfinite(state).all():
            raise ValueError('finite initial state [p,v,b_eff] of length 9 required')
        quat = np.array([1., 0., 0., 0.]) if initial_quaternion is None else np.asarray(initial_quaternion, dtype=float)
        if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) == 0:
            raise ValueError('nonzero finite initial quaternion required')
        self.data.qpos[:3] = state[:3]
        self.data.qpos[3:7] = quat/np.linalg.norm(quat)
        self._set_velocity_and_b(state[3:6], state[6:9])
        # No body position is overwritten again except the micrometre-scale
        # event-plane correction after a bracketed real-engine crossing.

    def _rotation(self):
        out = np.empty(9)
        self.mujoco.mju_quat2Mat(out, self.data.qpos[3:7])
        return out.reshape(3, 3)

    def _set_velocity_and_b(self, velocity_world, b_world):
        self.data.qvel[:3] = velocity_world
        # Freejoint linear velocity is world-frame; angular velocity is in the
        # body frame. Explicit rotation avoids treating local angular axes as
        # world spin axes. This relation is independently checked in the tool.
        self.data.qvel[3:6] = self._rotation().T @ (np.asarray(b_world)/self.scale)

    def state(self):
        return np.r_[self.data.qpos[:3].copy(), self.data.qvel[:3].copy(),
                     self.scale*(self._rotation() @ self.data.qvel[3:6])]

    def _control(self, model, data):
        # Callback runs at each RK4 stage with stage-specific qpos/qvel.
        rotation = data.xmat[self.ball_id].reshape(3, 3)
        v = data.qvel[:3]
        b = self.scale*(rotation @ data.qvel[3:6])
        force = self.mass*(-self.k_drag*np.linalg.norm(v)*v+np.cross(b, v))
        data.xfrc_applied[self.ball_id, :3] = force
        data.xfrc_applied[self.ball_id, 3:6] = 0
        self.callback_calls += 1
        if self.record_callback_samples:
            self.callback_samples.append({'t_s': float(data.time),
                'velocity_world_m_s': v.copy().tolist(), 'b_eff_per_s': b.tolist(),
                'air_force_world_N': force.tolist()})

    def __enter__(self):
        if self._active:
            raise RuntimeError('standalone callback context already active')
        self._old_callback = self.mujoco.get_mjcb_control()
        self.mujoco.set_mjcb_control(self._control)
        self._active = True
        try:
            self.mujoco.mj_forward(self.model, self.data)
        except BaseException:
            self.mujoco.set_mjcb_control(self._old_callback)
            self._active = False
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.mujoco.set_mjcb_control(self._old_callback)
        self._active = False

    def _capture(self):
        return (float(self.data.time), self.data.qpos.copy(), self.data.qvel.copy())

    def _restore(self, capture):
        self.data.time, self.data.qpos[:], self.data.qvel[:] = capture
        self.mujoco.mj_forward(self.model, self.data)

    def _integrate(self, duration):
        if duration <= 0:
            return
        old = float(self.model.opt.timestep)
        try:
            self.model.opt.timestep = duration
            self.mujoco.mj_step(self.model, self.data)
            self.engine_steps += 1
        finally:
            self.model.opt.timestep = old

    def _apply_first_impact(self, incoming, *, bracket_s):
        outgoing, diagnostics = impact_map(incoming, self.parameters, return_diagnostics=True)
        plane_correction = float(self.contact_height-self.data.qpos[2])
        self.data.qpos[2] = self.contact_height
        self._set_velocity_and_b(outgoing[3:6], outgoing[6:9])
        self.bounced = True
        self.mujoco.mj_forward(self.model, self.data)
        outgoing_actual = self.state()
        self.events.append({'event_type': 'first_descending_effective_plane_crossing',
            't_s': float(self.data.time), 'impact_mode': self.parameters['impact_mode'],
            'incoming_state': incoming.tolist(), 'outgoing_state': outgoing_actual.tolist(),
            'qpos_after_event': self.data.qpos.tolist(), 'qvel_after_event': self.data.qvel.tolist(),
            'momentum_impulse_N_s': (self.mass*(outgoing_actual[3:6]-incoming[3:6])).tolist(),
            'regime': str(diagnostics['regime']), 'native_contact_count': int(self.data.ncon),
            'position_plane_correction_m': plane_correction,
            'event_time_bracket_s': [float(v) for v in bracket_s],
            'definition': 'real MuJoCo incident state + manually applied shared instantaneous impulse; not native solver contact'})

    def _advance_one(self, duration):
        before = self.state()
        if not self.bounced and before[2] <= self.contact_height and before[5] < 0:
            self._apply_first_impact(before, bracket_s=[self.data.time, self.data.time])
            before = self.state()
        capture = self._capture()
        self._integrate(duration)
        end = self.state()
        crossing = (not self.bounced and before[2] > self.contact_height
                    and end[2] <= self.contact_height and end[5] < 0)
        if not crossing:
            return
        # Root-find by rerunning the *MuJoCo* dynamics from its captured state.
        # The ODE predictor is not called to manufacture any engine positions.
        low, high = 0., duration
        for _ in range(24):
            middle = (low+high)/2
            self._restore(capture)
            self._integrate(middle)
            if self.data.qpos[2] > self.contact_height:
                low = middle
            else:
                high = middle
        event_dt = (low+high)/2
        self._restore(capture)
        self._integrate(event_dt)
        incoming = self.state()
        self._apply_first_impact(incoming, bracket_s=[capture[0]+low, capture[0]+high])
        self._integrate(duration-event_dt)

    def advance_to(self, time_s):
        if not self._active:
            raise RuntimeError('use StandaloneBall as a context manager to install stage air force')
        target = float(time_s)
        if not np.isfinite(target) or target < self.data.time-1e-12:
            raise ValueError('target time must be finite and nondecreasing')
        while target-self.data.time > 1e-12:
            self._advance_one(min(self.timestep, target-self.data.time))
        return self.state()

    def replay(self, times_s):
        times = np.asarray(times_s, dtype=float)
        if times.ndim != 1 or np.any(np.diff(times) < 0):
            raise ValueError('monotonic one-dimensional output times required')
        states, qpos, qvel, ncon = [], [], [], []
        for t in times:
            states.append(self.advance_to(t))
            qpos.append(self.data.qpos.copy())
            qvel.append(self.data.qvel.copy())
            self.mujoco.mj_forward(self.model, self.data)
            ncon.append(int(self.data.ncon))
        return {'times_s': times.copy(), 'states': np.asarray(states),
                'qpos': np.asarray(qpos), 'qvel': np.asarray(qvel), 'native_ncon': np.asarray(ncon),
                'events': list(self.events), 'callback_calls': self.callback_calls,
                'engine_steps_including_contact_search': self.engine_steps}
