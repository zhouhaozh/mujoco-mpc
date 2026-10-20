"""Freeze received precontact observations, then play an independent rebound."""
from __future__ import annotations

import numpy as np


class BounceReplay:
    def __init__(self, trace, selected, simulation, bank, model_id='gp', gap_s=.04):
        self.trace, self.selected, self.sim = trace, selected, simulation
        self.bank, self.model_id, self.gap_s = bank, model_id, float(gap_s)
        self.prepared = bank.prepare_prefix(trace, gap_s)
        self.valid = bool(self.prepared['valid'])
        self.times = np.asarray(trace['raw_times'], float)
        self.positions = np.asarray(trace['raw_xyz'], float)
        self.elapsed = 0.
        self.audit, self.filter_trail = [], []
        self.forecast = np.empty((0, 3))
        self.comparison = np.empty((0, 3))
        self.forecast_t = None
        self.forecast_updates = 0
        self.snapshot = None
        self.initialized = False
        self.cursor = len(self.prepared.get('raw_prefix_times', []))
        self.end_index = self.cursor
        self.status = ''
        self.origin = (float(self.prepared['fit']['capture_t_s']) if self.valid
                       else float(self.times[0]))
        end = min(float(self.times[-1]), float(trace.get('eval_scope_end_s') or self.times[-1]))
        reference = trace.get('bounce_reference_s')
        if reference is not None:
            end = min(end, float(reference)+.5)
        self.duration = max(.001, end-self.origin)
        self.sim.mujoco.mj_resetData(self.sim.model, self.sim.data)
        self.sim.data.qpos[:3] = self.positions[0]
        self.sim.data.qpos[3:7] = [1., 0., 0., 0.]
        self.sim.data.qvel[:] = 0
        self.sim.bounced = False
        self.sim.events.clear()
        self.sim.callback_calls = self.sim.engine_steps = 0
        self.sim.collision_response = None
        self.sim.collision_model_id = None
        if not self.valid:
            self.status = ('此样本没有触地参考，请选择其他样本或在线模式。'
                           if reference is None else '此触地前观测暂无法拟合，请调整缺失时长或样本。')
            self.sim.mujoco.mj_forward(self.sim.model, self.sim.data)
            return
        initial = np.asarray(self.prepared['fit']['state'], float)
        self.sim.data.qpos[:3] = initial[:3]
        self.sim._set_velocity_and_b(initial[3:6], initial[6:9])
        self.sim.collision_response = lambda incoming: bank.apply_state(incoming, model_id)
        self.sim.collision_model_id = model_id
        self.sim.mujoco.mj_forward(self.sim.model, self.sim.data)
        self.initialized = True
        self.forecast_t = self.origin
        self.target_times = self.origin + np.linspace(0., self.duration, 71)
        self.forecast = bank.predict(self.prepared, model_id, self.target_times)[:, :3]
        self.comparison = bank.predict(self.prepared, 'fixed', self.target_times)[:, :3]
        self.forecast_updates = 1
        self.input_sha256 = self.prepared['input_sha256']
        self.status = f'已冻结 {self.cursor} 个触地前观测；反弹后不再接收新观测。'
        self.advance(0.)

    def advance(self, elapsed):
        if not self.valid:
            return
        self.elapsed = min(max(float(elapsed), self.elapsed), self.duration)
        predicted = self.bank.predict(self.prepared, self.model_id,
                                      [self.origin+self.elapsed])[0]
        self.sim.advance_to(self.elapsed)
        self.snapshot = {'initialized': True, 'state': predicted,
                         't_s': self.origin+self.elapsed,
                         'bounced': bool(self.origin+self.elapsed >= self.prepared['contact']['contact_t_s'])}
        self.filter_trail.append(predicted[:3].tolist())
        self.audit.append({'elapsed_s': self.elapsed, 'capture_t_s': self.origin,
            'raw_rows_received': self.cursor, 'qpos': self.sim.data.qpos.tolist(),
            'qvel': self.sim.data.qvel.tolist(), 'simulation_state': self.sim.state().tolist(),
            'filter_state': predicted.tolist(), 'simulation_bounced': self.sim.bounced,
            'filter_bounced': self.snapshot['bounced'], 'forecast_capture_t_s': self.origin,
            'forecast_updates': 1, 'view_mode': 'bounce', 'collision_model_id': self.model_id,
            'input_sha256': self.input_sha256, 'uses_post_contact_observations': False})

    def info(self):
        contact = self.prepared.get('contact') or {}
        return {'valid': self.valid, 'model_id': self.model_id, 'gap_s': self.gap_s,
                'input_sha256': self.prepared.get('input_sha256'),
                'cut_t_s': self.prepared.get('cut_t_s'), 'capture_t_s': self.origin,
                'predicted_contact_t_s': contact.get('contact_t_s'),
                'reference_contact_t_s': self.trace.get('bounce_reference_s'),
                'target_clock': 'same absolute observation clock for every collision model',
                'uses_post_contact_observations': False, 'status': self.status}
