"""Native replay controls; seeking replays received observations causally."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

SPEED_PRESETS = (.1, .25, .5, 1., 2., 4.)
LAYERS = ('reference', 'received', 'estimate', 'forecast', 'ball', 'court')


class PlayerSession:
    def __init__(self, simulation, traces, selected, trajectory_id, pool_ids,
                 *, speed=.5, loop=True, bounce_bank=None):
        self.sim, self.selected = simulation, selected
        self.bounce_bank = bounce_bank
        self.traces = list(traces)
        self.by_id = {t['id']: t for t in traces}
        self.pool_ids = list(pool_ids)
        self.controls = {'paused': False, 'speed': self.checked_speed(speed),
                         'loop': bool(loop), 'auto_next': False,
                         'visible': {layer: True for layer in LAYERS},
                         'camera': 'overview', 'view_mode': 'online',
                         'bounce_model': 'gp', 'bounce_gap_s': .04,
                         'bounce_compare': True}
        self.camera_dirty = True
        self.closed = False
        self.select(trajectory_id)

    @staticmethod
    def checked_speed(value):
        speed = float(value)
        if not np.isfinite(speed) or not .1 <= speed <= 4.:
            raise ValueError('Playback speed must be between 0.1 and 4.0')
        return speed

    def select(self, identity):
        if identity not in self.by_id:
            raise ValueError('Unknown trajectory identity')
        from .identified_player import Replay
        from .player_scene import configure_scene
        if hasattr(self, 'replay'):
            # A new replay resets shared simulation counters/events. Preserve
            # the old interval's real-engine evidence before that reset.
            self.replay.simulation_evidence = {
                'callback_calls': self.sim.callback_calls,
                'engine_steps': self.sim.engine_steps,
                'events': deepcopy(self.sim.events),
            }
        self.trajectory_id = identity
        if self.controls['view_mode'] == 'bounce':
            from .bounce_replay import BounceReplay
            if self.bounce_bank is None:
                raise ValueError('Frozen bounce checkpoints are unavailable')
            self.replay = BounceReplay(self.by_id[identity], self.selected, self.sim,
                                      self.bounce_bank, self.controls['bounce_model'],
                                      self.controls['bounce_gap_s'])
            if not self.replay.valid:
                self.controls['paused'] = True
        else:
            self.sim.collision_response = None
            self.sim.collision_model_id = None
            self.replay = Replay(self.by_id[identity], self.selected, self.sim)
        self.style = configure_scene(self.sim, self.by_id[identity],
                                     self.selected['contact_height'])
        if self.controls['view_mode'] == 'bounce' and self.replay.valid:
            low, high = np.min(self.replay.forecast, axis=0), np.max(self.replay.forecast, axis=0)
            self.style['camera']['lookat'] = ((low+high)/2).tolist()
            self.style['camera']['lookat'][2] = max(.6, self.style['camera']['lookat'][2])
            self.style['camera']['distance'] = max(5., float(np.linalg.norm(high-low))*1.8)
        self.camera_dirty = True

    def seek(self, fraction):
        fraction = float(fraction)
        if not np.isfinite(fraction) or not 0. <= fraction <= 1.:
            raise ValueError('Seek fraction must be between zero and one')
        target = fraction * self.replay.duration
        if target < self.replay.elapsed:
            # Reinitialization removes all future EKF/history/physics state.
            self.select(self.trajectory_id)
        self.replay.advance(target)
        self.controls['paused'] = True

    def adjacent(self, offset):
        if not self.pool_ids:
            return
        index = self.pool_ids.index(self.trajectory_id) if self.trajectory_id in self.pool_ids else 0
        self.select(self.pool_ids[(index + offset) % len(self.pool_ids)])

    def dispatch(self, command):
        action = command['action']
        if action == 'view_mode':
            value = command['value']
            if value not in ('online', 'bounce') or (value == 'bounce' and self.bounce_bank is None):
                raise ValueError('Unknown or unavailable observation mode')
            self.controls['view_mode'] = value
            self.select(self.trajectory_id)
        elif action == 'bounce_model':
            value = command['value']
            if value not in ('fixed', 'ridge1', 'ridge2', 'gp'):
                raise ValueError('Unknown rebound model')
            self.controls['bounce_model'] = value
            if self.controls['view_mode'] == 'bounce':
                self.select(self.trajectory_id)
        elif action == 'bounce_gap':
            value = float(command['seconds'])
            if value not in (0., .005, .01, .02, .04, .08):
                raise ValueError('Unsupported missing-tail scenario')
            self.controls['bounce_gap_s'] = value
            if self.controls['view_mode'] == 'bounce':
                self.select(self.trajectory_id)
        elif action == 'bounce_compare':
            self.controls['bounce_compare'] = bool(command['enabled'])
        elif action == 'select':
            self.select(command['trajectory_id'])
        elif action == 'trajectory_pool':
            ids = list(dict.fromkeys(command['trajectory_ids']))
            if any(i not in self.by_id for i in ids):
                raise ValueError('Trajectory pool contains unknown identities')
            self.pool_ids = ids
            if ids and self.trajectory_id not in ids:
                self.select(ids[0])
        elif action in ('next', 'previous'):
            self.adjacent(1 if action == 'next' else -1)
        elif action == 'pause':
            self.controls['paused'] = bool(command['paused'])
        elif action == 'restart':
            self.select(self.trajectory_id)
            self.controls['paused'] = False
        elif action == 'speed':
            self.controls['speed'] = self.checked_speed(command['value'])
        elif action in ('loop', 'auto_next'):
            self.controls[action] = bool(command['enabled'])
        elif action == 'seek':
            self.seek(command['fraction'])
        elif action == 'step':
            target = np.clip(self.replay.elapsed + float(command['seconds']), 0., self.replay.duration)
            self.seek(float(target / self.replay.duration))
        elif action == 'visibility':
            if command['layer'] not in LAYERS:
                raise ValueError('Unknown visual layer')
            self.controls['visible'][command['layer']] = bool(command['enabled'])
        elif action == 'camera':
            if command['preset'] not in ('overview', 'top', 'side', 'follow'):
                raise ValueError('Unknown camera preset')
            self.controls['camera'] = command['preset']
            self.camera_dirty = True
        elif action == 'panel_close':
            self.closed = True
        else:
            raise ValueError('Unknown player action: ' + str(action))

    def key(self, key):
        if key == 32:
            self.dispatch({'action': 'pause', 'paused': not self.controls['paused']})
        elif key in (78, 80, 82):
            self.dispatch({'action': {78: 'next', 80: 'previous', 82: 'restart'}[key]})
        elif key in (43, 61, 334, 45, 333):
            value = self.controls['speed']
            faster = key in (43, 61, 334)
            candidates = [s for s in SPEED_PRESETS if s > value + 1e-8] if faster else [s for s in SPEED_PRESETS if s < value - 1e-8]
            self.controls['speed'] = (min(candidates) if faster else max(candidates)) if candidates else value
        elif key == 76:
            self.controls['loop'] = not self.controls['loop']
        elif key in (263, 262, 91, 93):
            self.dispatch({'action': 'step', 'seconds': -.02 if key in (263, 91) else .02})
        elif key in (79, 84, 83, 70):
            self.dispatch({'action': 'camera', 'preset': {79: 'overview', 84: 'top', 83: 'side', 70: 'follow'}[key]})

    def advance(self, wall_delta_s):
        if self.controls['paused']:
            return
        if not getattr(self.replay, 'valid', True):
            self.controls['paused'] = True
            return
        self.replay.advance(self.replay.elapsed + max(0., wall_delta_s) * self.controls['speed'])
        if self.replay.elapsed < self.replay.duration - 1e-9:
            return
        if self.controls['auto_next'] and self.pool_ids:
            index = self.pool_ids.index(self.trajectory_id) if self.trajectory_id in self.pool_ids else 0
            if index + 1 < len(self.pool_ids):
                self.select(self.pool_ids[index + 1])
            elif self.controls['loop']:
                self.select(self.pool_ids[0])
            else:
                self.controls['paused'] = True
        elif self.controls['loop']:
            self.select(self.trajectory_id)
        else:
            self.controls['paused'] = True

    def state(self):
        r = self.replay
        return {'trajectory_id': self.trajectory_id,
                'trajectory_index': next(i for i,t in enumerate(self.traces) if t['id'] == self.trajectory_id),
                'pool_ids': list(self.pool_ids), 'pool_total': len(self.pool_ids),
                'pool_index': self.pool_ids.index(self.trajectory_id) if self.trajectory_id in self.pool_ids else None,
                'elapsed_s': r.elapsed, 'duration_s': r.duration,
                'raw_rows_received': r.cursor, 'forecast_updates': r.forecast_updates,
                'split': r.trace['split'],
                'bounce_status': getattr(r, 'status', ''),
                'bounce_info': r.info() if hasattr(r, 'info') else None,
                **self.controls}
