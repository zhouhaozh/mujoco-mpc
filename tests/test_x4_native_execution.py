import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'sckd_sim'
sys.path.insert(0, str(ROOT / 'tools'))
from run_x4_native_replay import native_environment, prepare_run_config


CONFIG = PROJECT / 'models/config/runtime_x4_950_replay.json'
CONFIG_DIR = PROJECT / 'models/config/measured_dynamics/current'
BINARY = PROJECT / 'build/x4_native_replay'
HIT_POINT = [3.395814208984375, -0.030633182525634767, 0.80078857421875]
HIT_TIME = .414
ACCELERATION_LIMIT = [8., 8., 30., 30., 180., 400., 200.]


class FakePredictor:
    def __init__(self, path):
        self.path = Path(path)
        self.error = None
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(str(self.path))
        self.server.listen(1)
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.thread.join(timeout=5)
        self.server.close()
        if exc_type is None:
            if self.thread.is_alive():
                raise RuntimeError('fake predictor did not receive close')
            if self.error:
                raise self.error

    @staticmethod
    def _response(request):
        tick = int(math.floor(float(request['t']) * 1000 + .5))
        if 20 <= tick <= 30:
            target = {'valid': True, 'p_hit_m': HIT_POINT, 't_hit_s': HIT_TIME,
                      'prediction_ms': 0.0}
        elif 31 <= tick <= 40:
            target = {'valid': True, 'p_hit_m': [1e6, 1e6, 1e6],
                      't_hit_s': HIT_TIME, 'prediction_ms': 0.0}
        else:
            target = {'valid': False}
        return {'ok': True, 'target': target, 'predicted_trajectory': [],
                'stats': {'received': tick + 1, 'accepted': tick + 1,
                          'rejected': 0, 'input_policy': 'fake'}}

    def _serve(self):
        try:
            connection, _ = self.server.accept()
            with connection, connection.makefile('rwb') as stream:
                for raw_line in stream:
                    request = json.loads(raw_line)
                    if request.get('op') == 'close':
                        response = {'ok': True}
                        stream.write((json.dumps(response) + '\n').encode())
                        stream.flush()
                        return
                    stream.write((json.dumps(self._response(request)) + '\n').encode())
                    stream.flush()
        except Exception as error:  # Propagate server failures in the test thread.
            self.error = error


class RampPlan:
    def __init__(self, solve):
        self.start = solve['t']
        self.state = solve['state']
        self.segment_dt = solve['segment_dt_s']
        increments = solve['velocity_increments']
        self.target_velocity = []
        velocity = self.state[7:].copy()
        for step in range(3):
            velocity = [velocity[axis] + increments[step][axis] for axis in range(7)]
            self.target_velocity.append(velocity.copy())
        total_s = self.segment_dt + self.segment_dt + self.segment_dt
        self.plan_ms = min(3000, int(total_s * 1000))
        self.positions = [[self.state[axis]] for axis in range(7)]
        self.velocities = [[self.state[7 + axis]] for axis in range(7)]
        first_boundary = int(self.segment_dt * 1000)
        second_boundary = int((self.segment_dt + self.segment_dt) * 1000)
        for tick in range(1, self.plan_ms + 1):
            step = 0 if tick <= first_boundary else 1 if tick <= second_boundary else 2
            for axis in range(7):
                old = self.velocities[axis][-1]
                limit = ACCELERATION_LIMIT[axis] * .001
                target = self.target_velocity[step][axis]
                velocity = min(old + limit, target) if old < target else max(old - limit, target)
                self.velocities[axis].append(velocity)
                self.positions[axis].append(self.positions[axis][-1] + velocity * .001)

    def sample(self, sample_time):
        elapsed = sample_time - self.start
        rounded = int(math.floor(max(0., elapsed) * 1000 + .5))
        index = min(rounded, self.plan_ms)
        positions = [self.positions[axis][index] for axis in range(7)]
        velocities = [self.velocities[axis][index] for axis in range(7)]
        endpoint = self.plan_ms * .001
        over = elapsed - endpoint
        if over > 0:
            duration = min(over, .15)
            for axis in range(2, 7):
                positions[axis] += velocities[axis] * duration
        if over > .15:
            velocities = [0.] * 7
        return index, positions, velocities, over


def command_validation(events, signs, offsets):
    successful_solves = [event for event in events
                         if event['kind'] == 'solve' and event['success']]
    failed_solves = [event for event in events
                     if event['kind'] == 'solve' and not event['success']]
    last_success_time = successful_solves[-1]['t'] if successful_solves else None
    first_failure_time = failed_solves[0]['t'] if failed_solves else math.inf
    plan = None
    result = {'accepted_solves': len(successful_solves), 'checked_commands': 0,
              'first_accept_index1': 0, 'sample_index_mismatches': 0,
              'plan_start_mismatches': 0, 'next_tick_time_mismatches': 0,
              'invalid_prediction_commands': 0, 'retained_after_failed_solve': 0,
              'max_q_error': 0., 'max_dq_error': 0.,
              'max_base_world_error': 0., 'max_base_body_error': 0.,
              'max_base_speed': 0.}
    boundary_samples = {}
    for event in events:
        if event['kind'] == 'solve':
            if event['success']:
                plan = RampPlan(event)
            continue
        if event['kind'] != 'command' or plan is None:
            continue
        result['plan_start_mismatches'] += abs(event['plan_start_s'] - plan.start) > 1e-12
        result['next_tick_time_mismatches'] += abs(event['sample_time_s'] - event['t'] - .001) > 1e-12
        index, positions, velocities, over = plan.sample(event['arm_sample_time_s'])
        base_index, _, base_velocities, _ = plan.sample(event['sample_time_s'])
        result['sample_index_mismatches'] += (
            event['arm_sample_index'] != index or event['base_sample_index'] != base_index)
        if abs(event['t'] - plan.start) < 1e-12:
            result['first_accept_index1'] += (
                event['arm_sample_index'] == 1 and event['base_sample_index'] == 1)
        for joint in range(5):
            expected_q = signs[joint] * (positions[2 + joint] - offsets[joint])
            expected_dq = signs[joint] * velocities[2 + joint]
            result['max_q_error'] = max(
                result['max_q_error'], abs(event['q_command'][joint] - expected_q))
            result['max_dq_error'] = max(
                result['max_dq_error'], abs(event['dq_command'][joint] - expected_dq))
        result['max_base_world_error'] = max(
            result['max_base_world_error'],
            abs(event['base_velocity_world'][0] - base_velocities[0]),
            abs(event['base_velocity_world'][1] - base_velocities[1]))
        result['max_base_speed'] = max(
            result['max_base_speed'], abs(base_velocities[0]), abs(base_velocities[1]))
        yaw = event['base_yaw_rad']
        expected_body = [
            math.cos(yaw) * base_velocities[0] + math.sin(yaw) * base_velocities[1],
            -math.sin(yaw) * base_velocities[0] + math.cos(yaw) * base_velocities[1], 0.]
        result['max_base_body_error'] = max(
            result['max_base_body_error'],
            *(abs(actual - expected)
              for actual, expected in zip(event['base_twist_command'], expected_body)))
        result['checked_commands'] += 1
        result['invalid_prediction_commands'] += not event['target_current']
        result['retained_after_failed_solve'] += (
            event['t'] > first_failure_time and event['plan_start_s'] == last_success_time)
        distance_ms = round(over * 1000)
        if distance_ms in (149, 150, 151):
            boundary_samples[distance_ms] = event
    result['follow_through_ms'] = sorted(boundary_samples)
    return result, boundary_samples


class X4NativeExecutionTests(unittest.TestCase):
    maxDiff = None

    def test_plan_lifecycle_and_commands_match_nmpc6_execution(self):
        self.assertTrue(BINARY.is_file(), f'missing prebuilt replay binary: {BINARY}')
        with tempfile.TemporaryDirectory(prefix='x4-native-execution-') as directory:
            root = Path(directory)
            trajectory = root / 'trajectory.txt'
            trajectory.write_text('\n'.join(
                f'{millisecond} 339.5814208984375 -3.0633182525634767 80.078857421875'
                for millisecond in range(701)) + '\n')

            source = json.loads(CONFIG.read_text())
            source.update(project_root=str(PROJECT), trajectory_path=str(trajectory),
                          prediction_period_ms=1, mpc_period_ms=0,
                          settle_s=.5, stop_s=.2)
            source_config = root / 'source.json'
            source_config.write_text(json.dumps(source))
            output = root / 'run'
            output.mkdir()
            runtime, cfg = prepare_run_config(source_config, output, CONFIG_DIR)

            socket_path = root / 'predictor.sock'
            with FakePredictor(socket_path):
                result = subprocess.run(
                    [str(BINARY), str(runtime), str(output), str(socket_path)],
                    cwd=output, env=native_environment(cfg), capture_output=True,
                    text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + '\n' + result.stderr)

            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            solves = [event for event in events if event['kind'] == 'solve']
            successes = [event for event in solves if event['success']]
            failures = [event for event in solves if not event['success']]
            self.assertTrue(successes, 'known target did not produce an accepted plan')
            self.assertTrue(failures, 'unreachable target did not exercise failed solves')
            last_success = successes[-1]
            self.assertTrue(all(event['t'] > last_success['t'] for event in failures))

            signs = cfg['mpc_joint_sign']
            offsets = cfg['mpc_joint_offset_rad']
            validation, boundary_samples = command_validation(events, signs, offsets)
            self.assertEqual(validation['sample_index_mismatches'], 0)
            self.assertEqual(validation['plan_start_mismatches'], 0)
            self.assertEqual(validation['next_tick_time_mismatches'], 0)
            self.assertEqual(validation['first_accept_index1'], len(successes))
            for field in ('max_q_error', 'max_dq_error',
                          'max_base_world_error', 'max_base_body_error'):
                self.assertLessEqual(validation[field], 1e-10, field)
            self.assertGreater(validation['checked_commands'], 500)
            self.assertGreater(validation['invalid_prediction_commands'], 0,
                               'prediction invalidation did not retain an executable plan')
            self.assertGreater(validation['retained_after_failed_solve'], 0,
                               'failed solves did not retain the last accepted plan')
            self.assertGreater(validation['max_base_speed'], 1e-8)
            self.assertEqual(set(boundary_samples), {149, 150, 151})
            self.assertTrue(any(abs(value) > 0 for value in boundary_samples[149]['dq_command']))
            self.assertTrue(any(abs(value) > 0 for value in boundary_samples[150]['dq_command']))
            self.assertEqual(boundary_samples[151]['dq_command'], [0.] * 5)
            self.assertEqual(boundary_samples[151]['base_velocity_world'], [0.] * 2)
            self.assertFalse(boundary_samples[151]['executing'])

            first_command = next(event for event in events
                                 if event['kind'] == 'command' and event['plan_start_s'] is not None)
            self.assertEqual(first_command['arm_sample_index'], 1)
            self.assertAlmostEqual(first_command['sample_time_s'] - first_command['plan_start_s'],
                                   .001, places=12)
            print(json.dumps(validation, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
