"""Build and run one X4 measured replay with isolated predictor/MPC libraries."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def native_environment(cfg):
    environment = os.environ.copy()
    native_lib = Path(cfg['native_mpc_library']).parent
    # Keep MPC's numerical runtime independent of the predictor and shell environment.
    dependencies = native_lib / 'runtime'
    environment['LD_LIBRARY_PATH'] = ':'.join([str(native_lib), str(dependencies)])
    environment['CASADIPATH'] = str(dependencies)
    environment['OMP_NUM_THREADS'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '1'
    return environment


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=repo / 'sckd_sim/models/config/runtime_x4_950_replay.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--view', action='store_true', help='open the window first, then display the simulation as it runs')
    parser.add_argument('--view-speed', type=float, default=0.25, help='live simulation speed multiplier (default: 0.25, four times slower)')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.view and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('--view requires a graphical desktop (DISPLAY or WAYLAND_DISPLAY)')
    if not 0 < args.view_speed < float('inf'):
        parser.error('--view-speed must be finite and positive')
    if not 0 < args.timeout < float('inf'):
        parser.error('--timeout must be finite and positive')
    config = args.config.resolve()
    cfg = json.loads(config.read_text())
    output = (args.output or repo / 'outputs/x4_native_replay' / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f'Output directory must be empty: {output}')
    project = (config.parent / cfg['project_root']).resolve()
    cfg['native_mpc_library'] = str((project / cfg['native_mpc_library']).resolve())
    build = project / 'build'
    if not args.no_build:
        subprocess.run([str(repo / '.venv/bin/python'), str(project / 'models/build_x4_replay.py')], check=True, cwd=project)
        subprocess.run(['cmake', '-S', str(project), '-B', str(build), '-DSCKD_BUILD_X4_NATIVE_REPLAY=ON'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'x4_native_replay', 'x4_shuttle_predictor', '-j4'], check=True)
    predictor_env = os.environ.copy()
    predictor_env['LD_LIBRARY_PATH'] = ':'.join(['/usr/local/lib', '/usr/local/TensorRT/lib', '/usr/local/cuda/lib64', predictor_env.get('LD_LIBRARY_PATH', '')])
    native_env = native_environment(cfg)
    with tempfile.TemporaryDirectory(prefix='x4-replay-') as temp, \
            (output / 'predictor.log').open('w') as predictor_log, \
            (output / 'run.log').open('w') as run_log:
        socket = Path(temp) / 'predictor.sock'
        predictor = subprocess.Popen([str(build / 'x4_shuttle_predictor'), str(socket), str(config)],
                                     cwd=output, env=predictor_env, stdout=predictor_log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while not socket.exists():
                if predictor.poll() is not None:
                    raise RuntimeError(f'Predictor exited {predictor.returncode}; see {output / "predictor.log"}')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Predictor startup exceeded 30 seconds')
                time.sleep(.05)
            if args.view:
                # The live viewer opens its window before starting the physics child.
                # It enforces the child timeout, then holds the final frame until closed.
                result = subprocess.Popen([str(repo / '.venv/bin/python'), str(repo / 'tools/live_x4_native_replay.py'),
                                           '--config', str(config), '--run', str(output), '--socket', str(socket),
                                           '--binary', str(build / 'x4_native_replay'), '--speed', str(args.view_speed),
                                           '--timeout', str(args.timeout)])
                try:
                    result.wait()
                finally:
                    stop_process(result)
            else:
                result = subprocess.run([str(build / 'x4_native_replay'), str(config), str(output), str(socket)],
                                        cwd=output, env=native_env, stdout=run_log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout)
            summary = output / 'summary.json'
            if summary.exists():
                print(summary.read_text())
            print(f'Run artifacts: {output}', flush=True)
            if result.returncode:
                raise RuntimeError(f'Simulation exited {result.returncode}; see {output / "run.log"}')
            if not summary.exists():
                print('Window closed; simulation stopped before completion.', flush=True)
                return
            predictor.wait(timeout=10)
            if predictor.returncode:
                raise RuntimeError(f'Run failed: simulation={result.returncode}, predictor={predictor.returncode}; inspect run.log')
        finally:
            stop_process(predictor)
    viewer_command = [str(repo / '.venv/bin/python'), str(repo / 'tools/view_x4_native_replay.py'),
                      '--run', str(output), '--speed', '0.25']
    if not args.view:
        import shlex
        print('Headless run finished. Open its replay window with:\n' + shlex.join(viewer_command), flush=True)


if __name__ == '__main__':
    main()
