#!/usr/bin/env python3
"""Plot measured-versus-simulated X4 command-replay comparisons."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    'measured': '#202020',
    'simulated': '#d1495b',
    'command': '#777777',
    'vicon': '#6f4aa8',
}


def required_columns():
    columns = ['t_s']
    columns += [f'{source}_base_{axis}_{unit}'
                for source in ('real', 'sim')
                for axis, unit in (('x', 'm'), ('y', 'm'), ('yaw', 'rad'))]
    columns += [f'{source}_base_{axis}_{unit}'
                for source in ('real', 'sim')
                for axis, unit in (('vx', 'm_s'), ('vy', 'm_s'), ('wz', 'rad_s'))]
    for joint in range(1, 6):
        columns += [f'{source}_q_j{joint}_rad' for source in ('real', 'sim', 'cmd')]
        columns += [f'{source}_dq_j{joint}_rad_s' for source in ('real', 'sim', 'cmd')]
    columns += [f'{source}_racket_{axis}_m'
                for source in ('real', 'sim', 'vicon') for axis in 'xyz']
    return columns


def read_comparison(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in required_columns() if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f'{path}: missing columns: {", ".join(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path}: comparison is empty')
    data = {}
    for name in required_columns():
        try:
            data[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{path}: column {name} contains a non-numeric value') from error
        if not np.isfinite(data[name]).all():
            raise ValueError(f'{path}: column {name} contains a non-finite value')
    if len(data['t_s']) > 1 and (np.diff(data['t_s']) <= 0).any():
        raise ValueError(f'{path}: t_s must be strictly increasing')
    return data


def decorate(ax, xlabel=False):
    ax.grid(alpha=.25)
    if xlabel:
        ax.set_xlabel('Time [s]')


def plot_joint_comparison(data, destination, title):
    time = data['t_s']
    fig, axes = plt.subplots(5, 2, figsize=(14, 15), sharex=True, layout='constrained')
    for index, joint in enumerate(range(1, 6)):
        angle = axes[index, 0]
        angle.plot(time, np.rad2deg(data[f'real_q_j{joint}_rad']),
                   color=COLORS['measured'], label='Measured')
        angle.plot(time, np.rad2deg(data[f'sim_q_j{joint}_rad']),
                   color=COLORS['simulated'], label='Simulated')
        angle.plot(time, np.rad2deg(data[f'cmd_q_j{joint}_rad']), '--',
                   color=COLORS['command'], label='Command')
        angle.set_ylabel(f'J{joint} angle [deg]')
        decorate(angle, index == 4)

        velocity = axes[index, 1]
        velocity.plot(time, data[f'real_dq_j{joint}_rad_s'],
                      color=COLORS['measured'], label='Measured')
        velocity.plot(time, data[f'sim_dq_j{joint}_rad_s'],
                      color=COLORS['simulated'], label='Simulated')
        velocity.plot(time, data[f'cmd_dq_j{joint}_rad_s'], '--',
                      color=COLORS['command'], label='Command')
        velocity.set_ylabel(f'J{joint} velocity [rad/s]')
        decorate(velocity, index == 4)
    axes[0, 0].legend(ncol=3, fontsize=9)
    axes[0, 1].legend(ncol=3, fontsize=9)
    fig.suptitle(f'{title}: joint command dynamics', fontsize=14)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def vector(data, prefix):
    return np.column_stack([data[f'{prefix}_{axis}_m'] for axis in 'xyz'])


def plot_paths(data, destination, title):
    time = data['t_s']
    real_base = np.column_stack([data['real_base_x_m'], data['real_base_y_m']])
    sim_base = np.column_stack([data['sim_base_x_m'], data['sim_base_y_m']])
    real_speed = np.hypot(data['real_base_vx_m_s'], data['real_base_vy_m_s'])
    sim_speed = np.hypot(data['sim_base_vx_m_s'], data['sim_base_vy_m_s'])
    real_racket = vector(data, 'real_racket')
    sim_racket = vector(data, 'sim_racket')
    vicon_racket = vector(data, 'vicon_racket')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
    ax = axes[0, 0]
    ax.plot(real_base[:, 0], real_base[:, 1], color=COLORS['measured'], label='Measured')
    ax.plot(sim_base[:, 0], sim_base[:, 1], color=COLORS['simulated'], label='Simulated')
    ax.scatter(real_base[0, 0], real_base[0, 1], color=COLORS['measured'], marker='o', s=25)
    ax.scatter(sim_base[0, 0], sim_base[0, 1], color=COLORS['simulated'], marker='x', s=35)
    ax.set_xlabel('World X [m]')
    ax.set_ylabel('World Y [m]')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Base XY path')
    ax.legend()
    decorate(ax)

    ax = axes[0, 1]
    ax.plot(time, real_speed, color=COLORS['measured'], label='Measured linear speed')
    ax.plot(time, sim_speed, color=COLORS['simulated'], label='Simulated linear speed')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Planar speed norm [m/s]')
    ax.set_title('Base planar speed')
    ax.legend()
    decorate(ax)

    ax = axes[1, 0]
    ax.plot(time, np.linalg.norm(sim_racket - real_racket, axis=1),
            color=COLORS['measured'], label='Simulated vs measured FK')
    ax.plot(time, np.linalg.norm(sim_racket - vicon_racket, axis=1),
            color=COLORS['vicon'], label='Simulated vs Vicon')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Racket-center position error [m]')
    ax.set_title('Racket-center error')
    ax.legend()
    decorate(ax)

    ax = axes[1, 1]
    ax.plot(time, np.linalg.norm(sim_base - real_base, axis=1),
            color=COLORS['simulated'], label='Simulated vs measured')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Base XY position error [m]')
    ax.set_title('Base XY error')
    ax.legend()
    decorate(ax)

    fig.suptitle(f'{title}: base and racket paths', fontsize=14)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, metavar='RUN_DIR')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    groups = sorted(path for path in run_dir.glob('group_*') if path.is_dir())
    if not groups:
        parser.error(f'no group_* directories found in {run_dir}')
    for group in groups:
        source = group / 'comparison.csv'
        if not source.is_file():
            raise FileNotFoundError(f'missing comparison: {source}')
        data = read_comparison(source)
        plot_joint_comparison(data, group / 'comparison.png', group.name)
        plot_paths(data, group / 'paths.png', group.name)
        print(f'Plotted {group.name}: {len(data["t_s"])} samples', flush=True)


if __name__ == '__main__':
    main()
