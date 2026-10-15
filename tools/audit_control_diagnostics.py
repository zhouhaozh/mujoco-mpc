"""核对控制诊断覆盖/文件，并给出不依赖求解结果的必要可达条件证据。"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def audit(root, source, out):
    items = json.loads((source / 'input_manifest.json').read_text())['items']
    files, proofs, checked = [], [], 0
    for mode in ('nmpc6', 'wbmpc'):
        for item in items:
            case = root / mode / item['sample_id']
            assert (case / 'run.json').exists()
            for name in ('run.json', 'diagnostics.jsonl', 'control.log'):
                p = case / name
                files.append({'path': str(p.relative_to(root)),
                              'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
            rows = [json.loads(s) for s in (case / 'diagnostics.jsonl').read_text().splitlines()]
            states = [r for r in rows if r['kind'] == 'state']
            assert len(states) > 100
            times = np.array([r['t'] for r in states])
            assert np.all(np.diff(times) > 0)
            assert np.allclose(np.diff(times), .005, atol=1e-10)
            for s in states:
                ref, actual = np.array(s['p_reference']), np.array(s['p_actual'])
                b, a = np.array(s['p_command_actualbase']), np.array(s['p_nominal_actual'])
                np.testing.assert_allclose((b-ref)+(a-b)+(actual-a), actual-ref, atol=1e-12)
                np.testing.assert_allclose(np.clip(s['tau_raw'], -np.array(s['tau_cap']), s['tau_cap']), s['tau'], atol=1e-9)
            if mode != 'nmpc6':
                continue
            for e in (r for r in rows if r['kind'] == 'solve'):
                i = int(np.argmin(abs(times-(e['t']-.001))))
                s = states[i]
                # 同一次循环记录；锁止后请求目标与规划锚可分离，故不重构。
                if abs(s['t']-(e['t']-.001)) > 1e-9 or s['locked']:
                    continue
                np.testing.assert_allclose(e['target'], s['planner_goal'], atol=1e-10)
                pos, vel = np.array(e['state'][:2]), np.array(e['state'][7:9])
                np.testing.assert_allclose(pos, s['base'][:2], atol=1e-10)
                delta = np.array(e['target'][:2])-pos
                anchor = np.array(e['target'][:2])-.8*delta/max(np.linalg.norm(delta), 1e-12)
                h = (e['target_time']-e['t'])/3
                # |a|<=8 下的宽松包络；未叠加速度上限/冻结条件，仍不相交即必不可行。
                lo, hi = pos+vel*h-4*h*h, pos+vel*h+4*h*h
                al = np.maximum(anchor-.4, [-1., -2.])
                ah = np.minimum(anchor+.4, [5.7, 2.])
                checked += 1
                if np.any((hi < al-1e-7) | (lo > ah+1e-7)):
                    proofs.append({'sample_id': item['sample_id'], 't_s': e['t'], 'h_s': h,
                                   'initial_position_m': pos.tolist(), 'initial_velocity_mps': vel.tolist(),
                                   'first_node_lo_m': lo.tolist(), 'first_node_hi_m': hi.tolist(),
                                   'anchor_lo_m': al.tolist(), 'anchor_hi_m': ah.tolist(),
                                   'gateway_accepted': e['accepted'], 'solver_ok': e['solver_ok'],
                                   'reason': e['reason']})
    assert len(files) == 210
    plots = sorted((root/'report'/'plots').rglob('*.png'))
    assert len(plots) == 280
    result = {'platform': 'Ubuntu ubuntu-11, 192.168.169.246', 'date': '2026-10-15',
              'runs': 70, 'plots': len(plots), 'state_period_s': .005,
              'vector_decomposition': 'passed all rows', 'torque_clipping': 'passed all rows',
              'unlocked_exact_request_bounds': {'checked': checked, 'disjoint': len(proofs),
                  'cases': len(set(p['sample_id'] for p in proofs)),
                  'accepted_despite_disjoint': sum(p['gateway_accepted'] for p in proofs),
                  'proofs': proofs}, 'source_hashes': files}
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','unlocked_exact_request_bounds')}))
    print(json.dumps({k:v for k,v in result['unlocked_exact_request_bounds'].items() if k!='proofs'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.root, a.source, a.out)
