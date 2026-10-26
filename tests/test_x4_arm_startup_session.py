#!/usr/bin/env python3
"""Ubuntu SDK sim only: startup cancellation, watchdog, and diagnostic evidence."""
import argparse
import json
import os
import signal
from pathlib import Path
from test_x4_arm_compare_session import Session

p = argparse.ArgumentParser()
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
for case in ('quit', 'disconnect', 'heartbeat_timeout'):
    s = Session(a.binary, a.config, a.output / case)
    try:
        s.wait(lambda: s.phase == 'startup_settling')
        if case == 'quit':
            s.send('quit')
        elif case == 'disconnect':
            s.p.stdin.close()
        else:
            os.kill(s.p.pid, signal.SIGSTOP)
            import time
            time.sleep(.65)
            os.kill(s.p.pid, signal.SIGCONT)
        s.heartbeat = False
        while s.p.poll() is None:
            s.pump()
        s.pump(0)
        assert s.p.returncode == 2, s.lines[-10:]
        result = json.loads((s.output / 'result.json').read_text())
        expected = 'startup_heartbeat_timeout' if case == 'heartbeat_timeout' else 'startup_cancelled'
        assert result['reason'] == expected, result
        rows = [json.loads(x) for x in (s.output / 'startup_feedback.jsonl').read_text().splitlines()]
        assert rows[-1]['reason'] == expected
        assert rows[-1]['feedback_observed']
        assert len(rows[-1]['q']) == len(rows[-1]['dq']) == 5
        assert any(all(t > 0 for t in r['stamp_s']) for r in rows)
        assert not any(r.get('initialized') for r in s.states)
    finally:
        if s.p.poll() is None:
            os.kill(s.p.pid, signal.SIGCONT)
            s.p.terminate()
            s.p.wait(timeout=5)
(a.output / 'verification.json').write_text(json.dumps({'passed': True, 'cases': 3, 'hardware_started': False}))
print('3 startup exception and evidence checks passed (SDK sim only)')
