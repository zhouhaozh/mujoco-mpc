#!/usr/bin/env python3
"""SDK模拟器UI协议集成：循环/暂停/停止/输入/断心跳，无硬件。"""
import argparse
import json
import selectors
import subprocess
import time
from pathlib import Path


class Session:
    def __init__(self, binary, config, output, step_deg=1):
        self.output = output
        self.p = subprocess.Popen([str(binary), '--config', str(config), '--backend', 'sim',
                                   '--step-deg', str(step_deg), '--ui', '--output', str(output)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.p.stdout, selectors.EVENT_READ)
        self.pending = b''
        self.states, self.lines = [], []
        self.phase, self.last_heartbeat = '', 0
        self.heartbeat = True

    def send(self, text):
        self.p.stdin.write((text + '\n').encode())

    def pump(self, duration=.03):
        if self.phase in ('startup_settling', 'initialized') and self.heartbeat and self.p.poll() is None and time.monotonic()-self.last_heartbeat>.1:
            self.send('heartbeat')
            self.last_heartbeat = time.monotonic()
        for key, _ in self.selector.select(duration):
            import os
            raw = os.read(key.fd, 16384)
            self.pending += raw
            while b'\n' in self.pending:
                line, self.pending = self.pending.split(b'\n', 1)
                text = line.decode(errors='replace')
                self.lines.append(text)
                if text.startswith('X4_UI_READY '):
                    self.phase = json.loads(text[12:])['phase']
                if text.startswith('X4_STATE '):
                    state = json.loads(text[9:], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                    assert not self.states or state['seq'] > self.states[-1]['seq']
                    self.states.append(state)

    def wait(self, predicate, timeout=15):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.pump()
            if predicate():
                return
            assert self.p.poll() is None, '\n'.join(self.lines[-20:])
        raise AssertionError('timeout: ' + '\n'.join(self.lines[-10:]))

    def ready(self):
        self.wait(lambda: self.states and self.states[-1]['settled'])

    def finish(self, command='quit', expected=0):
        if command:
            self.send(command)
        self.wait(lambda: self.p.poll() is not None, timeout=4)
        self.pump(0)
        assert self.p.returncode == expected, self.lines[-10:]
        self.output.joinpath('ui_transcript.txt').write_text('\n'.join(self.lines))
        self.selector.close()
        self.p.stdin.close()
        self.p.stdout.close()

    def events(self):
        return [json.loads(s) for s in self.output.joinpath('events.jsonl').read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    session = None
    try:
        session = Session(a.binary, a.config, a.output/'loop3')
        session.ready()
        origin = session.states[-1]['q_ref']
        session.send('heartbeat\n+\n+')
        session.wait(lambda: any('拒绝整批' in s for s in session.lines))
        session.send('heartbeat\nloop 3')
        session.wait(lambda: any(s['loop_active'] for s in session.states))
        session.send('select 2')  # loop期间不得换轴
        session.wait(lambda: not session.states[-1]['loop_active'] and session.states[-1]['settled'], timeout=40)
        session.finish()
        events = session.events()
        assert [e['detail'] for e in events if e['type']=='loop_move'] == ['J1+', 'J1-']*3
        assert sum(e['type']=='batch_rejected' for e in events) == 1
        for state in session.states:
            assert state['selected_joint']==1
            assert all(abs(state['q_ref'][j]-origin[j])<1e-10 for j in range(1,5))
            assert -1e-9 <= state['q_ref'][0]-origin[0] <= .017453293
            assert max(abs(x) for x in state['dq_ref']) <= .017453293
        assert max(abs(x-y) for x,y in zip(session.states[-1]['q_ref'], origin))<1e-10
        moves=[e for e in events if e['type']=='loop_move']
        assert all(y['t']-x['t']>=.75 for x,y in zip(moves,moves[1:]))

        session = Session(a.binary, a.config, a.output/'pause')
        session.ready()
        session.send('loop 10')
        session.wait(lambda: session.states[-1]['moving'])
        session.send('pause')
        session.wait(lambda: not session.states[-1]['loop_active'] and session.states[-1]['settled'])
        session.send('select 3')
        session.wait(lambda: session.states[-1]['selected_joint']==3)
        session.finish()
        assert [e['detail'] for e in session.events() if e['type']=='loop_move']==['J1+','J1-']

        session = Session(a.binary, a.config, a.output/'stop')
        session.ready()
        session.send('loop 3')
        session.wait(lambda: session.states[-1]['moving'])
        session.send('heartbeat\n+\nstop')
        session.wait(lambda: session.states[-1]['session_stopped'])
        before = session.states[-1]['q_ref']
        session.send('loop 1')
        session.wait(lambda: any('本轮已停止' in s for s in session.lines))
        session.finish()
        stopped = [s for s in session.states if s['session_stopped']]
        assert all(s['q_ref']==before and s['dq_ref']==[0]*5 and not s['loop_active'] for s in stopped)

        session = Session(a.binary, a.config, a.output/'lease')
        session.ready()
        session.send('loop 3')
        session.wait(lambda: session.states[-1]['moving'])
        session.heartbeat = False
        started = time.monotonic()
        session.finish(command='', expected=3)
        assert .2 < time.monotonic()-started < 1.5
        assert session.states[-1]['reason']=='ui_heartbeat_timeout'
        assert session.states[-1]['fault'] and session.states[-1]['session_stopped']
        assert session.states[-1]['dq_ref']==[0]*5
        assert len([e for e in session.events() if e['type']=='loop_move'])==1
        result = {'passed': True, 'backend': 'sim', 'cases': ['loop3', 'pause', 'stop', 'lease'],
                  'hardware_started': False}
        (a.output/'verification.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    finally:
        if session and session.p.poll() is None:
            session.p.terminate()
            session.p.wait(timeout=5)


if __name__=='__main__':
    main()
