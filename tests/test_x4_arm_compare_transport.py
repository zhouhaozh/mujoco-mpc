#!/usr/bin/env python3
"""无设备测试：传输失效停止和远端prepare范围。所有CAN/sudo调用均mock。"""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from x4_arm_compare import Transport, send_terminal
spec=importlib.util.spec_from_file_location('remote',ROOT/'tools/x4_arm_test/x4_arm_remote.py')
remote=importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


class DummyViewer:
    def __init__(self): self.states=[]
    def update(self,value):
        self.states.append(value)
        return True
    def blocked_reason(self): return ""


class Checks(unittest.TestCase):
    def test_terminal_axis_aliases_and_rate_rejection(self):
        child='''import sys
for line in sys.stdin:
 if line.strip()=='quit': break
 print('received:'+line.strip(),flush=True)
'''
        messages=io.StringIO()
        transport=Transport([sys.executable,'-u','-c',child],DummyViewer(),io.StringIO(),messages)
        try:
            transport.phase='initialized'
            for command in ('3','select3','select 3'):
                transport.last_action=0
                self.assertTrue(send_terminal(transport,command))
            self.assertFalse(send_terminal(transport,'+'))
            self.assertFalse(send_terminal(transport,'3\n+'))
            transport.phase='startup_settling'
            self.assertFalse(send_terminal(transport,'3'))
            self.assertFalse(transport.closed)
            self.assertTrue(transport.send('heartbeat'))
            deadline=time.monotonic()+2
            while messages.getvalue().count('received:select 3')<3 and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertEqual(messages.getvalue().count('received:select 3'),3)
            self.assertNotIn('received:+',messages.getvalue())
            transport.closed=True
            with self.assertRaises(RuntimeError): send_terminal(transport,'3')
        finally: transport.close()

    def test_transport_drains_and_rejects_duplicate_sequence(self):
        child='''import sys,json
print('SDK log',flush=True)
print('X4_UI_READY '+json.dumps({'schema':'X4_ARM_COMPARE/1','phase':'initialized'}),flush=True)
for line in sys.stdin:
 if line.strip()=='quit': break
 print('X4_STATE '+json.dumps({'schema':'X4_ARM_COMPARE/1','seq':0}),flush=True)
'''
        viewer=DummyViewer()
        transport=Transport([sys.executable,'-u','-c',child],viewer,io.StringIO())
        try:
            end=time.monotonic()+2
            while transport.phase!='initialized' and time.monotonic()<end: time.sleep(.01)
            self.assertEqual(transport.phase,'initialized')
            transport.send('heartbeat')
            end=time.monotonic()+2
            while not viewer.states and time.monotonic()<end: time.sleep(.01)
            self.assertEqual(len(viewer.states),1)
            transport.send('heartbeat')
            end=time.monotonic()+2
            while not transport.closed and time.monotonic()<end: time.sleep(.01)
            self.assertTrue(transport.closed)
            self.assertIn('sequence',transport.failure)
            with self.assertRaises(RuntimeError): transport.send('+')
        finally: transport.close()

    def test_preinit_cannot_move_or_start_early(self):
        child='import time; time.sleep(10)'
        transport=Transport([sys.executable,'-u','-c',child],DummyViewer(),io.StringIO())
        try:
            with self.assertRaises(ValueError): transport.send('+')
            with self.assertRaises(ValueError): transport.send('START-HOME')
            with self.assertRaises(ValueError): transport.send('loop 3\nloop 3')
        finally: transport.close()

    def test_invalid_viewer_packet_ends_transport_and_terminal_shares_gate(self):
        child = "import time; time.sleep(10)"
        viewer=DummyViewer()
        viewer.blocked_reason=lambda: "INVALID STATE"
        transport=Transport([sys.executable,'-u','-c',child],viewer,io.StringIO())
        try:
            transport.phase='initialized'
            with self.assertRaisesRegex(ValueError,'INVALID STATE'): transport.send('+')
            with self.assertRaisesRegex(ValueError,'INVALID STATE'): transport.send('loop 3')
        finally: transport.close()
        viewer=DummyViewer()
        viewer.update=lambda value: False
        packet=json.dumps({'schema':'X4_ARM_COMPARE/1','seq':0})
        child="import time; print("+repr('X4_STATE '+packet)+",flush=True); time.sleep(10)"
        transport=Transport([sys.executable,'-u','-c',child],viewer,io.StringIO())
        try:
            end=time.monotonic()+2
            while not transport.closed and time.monotonic()<end: time.sleep(.01)
            self.assertTrue(transport.closed)
            self.assertIn('viewer rejected',transport.failure)
        finally: transport.close()

    def state(self,up=False,rate=1000000):
        return {'flags':['UP'] if up else [],'linkinfo':{'info_data':{
            'state':'ERROR-ACTIVE' if up else 'STOPPED','ctrlmode':['FD','ONE-SHOT'],
            'bittiming':{'bitrate':rate},'data_bittiming':{'bitrate':2000000}}}}

    def test_prepare_never_starts_sdk_and_only_touches_arm_can(self):
        states={name:self.state() for name in remote.CHANNELS}
        calls=[]
        def run(cmd,**kwargs):
            calls.append(cmd)
            if cmd[:4]==['sudo','-n','ip','link'] and cmd[-1]=='up':
                states[cmd[5]]=self.state(True)
            return subprocess.CompletedProcess(cmd,0)
        with tempfile.TemporaryDirectory() as folder, patch.object(remote,'ROOT',Path(folder)), \
             patch.object(remote,'check_processes'), patch.object(remote,'can_info',side_effect=lambda n: states[n]), \
             patch.object(remote.subprocess,'run',side_effect=run), patch.object(sys,'argv',['remote','prepare']), \
             patch.object(sys.stdin,'isatty',return_value=True),patch.object(sys.stdout,'isatty',return_value=True):
            remote.main()
        self.assertEqual(len(calls),8) # inspect,sudo-v,each arm interface config/up
        self.assertNotIn('hardware',' '.join(' '.join(c) for c in calls))
        changed=[c[5] for c in calls if c[:4]==['sudo','-n','ip','link']]
        self.assertEqual(changed,['can4','can4','can5','can5','can6','can6'])

    def test_prepare_refuses_live_wrong_configuration_before_sudo(self):
        calls=[]
        with tempfile.TemporaryDirectory() as folder, patch.object(remote,'ROOT',Path(folder)), \
             patch.object(remote,'check_processes'), patch.object(remote,'can_info',return_value=self.state(True,500000)), \
             patch.object(remote.subprocess,'run',side_effect=lambda c,**k: calls.append(c)), \
             patch.object(sys,'argv',['remote','prepare']),patch.object(sys.stdin,'isatty',return_value=True), \
             patch.object(sys.stdout,'isatty',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'已UP'): remote.main()
        self.assertEqual(len(calls),1)
        self.assertNotIn('sudo',calls[0])

    def test_remote_run_no_can_refuses_before_exec(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(remote,'ROOT',Path(folder)), \
             patch.object(remote,'check_processes'),patch.object(remote,'can_info',return_value=self.state()), \
             patch.object(remote.os,'execve') as execute,patch.object(sys,'argv',['remote','run']), \
             patch.object(sys.stdin,'isatty',return_value=True),patch.object(sys.stdout,'isatty',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'CAN未准备'): remote.main()
            execute.assert_not_called()


if __name__=='__main__': unittest.main(verbosity=2)
