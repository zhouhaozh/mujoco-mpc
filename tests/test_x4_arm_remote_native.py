#!/usr/bin/env python3
"""Ubuntu原生remote入口的无设备集成检查；ip/sudo/控制器均为临时测试替身。"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import shutil
import subprocess
import sys
import time
import unittest

BINARY: Path
OUTPUT: Path

FAKE_IP = r'''#!/usr/bin/python3
import json,os,sys
from pathlib import Path
a=sys.argv[1:]
with open(os.environ['FAKE_COMMANDS'],'a') as f:f.write(json.dumps(['ip']+a)+'\n')
p=Path(os.environ['FAKE_CAN_STATE']);states=json.loads(p.read_text())
if a[:5]==['-details','-j','link','show',a[-1]] and len(a)==5:
 if a[-1] not in ('can4','can5','can6'):raise RuntimeError('non-arm interface requested')
 if os.environ.get('FAKE_IP_MODE')=='malformed':print('{broken');sys.exit(0)
 v=states[a[-1]]
 if os.environ.get('FAKE_IP_MODE')=='wrong_interface':v=dict(v,ifname='can7')
 print(json.dumps([v]));sys.exit(0)
if a[:3]!=['link','set',a[2]] or a[2] not in ('can4','can5','can6'):raise RuntimeError('unexpected ip command')
name=a[2]
if a[3:]==['type','can','bitrate','1000000','dbitrate','2000000','fd','on','one-shot','on']:
 states[name]['linkinfo']['info_data'].update(bittiming={'bitrate':1000000},data_bittiming={'bitrate':2000000},ctrlmode=['FD','ONE-SHOT'])
elif a[3:]==['up']:
 states[name]['flags']=['UP'];states[name]['linkinfo']['info_data']['state']='ERROR-ACTIVE'
else:raise RuntimeError('unexpected configuration')
p.write_text(json.dumps(states))
'''

FAKE_SUDO = r'''#!/usr/bin/python3
import json,os,sys
a=sys.argv[1:]
with open(os.environ['FAKE_COMMANDS'],'a') as f:f.write(json.dumps(['sudo']+a)+'\n')
if a==['-v']:sys.exit(0)
if a[:2]!=['-n','ip']:raise RuntimeError('unexpected sudo command')
os.execv(os.environ['FAKE_IP'],[os.environ['FAKE_IP']]+a[2:])
'''

FAKE_CONTROLLER = r'''#!/usr/bin/python3
import fcntl,json,os,select,sys,termios
from pathlib import Path
a=sys.argv[1:]
with open(os.environ['FAKE_COMMANDS'],'a') as f:f.write(json.dumps(['controller']+a)+'\n')
root=Path(__file__).resolve().parent.parent
locks=[]
for item in Path('/proc/self/fd').iterdir():
 try:
  if item.resolve()==root/'session.lock':locks.append({'fd':int(item.name),'cloexec':bool(fcntl.fcntl(int(item.name),fcntl.F_GETFD)&fcntl.FD_CLOEXEC)})
 except (FileNotFoundError,OSError):pass
seen={'argv':a,'ld_library_path':os.environ.get('LD_LIBRARY_PATH'),'stdin_tty':os.isatty(0),'stdout_tty':os.isatty(1),'lock_fds':locks,'sdk_started':False,'hardware_started':False}
if os.isatty(0):seen['echo']=bool(termios.tcgetattr(0)[3]&termios.ECHO)
if '--backend' in a:
 seen['unexpected_stdin']=os.read(0,4096).decode(errors='replace') if select.select([0],[],[],.1)[0] else ''
with open(os.environ['FAKE_SEEN'],'a') as f:f.write(json.dumps(seen)+'\n')
print(json.dumps({'native_fake_controller':True,**seen}),flush=True)
'''


def state(name, up=False, rate=1000000):
    return {'ifname': name, 'flags': ['UP'] if up else [], 'linkinfo': {'info_data': {
        'state': 'ERROR-ACTIVE' if up else 'STOPPED', 'ctrlmode': ['FD', 'ONE-SHOT'],
        'bittiming': {'bitrate': rate}, 'data_bittiming': {'bitrate': 2000000}}}}


class Checks(unittest.TestCase):
    def setUp(self):
        self.folder = OUTPUT / self._testMethodName
        self.package = self.folder / 'package'
        self.helpers = self.folder / 'fake_path'
        for directory in (self.package / 'bin', self.package / 'config', self.package / 'lib', self.helpers):
            directory.mkdir(parents=True, exist_ok=False)
        self.binary = self.package / 'bin/x4_arm_remote'
        shutil.copy2(BINARY, self.binary)
        (self.package / 'config/robot_8w4s.ini').write_text('# fake controller does not load SDK\n')
        self.commands = self.folder / 'commands.jsonl'
        self.seen = self.folder / 'controller_seen.jsonl'
        self.states = self.folder / 'can_state.json'
        self.set_states()
        for path, source in ((self.helpers / 'ip', FAKE_IP), (self.helpers / 'sudo', FAKE_SUDO),
                             (self.package / 'bin/x4_arm_axis_test', FAKE_CONTROLLER)):
            path.write_text(source)
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.helpers) + ':/usr/bin:/bin',
                        FAKE_COMMANDS=str(self.commands), FAKE_CAN_STATE=str(self.states),
                        FAKE_IP=str(self.helpers / 'ip'), FAKE_SEEN=str(self.seen))

    def set_states(self, **overrides):
        values = {name: state(name) for name in ('can4', 'can5', 'can6')}
        values.update(overrides)
        self.states.write_text(json.dumps(values))

    def ready(self):
        self.set_states(**{name: state(name, True) for name in ('can4', 'can5', 'can6')})

    def calls(self):
        return [json.loads(line) for line in self.commands.read_text().splitlines()] if self.commands.exists() else []

    def observations(self):
        return [json.loads(line) for line in self.seen.read_text().splitlines()] if self.seen.exists() else []

    def invoke(self, *args, tty=False):
        command = [str(self.binary), *args]
        if not tty:
            result = subprocess.run(command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=8)
        else:
            master, slave = pty.openpty()
            process = subprocess.Popen(command, env=self.env, stdin=slave, stdout=slave, stderr=slave,
                                       start_new_session=True)
            os.close(slave)
            chunks = []
            deadline = time.monotonic() + 8
            try:
                while True:
                    if time.monotonic() > deadline:
                        process.kill()
                        raise RuntimeError('native wrapper test timed out')
                    if select.select([master], [], [], .1)[0]:
                        try:
                            data = os.read(master, 65536)
                        except OSError as error:
                            if error.errno == errno.EIO:
                                break
                            raise
                        if not data:
                            break
                        chunks.append(data)
                    elif process.poll() is not None:
                        break
                result = subprocess.CompletedProcess(command, process.wait(timeout=2),
                                                     b''.join(chunks).decode(errors='replace'))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
                os.close(master)
        with (self.folder / 'output.txt').open('a') as log:
            log.write(json.dumps({'args': args, 'tty': tty, 'exit': result.returncode}, ensure_ascii=False) + '\n')
            log.write(result.stdout + '\n')
        return result

    def test_inspect_read_only_and_arguments(self):
        result = self.invoke('inspect', '--step-deg', '1', '--test-mode', 'tracking')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertFalse(any(call[0] == 'sudo' for call in self.calls()))
        self.assertEqual([call[-1] for call in self.calls() if call[0] == 'ip'], ['can4', 'can5', 'can6'])
        seen = self.observations()[0]
        self.assertNotIn('--backend', seen['argv'])
        self.assertEqual(seen['argv'][-4:], ['--step-deg', '1', '--test-mode', 'tracking'])
        self.assertEqual(seen['ld_library_path'], str(self.package / 'lib'))

    def test_default_and_explicit_thirty_degree_arguments(self):
        self.ready()
        for arguments, tty in ((('inspect',), False),
                               (('inspect', '--step-deg', '30'), False),
                               (('run', '--step-deg', '30'), True)):
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments, tty=tty)
                self.assertEqual(result.returncode, 0, result.stdout)
                seen = self.observations()[-1]
                argv = seen['argv']
                self.assertEqual(argv[argv.index('--step-deg') + 1], '30')
                self.assertEqual(argv[argv.index('--test-mode') + 1], 'direction')
                self.assertEqual('--backend' in argv, arguments[0] == 'run')
        self.assertFalse(any(call[0] == 'sudo' for call in self.calls()))

    def test_prepare_changes_only_down_arm_channels(self):
        self.set_states(can5=state('can5', True))
        result = self.invoke('prepare', tty=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        mutations = [call for call in self.calls() if call[:3] == ['ip', 'link', 'set']]
        self.assertEqual([call[3] for call in mutations], ['can4', 'can4', 'can6', 'can6'])
        self.assertEqual(sum(call == ['sudo', '-v'] for call in self.calls()), 1)
        self.assertTrue(all('--backend' not in seen['argv'] for seen in self.observations()))
        self.assertTrue(all('UP' in value['flags'] for value in json.loads(self.states.read_text()).values()))

    def test_prepare_refuses_up_wrong_bitrate_without_sudo(self):
        self.set_states(can4=state('can4', True, 500000))
        before = self.states.read_text()
        result = self.invoke('prepare', tty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('已UP', result.stdout)
        self.assertFalse(any(call[0] == 'sudo' for call in self.calls()))
        self.assertEqual(before, self.states.read_text())

    def test_run_requires_prepared_channels(self):
        result = self.invoke('run', tty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CAN未准备', result.stdout)
        self.assertEqual(self.observations(), [])

    def test_prepare_and_run_require_tty(self):
        for action in ('prepare', 'run'):
            with self.subTest(action=action):
                result = self.invoke(action)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('交互终端', result.stdout)
        self.assertEqual(self.calls(), [])

    def test_run_exec_arguments_library_echo_and_inherited_lock(self):
        self.ready()
        result = self.invoke('run', '--step-deg', '1', '--test-mode', 'tracking', tty=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        seen = self.observations()[0]
        self.assertTrue(seen['stdin_tty'] and seen['stdout_tty'])
        self.assertFalse(seen['echo'])
        self.assertEqual(seen['unexpected_stdin'], '')
        self.assertEqual(seen['ld_library_path'], str(self.package / 'lib'))
        self.assertTrue(seen['lock_fds'])
        self.assertTrue(all(not value['cloexec'] for value in seen['lock_fds']))
        argv = seen['argv']
        self.assertEqual(argv[argv.index('--backend') + 1], 'hardware')
        self.assertEqual(argv[argv.index('--step-deg') + 1], '1')
        self.assertEqual(argv[argv.index('--test-mode') + 1], 'tracking')
        self.assertIn('--ui', argv)
        self.assertEqual(argv[argv.index('--config') + 1], str(self.package / 'config/robot_8w4s.ini'))
        self.assertEqual(Path(argv[argv.index('--output') + 1]).parent, self.package / 'records')
        self.assertFalse(any(call[0] == 'sudo' for call in self.calls()))

    def test_busy_lock_refuses_before_external_commands(self):
        with (self.package / 'session.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.invoke('run', tty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('会话锁', result.stdout)
        self.assertEqual(self.calls(), [])

    def test_malformed_or_wrong_interface_json_refuses(self):
        for mode in ('malformed', 'wrong_interface'):
            with self.subTest(mode=mode):
                self.env['FAKE_IP_MODE'] = mode
                result = self.invoke('prepare', tty=True)
                self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call[0] == 'sudo' for call in self.calls()))

    def test_known_control_process_refuses_before_ip(self):
        child = subprocess.Popen(['/usr/bin/python3', '-c',
                                  "import ctypes,time; ctypes.CDLL(None).prctl(15,b'robotd',0,0,0); print('ready',flush=True); time.sleep(15)"],
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            result = self.invoke('prepare', tty=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('已有控制进程', result.stdout)
            self.assertEqual(self.calls(), [])
        finally:
            child.terminate()
            child.wait(timeout=3)
            child.stdout.close()

    def test_invalid_options_and_help_never_run_commands(self):
        for arguments in (('--step-deg', '2'), ('--test-mode', 'bad'), ('run', '--unknown')):
            self.assertNotEqual(self.invoke(*arguments).returncode, 0)
        self.assertEqual(self.invoke('--help').returncode, 0)
        self.assertEqual(self.calls(), [])

    def test_prepared_channels_need_no_sudo(self):
        self.ready()
        result = self.invoke('prepare', tty=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertFalse(any(call[0] == 'sudo' or call[:3] == ['ip', 'link', 'set'] for call in self.calls()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('仅在Ubuntu运行')
    global BINARY, OUTPUT
    BINARY, OUTPUT = args.binary.resolve(), args.output.resolve()
    if not BINARY.is_file():
        parser.error('native binary does not exist')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    report = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
              'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures],
              'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors],
              'binary': str(BINARY), 'platform': 'Ubuntu', 'hardware_started': False,
              'sdk_started': False, 'commands': 'temporary ip/sudo/controller fakes only'}
    (OUTPUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
