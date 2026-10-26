#!/usr/bin/env python3
"""Ubuntu真实桌面 sim 回归；切焦/最小化/注入阻塞，不提供hardware入口。"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scenario',choices=['idle','background','minimized','stall'],required=True)
    p.add_argument('--seconds',type=float,default=12)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expect',choices=['success','heartbeat_timeout','observe'],default='success')
    p.add_argument('--resume-loop',action='store_true',help='sim only: after focus restoration run one J2 cycle')
    p.add_argument('--background-loop',action='store_true',
                   help='sim only: run one J2 cycle while a test terminal owns focus; check actual rendered poses')
    p.add_argument('--step-deg',type=int,choices=(1,10),default=10)
    p.add_argument('--native-input',action='store_true',help='sim only: X11 mouse click and real 2/L key events')
    a=p.parse_args()
    if not sys.platform.startswith('linux'):
        p.error('仅在Ubuntu执行')
    if a.native_input and not a.resume_loop:
        p.error('--native-input requires --resume-loop (sim only)')
    if a.resume_loop and (a.scenario!='background' or a.seconds<30):
        p.error('--resume-loop requires background and enough time for the selected amplitude')
    if a.background_loop and (a.scenario!='background' or a.seconds<20 or a.resume_loop):
        p.error('--background-loop requires background, at least 20 seconds, and no --resume-loop')
    a.output.mkdir(parents=True,exist_ok=False)
    # 复用当前图形桌面，不沿用昨天的DISPLAY=:1。
    raw=subprocess.check_output(['systemctl','--user','show-environment'],text=True)
    for line in raw.splitlines():
        name,_,value=line.partition('=')
        if name in ('DISPLAY','WAYLAND_DISPLAY','XAUTHORITY','XDG_SESSION_TYPE','XDG_RUNTIME_DIR','DBUS_SESSION_BUS_ADDRESS'):
            os.environ[name]=value
    import x4_arm_compare_viewer as v
    import glfw
    import x4_arm_compare as app
    # 仅对本测试创建的SDK sim窗口注入输入；发键前核验实际X11焦点。
    def native_input(viewer, key=None):
        import ctypes as c
        x11=c.CDLL('libX11.so.6'); xtst=c.CDLL('libXtst.so.6')
        x11.XOpenDisplay.restype=c.c_void_p; x11.XOpenDisplay.argtypes=[c.c_char_p]
        x11.XCloseDisplay.argtypes=[c.c_void_p]
        x11.XRaiseWindow.argtypes=[c.c_void_p,c.c_ulong]
        x11.XFlush.argtypes=[c.c_void_p]
        x11.XGetInputFocus.argtypes=[c.c_void_p,c.POINTER(c.c_ulong),c.POINTER(c.c_int)]
        x11.XKeysymToKeycode.argtypes=[c.c_void_p,c.c_ulong]; x11.XKeysymToKeycode.restype=c.c_ubyte
        xtst.XTestFakeMotionEvent.argtypes=[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_ulong]
        xtst.XTestFakeButtonEvent.argtypes=[c.c_void_p,c.c_uint,c.c_int,c.c_ulong]
        xtst.XTestFakeKeyEvent.argtypes=[c.c_void_p,c.c_uint,c.c_int,c.c_ulong]
        display=x11.XOpenDisplay(None)
        if not display: raise RuntimeError('native probe requires X11 display')
        try:
            window=glfw.get_x11_window(viewer.window)
            if key is None:
                x11.XRaiseWindow(display,window)
                x,y=glfw.get_window_pos(viewer.window)
                xtst.XTestFakeMotionEvent(display,-1,x+200,y+200,0)
                xtst.XTestFakeButtonEvent(display,1,1,0)
                xtst.XTestFakeButtonEvent(display,1,0,0)
            else:
                focused=c.c_ulong();revert=c.c_int()
                x11.XGetInputFocus(display,c.byref(focused),c.byref(revert))
                if focused.value != window:
                    raise RuntimeError('refusing synthetic key: test window does not own focus')
                code=x11.XKeysymToKeycode(display,ord(key))
                xtst.XTestFakeKeyEvent(display,code,1,0)
                xtst.XTestFakeKeyEvent(display,code,0,0)
            x11.XFlush(display)
        finally:x11.XCloseDisplay(display)
    phase={'name':'startup','since':time.monotonic()}
    diagnostics=[]
    latencies={}
    beats=[]
    lifecycle=[]
    presentations=[]
    done=threading.Event()
    trace=(a.output/'stacks.txt').open('w')
    terminal=None
    started=[None]
    actions=set()
    restore_at=max(8.2,a.seconds-4) if a.background_loop else 8.2
    background_select_at=[None]
    def wrap(name,fn):
        def called(*args,**kwargs):
            before=time.monotonic()
            presentation=None
            if name=='draw':
                viewer=args[0]
                window=viewer.window
                state,q_ref,q_fb,received,_=viewer._snapshot()
                presentation={'time':before,
                              'control_started':started[0] is not None,
                              'focused':bool(glfw.get_window_attrib(window,glfw.FOCUSED)),
                              'visible':bool(glfw.get_window_attrib(window,glfw.VISIBLE)),
                              'iconified':bool(glfw.get_window_attrib(window,glfw.ICONIFIED)),
                              'seq_before':state.get('seq'), 'received_at':received,
                              'sdk_reference':q_ref.tolist(),'sdk_feedback':q_fb.tolist()}
            phase.update(name=name,since=before)
            try:return fn(*args,**kwargs)
            finally:
                if presentation is not None:
                    presentation.update(seq_after=viewer._snapshot()[0].get('seq'),
                                        reference_qpos=viewer.reference.qpos[viewer._qpos_adr].tolist(),
                                        feedback_qpos=viewer.feedback.qpos[viewer._qpos_adr].tolist())
                    # 若读线程恰好在绘制期间更新，不能把相邻两个包误当作映射错误。
                    if (presentation['received_at'] is not None and
                            presentation['seq_before']==presentation['seq_after']):
                        expected_ref=viewer._sign*q_ref[viewer._permutation]+viewer._offset
                        expected_fb=viewer._sign*q_fb[viewer._permutation]+viewer._offset
                        presentation['max_pose_error_rad']=max(
                            max(abs(float(x)-y) for x,y in zip(expected_ref,presentation['reference_qpos'])),
                            max(abs(float(x)-y) for x,y in zip(expected_fb,presentation['feedback_qpos'])))
                    presentations.append(presentation)
                latencies.setdefault(name,[]).append(time.monotonic()-before)
                phase.update(name='between_calls',since=time.monotonic())
        return called
    for name in ('swap_buffers','poll_events'):
        setattr(glfw,name,wrap(name,getattr(glfw,name)))
    v.CompareViewer._draw=wrap('draw',v.CompareViewer._draw)
    original_send=app.Transport.send
    def send(self,command):
        result=original_send(self,command)
        if command=='heartbeat':beats.append(time.monotonic())
        return result
    app.Transport.send=send
    original_render=v.CompareViewer.render
    def render(self):
        nonlocal terminal
        now=time.monotonic()
        if started[0] is None:started[0]=now
        age=now-started[0]
        if age>=2 and 'trigger' not in actions:
            actions.add('trigger')
            lifecycle.append({'event':'trigger','time':now,'scenario':a.scenario})
            if a.scenario=='background':
                terminal=subprocess.Popen(['gnome-terminal','--wait','--title=X4 background regression',
                                           '--','bash','-c',f'sleep {restore_at-2:.3f}'],
                                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            elif a.scenario=='minimized':glfw.iconify_window(self.window)
            elif a.scenario=='stall':time.sleep(.8) # 主线程停止，reader不能续租。
        if (a.background_loop and 'trigger' in actions and 'restore' not in actions and
                not glfw.get_window_attrib(self.window,glfw.FOCUSED)):
            state=self._snapshot()[0]
            if state.get('initialized') and state.get('settled') and 'background_select' not in actions:
                self._dispatch('select 2')
                actions.add('background_select')
                background_select_at[0]=now
                lifecycle.append({'event':'background_select','time':now,'focused':False})
            if (background_select_at[0] is not None and now-background_select_at[0]>=.25 and
                    'background_loop' not in actions):
                self._dispatch('loop 1')
                actions.add('background_loop')
                lifecycle.append({'event':'background_loop','time':now,'focused':False})
        if age>=restore_at and a.scenario in ('background','minimized') and 'restore' not in actions:
            actions.add('restore');glfw.restore_window(self.window)
            if a.native_input:native_input(self)
            else:glfw.focus_window(self.window)
            lifecycle.append({'event':'restore','time':time.monotonic()})
        if a.resume_loop and age>=8.5 and 'select' not in actions:
            actions.add('select')
            if a.native_input:native_input(self,'2')
            else:self._dispatch('select 2')
        if a.resume_loop and age>=8.8 and 'loop' not in actions:
            actions.add('loop')
            if a.native_input:native_input(self,'l')
            else:self._dispatch('loop 1')
        lifecycle.append({'event':'frame','time':time.monotonic(),
                          'focused':bool(glfw.get_window_attrib(self.window,glfw.FOCUSED)),
                          'iconified':bool(glfw.get_window_attrib(self.window,glfw.ICONIFIED))})
        return original_render(self)
    v.CompareViewer.render=render
    def monitor():
        last=''
        while not done.wait(.10):
            snapshot=dict(phase)
            if time.monotonic()-snapshot['since']>.25 and snapshot['name']!='between_calls':
                key=str(snapshot)
                if key!=last:
                    diagnostics.append(dict(snapshot,observed=time.monotonic()))
                    faulthandler.dump_traceback(file=trace,all_threads=True)
                    last=key
    observer=threading.Thread(target=monitor,daemon=True);observer.start()
    result=None
    app_error=None
    try:
        sys.argv=['x4_arm_compare.py','--backend','sim','--step-deg',str(a.step_deg),'--smoke-seconds',str(a.seconds),
                  '--output',str(a.output/'run')]
        result=app.main()
    except Exception as error:
        # 控制端按看门狗退出后，启动器可因失去管道而抛异常；记录并在下方判定。
        app_error=f'{type(error).__name__}: {error}'
    finally:
        done.set();observer.join(timeout=2);trace.close()
        if terminal is not None:
            try:terminal.wait(timeout=8)
            except subprocess.TimeoutExpired:terminal.terminate()
        path=a.output/'run/controller/result.json'
        controller=json.loads(path.read_text()) if path.exists() else None
        trigger_time=next((event['time'] for event in lifecycle if event['event']=='trigger'),float('inf'))
        restore_time=next((event['time'] for event in lifecycle if event['event']=='restore'),float('inf'))
        background_frames=[f for f in presentations if f['control_started'] and not f['focused']
                           and f['visible'] and not f['iconified'] and f['received_at'] is not None
                           and trigger_time<=f['time']<restore_time]
        pose_errors=[f['max_pose_error_rad'] for f in presentations if 'max_pose_error_rad' in f]
        report={'scenario':a.scenario,'app_exit':result,'app_error':app_error,'controller':controller,
                'glfw_version':glfw.get_version_string().decode(),
                'heartbeat_count':len(beats),'max_heartbeat_gap_s':max([b-x for x,b in zip(beats,beats[1:])],default=0),
                'max_call_s':{name:max(values) for name,values in latencies.items()},
                'slow_stages':diagnostics,'frames':lifecycle,
                'draw_count':len(presentations),
                'background_draw_count':len(background_frames),
                'hidden_or_minimized_draw_count':sum(f['control_started'] and
                                                    (not f['visible'] or f['iconified']) for f in presentations),
                'background_distinct_sequences':len({f['seq_before'] for f in background_frames}),
                'background_max_draw_gap_s':max([later['time']-earlier['time']
                                                for earlier,later in zip(background_frames,background_frames[1:])],default=0),
                'background_j2_reference_span_rad':(max(f['reference_qpos'][1] for f in background_frames)-
                                                    min(f['reference_qpos'][1] for f in background_frames)) if background_frames else 0,
                'background_j2_feedback_span_rad':(max(f['feedback_qpos'][1] for f in background_frames)-
                                                   min(f['feedback_qpos'][1] for f in background_frames)) if background_frames else 0,
                'coherent_pose_check_count':len(pose_errors),
                'max_pose_error_rad':max(pose_errors,default=None),
                'draw_after_restore_count':sum(f['time']>e['time'] for f in presentations
                                               for e in lifecycle if e['event']=='restore')}
        (a.output/'presentations.jsonl').write_text(''.join(json.dumps(frame)+'\n' for frame in presentations))
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:value for k,value in report.items() if k!='frames'},indent=2))
    if a.expect=='success':
        assert app_error is None and result==0 and controller['reason']=='quit',report
        assert len(beats)>=a.seconds*5,report
        assert report['max_heartbeat_gap_s']<.5,report
        if a.scenario=='background':assert any(x.get('focused') is False for x in lifecycle)
        if a.scenario=='minimized':assert any(x.get('iconified') is True for x in lifecycle)
        if a.scenario in ('background','minimized'):
            assert report['hidden_or_minimized_draw_count']==0,report
            assert report['draw_after_restore_count']>0,report
        if a.scenario=='background':
            assert report['background_draw_count']>10,report
            assert report['background_distinct_sequences']>10,report
            assert report['background_max_draw_gap_s']<.5,report
        if a.background_loop:
            import math
            assert {'background_select','background_loop'}<=actions,report
            assert report['background_j2_reference_span_rad']>=math.radians(a.step_deg)-1e-8,report
            assert report['background_j2_feedback_span_rad']>.8*math.radians(a.step_deg),report
            assert report['coherent_pose_check_count']>10 and report['max_pose_error_rad']<1e-12,report
        if a.resume_loop or a.background_loop:
            events=[json.loads(line) for line in (a.output/'run/controller/events.jsonl').read_text().splitlines()]
            assert [e['detail'] for e in events if e['type']=='loop_move']==['J2+','J2-']*(3 if a.native_input else 1),events
            states=[json.loads(line)['state'] for line in (a.output/'run/telemetry.jsonl').read_text().splitlines()]
            active=[s for s in states if s.get('initialized') and s.get('q_ref') is not None]
            assert active[-1]['settled'] and not active[-1]['loop_active'],active[-1]
            origin=active[0]['q_ref']
            assert max(abs(x-y) for x,y in zip(active[-1]['q_ref'],origin))<1e-9
            assert all(abs(s['q_ref'][j]-origin[j])<1e-9 for s in active for j in (0,2,3,4))
            import math
            assert abs(max(s['q_ref'][1]-origin[1] for s in active)-math.radians(a.step_deg))<1e-8
            assert max(abs(dq) for s in active for dq in s['dq_ref'])==0
            manifest=json.loads((a.output/'run/controller/manifest.json').read_text())
            assert manifest['step_deg']==a.step_deg
            assert manifest['command_profile']=='direct_position'
            values=[s['q_ref'][1]-origin[1] for s in active]
            assert all(min(abs(v),abs(v-math.radians(a.step_deg)))<1e-8 for v in values)
    elif a.expect=='heartbeat_timeout':
        assert controller['reason']=='ui_heartbeat_timeout',report


if __name__=='__main__':main()
