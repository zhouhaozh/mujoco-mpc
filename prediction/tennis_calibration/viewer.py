"""Ubuntu live causal trajectory replay and browser visualization.

The model consumes raw capture-time observations. Complete offline references
are only displayed; forecast errors enter the history after their target time
has been replayed. Dense curves are physics, MLP outputs remain discrete.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import platform
import sys
from threading import Event, RLock, Thread
from time import monotonic, perf_counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from .online_data import load_dataset, future_labels
from .online_predictor import OnlinePredictor
from .online_filter import forecast_batch

ROOT = Path(__file__).resolve().parents[1]


class LiveViewerPredictor(OnlinePredictor):
    """Separate capture updates from expensive forecasts, preserving ingestion.

    This adapter uses the frozen predictor's existing filter/history contract;
    physical parameters and MLP weights are never changed by the viewer.
    """
    def observe(self, capture_t_s, xyz):
        before = self.filter.last_t
        estimate = self.filter.update(capture_t_s, xyz)
        if self.filter.last_t is not None and self.filter.last_t != before:
            self._history.append((float(capture_t_s), np.asarray(xyz, dtype=float).copy()))
            while len(self._history) > 2 and self._history[1][0] < self.filter.last_t-.12:
                self._history.popleft()
        return estimate


def plain(value):
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class ViewerReplayEngine:
    def __init__(self, traces, model_dir, prediction_period_s=.02, predictor_factory=None):
        if prediction_period_s <= 0:
            raise ValueError('prediction_period_s must be positive')
        self.traces = {t['id']:t for t in traces}
        if not self.traces:
            raise ValueError('No replay traces')
        self.predictor = (predictor_factory or LiveViewerPredictor)(model_dir)
        self.period = float(prediction_period_s)
        self.lock = RLock()
        self.playing, self.speed, self.loop = True, 1., True
        self.revision = 0
        self.failure = None
        self.load(next((t['id'] for t in traces if t['split']=='test'), traces[0]['id']))

    def load(self, identity, wall_now=None):
        if identity not in self.traces:
            raise ValueError('Unknown trajectory')
        self.trace = self.traces[identity]
        self.identity = identity
        self.t0 = float(self.trace['raw_times'][0])
        scope = self.trace.get('eval_scope_end_s')
        end = float(self.trace['raw_times'][-1])
        if scope is not None:
            end = min(end,float(scope))
        self.count = max(1,int(np.searchsorted(self.trace['raw_times'],end,side='right')))
        self.duration = float(self.trace['raw_times'][self.count-1]-self.t0)
        select = self.trace['eval_times'] <= self.t0+self.duration+1e-12
        xyz = self.trace['eval_xyz'][select]
        bounds = xyz if len(xyz) else self.trace['raw_xyz'][:self.count]
        landing_xy = self.trace['bounce_reference_xy']
        height = self.predictor.physical['contact_height']
        self.truth = {'times_s':(self.trace['eval_times'][select]-self.t0).tolist(),
                      'xyz_m':xyz.tolist(),
                      'raw_times_s':(self.trace['raw_times'][:self.count]-self.t0).tolist(),
                      'raw_xyz_m':self.trace['raw_xyz'][:self.count].tolist(),
                      'bounce_relative_s':None if self.trace['bounce_reference_s'] is None else self.trace['bounce_reference_s']-self.t0,
                      'landing_xyz_m':None if landing_xy is None else np.r_[landing_xy,height].tolist(),
                      'bounds':{'min':np.min(bounds,axis=0).tolist(),'max':np.max(bounds,axis=0).tolist()},
                      'contact_height_m':height,
                      'reference_kind':'offline provisional cleaned position/contact reference'}
        self.reset(wall_now)

    def reset(self, wall_now=None):
        self.predictor.reset()
        self.index = 0
        self.wall_origin = monotonic() if wall_now is None else float(wall_now)
        self.playhead = 0.
        self.last_forecast_elapsed = -np.inf
        self.estimate = None
        self.prediction = None
        self.observed = []
        self.estimates = []
        self.pending = deque()
        self.errors = deque(maxlen=360)
        self.metrics = {'filter_ms':0.,'prediction_ms':0.,'total_ms':0.,
                        'prediction_count':0,'rejected_count':0,'observations_processed':0}
        self.failure = None
        self.revision += 1

    def cursor(self, wall_now=None):
        now = monotonic() if wall_now is None else float(wall_now)
        return self.playhead+(max(0.,now-self.wall_origin)*self.speed if self.playing else 0.)

    def set_playing(self, enabled, wall_now=None):
        now = monotonic() if wall_now is None else float(wall_now)
        self.playhead = min(self.cursor(now),self.duration)
        self.wall_origin = now
        self.playing = bool(enabled)
        self.revision += 1

    def set_speed(self, speed, wall_now=None):
        if speed not in (.25,.5,1.,2.):
            raise ValueError('Supported speeds: 0.25, 0.5, 1, 2')
        now = monotonic() if wall_now is None else float(wall_now)
        self.playhead = min(self.cursor(now),self.duration)
        self.wall_origin = now
        self.speed = float(speed)
        self.revision += 1

    def _mature_errors(self, stamp):
        still_pending = deque()
        for row in self.pending:
            if row['target_t_s'] > stamp+1e-12:
                still_pending.append(row)
                continue
            self.errors.append({k:v for k,v in row.items() if k!='target_t_s'})
        self.pending = still_pending

    def _forecast(self, snap):
        started = perf_counter()
        result = self.predictor.predict(horizons=[.1,.2,.5])
        times = np.linspace(0.,.5,51)
        curve = forecast_batch(snap['state'][None],np.array([snap['bounced']]),times,
                                **self.predictor.physical,config=self.predictor.config['filter'])[0]
        stamp = float(snap['t_s'])
        labels = future_labels(self.trace,stamp,[.1,.2,.5])
        points = []
        for forecast,label in zip(result['forecasts'],labels):
            target = None if not label['valid'] else np.asarray(label['target_xyz'])
            physical = forecast['physics_position_xyz_m']
            predicted = forecast['position_xyz_m']
            point = {'horizon_s':forecast['horizon_s'],
                     'position_xyz_m':predicted.tolist(),'physics_position_xyz_m':physical.tolist(),
                     'target_xyz_m':None if target is None else target.tolist(),
                     'error_physics_m':None,'error_mlp_m':None,'mlp_used':forecast['mlp_used'],
                     'mlp_reason':forecast['mlp_reason'],'label_reason':label['reason']}
            points.append(point)
            if target is not None:
                self.pending.append({'target_t_s':label['target_t_s'],'elapsed_s':stamp-self.t0,
                     'target_elapsed_s':label['target_t_s']-self.t0,'horizon_s':forecast['horizon_s'],
                     'physics_error_m':float(np.linalg.norm(physical-target)),
                     'mlp_error_m':float(np.linalg.norm(predicted-target))})
        land = result['landing']
        self.prediction = {'capture_t_s':stamp,'elapsed_s':stamp-self.t0,
            'relative_times_s':(times+stamp-self.t0).tolist(),'physics_xyz_m':curve.tolist(),
            'mlp_points':points,
            'landing':{'valid':bool(land.get('valid',False)),
                'reason':land.get('reason'),
                'physics_xyz_m':plain(land.get('physics_position_xyz_m')),
                'mlp_xyz_m':plain(land.get('position_xyz_m')),
                'physics_contact_relative_s':None if land.get('physics_contact_t_s') is None else land['physics_contact_t_s']-self.t0,
                'mlp_contact_relative_s':None if land.get('contact_t_s') is None else land['contact_t_s']-self.t0,
                'actual_contact_relative_s':self.truth['bounce_relative_s'],
                'mlp_used':land.get('mlp_used',False)}}
        self.last_forecast_elapsed = stamp-self.t0
        self.metrics['prediction_ms'] = (perf_counter()-started)*1000
        self.metrics['prediction_count'] += 1

    def _consume(self):
        started = perf_counter()
        stamp = float(self.trace['raw_times'][self.index])
        xyz = self.trace['raw_xyz'][self.index]
        filter_started = perf_counter()
        snap = self.predictor.observe(stamp,xyz)
        self.metrics['filter_ms'] = (perf_counter()-filter_started)*1000
        self.estimate = snap
        self.observed.append(xyz.tolist())
        if snap['initialized']:
            self.estimates.append({'t_s':stamp-self.t0,'p':snap['p'].tolist()})
            if stamp-self.t0 >= self.last_forecast_elapsed+self.period-1e-12:
                self._forecast(snap)
        if snap['reason']=='mahalanobis_outlier':
            self.metrics['rejected_count'] += 1
        self.metrics['observations_processed'] += 1
        self.metrics['total_ms'] = (perf_counter()-started)*1000
        self._mature_errors(stamp)
        self.index += 1
        self.revision += 1

    def advance(self, wall_now=None):
        if not self.playing:
            return
        now = monotonic() if wall_now is None else float(wall_now)
        due = self.cursor(now)
        while self.index < self.count and self.trace['raw_times'][self.index]-self.t0 <= due+1e-12:
            self._consume()
        if self.index == self.count:
            if self.loop and self.playing and due >= self.duration+.7:
                self.reset(now)
            elif not self.loop:
                self.set_playing(False,now)

    def step(self):
        self.set_playing(False)
        if self.index < self.count:
            self._consume()
            self.playhead = float(self.trace['raw_times'][self.index-1]-self.t0)
            self.wall_origin = monotonic()

    def trajectories(self):
        items = []
        for trace in sorted(self.traces.values(),key=lambda t:(t['split']!='test',t['id'])):
            t0,t1 = float(trace['raw_times'][0]),float(trace['raw_times'][-1])
            if trace.get('eval_scope_end_s') is not None:
                t1 = min(t1,float(trace['eval_scope_end_s']))
            bounce = trace['bounce_reference_s']
            items.append({'id':trace['id'],'split':trace['split'],'label':trace['id'].split('/')[-1],
                          'duration_s':t1-t0,'bounce_s_relative':None if bounce is None else bounce-t0})
        return {'trajectories':items,'selected_id':self.identity}

    def snapshot(self, include_truth=False):
        stamp = self.t0 if self.index==0 else float(self.trace['raw_times'][self.index-1])
        target = future_labels(self.trace,stamp,[0.])[0] if self.index else {'target_xyz':None}
        current_truth = target.get('target_xyz')
        estimate = None
        if self.estimate is not None:
            estimate = {k:self.estimate.get(k) for k in
                        ('p','v','a_res','bounced','bounce_confirmed','accepted','reason','initialized')}
        return plain({'revision':self.revision,'mode':'replay','playing':self.playing,
            'speed':self.speed,'loop':self.loop,'trajectory_id':self.identity,'split':self.trace['split'],
            'elapsed_s':stamp-self.t0,'duration_s':self.duration,'capture_t_s':stamp,
            'observation_index':self.index,'observation_count':self.count,
            'initialized':bool(self.estimate and self.estimate['initialized']),
            'truth':self.truth if include_truth else None,
            'observed_xyz_m':self.observed,'estimate_history':self.estimates,
            'current_truth_xyz_m':current_truth,'estimate':estimate,'prediction':self.prediction,
            'metrics':self.metrics,'error_history':list(self.errors),
            'failure':self.failure,'evaluation_clock':'forecast error revealed only after target capture timestamp'})

    def control(self, payload):
        action = payload.get('action')
        if action=='play': self.set_playing(True)
        elif action=='pause': self.set_playing(False)
        elif action=='reset': self.reset()
        elif action=='step': self.step()
        elif action=='select': self.load(payload.get('trajectory_id'))
        elif action=='speed': self.set_speed(float(payload.get('speed')))
        elif action=='loop': self.loop=bool(payload.get('loop'));self.revision+=1
        else: raise ValueError('Unknown control action')
        return self.snapshot(include_truth=True)


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self,address,engine):
        self.engine = engine
        self.closed = Event()
        super().__init__(address,ViewerHandler)

    def worker(self):
        while not self.closed.wait(.002):
            try:
                with self.engine.lock:
                    self.engine.advance()
            except Exception as error:
                with self.engine.lock:
                    self.engine.failure = type(error).__name__+': '+str(error)
                    self.engine.set_playing(False)


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self,format,*args):
        if args and str(args[1] if len(args)>1 else '') not in ('200','304'):
            super().log_message(format,*args)

    def send(self,status,payload,kind='application/json; charset=utf-8'):
        body = payload if isinstance(payload,bytes) else json.dumps(plain(payload),allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/','/viewer.html'):
            return self.send(200,(Path(__file__).parent/'viewer.html').read_bytes(),'text/html; charset=utf-8')
        with self.server.engine.lock:
            if parsed.path=='/api/trajectories':
                result = self.server.engine.trajectories()
            elif parsed.path=='/api/state':
                include = parse_qs(parsed.query).get('include_truth',['0'])[0]=='1'
                result = self.server.engine.snapshot(include)
            elif parsed.path=='/api/health':
                result = {'status':'ready','hostname':platform.node(),'pid':__import__('os').getpid(),
                          'mode':'raw causal replay','prediction_period_s':self.server.engine.period,
                          'model_dir':str(self.server.engine.predictor.run_dir)}
            else:
                return self.send(404,{'error':'not_found'})
        return self.send(200,result)

    def do_POST(self):
        if urlparse(self.path).path!='/api/control':
            return self.send(404,{'error':'not_found'})
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            return self.send(403,{'error':'cross_origin_control_rejected'})
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 8192:
                raise ValueError('Invalid request length')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload,dict):
                raise ValueError('Expected object')
            with self.server.engine.lock:
                result = self.server.engine.control(payload)
            return self.send(200,result)
        except (ValueError,TypeError,KeyError) as error:
            return self.send(400,{'error':str(error)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',type=Path,default=ROOT/'artifacts/tennis_online/20261008_final')
    parser.add_argument('--calibration',type=Path,default=ROOT/'artifacts/tennis_calibration/20261008_calibration')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=28771)
    parser.add_argument('--prediction-period',type=float,default=.02)
    args = parser.parse_args()
    if platform.system()!='Linux' or Path(sys.prefix).resolve()!=(ROOT/'.venv-tennis').resolve():
        raise RuntimeError('Run viewer in Ubuntu project .venv-tennis')
    traces = load_dataset(args.calibration)
    engine = ViewerReplayEngine(traces,args.model_dir,args.prediction_period)
    server = ViewerServer((args.host,args.port),engine)
    Thread(target=server.worker,daemon=True).start()
    print(json.dumps({'ready':True,'host':args.host,'port':args.port,'hostname':platform.node()}),flush=True)
    try:
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.closed.set()
        server.server_close()


if __name__=='__main__':
    main()
