"""核对高清视频的来源、因果缓存、结束几何与编码清单。"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from render_measured_hd import Playback
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
    manifest=json.loads((a.root/'manifest.json').read_text());assert manifest['expected']==manifest['validated']==105
    items={i['sample_id']:i for i in json.loads((a.root/'.source/input_manifest.json').read_text())['items']};records=[];groups={};total_frames=0;last_previews=[]
    for variant in ['chassis_model','chassis_recovery','chassis_guard']:
        metas=sorted((a.root/variant).glob('test_*.json'));assert len(metas)==35
        reproduced=json.loads((a.root/(variant+'_reproduction.json')).read_text());assert len(reproduced)==35 and all(r['same_states'] for r in reproduced)
        contacts=0
        for meta in metas:
            r=json.loads(meta.read_text());name=r['sample_id'];item=items[name];video=meta.with_suffix('.mp4');source=a.root/'.recording/runs'/variant/'nmpc6'/name
            assert r['validated'] and r['resolution']==[1920,1080] and r['fps']==60 and r['crf']==17 and r['playback_speed']==.25
            assert r['physical_hit_rate'] is None and r['camera_close']=='robot rear, follows base yaw; azimuth=180deg+yaw'
            assert digest(video)==r['sha256'] and digest(source/'frames.jsonl')==r['source_trace_sha256']
            assert digest(Path(item['csv']))==r['input_sha256']
            stream=r['probe']['streams'][0];assert stream['r_frame_rate']=='60/1' and int(stream['nb_frames'])==r['frames']
            assert r['frames']==math.ceil(r['end_time_s']/.25*60)+1
            rows=[json.loads(s) for s in (source/'frames.jsonl').read_text().splitlines()];cache=[json.loads(s) for s in (a.root/'.recording/cache'/(name+'.jsonl')).read_text().splitlines()]
            raw=np.loadtxt(item['csv'],delimiter=',',skiprows=1);last_seq=0
            for index,row in enumerate(rows):
                t=row['t'];expected=[np.interp(t,raw[:,0],raw[:,j]) for j in (1,2,3)]
                np.testing.assert_allclose(row['ball'],expected,rtol=0,atol=1e-10)
                expected_received=0 if index==0 and t==0 else int(np.searchsorted(raw[:,0],t+1e-10,side='right'))
                assert row['received_observations']==expected_received
                f=row['forecast'];seq=f['seq'];assert last_seq<=seq<=row['received_observations'];last_seq=seq
                if not seq:assert not f['grid']
                else:
                    cached=cache[seq-1];assert cached['t_s']<=t+1e-10
                    expected=[[v[0],v[1],v[2],v[3],v[6],v[7]] for v in cached['forecast']['grid']]
                    np.testing.assert_allclose(f['grid'],expected,rtol=0,atol=1e-10)
                total_frames+=1
            play=Playback(a.team,rows,raw);play.at(r['end_time_s']);gap=play.gap();earlier=None
            if r['contact_time_s'] is not None:
                contacts+=1;assert abs(r['end_time_s']-r['contact_time_s'])<1e-12 and gap<=0
                if r['end_time_s']>=.0005:play.at(r['end_time_s']-.0005);earlier=play.gap();assert earlier>0
            else:assert r['end_time_s']==r['original_scope_end_s']
            records.append(dict(variant=variant,sample=name,frames=r['frames'],end_s=r['end_time_s'],end_reason=r['end_reason'],last_gap_m=gap,prior_half_ms_gap_m=earlier,sha256=r['sha256']))
        groups[variant]={'videos':35,'first_geometry_cutoffs':contacts,'no_physical_hit_rate':True}
    result={'validated_videos':105,'recorded_frames_checked':total_frames,'groups':groups,'recorded_states_reproduced':True,'forecast_is_exact_consumed_causal_cache':True,'settings':'same1080p60fps0.25xCRF17/rear-camera/first-contact cutoff','records':records}
    (a.root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--root',type=Path,required=True);main(p.parse_args())
