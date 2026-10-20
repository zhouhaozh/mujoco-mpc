"""完整新预测器样本录像：27组×10球，另保留3个对侧发球演示。"""
import argparse,concurrent.futures,csv,hashlib,json,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
from evaluate_predictor_control import run as evaluate
from record_prediction_demo import render,trim_second_contact

MODES=('pfly','nmpc6','wbmpc');FAMILIES=('fixed','rand','gs');SEEDS=(2026,31,777)

def capture(job):
 team,prediction,out,mode,family,seed=job
 name=f'external_{mode}_{family}_{seed}';dest=out/'.records'/name
 a=SimpleNamespace(team=str(team),prediction=str(prediction),out=str(out/'.records'),trials=10,profile=False,video_trace=True)
 r=evaluate((a,'external',mode,family,seed))
 if 'hits' not in r:raise RuntimeError(r)
 return str(dest)

def validate_video(path):
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(path)],text=True))
 stream=probe['streams'][0]
 assert (stream['width'],stream['height'])==(1280,720)
 assert stream['codec_name']=='h264' and stream['r_frame_rate']=='60/1'
 subprocess.run(['ffmpeg','-threads','1','-v','error','-i',str(path),'-f','null','-'],check=True,timeout=120)
 return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,probe=probe)

def render_job(job):
 team,out,mode,source,trial,name,label=job
 target=out/mode/(name+'.mp4');meta=target.with_suffix('.json')
 if target.exists() and meta.exists():
  record=json.loads(meta.read_text())
  if record.get('validated') and record.get('sha256')==hashlib.sha256(target.read_bytes()).hexdigest():return record
 rows=[r for line in Path(source).read_text().splitlines() if (r:=json.loads(line))['trial']==trial]
 assert rows and rows[0]['t']==0
 record=render(team,out,mode,.25,rows_data=rows,output_path=target,label=label,previews=trial==0)
 record.update(validate_video(target));record.update(validated=True,source=str(source),trial=trial+1,sample_id=name)
 record['last_ground_contacts']=trim_second_contact(rows)[-1].get('ground_contacts')
 meta.write_text(json.dumps(record,indent=2)+'\n')
 return record

def main():
 p=argparse.ArgumentParser();p.add_argument('--team',required=True,type=Path);p.add_argument('--prediction',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--opposite',required=True,type=Path);p.add_argument('--workers',type=int,default=3);p.add_argument('--render-workers',type=int,default=3);p.add_argument('--capture-only',action='store_true');a=p.parse_args()
 for key in ('team','prediction','out','opposite'):setattr(a,key,getattr(a,key).resolve())
 a.out.mkdir(parents=True,exist_ok=True)
 for mode in MODES:(a.out/mode).mkdir(exist_ok=True)
 started=time.time();groups=[(a.team,a.prediction,a.out,m,f,s) for m in MODES for f in FAMILIES for s in SEEDS]
 with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:
  for n,result in enumerate(pool.map(capture,groups),1):print(f'capture {n}/27: {result}',flush=True)
 if a.capture_only:return
 jobs=[]
 for _,_,_,mode,family,seed in groups:
  source=a.out/'.records'/f'external_{mode}_{family}_{seed}'/'frames.jsonl'
  for trial in range(10):
   name=f'{family}_seed{seed}_sample{trial+1:02d}'
   jobs.append((a.team,a.out,mode,source,trial,name,f'{family} / {seed} / 样本{trial+1:02d}'))
 for mode in MODES:jobs.append((a.team,a.out,mode,a.opposite/mode/'frames.jsonl',0,'opposite_seed2026_sample01','对侧发球 / 样本01'))
 records=[]
 with concurrent.futures.ProcessPoolExecutor(a.render_workers) as pool:
  futures=[pool.submit(render_job,job) for job in jobs]
  for future in concurrent.futures.as_completed(futures):
   r=future.result();records.append(r);print(f"video {len(records)}/273: {r['mode']} {r['sample_id']}",flush=True)
   (a.out/'progress.json').write_text(json.dumps({'expected':273,'validated':len(records),'elapsed_s':time.time()-started},indent=2)+'\n')
 assert len(records)==273 and len({(r['mode'],r['sample_id']) for r in records})==273
 for mode in MODES:
  rows=sorted([r for r in records if r['mode']==mode],key=lambda r:r['sample_id']);assert len(rows)==91
  fields=['sample_id','hit','fallback','simulation_seconds','prediction_versions','bytes','sha256']
  with (a.out/mode/'index.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
 manifest={'expected':273,'validated':273,'per_method':91,'test_videos':270,'opposite_videos':3,'elapsed_s':time.time()-started,'records':sorted(records,key=lambda r:(r['mode'],r['sample_id']))}
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('COMPLETE 273/273',flush=True)
if __name__=='__main__':main()
