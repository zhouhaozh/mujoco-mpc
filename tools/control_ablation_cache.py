"""固定因果预测快照：逐原始帧生成，配对运行只可按已收到的序号读取。"""
import argparse,hashlib,json,os,socket,sys
from pathlib import Path

def prepare(prediction,source,out):
 sys.path.insert(0,str(prediction));from tennis_calibration.controller_service import ControllerPredictor
 selection=json.loads((prediction/'artifacts/tennis_controller/20261014/selected.json').read_text());transform=json.loads((source/'transform.json').read_text())
 out.mkdir(parents=True,exist_ok=True)
 items=json.loads((source/'input_manifest.json').read_text())['items']
 for item in items:
  dest=out/(item['sample_id']+'.jsonl')
  if dest.exists():continue
  predictor=ControllerPredictor(selection,transform);rows=[]
  import csv
  with open(item['csv']) as f:
   for seq,r in enumerate(csv.DictReader(f),1):
    values=list(map(float,r.values()));t=values[0];xyz=values[1:4];u=predictor.update(seq,t,xyz)
    row={'seq':seq,'t_s':t,'xyz':xyz,'update':u,'forecast':predictor.forecast(seq) if u['initialized'] else None};rows.append(row)
  tmp=dest.with_suffix('.partial');tmp.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));tmp.rename(dest)
  print(item['sample_id'],len(rows),flush=True)
 files=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.glob('*.jsonl'))]
 (out/'manifest.json').write_text(json.dumps({'source':str(source),'selection':selection,'causal':'each forecast computed immediately after that prefix update; server checks seq, capture time and xyz; future entries inaccessible through protocol','files':files},indent=2)+'\n')

def serve(cache,path):
 rows=[json.loads(r) for r in cache.read_text().splitlines()];seq=0
 if path.exists():raise FileExistsError(path)
 server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(path));server.listen(1)
 try:
  conn,_=server.accept()
  with conn,conn.makefile('rwb') as stream:
   while raw:=stream.readline():
    req=json.loads(raw);op=req['op'];assert req['schema']=='BALL_FORECAST/1'
    if op=='reset':seq=0;r={'initialized':False}
    elif op=='update':
     assert req['seq']==seq+1
     row=rows[seq];assert abs(row['t_s']-req['t_s'])<1e-10 and max(abs(x-y) for x,y in zip(row['xyz'],req['xyz']))<1e-10
     seq+=1;r=row['update']
    elif op=='forecast':assert req['seq']==seq and rows[seq-1]['forecast'] is not None;r=rows[seq-1]['forecast']
    else:raise ValueError(op)
    stream.write((json.dumps({'schema':'BALL_FORECAST/1','ok':True,**r})+'\n').encode());stream.flush()
 finally:server.close();path.unlink(missing_ok=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prediction',type=Path);p.add_argument('--source',type=Path);p.add_argument('--out',type=Path);p.add_argument('--serve',type=Path);p.add_argument('--socket',type=Path);a=p.parse_args()
 if a.serve:serve(a.serve,a.socket)
 else:prepare(a.prediction.resolve(),a.source.resolve(),a.out.resolve())
