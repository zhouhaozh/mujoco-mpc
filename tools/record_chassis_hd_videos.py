"""重录已验收的三个候选并沿用高清渲染器；先核对控制复现再编码。"""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np

VARIANTS=['chassis_model','chassis_recovery','chassis_guard']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_rows(path):
    with path.open() as f:return [r for s in f if (r:=json.loads(s))['kind']=='state']
def compare_runs(new,old):
    a=json.loads((new/'run.json').read_text());b=json.loads((old/'run.json').read_text())
    for k in ['binary_sha256','input_sha256','cache_sha256','identified_sha256','chassis_resources']:assert a[k]==b[k],(new,k)
    x=state_rows(new/'diagnostics.jsonl');y=state_rows(old/'diagnostics.jsonl');assert len(x)==len(y)
    for s,t in zip(x,y):
        for k in ['t','qpos','qvel','base','v_base_world','q','dq','goal','goal_time']:
            np.testing.assert_allclose(s[k],t[k],rtol=0,atol=1e-10)
        for k in ['active','fallback','planned']:assert s[k]==t[k]
    return {'sample':new.name,'state_rows':len(x),'same_states':True,'trace_sha256':a['video_trace_sha256']}
def main(a):
    team=a.team.resolve();previous=a.previous.resolve();original=a.source.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    recording=out/'.recording';source=out/'.source';source.mkdir(exist_ok=True);recording.mkdir(exist_ok=True)
    manifest=json.loads((original/'input_manifest.json').read_text());items=manifest['items'];assert len(items)==35
    (source/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,target in [('cache',previous/'cache'),('identification',previous/'identification')]:
        dest=recording/name
        if not dest.exists():dest.symlink_to(target,target_is_directory=True)
    for v in VARIANTS:
        (source/'.records'/v).mkdir(parents=True,exist_ok=True)
        for item in items:
            dest=source/'.records'/v/item['sample_id']
            if not dest.is_symlink():dest.symlink_to(recording/'runs'/v/'nmpc6'/item['sample_id'],target_is_directory=True)
    env=dict(os.environ,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def work(v):
        with (out/(v+'_record.log')).open('w') as f:
            subprocess.run([sys.executable,str(team/'tools/run_control_ablation.py'),'--team',str(team),'--source',str(original),'--root',str(recording),'--variants',v,'--modes','nmpc6','--video-trace'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        audit=[compare_runs(recording/'runs'/v/'nmpc6'/i['sample_id'],previous/'runs'/v/'nmpc6'/i['sample_id']) for i in items]
        (out/(v+'_reproduction.json')).write_text(json.dumps(audit,indent=2)+'\n');print(v,'recorded and reproduced',flush=True)
        with (out/(v+'_render.log')).open('w') as f:
            subprocess.run([sys.executable,str(team/'tools/render_measured_hd.py'),'--team',str(team),'--source',str(source),'--out',str(out),'--modes',v,'--workers','1'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        print(v,'35 HD videos complete',flush=True)
    with concurrent.futures.ThreadPoolExecutor(3) as pool:list(pool.map(work,VARIANTS))
    records=[]
    for v in VARIANTS:
        for item in items:
            meta=out/v/(item['sample_id']+'.json');r=json.loads(meta.read_text());run=recording/'runs'/v/'nmpc6'/item['sample_id'];info=json.loads((run/'run.json').read_text())
            r.update(controller='nmpc6',control_variant=v,source_trace_sha256=digest(run/'frames.jsonl'),source_binary_sha256=info['binary_sha256'],input_sha256=info['input_sha256'],forecast_cache_sha256=info['cache_sha256'],chassis_resources=info['chassis_resources'],reproduced_from=str(previous/'runs'/v/'nmpc6'/item['sample_id']),physical_hit_rate=None)
            meta.write_text(json.dumps(r,indent=2)+'\n');records.append(r)
    # 单组渲染器会写临时分组manifest；全部成功后用完整105条清单替换。
    (out/'manifest.json').write_text(json.dumps({'expected':105,'validated':105,'records':records,'source':'same35 measured causal input; controller states reproduced against accepted chassis ablation; forecasts only actual CachedVisualization','variants':VARIANTS,'settings':{'resolution':[1920,1080],'fps':60,'speed':.25,'crf':17,'cutoff':'first geometric racket contact else second ground/data end'}},indent=2)+'\n')
    html='''<!doctype html><meta charset="utf-8"><title>新NMPC6高清回放</title><style>body{background:#101824;color:#eef3fa;font:18px sans-serif;margin:24px}video{width:min(96vw,1600px)}select{font-size:18px;padding:8px;margin:8px}</style><h1>NMPC6 · 底盘模型更新</h1><p>实测test35条；1080p / 60fps / 0.25×。青：实测参考；橙：控制器实际请求的预测；粉：目标。几何接触即结束，不代表物理击球。各候选沿用评估参数，未调参选球。</p><select id="method"></select><select id="sample"></select><br><video id="player" controls></video><script>const methods=METHODS;for(const v of methods){let o=document.createElement('option');o.value=v;o.textContent=({'chassis_model':'底盘动态模型','chassis_recovery':'模型＋走廊恢复','chassis_guard':'模型＋恢复＋失效回退'})[v];method.append(o)}for(let i=1;i<=35;i++){let o=document.createElement('option');o.value='test_'+String(i).padStart(2,'0');o.textContent=o.value;sample.append(o)}function show(){player.src=method.value+'/'+sample.value+'.mp4'}method.onchange=sample.onchange=show;show()</script>'''.replace('METHODS',json.dumps(VARIANTS))
    (out/'index.html').write_text(html);print('COMPLETE 105',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
