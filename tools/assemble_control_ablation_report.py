"""把独立底盘反馈配对的图接入主浏览页；两组统计与来源分别保存。"""
import argparse,json,shutil
from pathlib import Path

def main(primary,extra):
 source=extra/'plots/base_feedback';assert len(list(source.rglob('*.png')))==140
 shutil.copytree(source,primary/'plots/base_feedback',dirs_exist_ok=True)
 shutil.copyfile(extra/'summary.json',primary/'base_feedback_summary.json')
 variants=[['baseline','原基线'],['gains','辨识增益候选'],['base_feedback','底盘反馈补充候选']]
 html='''<!doctype html><meta charset="utf-8"><title>pfly / nmpc6：末端与17轴电机</title><style>body{font:16px sans-serif;max-width:1500px;margin:20px auto;background:#f2f5f8;color:#17233a}select{font-size:18px;margin:8px}img{width:100%;margin:8px 0}p{line-height:1.6}</style><h1>pfly / nmpc6：控制目标、实际状态和17轴电机</h1><p>各35条：原基线/辨识增益候选；nmpc6另有底盘反馈补充。前三者共700张图。主385次与补充70次各自独立同二进制，全部使用同35球的因果预测快照。<a href="summary.json">主统计</a> · <a href="base_feedback_summary.json">补充统计</a></p><p>每组：末端位置、五臂轴、四舵、八轮。驱动轮显示真实轮速PI目标，不画无意义的零位置目标。力矩图显示前馈或PI命令、延迟后总需求、限幅输出。黑竖线为计划击球时刻；颜色以各图图例为准。</p><select id="variant"></select><select id="mode"><option>nmpc6</option><option>pfly</option></select><select id="sample"></select><div id="figures"></div><script>const vs=VARIANTS;for(const [v,l] of vs){let o=document.createElement('option');o.value=v;o.textContent=l;variant.append(o)}for(let i=1;i<=35;i++){let o=document.createElement('option');o.value='test_'+String(i).padStart(2,'0');o.textContent=o.value;sample.append(o)}function show(){if(variant.value==='base_feedback'&&mode.value==='pfly'){figures.textContent='底盘反馈补充只用于nmpc6；pfly比较原基线和辨识增益候选。';return}figures.innerHTML=['task','arm','steer','wheels'].map(k=>`<img src="plots/${variant.value}/${mode.value}/${sample.value}_${k}.png">`).join('')}variant.onchange=mode.onchange=sample.onchange=show;variant.value='gains';show();</script>'''.replace('VARIANTS',json.dumps(variants,ensure_ascii=False))
 (primary/'index.html').write_text(html)
 files=list((primary/'plots').rglob('*.png'));assert len(files)==700,len(files)
 (primary/'display_manifest.json').write_text(json.dumps({'plot_pages':len(files),'primary_trials':385,'extra_trials':70,'source_groups_kept_separate':True,'views':'140 main sample/profile groups +35 extra;4 pages per group'},indent=2)+'\n')
 print('700 figure pages assembled')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--primary',required=True,type=Path);p.add_argument('--extra',required=True,type=Path);a=p.parse_args();main(a.primary,a.extra)
