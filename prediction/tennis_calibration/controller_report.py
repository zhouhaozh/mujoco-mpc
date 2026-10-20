"""Plot audited grouped results with coverage; no model fitting."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);args=a.parse_args();out=args.out
 data=json.loads((out/'groupcv/summary.json').read_text())['oof']
 plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':11})
 fig,axs=plt.subplots(2,2,figsize=(12,7.6),gridspec_kw={'height_ratios':[3,1]},layout='constrained')
 colors={'baseline':'#8994a6','adaptive_protected':'#007f78'}
 for key,label in [('baseline','原 GP'),('adaptive_protected','所选 GP＋保护＋MLP')]:
  rows=[r for r in data[key]['by_horizon'] if r['horizon_s']<=1.5];x=[r['horizon_s'] for r in rows]
  for metric,style,desc in [('mean_m','-','均值'),('p95_m','--','P95')]:axs[0,0].plot(x,[100*r[metric] for r in rows],style,marker='o',color=colors[key],label=f'{label} · {desc}')
  rows=[r for r in data[key]['by_capture_time'] if r['horizon_s']==.5 and r['trajectories']>=10];x=[r['capture_relative_s'] for r in rows]
  for metric,style,desc in [('mean_m','-','均值'),('p95_m','--','P95')]:axs[0,1].plot(x,[100*r[metric] for r in rows],style,color=colors[key])
 for ax in axs[0]:ax.set_ylabel('三维位置误差（cm）');ax.grid(alpha=.2)
 axs[0,0].set_title('预测得越远，误差越大');axs[0,0].legend(fontsize=9)
 axs[0,1].set_title('始终预测未来 500 ms：触地附近升高，恢复后回落')
 for ax in axs[:,1]:ax.axvline(0,color='#555',lw=1);ax.axvspan(0,.1,color='#dddddd',alpha=.5)
 rows=[r for r in data['adaptive_protected']['by_horizon'] if r['horizon_s']<=1.5]
 axs[1,0].plot([r['horizon_s'] for r in rows],[r['trajectories'] for r in rows],'o-',color='#007f78');axs[1,0].set_xlabel('从当前观测向前预测的时长 H（s）')
 rows=[r for r in data['adaptive_protected']['by_capture_time'] if r['horizon_s']==.5 and r['trajectories']>0]
 axs[1,1].plot([r['capture_relative_s'] for r in rows],[r['trajectories'] for r in rows],color='#007f78');axs[1,1].set_xlabel('当前观测时刻相对参考触地（s；0 = 触地）')
 for ax in axs[1]:ax.set_ylabel('有标签轨迹数');ax.grid(alpha=.2);ax.set_ylim(0,150)
 fig.suptitle('五折留出共 140 条轨迹｜轨迹等权｜虚线 P95，实线均值',fontsize=15)
 fig.text(.02,-.03,'误差对照同设备留存观测；并非独立真值。右图误差仅画覆盖 ≥10 条轨迹的区间。灰带仅示意触地后 100 ms。',fontsize=10)
 for ext in ('png','pdf'):fig.savefig(out/f'controller_error_readable.{ext}',dpi=180,bbox_inches='tight')
if __name__=='__main__':main()
