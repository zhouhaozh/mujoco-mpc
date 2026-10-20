"""当前MuJoCo模型分轴辨识；由实测力矩/速度微分估计局部惯性和阻尼。"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
 entries=[];fig,axs=plt.subplots(5,3,figsize=(16,15))
 for j in range(5):
  d=np.genfromtxt(root/f'joint_{j+1}.csv',delimiter=',',names=True);t=d['t_s'];dt=.001
  # 行中的tau已产生该行速度；反向差分与其对齐。避开保持段与明显限幅。
  acc=np.r_[0,np.diff(d['dq'])/dt];mask=(t>2)&(t<10.8)&(abs(acc)>.02)&(abs(d['tau_raw'])<d['cap']-.01)
  x=np.c_[acc[mask],d['dq'][mask]];y=(d['tau']-d['gravity'])[mask]
  fit=least_squares(lambda v:x@v-y,[.2,.1],bounds=([1e-4,0],[10,30]),loss='huber',f_scale=.1)
  I,B=fit.x;kp=250 if j==0 else 120;kd=float(min(1.4*I/dt,max(.05,1.6*np.sqrt(kp*I)-B)))
  scale=max(float(np.std(y)),1e-9);residual=x@fit.x-y
  # 有限激励、当前姿态下的局部近似；耦合、接触与位置依赖性留在残差中。
  row={'axis':j+1,'inertia_kg_m2':float(I),'damping_Nms_rad':float(B),'fit_samples':int(mask.sum()),'residual_rmse_Nm':float(np.sqrt(np.mean(residual**2))),'normalized_residual':float(np.sqrt(np.mean(residual**2))/scale),'kp_candidate':kp,'kd_candidate':kd,'known_command_transport_s':.009,'step_position_error_p95_rad':float(np.percentile(abs(d['q_ref'][t>2]-d['q'][t>2]),95))}
  entries.append(row)
  axs[j,0].plot(t,d['q_ref'],label='issued q');axs[j,0].plot(t,d['q_effective'],ls=':',label='effective q');axs[j,0].plot(t,d['q'],label='actual q');axs[j,0].set_ylabel(f'J{j+1} rad')
  axs[j,1].plot(t,d['dq_ref'],label='target dq');axs[j,1].plot(t,d['dq'],label='actual dq');axs[j,1].set_ylabel('rad/s')
  axs[j,2].plot(t,d['tau']-d['gravity'],label='tau - gravity');axs[j,2].plot(t,I*acc+B*d['dq'],label='local I*a+B*v');axs[j,2].set_ylabel('Nm')
  for ax in axs[j]:ax.grid(alpha=.2);ax.set_xlim(1.8,10.8)
 for ax in axs[0]:ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(root/'identification.png');plt.close(fig)
 result={'source':'current MuJoCo simulated motor state, not real hardware identification','method':'Huber diagonal local regression tau-gravity=I*ddq+B*dq; backward dq difference; unsaturated samples','kp':[r['kp_candidate'] for r in entries],'kd':[r['kd_candidate'] for r in entries],'axes':entries,'limits':'unchanged mechanism and motor envelopes; no expansion','additional_lag':'do not conflate second-order transient with pure transport; transport from effective-command trace'}
 (root/'identified.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);main(p.parse_args().root)
