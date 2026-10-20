"""留出激励上的固定输入开环预测；不优化参数。"""
import argparse,json
from pathlib import Path
import casadi as ca
import numpy as np
def main(a):
    fn=ca.external('chassis_rollout',str((a.team/'sckd_sim/third_party/nmpc6/chassis_rollout.so').resolve()));rows=[]
    for trial in [3,4,5]:
        d=np.genfromtxt(a.root/f'excitation_{trial}.csv',delimiter=',',names=True)
        for horizon in [.05,.1,.2,.5]:
            n=round(horizon*1000)
            for i in range(1000,len(d)-n,20):
                # 输入在窗口内保持同一目标；排除未来外部改变指令的窗口。
                if np.ptp(d['cmd_x'][i:i+n+1])>1e-8 or np.ptp(d['cmd_y'][i:i+n+1])>1e-8:continue
                yaw=d['yaw'][i];c=np.cos(yaw);s=np.sin(yaw)
                vel=np.array([c*d['vx'][i]-s*d['vy'][i],s*d['vx'][i]+c*d['vy'][i]])
                cmd=np.array([c*d['cmd_x'][i]-s*d['cmd_y'][i],s*d['cmd_x'][i]+c*d['cmd_y'][i]])
                p=np.zeros(38);p[:2]=[d['x'][i],d['y'][i]];p[7:9]=vel;p[17:20]=horizon/3;p[25]=1
                p[27:29]=[d['ramp_x'][i],d['ramp_y'][i]];p[29:33]=[d[f'steer{j}'][i] for j in range(4)];p[33:37]=[d[f'wheel_ref{j}'][i] for j in range(4)];p[37]=yaw
                u=np.zeros(21);u[0]=cmd[0]-vel[0];u[3]=cmd[1]-vel[1]
                y=np.array(fn(u,p)[0]);pred=y[:2,-1];truth=np.array([d['x'][i+n],d['y'][i+n]])
                # 旧模型三个增量中仅首段非零，按受限斜坡积分；只比较其可表达输入。
                if np.max(abs(cmd-vel))>8*horizon/3:continue
                old=p[:2]+cmd*horizon-(cmd-vel)*abs(cmd-vel)/16
                rows.append(dict(trial=trial,horizon_s=horizon,old_m=float(np.linalg.norm(old-truth)),new_m=float(np.linalg.norm(pred-truth))))
    summary={}
    for h in [.05,.1,.2,.5]:
        rr=[r for r in rows if r['horizon_s']==h];summary[str(h)]={'n':len(rr)}
        for model in ['old','new']:
            e=np.array([r[model+'_m'] for r in rr]);summary[str(h)][model]={'mean_cm':float(e.mean()*100),'p95_cm':float(np.percentile(e,95)*100),'rmse_cm':float(np.sqrt(np.mean(e*e))*100)}
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'heldout_prediction.json').write_text(json.dumps({'protocol':'excitation 3/4/5; constant target input window; identical command increments; only legacy-feasible increments; current-state prefix; no future states used','horizons':summary},indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
