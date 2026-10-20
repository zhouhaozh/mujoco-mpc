"""独立激励0/1/2拟合；3/4/5留出，参数不使用网球test。"""
import argparse,json
from pathlib import Path
import numpy as np
def main(a):
    records=[np.genfromtxt(a.root/f'excitation_{i}.csv',delimiter=',',names=True) for i in range(6)]
    motion=[];steer=[]
    for d in records[:3]:
        w=np.array([d[f'wheel_ref{j}'] for j in range(4)]).T
        q=np.array([d[f'steer{j}'] for j in range(4)]).T
        inp=np.stack([(w*np.cos(q)).mean(axis=1),(w*np.sin(q)).mean(axis=1)],axis=1)
        v=np.stack([d['vx'],d['vy']],axis=1)
        motion.append((inp,v))
        for j in range(4):
            err=np.arctan2(np.sin(d[f'steer_cmd{j}']-d[f'steer{j}']),np.cos(d[f'steer_cmd{j}']-d[f'steer{j}']))
            dq=np.diff(d[f'steer{j}'])/.001
            mask=(np.abs(err[:-1])>.02)&(np.abs(err[:-1])<.5)&(d['t'][:-1]>1)
            steer.append((err[:-1][mask],dq[mask]))
    candidates=[]
    for tau in np.geomspace(.005,.4,100):
        loss=[]
        for inp,v in motion:
            pred=v[9:-1]+(inp[:-10]-v[9:-1])*(1-np.exp(-.001/tau))
            loss.extend(np.minimum((pred-v[10:])**2,.01).ravel())
        candidates.append((float(np.mean(loss)),float(tau)))
    tau=min(candidates)[1]
    e=np.concatenate([r[0] for r in steer]);dq=np.concatenate([r[1] for r in steer])
    staus=np.geomspace(.01,.4,100)
    stau=float(min(staus,key=lambda t:np.mean(np.minimum((np.clip(e/t,-8,8)-dq)**2,25))))
    out={'schema_version':1,'motion_tau_s':tau,'steer_tau_s':stau,'motion_gain':1.,'config':json.loads(a.config.read_text()),'training_excitation':[0,1,2],'heldout_excitation':[3,4,5],'transport_s':.009,'scope':'reduced planar response; fixed yaw, mean wheel per corner; not full tire/motor dynamics','fit_motion_one_step_loss':min(candidates)[0]}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
