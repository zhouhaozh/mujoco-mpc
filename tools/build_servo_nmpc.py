"""在Ubuntu项目中重建执行器状态预测候选；不覆盖原nlp.so。"""
import argparse,hashlib,json,subprocess
from pathlib import Path

def main(a):
 team=a.team.resolve();sckd=team/'sckd_sim';source=sckd/'third_party/nmpc6';dest=a.output.resolve() if a.output else source;dest.mkdir(parents=True,exist_ok=True);ident=a.identification.resolve() if a.identification else sckd/'models/config/servo_identified_20261015.json'
 casadi=a.casadi.resolve() if a.casadi else team/'deps/casadi';compiler=dest/'gen_servo_nlp'
 subprocess.run(['g++','-O2','-std=c++17','-I',str(casadi/'include'),'-L',str(casadi/'lib'),'-Wl,-rpath,'+str(casadi/'lib'),str(source/'gen_nlp.cc'),'-o',str(compiler),'-lcasadi'],check=True)
 subprocess.run([str(compiler),str(ident)],cwd=dest,check=True)
 for name in ('nlp_servo','actuator_rollout'):
  subprocess.run(['gcc','-fPIC','-shared','-O3',str(dest/(name+'.c')),'-o',str(dest/(name+'.so')),'-lm'],check=True)
 resources=[ident,source/'gen_nlp.cc',dest/'nlp_servo.so',dest/'actuator_rollout.so']
 (dest/'servo_build_manifest.json').write_text(json.dumps({'identification':str(ident),'resources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in resources],'scope':'local diagonal arm actuator model with inverse-dynamics feedforward and9ms transport; chassis nonlinear dynamics not embedded'},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--team',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--identification',type=Path);p.add_argument('--casadi',type=Path);p.add_argument('--output',type=Path);main(p.parse_args())
