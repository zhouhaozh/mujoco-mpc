"""构建独立nlp_chassis/chassis_rollout库，原nlp/servo库保持原样。"""
import argparse,hashlib,json,subprocess
from pathlib import Path
def main(a):
    team=a.team.resolve();src=team/'sckd_sim/third_party/nmpc6';deps=team/'deps/casadi';exe=src/'gen_chassis_nlp'
    subprocess.run(['g++','-O2','-std=c++17','-I',str(deps/'include'),'-L',str(deps/'lib'),'-Wl,-rpath,'+str(deps/'lib'),str(src/'gen_nlp.cc'),'-o',str(exe),'-lcasadi'],check=True)
    subprocess.run([str(exe),str(a.identification.resolve()),'chassis'],cwd=src,check=True)
    for name in ['nlp_chassis','chassis_rollout']:
        subprocess.run(['gcc','-fPIC','-shared','-O3',str(src/(name+'.c')),'-o',str(src/(name+'.so')),'-lm'],check=True)
    files=[a.identification,src/'gen_nlp.cc',src/'nlp_chassis.so',src/'chassis_rollout.so']
    (a.identification.parent/'build_manifest.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--identification',type=Path,required=True);main(p.parse_args())
