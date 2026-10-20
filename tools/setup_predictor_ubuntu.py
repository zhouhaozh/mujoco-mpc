"""Project-local Linux x86_64 dependencies; needs cmake/g++ and Eigen3 headers.
OSQP static archive is restored only from the user's preserved team archive.
No global Python installation is modified.
"""
import argparse,hashlib,os,platform,subprocess,sys,tarfile,venv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(*args,**kwargs):subprocess.run([str(a) for a in args],check=True,**kwargs)
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
 if platform.system()!='Linux' or platform.machine()!='x86_64':raise SystemExit('requires Ubuntu x86_64; OSQP archive is platform-specific')
 environment=ROOT/'.venv'
 if not (environment/'bin/python').exists():venv.create(environment,with_pip=True)
 python=environment/'bin/python';run(python,'-m','pip','install','mujoco==3.14.0','casadi==3.8.0')
 for name in ('mujoco','casadi'):
  folder=Path(subprocess.check_output([str(python),'-c',f'import {name};from pathlib import Path;print(Path({name}.__file__).parent)'],text=True).strip())
  target=ROOT/'deps'/name;target.mkdir(parents=True,exist_ok=True)
  for label,source in [('include',folder/'include'),('lib',folder)]:
   link=target/label
   if link.is_symlink() and link.resolve()==source.resolve():continue
   if link.exists() or link.is_symlink():raise RuntimeError(f'refusing to replace {link}')
   link.symlink_to(source,target_is_directory=True)
  if name=='mujoco' and not (folder/'libmujoco.so').exists():(folder/'libmujoco.so').symlink_to('libmujoco.so.3.14.0')
 lib=ROOT/'sckd_sim/third_party/osqp/libosqp.a';expected='a00af25b409510a548e5125544fef402b04d0eb494f6ff853a2fbe5c71700c39'
 if not lib.exists():
  with tarfile.open(a.archive) as t:
   f=t.extractfile('sim/sckd_sim/third_party/osqp/libosqp.a')
   if f is None:raise RuntimeError('missing archive member')
   data=f.read()
  if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('unexpected OSQP binary')
  lib.write_bytes(data)
 if hashlib.sha256(lib.read_bytes()).hexdigest()!=expected:raise RuntimeError('OSQP hash mismatch')
 source=ROOT/'sckd_sim';nlp=source/'third_party/nmpc6/nlp.so'
 if not nlp.exists():run('gcc','-fPIC','-shared','-O3',source/'third_party/nmpc6/nlp.c','-o',nlp,'-lm')
 run('cmake','-S',source,'-B',source/'build-predictor',f'-DMUJOCO_DIR={ROOT}/deps/mujoco',f'-DCASADI_DIR={ROOT}/deps/casadi','-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1')
 run('cmake','--build',source/'build-predictor','-j8');run('ctest','--test-dir',source/'build-predictor','--output-on-failure')
if __name__=='__main__':main()
