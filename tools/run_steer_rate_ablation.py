"""原始baseline控制器的舵限率配对实验，串行运行以减少旧10ms门的负载干扰。"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
CONDITIONS=['limited','direction_unlimited','feedforward_unlimited','both_unlimited']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
    team=a.team.resolve();root=a.root.resolve();prior=a.resources.resolve();root.mkdir(parents=True,exist_ok=True)
    files=[team/'sckd_sim/build-predictor/m2_intercept',team/'sckd_sim/third_party/nmpc6/nlp.so',team/'sckd_sim/models/config/chassis_m1.json',team/'sckd_sim/models/config/motors.json',team/'sckd_sim/models/generated/x4_tennis_demo.mjcf']
    hashes={str(f.relative_to(team)):digest(f) for f in files}
    (root/'experiment.json').write_text(json.dumps({'conditions':CONDITIONS,'controller_variant':'baseline','trials':280,'serial':True,'resources':hashes,'no_model_or_torque_changes':True,'software_rate_only':True,'legacy_solver_acceptance_preserved':True,'legacy_last100ms_wall_gate_preserved':True},indent=2)+'\n')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for condition in CONDITIONS:
        dest=root/condition;dest.mkdir(exist_ok=True)
        for name in ['cache','identification']:
            link=dest/name
            if not link.exists():link.symlink_to(prior/name,target_is_directory=True)
        print('START',condition,flush=True)
        with (dest/'run.log').open('w') as log:
            subprocess.run([sys.executable,str(team/'tools/run_control_ablation.py'),'--team',str(team),'--source',str(a.source.resolve()),'--root',str(dest),'--variants','baseline','--modes','nmpc6,pfly','--steer-rate-mode',condition,'--video-trace'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',condition,flush=True)
        (root/'progress.json').write_text(json.dumps({'last_completed':condition,'complete_conditions':sum((root/c/'runs').exists() and len(list((root/c/'runs').rglob('run.json')))==70 for c in CONDITIONS),'expected_conditions':4})+'\n')
    assert hashes=={str(f.relative_to(team)):digest(f) for f in files}
    print('COMPLETE 280',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--team',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--resources',type=Path,required=True);main(p.parse_args())
