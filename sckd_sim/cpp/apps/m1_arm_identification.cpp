// 当前模型的小幅分轴阶跃/加速辨识；不改质量、惯量或力矩包络。
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include "hal/sim_backend.h"
using namespace sckd;
int main(int argc,char**argv){
 if(argc!=2)return 2;
 SimBackend be("models/generated/x4_tennis_demo.mjcf","models/config/motors.json","models/config/x4_tennis.json","models/config/chassis_m1.json");
 if(!be.ok())return 3;
 std::filesystem::create_directories(argv[1]);
 for(int axis=0;axis<5;++axis){
  be.reset(0);be.enable(-1,true);be.hold_chassis(3);
  std::ofstream f(std::string(argv[1])+"/joint_"+std::to_string(axis+1)+".csv");f<<std::setprecision(14);f<<"t_s,q_ref,dq_ref,ddq_ref,q,dq,tau,tau_raw,cap,gravity,q_effective,dq_effective,kp,kd\n";
  for(int k=0;k<11000;++k){
   double t=k*.001;ArmHybridCmd ac;const double home[5]={0,-.6109,-.6109,0,0};
   for(int j=0;j<5;++j){ac.q[j]=home[j];ac.kp[j]=j==0?250:120;ac.kd[j]=j==0?3:j==4?1.5:8;}
   double ddq=0;
   if(t>=2&&t<4)ac.q[axis]+=.06;
   if(t>=4&&t<6)ac.q[axis]-=.06;
   if(t>=6){double w=2*M_PI*(t<8?.7:1.5),u=t-(t<8?6:8);ac.q[axis]+=.05*std::sin(w*u);ac.dq[axis]=.05*w*std::cos(w*u);ddq=-.05*w*w*std::sin(w*u);}
   double measured[5],g[5];for(int j=0;j<5;++j)measured[j]=be.joint(AxisArm(j)).q;
   be.arm_gravity(measured,g);for(int j=0;j<5;++j)ac.tau_ff[j]=g[j];be.set_arm_hybrid(ac);
   auto jr=be.joint(AxisArm(axis));auto eff=be.motor_effective(AxisArm(axis));
   f<<t<<','<<ac.q[axis]<<','<<ac.dq[axis]<<','<<ddq<<','<<jr.q<<','<<jr.dq<<','<<jr.tau_cmd<<','<<be.motor_raw(AxisArm(axis))<<','<<be.motor_cap(AxisArm(axis))<<','<<g[axis]<<','<<eff.q<<','<<eff.dq<<','<<ac.kp[axis]<<','<<ac.kd[axis]<<'\n';
   be.run(1);
  }
 }
}
