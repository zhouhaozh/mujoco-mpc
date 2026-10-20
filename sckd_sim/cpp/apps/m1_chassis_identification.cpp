// 独立底盘激励；不使用网球test轨迹拟合参数。
#include <cmath>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include "control/chassis_servo.h"
#include "hal/sim_backend.h"
using namespace sckd;
int main(int argc,char**argv){
 if(argc!=2)return 2;std::filesystem::create_directories(argv[1]);
 for(int trial=0;trial<6;++trial){
  SimBackend be("models/generated/x4_tennis_demo.mjcf","models/config/motors.json","models/config/x4_tennis.json","models/config/chassis_m1.json");
  if(!be.ok()||!be.reset(0))return 3;be.enable(-1,true);ChassisServo servo(be.chassis_config(),&be);
  std::ofstream f(std::string(argv[1])+"/excitation_"+std::to_string(trial)+".csv");f<<std::setprecision(15);
  f<<"t,cmd_x,cmd_y,ramp_x,ramp_y,x,y,yaw,vx,vy";
  for(int j=0;j<4;++j)f<<",steer"<<j<<",steer_cmd"<<j<<",wheel_ref"<<j;f<<'\n';
  for(int k=0;k<9000;++k){
   double t=k*.001;Twist cmd;
   if(t>=1){int seg=static_cast<int>((t-1)/1.0);double speed=trial<3?.7+trial*.35:.85+(trial-3)*.25;
    if(seg==0||seg==1)cmd={speed,0,0};else if(seg==2||seg==3)cmd={0,speed,0};else if(seg==4)cmd={-speed,0,0};else if(seg==5)cmd={speed*.707,-speed*.707,0};
    if(trial>=3){double a=(trial-2)*.31;double x=cmd.vx,y=cmd.vy;cmd.vx=cos(a)*x-sin(a)*y;cmd.vy=sin(a)*x+cos(a)*y;}
   }
   ArmHybridCmd ac;const double home[5]={0,-.6109,-.6109,0,0};double q[5],g[5];
   for(int j=0;j<5;++j)q[j]=be.joint(AxisArm(j)).q;be.arm_gravity(q,g);
   for(int j=0;j<5;++j){ac.q[j]=home[j];ac.kp[j]=j==0?250:120;ac.kd[j]=j==0?3:j==4?1.5:8;ac.tau_ff[j]=g[j];}be.set_arm_hybrid(ac);
   auto b=be.base_state();auto ramp=servo.ramped();f<<t<<','<<cmd.vx<<','<<cmd.vy<<','<<ramp.vx<<','<<ramp.vy<<','<<b.x<<','<<b.y<<','<<b.yaw<<','<<b.vx<<','<<b.vy;
   for(int j=0;j<4;++j){double mean=0;for(int h=0;h<2;++h){int a=h?AxisWheelR(j):AxisWheelL(j);mean+=be.chassis_config().wheel_sign[2*j+h]*servo.wheel_velocity_reference(a)*be.chassis_config().wheel_radius*.5;}
    f<<','<<be.joint(AxisSteer(j)).q<<','<<be.motor_issued(AxisSteer(j)).q<<','<<mean;}
   f<<'\n';servo.set_twist(cmd);servo.update(t);be.run(1);
  }
 }
}
