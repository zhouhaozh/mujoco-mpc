// M1 调试探针:直线 1 m/s,打印每轮 目标/实测线速度、力矩、舵角、整机 twist。
#include <cmath>
#include <cstdio>
#include <string>

#include <mujoco/mujoco.h>
#include "control/chassis_servo.h"
#include "hal/sim_backend.h"

using namespace sckd;

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  const std::string cj = argc > 4 ? argv[4] : "models/config/chassis_m1.json";
  SimBackend be(scene, argc > 2 ? argv[2] : "models/config/motors.json",
                argc > 3 ? argv[3] : "models/config/x4_tennis.json", cj);
  if (!be.ok() || !be.reset(0)) return 1;
  ChassisServo servo(be.chassis_config(), &be);
  be.enable(-1, true);
  be.hold_arm(60.0, 1.5);
  servo.set_twist(Twist{1.0, 0, 0});
  ChassisServo* sv = &servo;
  for (int t = 1; t <= 3000; ++t) {
    if (t == 1500) {  // 复现 m1_traj 的段间 reset + 新伺服
      be.reset(0);
      static ChassisServo s2(be.chassis_config(), &be);
      s2.set_twist(Twist{1.0, 0, 0});
      be.enable(-1, true);
      be.hold_arm(60.0, 1.5);
      sv = &s2;
    }
    sv->update(t * 0.001);
    be.run(1);
    if (t == 500 || t == 1500 || t == 1600 || t == 2999) {
      const BaseState b = be.base_state();
      {   // 原始 6 分量 vs 位置差分
        mjtNum res[6];
        const mjModel* m = be.model(); const mjData* d = be.data();
        int bid = mj_name2id(m, mjOBJ_BODY, "base_link");
        mj_objectVelocity(m, const_cast<mjData*>(d), mjOBJ_BODY, bid, res, 1);
        std::printf("   raw=[%+.3f %+.3f %+.3f | %+.3f %+.3f %+.3f]\n",
                    res[0],res[1],res[2],res[3],res[4],res[5]);
        const double* qpos = d->qpos;
        std::printf("   qpos=[%+.3f %+.3f %+.3f] qvel012=[%+.3f %+.3f %+.3f]\n",
                    qpos[0],qpos[1],qpos[2],d->qvel[0],d->qvel[1],d->qvel[2]);
      }
      std::printf("t=%d  pos: x=%.3f y=%.3f yaw=%.3f | twist: vx=%.3f vy=%.3f wz=%.3f\n", t, b.x, b.y, b.yaw, b.vx, b.vy, b.wz);
      const char* corner[4] = {"FL", "FR", "RL", "RR"};
      for (int i = 0; i < 4; ++i) {
        const JointReport st = be.joint(AxisSteer(i));
        const JointReport wl = be.joint(AxisWheelL(i));
        const JointReport wr = be.joint(AxisWheelR(i));
        const double r = be.chassis_config().wheel_radius;
        const double sgn_l = be.chassis_config().wheel_sign[2 * i];
        const double sgn_r = be.chassis_config().wheel_sign[2 * i + 1];
        std::printf("  %s steer q=%+.3f dq=%+.2f | L: dq=%+7.2f v=%+.3f tau=%+6.2f | "
                    "R: dq=%+7.2f v=%+.3f tau=%+6.2f\n",
                    corner[i], st.q, st.dq, wl.dq, sgn_l * wl.dq * r, wl.tau_cmd, wr.dq,
                    sgn_r * wr.dq * r, wr.tau_cmd);
      }
    }
  }
  return 0;
}
