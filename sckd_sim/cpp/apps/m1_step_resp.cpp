// M1 底盘阶跃速度响应采集(2026-10-16):给定阶跃 twist 指令序列,逐毫秒记录
// 原始指令 / 伺服斜坡后指令 / 实测 twist / 位姿 / 四角舵角 / 轮力矩 → CSV,
// 供 python 绘制跟踪曲线(延迟/上升/超调/稳态误差/舵翻转动态)。
// 时序(ms):0-600 零速基线 → 600 vx=+1 → 1600 vx=+4 → 2600 vx=−2 →
// 3600 vx=0 → 4600 vy=+1.5(舵转 90°) → 5600 vy=0 → 6600 wz=+1.5 → 7600 wz=0
// → 8600 结束。臂全程 hold(60,1.5)隔离臂动力学(m1_dbg_chassis 同口径)。
#include <cmath>
#include <cstdio>
#include <string>

#include "control/chassis_servo.h"
#include "hal/sim_backend.h"

using namespace sckd;

namespace {

// 阶跃计划:返回 t(s) 时刻的原始(未斜坡)twist 指令
Twist CmdAt(double t) {
  Twist c{0, 0, 0};
  if (t >= 0.6 && t < 1.6) c.vx = 1.0;
  else if (t >= 1.6 && t < 2.6) c.vx = 4.0;
  else if (t >= 2.6 && t < 3.6) c.vx = -2.0;
  else if (t >= 4.6 && t < 5.6) c.vy = 1.5;
  else if (t >= 6.6 && t < 7.6) c.wz = 1.5;
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string scene =
      argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  const std::string out = argc > 2 ? argv[2] : "/tmp/step_resp.csv";
  SimBackend be(scene, "models/config/motors.json",
                "models/config/x4_tennis.json", "models/config/chassis_m1.json");
  if (!be.ok() || !be.reset(0)) return 1;
  ChassisServo servo(be.chassis_config(), &be);
  be.enable(-1, true);
  be.hold_arm(60.0, 1.5);

  FILE* f = std::fopen(out.c_str(), "w");
  if (!f) { std::printf("[FAIL] open %s\n", out.c_str()); return 1; }
  std::fprintf(f, "t,cmd_vx,cmd_vy,cmd_wz,ramp_vx,ramp_vy,ramp_wz,"
                  "meas_vx,meas_vy,meas_wz,x,y,yaw,"
                  "steer_fl,steer_fr,steer_rl,steer_rr,tau_fll\n");

  const int kTotalMs = 8600;
  for (int t = 1; t <= kTotalMs; ++t) {
    const double now = t * 0.001;
    servo.set_twist(CmdAt(now));
    servo.update(now);
    be.run(1);

    const Twist cmd = CmdAt(now), rp = servo.ramped();
    const BaseState b = be.base_state();
    double steer[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) steer[i] = be.joint(AxisSteer(i)).q;
    const double tau_fll = be.joint(AxisWheelL(0)).tau_cmd;
    std::fprintf(f,
                 "%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
                 "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n",
                 now, cmd.vx, cmd.vy, cmd.wz, rp.vx, rp.vy, rp.wz,
                 b.vx, b.vy, b.wz, b.x, b.y, b.yaw,
                 steer[0], steer[1], steer[2], steer[3], tau_fll);
  }
  std::fclose(f);
  std::printf("[OK] %d rows -> %s\n", kTotalMs, out.c_str());
  return 0;
}
