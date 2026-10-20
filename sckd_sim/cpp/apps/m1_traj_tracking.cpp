// M1 验收 1/3:底盘轨迹跟踪(伺服层,速度前馈口径)。
// 场景:圆弧(v/wz 前馈)、反向切换(舵反转+cos 投影)、急停;附舵阶跃 T_steer 辨识
// (供 MPC 等效模型用,§3.4)。判据见 KPass* 常量;CSV -> eval/reports/m1_traj.csv。
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "control/chassis_servo.h"
#include "hal/sim_backend.h"

using namespace sckd;

namespace {

struct Args {
  std::string scene, mj, vj, cj;
};

int RunTracking(const Args& a) {
  SimBackend be(a.scene, a.mj, a.vj, a.cj);
  if (!be.ok() || !be.reset(0)) {
    std::printf("[FAIL] backend init\n");
    return 1;
  }
  ChassisServo servo(be.chassis_config(), &be);
  be.enable(-1, true);
  be.hold_arm(60.0, 1.5);

  std::filesystem::create_directories("eval/reports");
  std::ofstream csv("eval/reports/m1_traj.csv");
  csv << "phase,t,ref_vx,ref_wz,vx,vy,wz,radial_err\n";

  const double v_circle = 1.0, r_circle = 1.5, wz_circle = v_circle / r_circle;
  const double kPassVelRms = 0.15;      // m/s
  const double kPassRadialRms = 0.15;   // m
  const double kPassStopV = 0.05;       // m/s

  double sum_v_err2 = 0, sum_rad2 = 0, sum_wz_err2 = 0, n = 0;
  double max_v_err = 0, stop_v_max = 0;
  bool pass = true;

  // --- 舵阶跃 T_steer 辨识(wz 阶跃):FL 舵角达 63% 时间 ---
  double steer63 = -1;
  {
    servo.set_twist(Twist{v_circle, 0, 0});
    for (int t = 1; t <= 2000; ++t) {  // 2s 直行:舵对准 0
      servo.update(t * 0.001);
      be.run(1);
    }
    const double steer0 = be.joint(AxisSteer(0)).q;
    servo.set_twist(Twist{v_circle, 0, wz_circle});       // 阶跃
    // FL 真实目标角 = atan2(vy + wz·x, vx − wz·y) @ FL(0.475, 0.475)
    const double target = std::atan2(wz_circle * 0.475, v_circle - wz_circle * 0.475);
    bool hit63 = false;
    for (int t = 2001; t <= 8000 && !hit63; ++t) {
      servo.update(t * 0.001);
      be.run(1);
      const double q = be.joint(AxisSteer(0)).q;
      if (std::fabs(q - steer0) >= 0.63 * std::fabs(target - steer0)) {
        steer63 = (t - 2000) * 0.001;
        hit63 = true;
      }
    }
  }

  // --- 圆弧 20 s ---
  be.reset(0);
  {
    ChassisServo s2(be.chassis_config(), &be);
    be.enable(-1, true);
    be.hold_arm(60.0, 1.5);
    s2.set_twist(Twist{v_circle, 0, wz_circle});
    for (int t = 1; t <= 20000; ++t) {
      s2.update(t * 0.001);
      be.run(1);
      if (t < 3000) continue;  // 入口过渡不计入指标
      const BaseState b = be.base_state();
      const double v_err = std::hypot(b.vx - v_circle, b.vy);
      // 前馈圆弧圆心在 (0, r)(起点原点、左转);横向误差相对该圆度量
      const double radial = std::fabs(std::hypot(b.x, b.y - r_circle) - r_circle);
      sum_v_err2 += v_err * v_err;
      sum_wz_err2 += (b.wz - wz_circle) * (b.wz - wz_circle);
      sum_rad2 += radial * radial;
      max_v_err = std::fmax(max_v_err, v_err);
      ++n;
      if (t % 100 == 0)
        csv << "circle," << t * 0.001 << ',' << v_circle << ',' << wz_circle << ','
            << b.vx << ',' << b.vy << ',' << b.wz << ',' << radial << '\n';
    }
  }

  // --- 反向切换 + 急停(有限性与停稳检查) ---
  {
    be.reset(0);
    ChassisServo s3(be.chassis_config(), &be);
    be.enable(-1, true);
    be.hold_arm(60.0, 1.5);
    for (int t = 1; t <= 8000; ++t) {
      Twist cmd{1.0, 0, 0};
      if (t > 2000 && t <= 4000) cmd = Twist{-1.0, 0, 0};  // 反向(舵保持,轮速反号)
      if (t > 4000) cmd = Twist{0, 0, 0};                   // 急停
      if (t % 4 == 0) s3.set_twist(cmd);
      s3.update(t * 0.001);
      be.run(1);
      const BaseState b = be.base_state();
      if (t % 100 == 0) {
        csv << "reverse_stop," << t * 0.001 << ',' << cmd.vx << ',' << cmd.wz << ','
            << b.vx << ',' << b.vy << ',' << b.wz << ",0\n";
      }
      if (t > 4500) stop_v_max = std::fmax(stop_v_max, std::hypot(b.vx, b.vy));
      const JointReport fl = be.joint(0);
      if (!std::isfinite(fl.q) || !std::isfinite(b.vx)) pass = false;
    }
  }
  csv.close();

  const double v_rms = std::sqrt(sum_v_err2 / n);
  const double wz_rms = std::sqrt(sum_wz_err2 / n);
  const double rad_rms = std::sqrt(sum_rad2 / n);
  std::printf("T_steer(63%%) = %s s(FL 阶跃;MPC 等效模型输入)\n",
              steer63 > 0 ? std::to_string(steer63).c_str() : ">6s 未达");
  std::printf("circle: v_rms=%.3f m/s (max %.3f)  wz_rms=%.3f rad/s  radial_rms=%.3f m\n",
              v_rms, max_v_err, wz_rms, rad_rms);
  std::printf("stop: |v|max(4.5s 后)=%.3f m/s  finite=%d\n", stop_v_max, pass);
  const bool ok = v_rms < kPassVelRms && rad_rms < kPassRadialRms &&
                  stop_v_max < kPassStopV && pass;
  if (ok) {
    std::printf("[PASS] m1_traj_tracking\n");
    return 0;
  }
  std::printf("[FAIL] m1_traj_tracking\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  a.scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  a.mj = argc > 2 ? argv[2] : "models/config/motors.json";
  a.vj = argc > 3 ? argv[3] : "models/config/x4_tennis.json";
  a.cj = argc > 4 ? argv[4] : "models/config/chassis_m1.json";
  return RunTracking(a);
}
