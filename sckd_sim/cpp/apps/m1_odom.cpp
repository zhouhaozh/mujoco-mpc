// M1 验收 3/3:里程计有/无 IMU 对照(§5.2 指标)。
// 场景:圆弧(巡航)+ 反向切换/急停(高机动段)。指标:
//   平移漂移 %(路径归一)、航向误差、速度 RMSE、滑移残差;输出对照表。
#include <cmath>
#include <cstdio>
#include <string>

#include "control/chassis_servo.h"
#include "estimation/wheel_odom.h"
#include "hal/sim_backend.h"

using namespace sckd;

namespace {

struct OdomStats {
  double drift_xy = 0, yaw_err = 0, v_rmse = 0, path_len = 0;
};

int Run(const std::string& scene, const std::string& mj, const std::string& vj,
        const std::string& cj) {
  SimBackend be(scene, mj, vj, cj);
  if (!be.ok() || !be.reset(0)) {
    std::printf("[FAIL] backend init\n");
    return 1;
  }
  ChassisServo servo(be.chassis_config(), &be);
  WheelOdom o_wheel(be.chassis_config());
  WheelOdom o_fused(be.chassis_config());
  be.enable(-1, true);
  be.hold_arm(60.0, 1.5);

  auto collect = [&](WheelOdom* ow, WheelOdom* of, double* v_rmse, double* path) {
    double sum_ve2 = 0;
    int n = 0;
    double prev_x = be.base_state().x, prev_y = be.base_state().y;
    *path = 0;
    for (int t = 1; t <= 20000; ++t) {
      if (t % 4 == 0) servo.update(t * 0.001);
      be.run(1);
      if (t % 4 == 0) {
        double w[kDriveWheels], steer[kSwerveCorners];
        int wi = 0;
        for (int c = 0; c < kSwerveCorners; ++c) {
          steer[c] = be.joint_measured(AxisSteer(c)).q;
          w[wi++] = be.joint_measured(AxisWheelL(c)).dq;
          w[wi++] = be.joint_measured(AxisWheelR(c)).dq;
        }
        ImuSample imu;
        const bool have = be.get_imu(&imu);
        ow->update(0.004, w, steer, nullptr, be.tick() * 1000);
        of->update(0.004, w, steer, have ? &imu : nullptr, be.tick() * 1000);
        const BaseState b = be.base_state();
        const double ve = std::hypot(ow->twist().vx - b.vx, ow->twist().vy - b.vy);
        sum_ve2 += ve * ve;
        ++n;
        *path += std::hypot(b.x - prev_x, b.y - prev_y);
        prev_x = b.x;
        prev_y = b.y;
      }
    }
    *v_rmse = std::sqrt(sum_ve2 / n);
  };

  // 圆弧巡航 20 s
  servo.set_twist(Twist{1.0, 0, 1.0 / 1.5});
  double v_rmse = 0, path = 0;
  collect(&o_wheel, &o_fused, &v_rmse, &path);
  const BaseState bc = be.base_state();

  // 反向+急停(高机动)8 s
  struct { int t; Twist cmd; } dyn[] = {{0, {1.0, 0, 0}}, {2000, {-1.0, 0, 0}}, {4000, {0, 0, 0}}};
  int di = 0;
  for (int t = 1; t <= 8000; ++t) {
    while (di < 2 && t > dyn[di + 1].t) ++di;
    if (t % 4 == 0) {
      servo.set_twist(dyn[di].cmd);
      servo.update(t * 0.001);
    }
    be.run(1);
    if (t % 4 == 0) {
      double w[kDriveWheels], steer[kSwerveCorners];
      int wi = 0;
      for (int c = 0; c < kSwerveCorners; ++c) {
        steer[c] = be.joint_measured(AxisSteer(c)).q;
        w[wi++] = be.joint_measured(AxisWheelL(c)).dq;
        w[wi++] = be.joint_measured(AxisWheelR(c)).dq;
      }
      ImuSample imu;
      const bool have = be.get_imu(&imu);
      o_wheel.update(0.004, w, steer, nullptr, be.tick() * 1000);
      o_fused.update(0.004, w, steer, have ? &imu : nullptr, be.tick() * 1000);
    }
  }
  const BaseState b = be.base_state();
  const double total_path = path + 8.0;
  const double drift_w = std::hypot(o_wheel.x() - b.x, o_wheel.y() - b.y);
  const double drift_f = std::hypot(o_fused.x() - b.x, o_fused.y() - b.y);
  const double yaw_err_w = std::fabs(WrapPi(o_wheel.yaw() - b.yaw));
  const double yaw_err_f = std::fabs(WrapPi(o_fused.yaw() - b.yaw));
  const double cruise_drift = std::hypot(o_wheel.x() - bc.x, o_wheel.y() - bc.y);

  std::printf("=== 里程计对照表(§5.2;测量注入:量化+噪声,舵偏置=0)===\n");
  std::printf("巡航段(圆弧 20 s): 路径 %.1f m  漂移 %.3f m  vRMSE %.3f m/s\n", path,
              cruise_drift, v_rmse);
  std::printf("全程(巡航+高机动): 路径 %.1f m\n", total_path);
  std::printf("  wheel-only: 漂移 %.3f m (%.2f%%)  航向误差 %.3f rad\n", drift_w,
              100.0 * drift_w / total_path, yaw_err_w);
  std::printf("  gyro-fused: 漂移 %.3f m (%.2f%%)  航向误差 %.3f rad  滑移残差 %.4f rad/s\n",
              drift_f, 100.0 * drift_f / total_path, yaw_err_f, o_fused.slip_residual());
  std::printf("[INFO] m1_odom 完成(基线建立,漂移%%记录供回归对比)\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  const std::string mj = argc > 2 ? argv[2] : "models/config/motors.json";
  const std::string vj = argc > 3 ? argv[3] : "models/config/x4_tennis.json";
  const std::string cj = argc > 4 ? argv[4] : "models/config/chassis_m1.json";
  return Run(scene, mj, vj, cj);
}
