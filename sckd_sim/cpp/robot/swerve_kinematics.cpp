#include "robot/swerve_kinematics.h"

#include <cmath>

namespace sckd {

double WrapPi(double a) {
  while (a >= M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

double AngleDiff(double to, double from) { return WrapPi(to - from); }

SwerveKinematics::SwerveKinematics(const SwerveGeometry& g, const CoordinationParams& c)
    : g_(g), c_(c) {}

void SwerveKinematics::Allocate(const Twist& t, const double steer_meas[kSwerveCorners],
                                bool reversed[kSwerveCorners],
                                CornerCmd out[kSwerveCorners]) const {
  for (int i = 0; i < kSwerveCorners; ++i) {
    // 舵轴中心刚体速度:v_i = (vx − wz·y_i, vy + wz·x_i)
    const double cx = t.vx - t.wz * g_.corner_y[i];
    const double cy = t.vy + t.wz * g_.corner_x[i];
    const double speed = std::hypot(cx, cy);

    if (speed < c_.zero_speed) {  // 速度死区:保持当前舵角(实测:防零速自指振荡)
      out[i].steer_target = steer_meas[i];
      out[i].drive_mps[0] = out[i].drive_mps[1] = 0.0;
      continue;
    }

    const double desired = std::atan2(cy, cx);
    const double err_fwd = AngleDiff(desired, steer_meas[i]);
    const double err_rev = AngleDiff(desired, steer_meas[i] + M_PI);

    // 反转判定(迟滞,防 >90° 临界抖动):反转体现在轮速符号,目标角仍取最近角
    bool rev = reversed[i];
    if (!rev) {
      if (std::fabs(err_fwd) > c_.reverse_threshold &&
          std::fabs(err_rev) < std::fabs(err_fwd)) {
        rev = true;
      }
    } else if (std::fabs(err_rev) > c_.reverse_threshold + c_.reverse_hysteresis &&
               std::fabs(err_fwd) < std::fabs(err_rev)) {
      rev = false;
    }
    reversed[i] = rev;

    const double err = rev ? err_rev : err_fwd;
    double signed_speed = rev ? -speed : speed;

    if (c_.enable_cos_projection) {  // 舵角误差大 -> 驱动速度按投影收缩
      const double mismatch = std::fabs(err);
      double proj = 1.0;
      if (mismatch > c_.cos_deadband) {
        proj = std::cos(mismatch);
        if (proj < c_.cos_floor) proj = c_.cos_floor;
      }
      signed_speed *= proj;
    }

    out[i].steer_target = steer_meas[i] + err;
    out[i].drive_mps[0] = out[i].drive_mps[1] = signed_speed;
  }
}

void SwerveKinematics::DualWheel(double module_mps, double wz, double steer_rate,
                                 double dual_offset, double out_mps[kDrivesPerCorner]) {
  // j=0(wheel_L)在 +n 侧:d=+offset/2;θ̇ 与 wz 对 +n 侧轮同为减速项
  const double d0 = 0.5 * dual_offset;
  out_mps[0] = module_mps - d0 * (wz + steer_rate);
  out_mps[1] = module_mps + d0 * (wz + steer_rate);
}

Twist SwerveKinematics::EstimateTwist(const double drive_mps[kDriveWheels],
                                      const double steer_meas[kSwerveCorners]) const {
  // 轮速度矢量方向 u=(cosθ,sinθ);轮位置 p_ij = 舵轴 ± d·n(θ),n=(−sinθ,cosθ);
  // 刚体投影方程:v·cosθ = vx − wz·p_y ; v·sinθ = vy + wz·p_x
  Eigen::Matrix<double, 2 * kDriveWheels, 3> A;
  Eigen::Matrix<double, 2 * kDriveWheels, 1> b;
  int row = 0;
  for (int i = 0; i < kSwerveCorners; ++i) {
    const double ux = std::cos(steer_meas[i]);
    const double uy = std::sin(steer_meas[i]);
    for (int j = 0; j < kDrivesPerCorner; ++j) {
      const double d = (j == 0 ? 0.5 : -0.5) * g_.dual_offset;
      const double px = g_.corner_x[i] - d * uy;  // +n = (−sinθ, cosθ)
      const double py = g_.corner_y[i] + d * ux;
      const double v = drive_mps[i * kDrivesPerCorner + j];
      A.row(row) << 1.0, 0.0, -py;
      b(row++) = v * ux;
      A.row(row) << 0.0, 1.0, px;
      b(row++) = v * uy;
    }
  }
  Twist out;
  const Eigen::Matrix<double, 3, 1> x =
      (A.transpose() * A).ldlt().solve(A.transpose() * b);
  out.vx = x(0);
  out.vy = x(1);
  out.wz = x(2);
  return out;
}

}  // namespace sckd
