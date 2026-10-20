#include "estimation/wheel_odom.h"

#include <cmath>

namespace sckd {

WheelOdom::WheelOdom(const ChassisConfig& cfg)
    : cfg_(cfg), kin_(SwerveKinematics::FromChassisConfig(cfg)) {}

void WheelOdom::update(double dt, const double wheels_rad_s[kDriveWheels],
                       const double steer[kSwerveCorners], const ImuSample* imu,
                       uint64_t now_us) {
  double mps[kDriveWheels];
  for (int j = 0; j < kDriveWheels; ++j)
    mps[j] = cfg_.wheel_sign[j] * wheels_rad_s[j] * cfg_.wheel_radius;  // 镜像轴符号
  tw_ = kin_.EstimateTwist(mps, steer);

  // 航向:IMU 新鲜(龄期 < 回退阈值)用陀螺,否则轮式 wz(实机同语义)
  gyro_active_ = false;
  if (imu != nullptr) {
    const uint64_t age_us = now_us > imu->t_us ? now_us - imu->t_us : 0;
    if (age_us <= static_cast<uint64_t>(cfg_.odom.gyro_fallback_ms * 1000.0)) {
      yaw_ += imu->gyr[2] * dt;
      slip_ = std::fabs(imu->gyr[2] - tw_.wz);
      gyro_active_ = true;
    }
  }
  if (!gyro_active_) {
    yaw_ += tw_.wz * dt;
    slip_ = 0.0;
  }
  yaw_ = WrapPi(yaw_);

  // 体 -> 世界(按融合航向旋转)
  const double c = std::cos(yaw_), s = std::sin(yaw_);
  x_ += (tw_.vx * c - tw_.vy * s) * dt;
  y_ += (tw_.vx * s + tw_.vy * c) * dt;
}

}  // namespace sckd
