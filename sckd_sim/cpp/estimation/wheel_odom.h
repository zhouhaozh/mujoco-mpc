#pragma once
// 舵轮里程计(§5.1/§5.2):8 轮+4 舵最小二乘正解 -> 位姿积分;
// 可选 IMU 陀螺航向融合(实机"第二刀"方案:ψ 用陀螺,x/y 按融合航向体->世界旋转,
// 龄期超时回退轮式 wz)。slip_residual = |gyro_wz − wheel_wz| 作滑移指示。
#include "core/config.h"
#include "core/types.h"
#include "robot/swerve_kinematics.h"

namespace sckd {

class WheelOdom {
 public:
  explicit WheelOdom(const ChassisConfig& cfg);

  // wheels:契约序 8 驱动轮【关节】角速度(rad/s;镜像轴符号在内部换算)
  // imu:可空(空 = 纯轮式);now_us 用于 IMU 龄期判定
  void update(double dt, const double wheels_rad_s[kDriveWheels],
              const double steer[kSwerveCorners], const ImuSample* imu,
              uint64_t now_us);

  double x() const { return x_; }
  double y() const { return y_; }
  double yaw() const { return yaw_; }
  const Twist& twist() const { return tw_; }
  double slip_residual() const { return slip_; }   // rad/s(滑移指示)
  bool gyro_active() const { return gyro_active_; }

 private:
  ChassisConfig cfg_;
  SwerveKinematics kin_;
  double x_ = 0, y_ = 0, yaw_ = 0;
  Twist tw_{};
  double slip_ = 0;
  bool gyro_active_ = false;
};

}  // namespace sckd
