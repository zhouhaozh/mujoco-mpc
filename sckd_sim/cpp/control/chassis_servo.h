#pragma once
// 底盘伺服(L2):语义命令(车体 twist)-> 轴级电机命令。
// 双拍结构(2026-10-05 重构,修 250Hz 轮速环 + 9ms 延迟下的振荡):
//   分配节拍(period_ms=4ms,对齐实机):斜坡 -> 分配(协同) -> 舵 PD 命令 + 轮目标速度
//   每拍(1kHz):轮速 PI(目标按加速度限幅趋近;打滑时轮惯量骤降,kp 取小、ki 主导)
// 增益依据:9ms 延迟 + ZOH 的稳定交越 ≈28 rad/s —— 舵 kp12(与实机台架整定一致),
// 轮 kp0.1/ki2.0。本层整体是 MPC 的"被建模对象"(等效一阶模型,§3.4)。
#include "core/config.h"
#include "hal/robot_interface.h"
#include "robot/swerve_kinematics.h"

namespace sckd {

class ChassisServo {
 public:
  ChassisServo(const ChassisConfig& cfg, RobotInterface* hw);

  void set_twist(const Twist& cmd);   // 语义命令(m/s, rad/s, 车体系)
  // 仿真实验开关:分别控制平移方向角限率和舵速度前馈钳位;默认保持原行为。
  // 不改变车体 yaw 斜坡、轮目标斜坡或电机物理保护。
  void set_steering_rate_limits(bool direction_limited, bool feedforward_limited);
  void update(double now_s);          // 每 1 ms 调用(与物理步同步)

  const Twist& ramped() const { return cur_; }
  double wheel_velocity_reference(int axis)const {
    const int idx=(axis/3)*2+(axis%3==2?1:0);
    return cfg_.wheel_sign.at(idx)*target_lin_[idx]/cfg_.wheel_radius;
  }

 private:
  void AllocateTick(double dt);       // 分配节拍:斜坡+分配+舵命令+轮目标
  void WheelTick(double dt);          // 1 kHz:目标限幅 + 轮速 PI -> tau_ff
  double MoveToward(double v, double target, double max_dv) const;

  ChassisConfig cfg_;
  RobotInterface* hw_;
  SwerveKinematics kin_;
  Twist cmd_{}, cur_{};
  bool direction_limited_ = true;
  bool feedforward_limited_ = true;
  bool reversed_[kSwerveCorners] = {};
  double wheel_int_[kDriveWheels] = {};     // PI 积分(幅值钳制抗饱和)
  double alloc_lin_[kDriveWheels] = {};     // 分配节拍轮目标(m/s,线速度)
  double target_lin_[kDriveWheels] = {};    // 限幅后当前目标(1kHz 趋近)
  double prev_steer_[kSwerveCorners] = {0, 0, 0, 0};
  double last_alloc_s_ = -1.0;
  double last_tick_s_ = -1.0;
  bool started_ = false;
};

}  // namespace sckd
