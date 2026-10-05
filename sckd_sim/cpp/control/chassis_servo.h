#pragma once
// 底盘伺服(L2):语义命令(车体 twist)-> 轴级电机命令。
// 组成:指令斜坡(加速度/方向角率限幅) -> 分配(协同策略,§3.4)
//       -> 舵位置 PD(混合律,等效 CSP) + 轮速 PI -> tau_ff(纯前馈显式环路)。
// 本层整体是"被建模对象"的一部分:MPC 见 等效一阶模型(T_steer/加速度限),
// 可整体替换本层(设计 §3.4 控制模式决策,2026-10-05)。
// 周期 period_ms(默认 4ms,对齐实机 250Hz);调用方在 1ms 仿真循环内按周期调用。
#include "core/config.h"
#include "hal/robot_interface.h"
#include "robot/swerve_kinematics.h"

namespace sckd {

class ChassisServo {
 public:
  ChassisServo(const ChassisConfig& cfg, RobotInterface* hw);

  void set_twist(const Twist& cmd);   // 语义命令(m/s, rad/s, 车体系)
  void update(double now_s);          // 每伺服周期调用(仿真秒)

  const Twist& ramped() const { return cur_; }

 private:
  void Ramp(double dt);
  double MoveToward(double v, double target, double max_dv) const;

  ChassisConfig cfg_;
  RobotInterface* hw_;
  SwerveKinematics kin_;
  Twist cmd_{}, cur_{};
  bool reversed_[kSwerveCorners] = {};
  double wheel_int_[kDriveWheels] = {};              // 轮速 PI 积分(条件积分抗饱和)
  double prev_steer_[kSwerveCorners] = {0, 0, 0, 0}; // 舵速率估计(差分)
  double last_update_s_ = -1.0;
  bool started_ = false;
};

}  // namespace sckd
