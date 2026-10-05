#include "control/chassis_servo.h"

#include <algorithm>
#include <cmath>

namespace sckd {
namespace {
constexpr double kWheelTauCap = 36.0;  // RS06 底盘电机机构限矩(N·m,§3.3)
}

ChassisServo::ChassisServo(const ChassisConfig& cfg, RobotInterface* hw)
    : cfg_(cfg), hw_(hw), kin_(SwerveKinematics::FromChassisConfig(cfg)) {}

void ChassisServo::set_twist(const Twist& cmd) { cmd_ = cmd; }

double ChassisServo::MoveToward(double v, double target, double max_dv) const {
  const double dv = target - v;
  if (std::fabs(dv) <= max_dv) return target;
  return v + (dv > 0 ? max_dv : -max_dv);
}

void ChassisServo::Ramp(double dt) {
  const double a_dt = cfg_.ramp.accel_limit_mps2 * dt;
  const double cur_mag = std::hypot(cur_.vx, cur_.vy);
  const double cmd_mag = std::hypot(cmd_.vx, cmd_.vy);
  if (cur_mag > 1e-6 && cmd_mag > 1e-6) {
    // 方向角率限幅:扫弯换向代替急停甩头(实机标定 3 rad/s)
    const double a0 = std::atan2(cur_.vy, cur_.vx);
    const double a1 = std::atan2(cmd_.vy, cmd_.vx);
    double da = AngleDiff(a1, a0);
    const double da_max = cfg_.ramp.dir_rate_limit_rad_s * dt;
    da = MoveToward(0.0, da, da_max);
    const double new_mag = MoveToward(cur_mag, cmd_mag, a_dt);
    cur_.vx = new_mag * std::cos(a0 + da);
    cur_.vy = new_mag * std::sin(a0 + da);
  } else {
    // 零速起步/停车:直接分量级限幅(方向由指令给定)
    cur_.vx = MoveToward(cur_.vx, cmd_.vx, a_dt);
    cur_.vy = MoveToward(cur_.vy, cmd_.vy, a_dt);
  }
  cur_.wz = MoveToward(cur_.wz, cmd_.wz, cfg_.ramp.ang_accel_limit_rad_s2 * dt);
}

void ChassisServo::update(double now_s) {
  const double period = cfg_.servo.period_ms * 0.001;
  double dt = period;
  if (started_) {
    dt = now_s - last_update_s_;
    if (dt <= 0.0) return;                 // 同拍重复调用
    if (dt > 10.0 * period) dt = period;   // 长间断(暂停后恢复)按单拍处理
  }
  started_ = true;
  last_update_s_ = now_s;

  Ramp(dt);

  // 测量视角(量化+舵角零位偏置+轮速噪声);舵命令回减偏置 = 标定
  double steer_meas[kSwerveCorners];
  for (int i = 0; i < kSwerveCorners; ++i)
    steer_meas[i] = hw_->joint_measured(AxisSteer(i)).q;

  CornerCmd cc[kSwerveCorners];
  kin_.Allocate(cur_, steer_meas, reversed_, cc);

  for (int i = 0; i < kSwerveCorners; ++i) {
    // 舵位置 PD(混合律;最近角语义,回减零位偏置到关节系)
    MotorCmd sc;
    sc.q = cc[i].steer_target - cfg_.odom.steer_offset_rad[i];
    sc.kp = cfg_.servo.steer_kp;
    sc.kd = cfg_.servo.steer_kd;
    hw_->set_axis_cmd(AxisSteer(i), sc);

    // 舵速率估计(用于双轮差速)
    const double steer_rate = AngleDiff(cc[i].steer_target, prev_steer_[i]) / dt;
    prev_steer_[i] = cc[i].steer_target;

    // 双轮差速 + 轮速 PI -> tau_ff(显式速度环路,条件积分抗饱和)
    double dm[kDrivesPerCorner];
    SwerveKinematics::DualWheel(cc[i].drive_mps[0], cur_.wz, steer_rate,
                                cfg_.dual_offset, dm);
    const int axes[2] = {AxisWheelL(i), AxisWheelR(i)};
    for (int j = 0; j < kDrivesPerCorner; ++j) {
      const int idx = i * kDrivesPerCorner + j;
      const double sgn = cfg_.wheel_sign[idx];              // 镜像轴符号(§4.2)
      const double v_target = dm[j];                        // 线速度 m/s(沿舵向)
      const double v_meas = sgn * hw_->joint_measured(axes[j]).dq * cfg_.wheel_radius;
      const double err = (v_target - v_meas) / cfg_.wheel_radius;  // rad/s 等效
      double tau = sgn * (cfg_.servo.wheel_kp * err + cfg_.servo.wheel_ki * wheel_int_[idx]);
      if (std::fabs(tau) <= kWheelTauCap) {
        wheel_int_[idx] += err * dt;          // 未饱和才积分
      } else {
        tau = tau > 0 ? kWheelTauCap : -kWheelTauCap;
      }
      const double imax = cfg_.servo.wheel_int_max_rad;
      wheel_int_[idx] = std::clamp(wheel_int_[idx], -imax, imax);  // 幅值钳制防缠绕
      MotorCmd wc;
      wc.tau_ff = tau;                        // kp=0/kd=0:纯前馈,环路显式
      hw_->set_axis_cmd(axes[j], wc);
    }
  }
}

}  // namespace sckd
