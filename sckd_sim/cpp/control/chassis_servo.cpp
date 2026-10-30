#include "control/chassis_servo.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sckd {

ChassisServo::ChassisServo(const ChassisConfig& cfg, RobotInterface* hw)
    : cfg_(cfg), hw_(hw), kin_(SwerveKinematics::FromChassisConfig(cfg)) {}

void ChassisServo::set_twist(const Twist& cmd) { cmd_ = cmd; }

void ChassisServo::set_steering_rate_limits(bool direction_limited,
                                           bool feedforward_limited) {
  direction_limited_ = direction_limited;
  feedforward_limited_ = feedforward_limited;
}

void ChassisServo::initialize_consistent_state(const Twist& ramped_body_twist) {
  if (started_)
    throw std::logic_error("chassis servo state must be initialized before the first update");
  if (!std::isfinite(ramped_body_twist.vx) || !std::isfinite(ramped_body_twist.vy) ||
      !std::isfinite(ramped_body_twist.wz))
    throw std::invalid_argument("chassis servo ramp state must be finite");

  cur_ = ramped_body_twist;
  cmd_ = ramped_body_twist;
  double steer_meas[kSwerveCorners];
  for (int i = 0; i < kSwerveCorners; ++i)
    steer_meas[i] = hw_->joint_measured(AxisSteer(i)).q;

  CornerCmd cc[kSwerveCorners];
  kin_.Allocate(cur_, steer_meas, reversed_, cc);
  for (int i = 0; i < kSwerveCorners; ++i) {
    prev_steer_[i] = cc[i].steer_target;
    double wheel_speed[kDrivesPerCorner];
    SwerveKinematics::DualWheel(cc[i].drive_mps[0], cur_.wz, 0.0,
                                cfg_.dual_offset, wheel_speed);
    for (int j = 0; j < kDrivesPerCorner; ++j) {
      const int idx = kDrivesPerCorner * i + j;
      alloc_lin_[idx] = wheel_speed[j];
      target_lin_[idx] = wheel_speed[j];
      wheel_int_[idx] = 0.0;
    }
  }
}

double ChassisServo::MoveToward(double v, double target, double max_dv) const {
  const double dv = target - v;
  if (std::fabs(dv) <= max_dv) return target;
  return v + (dv > 0 ? max_dv : -max_dv);
}

void ChassisServo::AllocateTick(double dt) {
  // 指令斜坡:幅值按加速度限;方向角按角率限(扫弯换向,实机标定 3 rad/s)
  const double a_dt = cfg_.ramp.accel_limit_mps2 * dt;
  const double cur_mag = std::hypot(cur_.vx, cur_.vy);
  const double cmd_mag = std::hypot(cmd_.vx, cmd_.vy);
  if (cur_mag > 1e-6 && cmd_mag > 1e-6) {
    const double a0 = std::atan2(cur_.vy, cur_.vx);
    const double a1 = std::atan2(cmd_.vy, cmd_.vx);
    double da = AngleDiff(a1, a0);
    if (direction_limited_)
      da = MoveToward(0.0, da, cfg_.ramp.dir_rate_limit_rad_s * dt);
    const double new_mag = MoveToward(cur_mag, cmd_mag, a_dt);
    cur_.vx = new_mag * std::cos(a0 + da);
    cur_.vy = new_mag * std::sin(a0 + da);
  } else {
    cur_.vx = MoveToward(cur_.vx, cmd_.vx, a_dt);
    cur_.vy = MoveToward(cur_.vy, cmd_.vy, a_dt);
  }
  cur_.wz = MoveToward(cur_.wz, cmd_.wz, cfg_.ramp.ang_accel_limit_rad_s2 * dt);

  double steer_meas[kSwerveCorners];
  for (int i = 0; i < kSwerveCorners; ++i)
    steer_meas[i] = hw_->joint_measured(AxisSteer(i)).q;

  CornerCmd cc[kSwerveCorners];
  kin_.Allocate(cur_, steer_meas, reversed_, cc);

  for (int i = 0; i < kSwerveCorners; ++i) {
    const double steer_rate = AngleDiff(cc[i].steer_target, prev_steer_[i]) / dt;
    prev_steer_[i] = cc[i].steer_target;

    MotorCmd sc;  // 舵位置 PD + 速率前馈(混合律;电机模型 1kHz 连续作用)
    sc.q = cc[i].steer_target - cfg_.odom.steer_offset_rad[i];
    sc.dq = feedforward_limited_ ? std::clamp(steer_rate, -8.0, 8.0)
                                : steer_rate;  // 舵速率前馈,非实际舵速硬限制
    sc.kp = cfg_.servo.steer_kp;
    sc.kd = cfg_.servo.steer_kd;
    hw_->set_axis_cmd(AxisSteer(i), sc);

    double dm[kDrivesPerCorner];
    SwerveKinematics::DualWheel(cc[i].drive_mps[0], cur_.wz, steer_rate,
                                cfg_.dual_offset, dm);
    alloc_lin_[2 * i] = dm[0];
    alloc_lin_[2 * i + 1] = dm[1];
  }
}

void ChassisServo::WheelTick(double dt) {
  for (int i = 0; i < kSwerveCorners; ++i) {
    const int ax[2] = {AxisWheelL(i), AxisWheelR(i)};
    for (int j = 0; j < kDrivesPerCorner; ++j) {
      const int idx = 2 * i + j;
      const double sgn = cfg_.wheel_sign[idx];  // 镜像轴符号(§4.2)
      // 目标按加速度限幅趋近(≈实机驱动器 97 rad/s² × r);记录实际斜率作前馈
      const double prev_tgt = target_lin_[idx];
      target_lin_[idx] = MoveToward(target_lin_[idx], alloc_lin_[idx],
                                    cfg_.servo.wheel_target_accel_mps2 * dt);
      const double a_tgt = (target_lin_[idx] - prev_tgt) / dt;  // m/s²
      const JointReport wheel = hw_->joint_measured(ax[j]);
      const double v_meas = sgn * wheel.dq * cfg_.wheel_radius;
      if (std::fabs(alloc_lin_[idx]) < 1e-6 && std::fabs(v_meas) < 0.05) {
        // 停稳吸附:目标零且低速 -> 清积分/零力矩,消除残余爬行
        wheel_int_[idx] = 0.0;
        MotorCmd wc0;
        hw_->set_axis_cmd(ax[j], wc0);
        continue;
      }
      const double err = (target_lin_[idx] - v_meas) / cfg_.wheel_radius;  // rad/s 等效
      // 加速度前馈:整机 m/8 轮均摊(斜坡制动不再依赖慢积分建矩;打滑时 PI 微调)
      const double tau_ff_a = cfg_.robot_mass_kg / kDriveWheels * a_tgt * cfg_.wheel_radius;
      double tau = sgn * (cfg_.servo.wheel_kp * err + cfg_.servo.wheel_ki * wheel_int_[idx]
                          + tau_ff_a);
      const double tau_cap = hw_->motor_torque_cap(ax[j], tau, wheel.dq);
      // 积分条件:未饱和 且 误差在带内(|err|<4 rad/s)——斜坡/急停大误差期间
      // 不积分,防缠绕过零反弹(2026-10-05 实测:0.6 m/s 反向回弹)
      const double imax = cfg_.servo.wheel_int_max_rad;
      if (std::fabs(tau) <= tau_cap && std::fabs(err) < 4.0) {
        const double next_int = std::clamp(wheel_int_[idx] + err * dt, -imax, imax);
        if (!hw_->has_dynamic_motor_torque_cap()) {
          // 网球版保持旧积分时序：本拍输出不重算，下拍再依 36 N·m 判断。
          wheel_int_[idx] = next_int;
        } else {
          const double next_tau = sgn * (cfg_.servo.wheel_kp * err +
                                          cfg_.servo.wheel_ki * next_int + tau_ff_a);
          const double next_cap = hw_->motor_torque_cap(ax[j], next_tau, wheel.dq);
          if (std::fabs(next_tau) <= next_cap) wheel_int_[idx] = next_int;
        }
      }
      if (std::fabs(tau) > tau_cap) tau = std::copysign(tau_cap, tau);
      MotorCmd wc;
      wc.tau_ff = tau;  // kp=0/kd=0:纯前馈,环路显式(MPC 可整体替换)
      hw_->set_axis_cmd(ax[j], wc);
    }
  }
}

void ChassisServo::update(double now_s) {
  const double period = cfg_.servo.period_ms * 0.001;
  if (!started_) {
    started_ = true;
    last_tick_s_ = now_s;
    last_alloc_s_ = now_s;
    AllocateTick(period);
    WheelTick(0.001);
    return;
  }
  double dt = now_s - last_tick_s_;
  if (dt <= 0.0) return;
  if (dt > 0.1) dt = 0.1;  // 异常大间隔截断(暂停后恢复)
  last_tick_s_ = now_s;
  if (now_s - last_alloc_s_ >= period - 1e-9) {
    double dt_a = now_s - last_alloc_s_;
    if (dt_a > 10.0 * period) dt_a = period;
    last_alloc_s_ = now_s;
    AllocateTick(dt_a);
  }
  WheelTick(dt);
}

}  // namespace sckd
