#include "robot/motor_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace sckd {

MotorModel::MotorModel(const MotorSystemConfig& cfg) : cfg_(cfg) {
  delay_slots_ = std::max(0, static_cast<int>(std::lround(
                                 cfg_.command_delay_ms / (cfg_.control_dt_s * 1000.0))));
  for (auto& st : st_) st.pipe.assign(static_cast<size_t>(delay_slots_), MotorCmd{});
}

const MotorTypeConfig& MotorModel::motor_of(int axis) const {
  const std::string& want = cfg_.joints.at(axis).motor;
  for (const auto& m : cfg_.motors)
    if (m.name == want) return m;
  throw std::runtime_error("unknown motor type: " + want);
}

const JointConfig& MotorModel::joint_of(int axis) const { return cfg_.joints.at(axis); }

void MotorModel::set_cmd(int axis, const MotorCmd& cmd_in) {
  if (axis < 0 || axis >= kNumAxes) return;
  const auto& m = motor_of(axis);
  MotorCmd c = cmd_in;
  // 协议范围截断(手册运控模式量程;量程 != 机构安全范围)
  c.kp = std::clamp(c.kp, m.kp_min, m.kp_max);
  c.kd = std::clamp(c.kd, m.kd_min, m.kd_max);
  c.q  = std::clamp(c.q, -m.hybrid_position_limit_rad, m.hybrid_position_limit_rad);
  st_[axis].issued = c;  // 锁存,经管道延迟后生效
}

bool MotorModel::enable(int axis, bool on) {
  if (axis >= 0) {
    if (axis >= kNumAxes) return false;
    bool prev = st_[axis].enabled;
    st_[axis].enabled = on;
    if (!on) st_[axis].last_tau = 0.0;
    return prev;
  }
  bool prev = st_[0].enabled;
  for (auto& st : st_) {
    prev = st.enabled;
    st.enabled = on;
    if (!on) st.last_tau = 0.0;
  }
  return prev;
}

bool MotorModel::enabled(int axis) const {
  return (axis >= 0 && axis < kNumAxes) ? st_[axis].enabled : false;
}

void MotorModel::tick() {
  // 每周期把锁存命令压入管道并弹出最旧者 -> 生效命令滞后 delay_slots 拍;
  // 断流时 issued 不变 -> 管道被同一命令填满 = 目标锁存(看门狗语义,设计 §3.3)
  for (auto& st : st_) {
    st.pipe.push_back(st.issued);
    if (st.pipe.size() > static_cast<size_t>(delay_slots_)) st.pipe.pop_front();
  }
}

double MotorModel::envelope(const MotorTypeConfig& m, double abs_speed) {
  const auto& p = m.motoring;
  if (abs_speed <= p.front().speed_rad_s) return p.front().torque_nm;
  if (abs_speed >= p.back().speed_rad_s) return p.back().torque_nm;  // 末点 = 空载 0
  for (size_t i = 1; i < p.size(); ++i) {
    if (abs_speed <= p[i].speed_rad_s) {
      const double f = (abs_speed - p[i - 1].speed_rad_s) /
                       (p[i].speed_rad_s - p[i - 1].speed_rad_s);
      return p[i - 1].torque_nm + f * (p[i].torque_nm - p[i - 1].torque_nm);
    }
  }
  return p.back().torque_nm;
}

double MotorModel::compute(int axis, double q_meas, double dq_meas) {
  if (axis < 0 || axis >= kNumAxes) return 0.0;
  auto& st = st_[axis];
  if (!st.enabled) {  // 失能:零输出,物理继续
    st.last_tau = 0.0;
    return 0.0;
  }
  const auto& m = motor_of(axis);
  const auto& jc = joint_of(axis);
  const MotorCmd& c = st.pipe.empty() ? st.issued : st.pipe.front();  // 延迟后的生效命令

  // 手册混合控制律(RS04 p43 / RS06 p41)
  double tau = c.kp * (c.q - q_meas) + c.kd * (c.dq - dq_meas) + c.tau_ff;

  // 限幅链:机构限矩 -> 峰值/连续档 -> T-N 方向相关包络
  double cap = std::min(jc.mechanism_limit_nm,
                        cfg_.profile_peak ? m.peak_torque_nm : m.continuous_torque_nm);
  const bool motoring = tau * dq_meas > 0.0;
  if (motoring) {
    cap = std::min(cap, envelope(m, std::fabs(dq_meas)));
    // 软件速度保护:超过运控速度量程且仍在推动 -> 停止推动
    if (std::fabs(dq_meas) > m.hybrid_speed_limit_rad_s) cap = 0.0;
  } else {
    cap = std::min(cap, m.braking_torque_nm);
  }
  st.last_raw=tau;st.last_cap=cap;
  st.last_tau = std::clamp(tau, -cap, cap);
  return st.last_tau;
}

double MotorModel::last_tau_cmd(int axis) const {
  return (axis >= 0 && axis < kNumAxes) ? st_[axis].last_tau : 0.0;
}

}  // namespace sckd
