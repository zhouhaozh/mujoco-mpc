#include "hal/sim_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace sckd {

SimBackend::SimBackend(const std::string& scene_xml, const std::string& motors_json,
                       const std::string& variant_json, const std::string& chassis_json) {
  cfg_ = LoadMotorConfig(motors_json, variant_json);
  chcfg_ = LoadChassisConfig(chassis_json);

  char err[1024] = "";
  m_ = mj_loadXML(scene_xml.c_str(), nullptr, err, sizeof(err));
  if (!m_) {
    std::fprintf(stderr, "[SimBackend] mj_loadXML failed: %s\n", err);
    return;
  }
  d_ = mj_makeData(m_);
  // 物理步长以配置为准(设计 §1.2:1 ms;0.5 ms 用于接触校核场景)
  const std::string model_name = m_->names ? m_->names : "";
  const bool involves_x4 = model_name == "X4_950" || cfg_.robot_variant == "x4_950" ||
                           chcfg_.robot_variant == "x4_950";
  if (involves_x4 && (model_name != "X4_950" || cfg_.robot_variant != "x4_950" ||
                      chcfg_.robot_variant != "x4_950")) {
    std::fprintf(stderr,
                 "[SimBackend] X4 variant mismatch: model='%s', motor='%s', chassis='%s'\n",
                 model_name.c_str(), cfg_.robot_variant.c_str(), chcfg_.robot_variant.c_str());
    mj_deleteData(d_);
    mj_deleteModel(m_);
    d_ = nullptr;
    m_ = nullptr;
    return;
  }
  if (!involves_x4 && std::fabs(m_->opt.timestep - cfg_.control_dt_s) > 1e-12) {
    std::fprintf(stderr, "[SimBackend] note: XML timestep %.5g != config %.5g, using config\n",
                 m_->opt.timestep, cfg_.control_dt_s);
    m_->opt.timestep = cfg_.control_dt_s;
  }
  dt_ = m_->opt.timestep;
  if (!build_maps()) {
    mj_deleteData(d_);
    mj_deleteModel(m_);
    d_ = nullptr;
    m_ = nullptr;
    return;
  }
  motor_ = std::make_unique<MotorModel>(cfg_);
  if (involves_x4 && !validate_x4_runtime()) {
    motor_.reset();
    mj_deleteData(d_);
    mj_deleteModel(m_);
    d_ = nullptr;
    m_ = nullptr;
    return;
  }
  scratch_ = mj_makeData(m_);   // 重力前馈专用(悬空姿态,无接触)
}

SimBackend::~SimBackend() {
  if (scratch_) mj_deleteData(scratch_);
  if (d_) mj_deleteData(d_);
  if (m_) mj_deleteModel(m_);
}

bool SimBackend::build_maps() {
  for (int i = 0; i < kNumAxes; ++i) {
    jid_[i] = mj_name2id(m_, mjOBJ_JOINT, kAxisJointName[i]);
    aid_[i] = mj_name2id(m_, mjOBJ_ACTUATOR, kAxisJointName[i]);
    if (jid_[i] < 0 || aid_[i] < 0) {
      std::fprintf(stderr, "[SimBackend] missing joint/actuator '%s' in model\n",
                   kAxisJointName[i]);
      return false;
    }
    qadr_[i] = m_->jnt_qposadr[jid_[i]];
    vadr_[i] = m_->jnt_dofadr[jid_[i]];
  }
  base_bid_ = mj_name2id(m_, mjOBJ_BODY, "base_link");
  const int root_jid = mj_name2id(m_, mjOBJ_JOINT, "root");
  if (base_bid_ < 0 || root_jid < 0) {
    std::fprintf(stderr, "[SimBackend] missing base_link body or root freejoint\n");
    return false;
  }
  base_qadr_ = m_->jnt_qposadr[root_jid];
  // 球(可选;demo/m2 场景含 ball body)
  ball_bid_ = mj_name2id(m_, mjOBJ_BODY, "ball");
  if (ball_bid_ >= 0) {
    const int bj = mj_name2id(m_, mjOBJ_JOINT, "ball_free");
    if (bj >= 0) ball_qadr_ = m_->jnt_qposadr[bj];
    else ball_bid_ = -1;
  }
  return true;
}

bool SimBackend::validate_x4_runtime() const {
  const auto& odom = chcfg_.odom;
  const bool zero_steer_offsets = std::all_of(
      odom.steer_offset_rad.begin(), odom.steer_offset_rad.end(),
      [](double offset) { return std::fabs(offset) <= 1e-12; });
  if (std::fabs(cfg_.control_dt_s - 0.001) > 1e-12 ||
      std::fabs(m_->opt.timestep - 0.001) > 1e-12 ||
      std::fabs(cfg_.command_delay_ms) > 1e-12 || cfg_.encoder_bits != 0 ||
      !cfg_.truth_feedback || cfg_.profile_peak ||
      std::fabs(chcfg_.servo.period_ms - 1.0) > 1e-12 ||
      std::fabs(odom.imu_rate_hz - 1000.0) > 1e-12 ||
      std::fabs(odom.imu_latency_ms) > 1e-12 ||
      std::fabs(odom.wheel_speed_noise_rad_s) > 1e-12 ||
      std::fabs(odom.gyro_bias_rad_s) > 1e-12 ||
      std::fabs(odom.gyro_noise_rad_s) > 1e-12 ||
      std::fabs(odom.acc_noise_mps2) > 1e-12 || !zero_steer_offsets) {
    std::fprintf(stderr,
                 "[SimBackend] X4 runtime requires 1 ms, zero delay, encoder_bits=0, "
                 "truth feedback, continuous torque profile, and zero-noise 1 kHz chassis "
                 "feedback\n");
    return false;
  }
  if (m_->nu != kNumAxes) {
    std::fprintf(stderr, "[SimBackend] X4 requires exactly %d actuators, got %lld\n",
                 kNumAxes, static_cast<long long>(m_->nu));
    return false;
  }
  for (int axis = 0; axis < kNumAxes; ++axis) {
    const int actuator = aid_[axis];
    const int ctrl = m_->actuator_ctrladr[actuator];
    const int out = m_->actuator_outadr[actuator];
    const double cap = motor_->torque_cap(axis, 0.0, 0.0);
    bool gear_ok = out >= 0 && m_->actuator_outnum[actuator] == 1 &&
                   std::fabs(m_->actuator_gear[6 * out] - 1.0) <= 1e-12;
    for (int k = 1; k < 6 && gear_ok; ++k)
      gear_ok = std::fabs(m_->actuator_gear[6 * out + k]) <= 1e-12;
    const bool ok = m_->actuator_trntype[actuator] == mjTRN_JOINT &&
                    m_->actuator_trnid[2 * actuator] == jid_[axis] &&
                    m_->actuator_ctrlnum[actuator] == 1 && ctrl >= 0 &&
                    m_->actuator_ctrllimited[ctrl] && gear_ok &&
                    std::fabs(m_->actuator_ctrlrange[2 * ctrl] + cap) <= 1e-9 &&
                    std::fabs(m_->actuator_ctrlrange[2 * ctrl + 1] - cap) <= 1e-9;
    if (!ok) {
      std::fprintf(stderr,
                   "[SimBackend] X4 actuator contract mismatch on axis %d ('%s'), "
                   "expected joint binding, gear=1 and ctrlrange=[%.6g, %.6g]\n",
                   axis, kAxisJointName[axis], -cap, cap);
      return false;
    }
  }
  return true;
}

bool SimBackend::reset(int keyframe) {
  if (!ok()) return false;
  if (keyframe < 0 || keyframe >= static_cast<int>(m_->nkey)) {
    std::fprintf(stderr, "[SimBackend] keyframe %d out of range (nkey=%d)\n", keyframe,
                 static_cast<int>(m_->nkey));
    return false;
  }
  mj_resetDataKeyframe(m_, d_, keyframe);
  mj_forward(m_, d_);
  motor_ = std::make_unique<MotorModel>(cfg_);  // 电机状态(延迟管道/使能)一并复位
  rng_.seed(12345);                              // 噪声流固定 -> 确定性可回归
  imu_valid_ = false;
  next_imu_tick_ = 0;
  ball_obs_valid_ = false;
  next_ball_obs_tick_ = 0;
  std::fill(std::begin(ball_hist_tick_), std::end(ball_hist_tick_), ~0ull);
  std::fill(std::begin(ball_hist_p_[0]), std::begin(ball_hist_p_[0]) + 3 * kBallHist, 0.0);
  last_bvx_ = last_bvy_ = last_bvz_ = 0;
  tick_ = 0;
  return true;
}

void SimBackend::hold_chassis(double kd) {
  for (int i = 0; i < kNumChassis; ++i) {
    MotorCmd c;
    c.kd = kd;  // kp=0:速度阻尼保持(M1 换正式舵轮分配器)
    motor_->set_cmd(i, c);
  }
}

void SimBackend::hold_arm(double kp, double kd) {
  for (int j = 0; j < kNumArm; ++j) {
    MotorCmd c;
    c.kp = kp;
    c.kd = kd;
    motor_->set_cmd(AxisArm(j), c);
  }
}

bool SimBackend::enable(int axis, bool on) { return motor_->enable(axis, on); }

void SimBackend::set_axis_cmd(int axis, const MotorCmd& cmd) { motor_->set_cmd(axis, cmd); }

JointReport SimBackend::joint_measured(int axis) {
  if (cfg_.truth_feedback) return joint(axis);
  JointReport r = joint(axis);
  r.q = quantize(r.q);
  if (axis >= 0 && axis < kNumChassis) {
    if (axis % 3 == 0) {
      // 舵角:模型侧零位偏置(装配误差;实机 x4-04 实测约 ±45°)
      r.q += chcfg_.odom.steer_offset_rad[axis / 3];
    } else {
      // 驱动轮:速度测量噪声
      r.dq += gauss_(rng_) * chcfg_.odom.wheel_speed_noise_rad_s;
    }
  }
  return r;
}

bool SimBackend::get_imu(ImuSample* out) {
  if (!ok() || !imu_valid_ || !out) return false;
  *out = imu_;
  return true;
}

bool SimBackend::arm_gravity(const double q[kNumArm], double tau_ff[kNumArm]) {
  if (!ok()) return false;
  mj_resetDataKeyframe(m_, scratch_, 0);
  scratch_->qpos[2] += 0.5;  // 悬空:排除地面接触对逆动力学的影响
  for (int j = 0; j < kNumArm; ++j) scratch_->qpos[qadr_[AxisArm(j)]] = q[j];
  // reset 后 qvel=qacc=0 -> qfrc_inverse 即纯重力项
  mj_inverse(m_, scratch_);
  for (int j = 0; j < kNumArm; ++j) tau_ff[j] = scratch_->qfrc_inverse[vadr_[AxisArm(j)]];
  return true;
}

bool SimBackend::arm_inverse(const double q[5],const double dq[5],const double ddq[5],double tau[5]) {
  if(!ok())return false;
  mj_resetDataKeyframe(m_,scratch_,0);scratch_->qpos[2]+=.5;
  for(int j=0;j<5;++j){int a=AxisArm(j);scratch_->qpos[qadr_[a]]=q[j];scratch_->qvel[vadr_[a]]=dq[j];scratch_->qacc[vadr_[a]]=ddq[j];}
  mj_inverse(m_,scratch_);
  for(int j=0;j<5;++j)tau[j]=scratch_->qfrc_inverse[vadr_[AxisArm(j)]];
  return true;
}

void SimBackend::set_arm_hybrid(const ArmHybridCmd& cmd) {
  for (int j = 0; j < kNumArm; ++j) {
    MotorCmd c;
    c.q = cmd.q[j];
    c.dq = cmd.dq[j];
    c.kp = cmd.kp[j];
    c.kd = cmd.kd[j];
    c.tau_ff = cmd.tau_ff[j];
    motor_->set_cmd(AxisArm(j), c);
  }
}

double SimBackend::quantize(double q) const {
  if (cfg_.encoder_bits <= 0) return q;
  const double res = 2.0 * M_PI / static_cast<double>(1 << cfg_.encoder_bits);
  return std::round(q / res) * res;
}

void SimBackend::write_ctrl() {
  for (int i = 0; i < kNumAxes; ++i) {
    const double qm = quantize(d_->qpos[qadr_[i]]);      // 固件视角:量化位置
    const double dvm = d_->qvel[vadr_[i]];
    d_->ctrl[aid_[i]] = motor_->compute(i, qm, dvm);     // 电机模型 -> 执行器力矩
  }
}

bool SimBackend::run(int steps) {
  if (!ok()) return false;
  for (int k = 0; k < steps; ++k) {
    motor_->tick();   // 推进命令延迟管道(9 ms @1 kHz)
    write_ctrl();     // 电机模型(混合律+限幅链) -> ctrl
    mj_step(m_, d_);
    ++tick_;
    sample_imu();     // 200Hz 采样(带龄期戳/偏置/噪声)
    sample_ball_obs();// 300Hz(带龄期戳/噪声)
  }
  return true;
}

void SimBackend::sample_imu() {
  const uint64_t period = std::max<uint64_t>(
      1, static_cast<uint64_t>(std::lround(1000.0 / chcfg_.odom.imu_rate_hz)));  // ticks(ms)
  if (tick_ < next_imu_tick_) return;
  next_imu_tick_ = tick_ + period;

  const BaseState s = base_state();
  const bool first = !imu_valid_;
  const double ax = first ? 0.0 : (s.vx - last_bvx_) / dt_;
  const double ay = first ? 0.0 : (s.vy - last_bvy_) / dt_;
  const auto& p = chcfg_.odom;
  imu_.t_us = tick_ * 1000 - static_cast<uint64_t>(p.imu_latency_ms * 1000.0);
  imu_.acc[0] = ax + gauss_(rng_) * p.acc_noise_mps2;
  imu_.acc[1] = ay + gauss_(rng_) * p.acc_noise_mps2;
  imu_.acc[2] = -m_->opt.gravity[2] + gauss_(rng_) * p.acc_noise_mps2;  // 静止读 +g
  imu_.gyr[0] = gauss_(rng_) * p.gyro_noise_rad_s;
  imu_.gyr[1] = gauss_(rng_) * p.gyro_noise_rad_s;
  imu_.gyr[2] = s.wz + p.gyro_bias_rad_s + gauss_(rng_) * p.gyro_noise_rad_s;
  last_bvx_ = s.vx;
  last_bvy_ = s.vy;
  last_bvz_ = 0.0;
  imu_valid_ = true;
}

BaseState SimBackend::base_state() const {
  BaseState s;
  if (!ok()) return s;
  const double* p = &d_->qpos[base_qadr_];
  s.x = p[0];
  s.y = p[1];
  s.z = p[2];
  const double w = p[3], x = p[4], y = p[5], z = p[6];  // MuJoCo 四元数 (w,x,y,z)
  s.roll  = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  s.pitch = std::asin(std::clamp(2 * (w * y - z * x), -1.0, 1.0));
  s.yaw   = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  // 车体系速度:用 freejoint qvel(无歧义)——线速度为世界系,按 yaw 旋转到车体系;
  // 角速度 z 分量平面运动下两系一致。
  // (注:mj_objectVelocity flg_local=1 的线速度实测与车体系不符(疑似含参考系旋转),
  //  弃用;qvel 路径与位置差分/轮速正解一致,2026-10-05 验证)
  const double wx = d_->qvel[0], wy = d_->qvel[1];
  const double c = std::cos(s.yaw), sn = std::sin(s.yaw);
  s.vx = c * wx + sn * wy;
  s.vy = -sn * wx + c * wy;
  s.wz = d_->qvel[5];
  return s;
}

void SimBackend::sample_ball_obs() {
  if (ball_bid_ < 0 || ball_qadr_ < 0) return;
  // 每 ms 入环(本 tick 末的真值位置)
  const int slot = static_cast<int>(tick_ % kBallHist);
  for (int i = 0; i < 3; ++i) ball_hist_p_[slot][i] = d_->qpos[ball_qadr_ + i];
  ball_hist_tick_[slot] = tick_;

  const uint64_t period = std::max<uint64_t>(
      1, static_cast<uint64_t>(std::lround(1000.0 / ball_obs_rate_hz_)));
  if (tick_ < next_ball_obs_tick_) return;
  next_ball_obs_tick_ = tick_ + period;
  // 延迟回放:取 latency 前那个 tick 的位置,时间戳与其一致
  const uint64_t lat = std::max<uint64_t>(
      0, static_cast<uint64_t>(std::lround(ball_obs_latency_ms_)));
  if (tick_ < lat) return;                        // 历史不足(刚 reset),本帧丢弃
  const uint64_t want = tick_ - lat;
  const int idx = static_cast<int>(want % kBallHist);
  if (ball_hist_tick_[idx] != want) return;       // 环溢出/复位后残留,丢弃
  ball_obs_.t = want * 0.001;
  for (int i = 0; i < 3; ++i)
    ball_obs_.p[i] = ball_hist_p_[idx][i] + gauss_(rng_) * ball_obs_noise_;
  ball_obs_valid_ = true;
}

bool SimBackend::get_ball_obs(BallObs* out) {
  if (cfg_.truth_feedback) {
    if (!ok() || !out || ball_bid_ < 0 || ball_qadr_ < 0) return false;
    out->t = d_->time;
    for (int i = 0; i < 3; ++i) out->p[i] = d_->qpos[ball_qadr_ + i];
    return true;
  }
  if (!ok() || !ball_obs_valid_ || !out) return false;
  *out = ball_obs_;
  return true;
}

double SimBackend::motor_torque_cap(int axis, double requested_tau_nm,
                                    double dq_rad_s) const {
  if (!has_dynamic_motor_torque_cap() || !motor_) return 36.0;
  return motor_->torque_cap(axis, requested_tau_nm, dq_rad_s);
}

bool SimBackend::has_dynamic_motor_torque_cap() const {
  return cfg_.robot_variant == "x4_950";
}

JointReport SimBackend::joint(int axis) const {
  JointReport r;
  if (!ok() || axis < 0 || axis >= kNumAxes) return r;
  r.q = d_->qpos[qadr_[axis]];
  r.dq = d_->qvel[vadr_[axis]];
  r.tau_cmd = motor_->last_tau_cmd(axis);
  r.enabled = motor_->enabled(axis);
  return r;
}

}  // namespace sckd
