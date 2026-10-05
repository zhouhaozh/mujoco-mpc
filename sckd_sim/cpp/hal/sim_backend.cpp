#include "hal/sim_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace sckd {

SimBackend::SimBackend(const std::string& scene_xml, const std::string& motors_json,
                       const std::string& variant_json) {
  cfg_ = LoadMotorConfig(motors_json, variant_json);

  char err[1024] = "";
  m_ = mj_loadXML(scene_xml.c_str(), nullptr, err, sizeof(err));
  if (!m_) {
    std::fprintf(stderr, "[SimBackend] mj_loadXML failed: %s\n", err);
    return;
  }
  d_ = mj_makeData(m_);
  // 物理步长以配置为准(设计 §1.2:1 ms;0.5 ms 用于接触校核场景)
  if (std::fabs(m_->opt.timestep - cfg_.control_dt_s) > 1e-12) {
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
}

SimBackend::~SimBackend() {
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
  }
  return true;
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
  // 车体系速度:mj_objectVelocity 为只读计算,但签名取非常量 mjData
  mjtNum res[6];
  mj_objectVelocity(m_, const_cast<mjData*>(d_), mjOBJ_BODY, base_bid_, res, /*flg_local=*/1);
  s.vx = res[0];
  s.vy = res[1];
  s.wz = res[5];  // 近平面运动时车体 z ≈ 世界 z
  return s;
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
