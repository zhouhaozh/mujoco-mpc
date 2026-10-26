#include "runtime/x4_arm_test_backend.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "arm/server.h"
#include "arm/server_sim.h"
#include "core/config.h"

namespace sckd::io::axischeck {
namespace {

bool Fail(std::string* error, const std::string& reason) {
  if (error) *error = reason;
  return false;
}

std::string Trim(std::string text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

std::vector<std::string> RequiredList(const core::Config& ini,
                                      const std::string& key, std::size_t count) {
  const auto value = ini.Str("Arm", key);
  if (!value) throw std::invalid_argument("missing [Arm]." + key);
  std::vector<std::string> result;
  std::istringstream input(*value);
  std::string item;
  while (std::getline(input, item, ',')) result.push_back(Trim(item));
  if (value->empty() || value->back() == ',' || result.size() != count ||
      std::any_of(result.begin(), result.end(), [](const auto& x) { return x.empty(); }))
    throw std::invalid_argument("[Arm]." + key + " requires " + std::to_string(count) + " values");
  return result;
}

std::vector<double> RequiredNumbers(const core::Config& ini,
                                    const std::string& key, std::size_t count) {
  const auto fields = RequiredList(ini, key, count);
  std::vector<double> result;
  for (const auto& field : fields) {
    std::size_t consumed = 0;
    double value;
    try { value = std::stod(field, &consumed); }
    catch (const std::exception&) {
      throw std::invalid_argument("invalid numeric [Arm]." + key);
    }
    if (consumed != field.size() || !std::isfinite(value) ||
        std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
      throw std::invalid_argument("invalid finite float [Arm]." + key);
    result.push_back(value);
  }
  return result;
}

// LoadArmConfig 允许字段缺失/非法后回落默认，测试工具必须先拒绝这种情况。
void ValidateExplicitFields(const core::Config& ini) {
  RequiredNumbers(ini, "dimens", 4);
  RequiredNumbers(ini, ini.Has("Arm", "kp_hold") ? "kp_hold" : "kp", 5);
  RequiredNumbers(ini, ini.Has("Arm", "kd_hold") ? "kd_hold" : "kd", 5);
  RequiredNumbers(ini, "kp_exit", 5);
  RequiredNumbers(ini, "kd_exit", 5);
  RequiredNumbers(ini, "home_deg", 5);
  for (double sign : RequiredNumbers(ini, "joint_sign", 5))
    if (sign != 1 && sign != -1)
      throw std::invalid_argument("[Arm].joint_sign must contain only +1/-1");
  RequiredNumbers(ini, "joint_offset_deg", 5);
  if (RequiredNumbers(ini, "control_period_ms", 1)[0] <= 0)
    throw std::invalid_argument("[Arm].control_period_ms must be positive");
  const auto channels = RequiredList(ini, "can_if", 3);
  for (int ch = 0; ch < 3; ++ch)
    if (channels[ch] != "can" + std::to_string(ch + 4))
      throw std::invalid_argument("arm test requires explicit can4,can5,can6 in this order");
  if (ini.Has("Arm", "enable")) {
    const auto enabled = ini.Bool("Arm", "enable");
    if (!enabled || !*enabled) throw std::invalid_argument("[Arm].enable must be true if present");
  }
}

// 除检查原文本，还与 SDK 实际加载值对拍，避免加载器默认值掩盖配置意图。
void VerifyLoadedFields(const core::Config& ini, const arm::ArmConfig& cfg) {
  const auto same = [&](const std::string& key, const float* values,
                        std::size_t count, double scale = 1.) {
    const auto requested = RequiredNumbers(ini, key, count);
    for (std::size_t i = 0; i < count; ++i) {
      const double expected = requested[i] * scale;
      if (!std::isfinite(values[i]) ||
          std::abs(values[i] - expected) > 2e-6 * std::max(1., std::abs(expected)))
        throw std::invalid_argument("SDK config value differs from explicit [Arm]." + key);
    }
  };
  same("dimens", cfg.dims, 4);
  same(ini.Has("Arm", "kp_hold") ? "kp_hold" : "kp", cfg.kp, 5);
  same(ini.Has("Arm", "kd_hold") ? "kd_hold" : "kd", cfg.kd, 5);
  same("kp_exit", cfg.kp_exit, 5);
  same("kd_exit", cfg.kd_exit, 5);
  same("home_deg", cfg.home, 5, kDeg);
  same("joint_sign", cfg.joint_sign, 5);
  same("joint_offset_deg", cfg.joint_offset, 5, kDeg);
  same("control_period_ms", &cfg.control_period_ms, 1);
}

bool FloatFits(double x) {
  return std::isfinite(x) &&
      std::abs(x) <= static_cast<double>(std::numeric_limits<float>::max());
}

bool ValidateCommand(const Joints& q, const Joints& dq, const Joints& kp,
                     const Joints& kd, const Setup& setup, std::string* error) {
  for (int j = 0; j < arm::kJoints; ++j) {
    const std::string suffix = ":J" + std::to_string(j + 1);
    for (double value : {q[j], dq[j], kp[j], kd[j]})
      if (!FloatFits(value)) return Fail(error, "nonfinite_or_float_overflow" + suffix);
    if (kp[j] <= 0 || kd[j] <= 0 ||
        static_cast<float>(kp[j]) <= 0 || static_cast<float>(kd[j]) <= 0)
      return Fail(error, "nonpositive_hold_or_exit_gain" + suffix);
    const auto& limits = arm::kJointModels[j];
    // 同时验证 double 表达式及 SDK 实际 float 运算路径，拒绝静默编码钳位。
    const double motor_q = setup.sign[j] * (q[j] - setup.offset[j]);
    const double motor_dq = setup.sign[j] * dq[j];
    const float motor_q_float = static_cast<float>(setup.sign[j]) *
        (static_cast<float>(q[j]) - static_cast<float>(setup.offset[j]));
    const float motor_dq_float = static_cast<float>(setup.sign[j]) * static_cast<float>(dq[j]);
    if (!std::isfinite(motor_q) || motor_q < limits.pmin || motor_q > limits.pmax ||
        !std::isfinite(motor_q_float) || motor_q_float < limits.pmin || motor_q_float > limits.pmax)
      return Fail(error, "position_encoding_range" + suffix);
    if (!std::isfinite(motor_dq) || motor_dq < limits.vmin || motor_dq > limits.vmax ||
        !std::isfinite(motor_dq_float) || motor_dq_float < limits.vmin || motor_dq_float > limits.vmax)
      return Fail(error, "velocity_encoding_range" + suffix);
    if (kp[j] < limits.kpmin || kp[j] > limits.kpmax ||
        kd[j] < limits.kdmin || kd[j] > limits.kdmax ||
        static_cast<float>(kp[j]) < limits.kpmin || static_cast<float>(kp[j]) > limits.kpmax ||
        static_cast<float>(kd[j]) < limits.kdmin || static_cast<float>(kd[j]) > limits.kdmax)
      return Fail(error, "gain_encoding_range" + suffix);
  }
  return true;
}

Setup DescribeAndValidate(const arm::ArmConfig& cfg) {
  Setup setup;
  for (int j = 0; j < arm::kJoints; ++j) {
    setup.home[j] = cfg.home[j];
    setup.kp[j] = cfg.kp[j];
    setup.kd[j] = cfg.kd[j];
    setup.kp_exit[j] = cfg.kp_exit[j];
    setup.kd_exit[j] = cfg.kd_exit[j];
    setup.sign[j] = cfg.joint_sign[j];
    setup.offset[j] = cfg.joint_offset[j];
    if ((setup.sign[j] != 1 && setup.sign[j] != -1) || !FloatFits(setup.offset[j]))
      throw std::invalid_argument("invalid joint_sign/offset at J" + std::to_string(j + 1));
  }
  for (int k = 0; k < 4; ++k) {
    setup.dims[k] = cfg.dims[k];
    if (!std::isfinite(setup.dims[k])) throw std::invalid_argument("nonfinite Arm dimension");
  }
  for (int ch = 0; ch < 3; ++ch) {
    setup.can_if[ch] = cfg.can_if[ch];
    if (setup.can_if[ch] != "can" + std::to_string(ch + 4))
      throw std::invalid_argument("arm test requires can4,can5,can6 in this order");
  }
  setup.period_s = static_cast<double>(cfg.control_period_ms) * .001;
  if (!std::isfinite(setup.period_s) || setup.period_s <= 0 || !cfg.enable)
    throw std::invalid_argument("invalid arm period or disabled arm");
  std::string error;
  if (!ValidateCommand(setup.home, Joints{}, setup.kp, setup.kd, setup, &error) ||
      !ValidateCommand(setup.home, Joints{}, setup.kp_exit, setup.kd_exit, setup, &error))
    throw std::invalid_argument("invalid Arm startup/exit configuration: " + error);
  return setup;
}

}  // namespace

struct SdkArmBackend::Impl {
  explicit Impl(const std::string& path) {
    core::Config ini;
    std::string error;
    if (!core::Config::LoadFile(path, &ini, &error))
      throw std::invalid_argument("cannot load arm config: " + error);
    ValidateExplicitFields(ini);
    if (!arm::LoadArmConfig(ini, &cfg)) throw std::invalid_argument("LoadArmConfig rejected config");
    VerifyLoadedFields(ini, cfg);
    setup = DescribeAndValidate(cfg);
  }
  arm::ArmConfig cfg;
  Setup setup;
  std::unique_ptr<arm::ArmServer> server;
  bool attempted = false, stop_called = false, seeded = false, simulation = false;
  bool targets_enabled = false;

  bool Start(bool use_simulation, const std::string& log_dir, std::string* error) {
    if (attempted) return Fail(error, "backend_start_already_attempted");
    attempted = true;
    simulation = use_simulation;
    try {
      if (simulation) {
        arm::ArmServerSim::Params params;
        for (int j = 0; j < arm::kJoints; ++j) {
          params.gravity_A[j] = static_cast<float>(setup.sim_gravity_amplitude[j]);
          params.inertia[j] = static_cast<float>(setup.sim_inertia[j]);
          params.damping[j] = static_cast<float>(setup.sim_damping[j]);
        }
        params.noise_deg = static_cast<float>(setup.sim_noise_deg);
        params.cmd_delay_ms = setup.sim_command_delay_ms;
        params.floor_check = setup.sim_floor_check;
        params.log_dir = log_dir;
        server = std::make_unique<arm::ArmServerSim>(cfg, params);
        // 指定SDK的Loop会在未seed时把初始cmd_=0写入9ms延迟队列；
        // 第一条正常命令到来后可短暂消费这些旧零帧。仅对模拟后端在线程
        // 启动前预置home/零速/hold增益，保持完整9ms延迟且不放宽监测阈值。
        arm::JointCmd home_cmd[arm::kJoints];
        for (int j = 0; j < arm::kJoints; ++j)
          home_cmd[j] = {cfg.home[j], 0.f, cfg.kp[j], cfg.kd[j]};
        server->SetJointTargets(home_cmd);
        seeded = true;
      } else {
        server = std::make_unique<arm::ArmServer>(cfg);
      }
      if (!server->Start(error)) { Stop(); return false; }
      if (!server->IsRunning()) { Stop(); return Fail(error, "server_not_running_after_start"); }
      if (!simulation) {
        const auto status = server->Snapshot();
        for (bool open : status.can_open)
          if (!open) { Stop(); return Fail(error, "partial_can_start"); }
      }
      if (error) error->clear();
      return true;
    } catch (const std::exception& ex) {
      Stop();
      return Fail(error, std::string("backend_start_exception: ") + ex.what());
    }
  }

  void Stop() {
    if (!server || stop_called) return;
    stop_called = true;
    // 不调用 Hold；该方法可能把陈旧/缺失反馈变成新的保持目标。
    server->Stop();
  }
};

SdkArmBackend::SdkArmBackend(const std::string& config_path)
    : impl_(std::make_unique<Impl>(config_path)) {}
SdkArmBackend::~SdkArmBackend() {
  try { Stop(); } catch (...) { /* 析构不传播异常；硬件急停由现场独立链路负责。 */ }
}
const Setup& SdkArmBackend::setup() const { return impl_->setup; }
bool SdkArmBackend::StartSim(const std::string& log_dir, std::string* error) {
  if (log_dir.empty()) return Fail(error, "empty_sim_log_directory");
  return impl_->Start(true, log_dir, error);
}
bool SdkArmBackend::StartHardware(std::string* error) { return impl_->Start(false, {}, error); }

Feedback SdkArmBackend::Read() const {
  Feedback out;
  if (!impl_->server) return out;
  out.running = !impl_->stop_called && impl_->server->IsRunning();
  out.mode_state_synthetic = impl_->simulation && out.running;
  auto status = impl_->server->Snapshot();
  out.snapshot_attempts = 1;
  // SDK Snapshot对seqlock竞争只读一次，失败的关节槽保持全零。
  // 仅重读该特征，最多3次；不缓存、不补时间戳、不改实际mode。
  for(unsigned attempt=1;attempt<3;++attempt) {
    bool missing=false, invalid=!out.running;
    for(bool open:status.can_open) if(!open) invalid=true;
    timespec ts{};
    if(clock_gettime(CLOCK_MONOTONIC,&ts)!=0) break;
    const double now=ts.tv_sec+ts.tv_nsec*1e-9;
    for(const auto& s:status.joints) {
      const bool empty=s.pos==0 && s.speed==0 && s.torque==0 && s.temperature==0 &&
                       s.stamp_us==0 && s.mode_state==0 && s.fault_bits==0;
      missing=missing || empty;
      if(!empty && (!std::isfinite(s.pos) || !std::isfinite(s.speed) ||
          !std::isfinite(s.torque) || !std::isfinite(s.temperature) ||
          s.fault_bits!=0 || s.stamp_us==0 ||
          now-static_cast<double>(s.stamp_us)*1e-6>Policy{}.freshness_s ||
          static_cast<double>(s.stamp_us)*1e-6>now ||
          (!impl_->simulation && s.mode_state!=2) ||
          std::abs(s.speed)>Policy{}.max_feedback_speed_rad_s)) invalid=true;
    }
    if(!missing || invalid) break;
    status=impl_->server->Snapshot();
    ++out.snapshot_attempts;
  }
  for (int ch = 0; ch < 3; ++ch) out.can_open[ch] = status.can_open[ch];
  for (int j = 0; j < arm::kJoints; ++j) {
    const auto& state = status.joints[j];
    out.sample.q[j] = state.pos;
    out.sample.dq[j] = state.speed;
    // status.stamp_us 是取快照时间，不能代表每轴最新反馈。
    out.sample.stamp_s[j] = static_cast<double>(state.stamp_us) * 1e-6;
    out.sample.fault_bits[j] = state.fault_bits;
    out.raw_mode_state[j] = state.mode_state;
    // SDK 仿真未实现电机mode；只为模拟流程合成Motor状态，硬件原样保留。
    // 不修改stamp/fault，且CLI必须在清单和采样日志标明此来源。
    out.sample.mode_state[j] = out.mode_state_synthetic ? 2 : state.mode_state;
    out.torque[j] = state.torque;
    out.temperature[j] = state.temperature;
  }
  return out;
}

Joints SdkArmBackend::LastTarget() const {
  if (!impl_->server || !impl_->seeded) return impl_->setup.home;
  arm::JointCmd cmd[arm::kJoints];
  impl_->server->LastJointCmds(cmd);
  Joints result{};
  for (int j = 0; j < arm::kJoints; ++j) result[j] = cmd[j].pos;
  return result;
}

bool SdkArmBackend::Write(const Output& output, std::string* error) {
  if (!impl_->server || impl_->stop_called || !impl_->server->IsRunning())
    return Fail(error, "backend_not_running");
  if (!output.initialized) return Fail(error, "output_not_initialized");
  if (output.fault)
    for (double velocity : output.dq)
      if (velocity != 0) return Fail(error, "fault_output_velocity_nonzero");
  if (!ValidateCommand(output.q, output.dq, impl_->setup.kp,
                       impl_->setup.kd, impl_->setup, error)) return false;
  arm::JointCmd cmds[arm::kJoints];
  for (int j = 0; j < arm::kJoints; ++j)
    cmds[j] = {static_cast<float>(output.q[j]), static_cast<float>(output.dq[j]),
               static_cast<float>(impl_->setup.kp[j]), static_cast<float>(impl_->setup.kd[j])};
  impl_->server->SetJointTargets(cmds);
  impl_->seeded = true;
  if (!impl_->targets_enabled) {
    // 硬件SDK默认enabled_=false：ChannelLoop仍发home，即便cmd_已更新。
    // 先完整提交并校验目标，再接管目标通道；Start/home/启动稳定窗不变。
    // false不是失能/停止，退出时不可用SetEnabled(false)代替SDK Stop。
    impl_->server->SetEnabled(true);
    impl_->targets_enabled = true;
  }
  if (error) error->clear();
  return true;
}
void SdkArmBackend::Stop() { impl_->Stop(); }

}  // namespace sckd::io::axischeck
