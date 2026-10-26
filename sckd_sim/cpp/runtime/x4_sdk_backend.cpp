#include "runtime/x4_sdk_backend.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "arm/server_sim.h"
#include "chassis/server_sim.h"
#include "core/clock.h"
#include "core/config.h"
#include "sdk/robot.h"

namespace sckd::io {
namespace {
using Endpoint = testing::X4SdkEndpoint;
using Status = testing::X4SdkEndpointStatus;

bool FloatFits(double value, double scale = 1.0) {
  return std::isfinite(value) &&
         std::abs(value) <= static_cast<double>(std::numeric_limits<float>::max()) / scale;
}

bool EncodingValid(const X4SdkJointEncoding& encoding) {
  for (int j = 0; j < arm::kJoints; ++j)
    if ((encoding.sign[j] != 1.0 && encoding.sign[j] != -1.0) ||
        !FloatFits(encoding.offset_rad[j])) return false;
  return true;
}
X4SdkJointEncoding EncodingFromConfig(const arm::ArmConfig& cfg) {
  X4SdkJointEncoding encoding;
  for (int j = 0; j < arm::kJoints; ++j) {
    encoding.sign[j] = cfg.joint_sign[j];
    encoding.offset_rad[j] = cfg.joint_offset[j];
  }
  return encoding;
}

WriteResult Preflight(const Command& c, Time now, const Capabilities& caps,
                      const X4SdkJointEncoding& encoding,
                      std::uint64_t previous_sequence) {
  if (now.clock != Clock::kMonotonic) return {false, "clock_mismatch"};
  const auto common = Validate(c, now, caps);
  if (!common.submitted) return common;
  if (!EncodingValid(encoding)) return {false, "invalid_joint_encoding_context"};
  if (c.sequence <= previous_sequence) return {false, "nonincreasing_sequence"};
  // 包括 facade 内部 m/s -> mm/s 换算，避免有限 double 在 SDK 变成 inf。
  for (int k = 0; k < 3; ++k) {
    const double scale = k < 2 ? 1000.0 : 1.0;
    if (!FloatFits(c.body_twist[k], scale))
      return {false, "float_overflow"};
    // facade 先接收 float m/s 再换算；转换的舍入上浮也不能越过 mm/s 上界。
    if (!FloatFits(static_cast<float>(c.body_twist[k]), scale))
      return {false, "float_overflow"};
  }
  for (int j = 0; j < 5; ++j) {
    if (c.tau_ff[j] != 0) return {false, "unsupported_tau_ff"};
    if (c.kp[j] <= 0 || c.kd[j] <= 0) return {false, "nonpositive_gain"};
    for (double v : {c.q[j], c.dq[j], c.kp[j], c.kd[j]})
      if (!FloatFits(v)) return {false, "float_overflow"};
    if (static_cast<float>(c.kp[j]) <= 0 || static_cast<float>(c.kd[j]) <= 0)
      return {false, "gain_float_underflow"};
    const auto& limits = arm::kJointModels[j];
    // 先在model->motor边界转换，再核对MIT编码范围。禁止静默钳位；
    // 同时检查SDK实际float运算路径，防止边界舍入或相减溢出。
    const double motor_q = encoding.sign[j] * (c.q[j] - encoding.offset_rad[j]);
    const double motor_dq = encoding.sign[j] * c.dq[j];
    const float motor_q_float = static_cast<float>(encoding.sign[j]) *
        (static_cast<float>(c.q[j]) - static_cast<float>(encoding.offset_rad[j]));
    const float motor_dq_float = static_cast<float>(encoding.sign[j]) * static_cast<float>(c.dq[j]);
    if (!std::isfinite(motor_q) || motor_q < limits.pmin || motor_q > limits.pmax ||
        !std::isfinite(motor_q_float) || motor_q_float < limits.pmin || motor_q_float > limits.pmax)
      return {false, "position_encoding_range"};
    if (!std::isfinite(motor_dq) || motor_dq < limits.vmin || motor_dq > limits.vmax ||
        !std::isfinite(motor_dq_float) || motor_dq_float < limits.vmin || motor_dq_float > limits.vmax)
      return {false, "velocity_encoding_range"};
    if (c.kp[j] < limits.kpmin || c.kp[j] > limits.kpmax ||
        c.kd[j] < limits.kdmin || c.kd[j] > limits.kdmax)
      return {false, "gain_out_of_range"};
  }
  return {true, "validated"};
}

void ToJointCommands(const Command& c, arm::JointCmd (&out)[arm::kJoints]) {
  for (int j = 0; j < arm::kJoints; ++j)
    out[j] = {static_cast<float>(c.q[j]), static_cast<float>(c.dq[j]),
              static_cast<float>(c.kp[j]), static_cast<float>(c.kd[j])};
}

JointFeedback JointFromSdk(const arm::JointState& state, bool ready, Source source) {
  JointFeedback out;
  out.q = state.pos;
  out.dq = state.speed;
  out.torque = state.torque;
  out.temperature = state.temperature;
  out.fault_bits = state.fault_bits;
  out.stamp = {static_cast<double>(state.stamp_us) * 1e-6, Clock::kMonotonic};
  out.stamp_known = state.stamp_us != 0;
  out.available = ready && out.stamp_known && std::isfinite(out.q) &&
                  std::isfinite(out.dq) && std::isfinite(out.torque) &&
                  std::isfinite(out.temperature);
  if (out.available) out.position_source = out.velocity_source = source;
  return out;
}

bool FiniteBase(const Observation& out) {
  for (double x : out.base_pose) if (!std::isfinite(x)) return false;
  for (double x : out.body_twist) if (!std::isfinite(x)) return false;
  return true;
}

class BorrowedEndpoint final : public Endpoint {
 public:
  BorrowedEndpoint(sdk::Robot& robot, const X4SdkJointEncoding& encoding, std::string frame)
      : robot_(robot), frame_(std::move(frame)), encoding_(encoding) {}
  Capabilities capabilities() const override { return {false, true, frame_}; }
  X4SdkJointEncoding joint_encoding() const override { return encoding_; }
  Status status() const override {
    return {robot_.ready(), robot_.arm_degraded(), robot_.gamepad_active()};
  }
  void Submit(const Command& c) override {
    arm::JointCmd joints[arm::kJoints];
    ToJointCommands(c, joints);
    robot_.SetJoints(joints);
    robot_.SetTwist(static_cast<float>(c.body_twist[0]),
                    static_cast<float>(c.body_twist[1]),
                    static_cast<float>(c.body_twist[2]));
  }
  Observation Snapshot(Time now) override {
    Observation out;
    const auto s = status();
    out.received = now;
    out.frame_id = frame_;
    out.ready = s.ready;
    out.arm_degraded = s.arm_degraded;
    out.manual_takeover = s.manual_takeover;
    out.base_stamp.clock = Clock::kMonotonic;
    // facade 不暴露底盘采样时间；不能用调用时间伪造 base_stamp。
    if (s.ready) {
      const auto pose = robot_.Odom();
      const auto twist = robot_.TwistFeedback();
      out.base_pose = {pose.x, pose.y, pose.w};
      out.body_twist = {twist.vx, twist.vy, twist.w};
      out.base_available = FiniteBase(out);
      if (out.base_available) out.base_source = Source::kCommandIntegrated;
    }
    if (s.ready && !s.arm_degraded)
      for (int j = 0; j < arm::kJoints; ++j)
        out.joints[j] = JointFromSdk(robot_.Joint(j), true, Source::kMeasured);
    out.received = X4SdkMonotonicNow();
    return out;  // 逐关节/底盘分别读取，coherent_snapshot 保持 false。
  }
 private:
  sdk::Robot& robot_;
  std::string frame_;
  X4SdkJointEncoding encoding_;
};

class SimulationEndpoint final : public Endpoint {
 public:
  SimulationEndpoint(const chassis::ChassisConfig& cc, const arm::ArmConfig& ac,
                     const arm::ArmServerSim::Params& params)
      : chassis_(std::make_unique<chassis::ChassisServerSim>(cc)),
        arm_(std::make_unique<arm::ArmServerSim>(ac, params)),
        encoding_(EncodingFromConfig(ac)) {}
  ~SimulationEndpoint() override {
    // 仅停止自有模拟线程；这里没有 Robot 或真实服务对象。
    if (arm_) arm_->Stop();
    if (chassis_) chassis_->Stop();
  }
  bool Start(std::string* err) {
    if (!chassis_->Start(err)) return false;
    if (!arm_->Start(err)) { chassis_->Stop(); return false; }
    started_ = true;
    return true;
  }
  Capabilities capabilities() const override { return {false, false, "sdk_model"}; }
  X4SdkJointEncoding joint_encoding() const override { return encoding_; }
  Status status() const override { return {started_ && arm_->IsRunning(), false, false}; }
  void Submit(const Command& c) override {
    arm::JointCmd joints[arm::kJoints];
    ToJointCommands(c, joints);
    arm_->SetJointTargets(joints);
    chassis_->SetTwist({static_cast<float>(c.body_twist[0] * 1000.0),
                        static_cast<float>(c.body_twist[1] * 1000.0),
                        static_cast<float>(c.body_twist[2])});
  }
  Observation Snapshot(Time now) override {
    Observation out;
    out.received = now;
    out.frame_id = "sdk_model";
    out.ready = status().ready;
    out.base_stamp.clock = Clock::kMonotonic;
    if (!out.ready) return out;
    const auto base = chassis_->Snapshot();
    const auto joints = arm_->Snapshot();
    out.base_pose = {base.odom.x * 0.001, base.odom.y * 0.001, base.odom.yaw};
    out.body_twist = {base.twist.vx * 0.001, base.twist.vy * 0.001, base.twist.w};
    out.base_stamp = {static_cast<double>(base.stamp_us) * 1e-6, Clock::kMonotonic};
    out.base_stamp_known = base.stamp_us != 0;
    out.base_available = out.base_stamp_known && FiniteBase(out);
    if (out.base_available) out.base_source = Source::kSimulationTruth;
    // SDK 臂快照可含配置噪声；不把这些合成反馈标成无噪声物理真值。
    for (int j = 0; j < arm::kJoints; ++j)
      out.joints[j] = JointFromSdk(joints.joints[j], true, Source::kEstimated);
    out.received = X4SdkMonotonicNow();
    return out;
  }
 private:
  std::unique_ptr<chassis::ChassisServerSim> chassis_;
  std::unique_ptr<arm::ArmServerSim> arm_;
  X4SdkJointEncoding encoding_;
  bool started_ = false;
};

// PIMPL：锁、序号和实际 SDK endpoint 全部由实现拥有。
class X4SdkBackend final : public RobotBackend {
 public:
  explicit X4SdkBackend(std::unique_ptr<Endpoint> endpoint);
  ~X4SdkBackend() override;
  Capabilities capabilities() const override;
  WriteResult Write(const Command& c, Time now) override;
  Observation Read(Time now) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
struct X4SdkBackend::Impl {
  explicit Impl(std::unique_ptr<Endpoint> e) : endpoint(std::move(e)) {}
  std::unique_ptr<Endpoint> endpoint;
  std::mutex mutex;
  std::uint64_t last_submitted = 0, observations = 0;
};
X4SdkBackend::X4SdkBackend(std::unique_ptr<Endpoint> e)
    : impl_(std::make_unique<Impl>(std::move(e))) {}
X4SdkBackend::~X4SdkBackend() = default;
Capabilities X4SdkBackend::capabilities() const { return impl_->endpoint->capabilities(); }
WriteResult X4SdkBackend::Write(const Command& c, Time now) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const auto checked = ValidateX4SdkCommand(c, now, capabilities(),
                                           impl_->endpoint->joint_encoding(), impl_->last_submitted);
  if (!checked.submitted) return checked;
  const auto s = impl_->endpoint->status();
  if (!s.ready) return {false, "backend_not_ready"};
  if (s.arm_degraded) return {false, "arm_degraded"};
  if (s.manual_takeover) return {false, "manual_takeover"};
  // 派发前消耗序号：即使底层抛异常，也不允许重复重试潜在的半次下发。
  impl_->last_submitted = c.sequence;
  try {
    impl_->endpoint->Submit(c);
  } catch (const std::exception& e) {
    return {false, std::string("submission_exception: ") + e.what()};
  }
  return {true, "submitted"};
}
Observation X4SdkBackend::Read(Time now) {
  if (now.clock != Clock::kMonotonic || !std::isfinite(now.seconds))
    throw std::invalid_argument("X4 SDK Read requires finite CLOCK_MONOTONIC time");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto out = impl_->endpoint->Snapshot(now);
  out.sequence = ++impl_->observations;
  out.coherent_snapshot = false;
  return out;
}

bool LoadSimParams(const core::Config& ini, arm::ArmServerSim::Params* p,
                   std::ostringstream* manifest, std::string* err) {
  auto fail = [&](const std::string& key) {
    if (err) *err = "invalid [Sim] " + key;
    return false;
  };
  auto array = [&](const char* key, float* values, bool positive, bool nonnegative) {
    const bool supplied = ini.Has("Sim", key);
    if (supplied) {
      const auto v = ini.Vec<double>("Sim", key);
      if (!v || v->size() != arm::kJoints) return fail(key);
      for (int j = 0; j < arm::kJoints; ++j) {
        const double x = (*v)[j];
        if (!FloatFits(x) || (positive && x <= 0) || (nonnegative && x < 0) ||
            (positive && static_cast<float>(x) <= 0)) return fail(key);
        values[j] = static_cast<float>(x);
      }
    }
    *manifest << key << "=";
    for (int j = 0; j < arm::kJoints; ++j) *manifest << (j ? "," : "") << values[j];
    *manifest << (supplied ? " [config]\n" : " [SDK header default]\n");
    return true;
  };
  auto scalar = [&](const char* key, float* value, bool positive, bool nonnegative) {
    const bool supplied = ini.Has("Sim", key);
    if (supplied) {
      const auto x = ini.Num("Sim", key);
      if (!x || !FloatFits(*x) || (positive && *x <= 0) || (nonnegative && *x < 0) ||
          (positive && static_cast<float>(*x) <= 0)) return fail(key);
      *value = static_cast<float>(*x);
    }
    *manifest << key << "=" << *value
              << (supplied ? " [config]\n" : " [SDK header default]\n");
    return true;
  };
  auto integer = [&](const char* key, int* value, int low, int high) {
    const bool supplied = ini.Has("Sim", key);
    if (supplied) {
      const auto x = ini.Int("Sim", key);
      if (!x || *x < low || *x > high) return fail(key);
      *value = static_cast<int>(*x);
    }
    *manifest << key << "=" << *value
              << (supplied ? " [config]\n" : " [SDK header default]\n");
    return true;
  };
  if (!array("arm_gravity_A", p->gravity_A, false, false) ||
      !array("arm_gravity_eq_deg", p->gravity_eq_deg, false, false) ||
      !array("arm_gravity_couple", p->gravity_couple, false, false) ||
      !array("arm_inertia", p->inertia, true, false) ||
      !array("arm_damping", p->damping, false, true) ||
      !scalar("arm_vel_limit_dps", &p->vel_limit_dps, true, false) ||
      !scalar("arm_noise_deg", &p->noise_deg, false, true) ||
      !scalar("floor_delta_m", &p->floor_delta_m, false, false) ||
      !scalar("tip_extra_m", &p->tip_extra_m, false, true) ||
      // SDK 1kHz 延迟环仅 64 槽；暂限定 0..63ms，避免依赖未知截断语义。
      !integer("arm_cmd_delay_ms", &p->cmd_delay_ms, 0, 63) ||
      !integer("log_hz", &p->log_hz, 1, 1000)) return false;
  const bool supplied = ini.Has("Sim", "floor_check");
  if (supplied) {
    const auto b = ini.Bool("Sim", "floor_check");
    if (!b) return fail("floor_check");
    p->floor_check = *b;
  }
  *manifest << "floor_check=" << p->floor_check
            << (supplied ? " [config]\n" : " [SDK header default]\n");
  return true;
}
}  // namespace

Time X4SdkMonotonicNow() {
  return {static_cast<double>(core::NowUs()) * 1e-6, Clock::kMonotonic};
}
WriteResult ValidateX4SdkCommand(const Command& c, Time now, const Capabilities& caps,
                                 const X4SdkJointEncoding& encoding,
                                 std::uint64_t previous_sequence) {
  return Preflight(c, now, caps, encoding, previous_sequence);
}
bool LoadX4SdkJointEncoding(const std::string& config_ini,
                           X4SdkJointEncoding* out, std::string* err) {
  if (err) err->clear();
  if (!out) { if (err) *err = "null_joint_encoding_output"; return false; }
  *out = {};  // 失败后保持显式不可用，不能沿用上次加载的转换。
  core::Config ini;
  if (!core::Config::LoadFile(config_ini, &ini, err)) return false;
  arm::ArmConfig cfg;
  if (!arm::LoadArmConfig(ini, &cfg)) {
    if (err) *err = "invalid_arm_config";
    return false;
  }
  const auto encoding = EncodingFromConfig(cfg);
  if (!EncodingValid(encoding)) {
    if (err) *err = "invalid_joint_encoding_context";
    return false;
  }
  *out = encoding;
  return true;
}
std::unique_ptr<RobotBackend> BorrowX4Robot(sdk::Robot& robot,
                                         const X4SdkJointEncoding& encoding, std::string frame_id) {
  if (frame_id.empty()) throw std::invalid_argument("empty X4 SDK frame_id");
  if (!EncodingValid(encoding)) throw std::invalid_argument("invalid X4 SDK joint encoding context");
  return testing::WrapX4SdkEndpoint(std::make_unique<BorrowedEndpoint>(robot, encoding, std::move(frame_id)));
}
std::unique_ptr<RobotBackend> CreateX4SdkSimulation(
    const std::string& config_ini, const std::string& log_dir, std::string* err) {
  if (err) err->clear();
  auto fail = [&](const std::string& reason) -> std::unique_ptr<RobotBackend> {
    if (err) *err = reason;
    return nullptr;
  };
  try {
    core::Config ini;
    if (!core::Config::LoadFile(config_ini, &ini, err)) return nullptr;
    const auto enabled = ini.Bool("Sim", "enabled");
    if (!enabled || !*enabled) return fail("simulation_factory_requires_Sim_enabled_true");
    chassis::ChassisConfig cc;
    arm::ArmConfig ac;
    if (!chassis::LoadChassisConfig(ini, &cc, err)) return nullptr;
    if (!arm::LoadArmConfig(ini, &ac)) return fail("invalid_arm_config");
    const auto encoding = EncodingFromConfig(ac);
    if (!EncodingValid(encoding)) return fail("invalid_joint_encoding_context");
    if (!cc.enable || !ac.enable) return fail("simulation_requires_chassis_and_arm_enabled");
    if (log_dir.empty()) return fail("simulation_requires_explicit_log_dir");
    arm::ArmServerSim::Params params;
    std::ostringstream manifest;
    manifest << std::setprecision(9)
             << "backend=SDK_DEVICE_SIMULATION_ONLY\nframe_id=sdk_model\n"
             << "clock=CLOCK_MONOTONIC\nconfig=" << config_ini << "\n"
             << "configuration_loaders=core::Config,LoadChassisConfig,LoadArmConfig\n"
             << "chassis=ideal_kinematic_integration\narm_feedback=synthetic_with_optional_noise\n"
             << "unused_sections=Gamepad,Imu,Vision,Motion,Mpc,MpcHil\n";
    for (int j = 0; j < arm::kJoints; ++j)
      manifest << "joint_encoding_" << j << "=sign:" << encoding.sign[j]
               << ",offset_rad:" << encoding.offset_rad[j] << " [loaded ArmConfig]\n";
    if (!LoadSimParams(ini, &params, &manifest, err)) return nullptr;
    params.log_dir = std::filesystem::absolute(log_dir).string();
    if (std::filesystem::exists(params.log_dir) &&
        (!std::filesystem::is_directory(params.log_dir) ||
         !std::filesystem::is_empty(params.log_dir)))
      return fail("simulation_log_dir_must_be_new_or_empty");
    std::filesystem::create_directories(params.log_dir);
    manifest << "log_dir=" << params.log_dir << " [factory argument]\n";
    const auto manifest_path = std::filesystem::path(params.log_dir) / "sdk_sim_config_manifest.txt";
    std::FILE* log = std::fopen(manifest_path.c_str(), "wx");
    if (!log) return fail(std::string("cannot_exclusively_create_sim_config_manifest: ") + std::strerror(errno));
    const auto contents = manifest.str();
    const bool written = std::fwrite(contents.data(), 1, contents.size(), log) == contents.size();
    const int closed = std::fclose(log);
    if (!written || closed != 0) return fail("cannot_write_sim_config_manifest");
    auto endpoint = std::make_unique<SimulationEndpoint>(cc, ac, params);
    if (!endpoint->Start(err)) return nullptr;
    return testing::WrapX4SdkEndpoint(std::move(endpoint));
  } catch (const std::exception& e) {
    return fail(std::string("simulation_factory_exception: ") + e.what());
  }
}
namespace testing {
std::unique_ptr<RobotBackend> WrapX4SdkEndpoint(std::unique_ptr<X4SdkEndpoint> endpoint) {
  if (!endpoint) throw std::invalid_argument("null X4 SDK endpoint");
  return std::make_unique<X4SdkBackend>(std::move(endpoint));
}
}  // namespace testing
}  // namespace sckd::io
