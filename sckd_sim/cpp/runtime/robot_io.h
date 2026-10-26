#pragma once
// MPC/仿真/SDK 共用边界。SI 单位，底盘速度为车体系；不依赖任何设备库。
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace sckd::io {
enum class Clock { kSimulation, kMonotonic };
struct Time { double seconds = 0; Clock clock = Clock::kSimulation; };
enum class Source { kUnavailable, kSimulationTruth, kMeasured, kEstimated, kCommandIntegrated };
struct Command {
  std::uint64_t sequence = 0, plan_id = 0;
  Time issued, expires;
  std::string frame_id = "mpc_model";
  std::array<double, 3> body_twist{};  // vx, vy, wz: m/s, m/s, rad/s
  std::array<double, 5> q{}, dq{}, kp{}, kd{}, tau_ff{};
};
struct JointFeedback {
  double q = 0, dq = 0, torque = 0, temperature = 0;
  Time stamp;
  bool available = false, stamp_known = false;
  unsigned fault_bits = 0;
  Source position_source = Source::kUnavailable, velocity_source = Source::kUnavailable;
};
struct Observation {
  std::uint64_t sequence = 0;
  Time received;
  std::string frame_id;
  std::array<double, 3> base_pose{}, body_twist{};  // world x/y/yaw; body vx/vy/wz
  Time base_stamp;
  bool base_available = false, base_stamp_known = false;
  Source base_source = Source::kUnavailable;
  std::array<JointFeedback, 5> joints{};
  bool ready = false, arm_degraded = false, manual_takeover = false;
  // false:逐关节读取，不宣称整机同一采样时刻。
  bool coherent_snapshot = false;
};
struct Capabilities {
  bool torque_feedforward = false;
  bool hardware = false;
  std::string frame_id;
};
struct WriteResult {
  bool submitted = false;  // 已交给后端，不代表驱动器确认执行。
  std::string reason;
};

inline WriteResult Validate(const Command& c, Time now, const Capabilities& caps) {
  if (c.sequence == 0) return {false, "invalid_sequence"};
  if (c.issued.clock != now.clock || c.expires.clock != now.clock)
    return {false, "clock_mismatch"};
  if (!std::isfinite(now.seconds) || !std::isfinite(c.issued.seconds) ||
      !std::isfinite(c.expires.seconds) || c.expires.seconds <= c.issued.seconds)
    return {false, "invalid_time"};
  if (now.seconds < c.issued.seconds) return {false, "not_yet_valid"};
  if (now.seconds >= c.expires.seconds) return {false, "expired"};
  if (c.frame_id.empty() || c.frame_id != caps.frame_id) return {false, "frame_mismatch"};
  for (double v : c.body_twist) if (!std::isfinite(v)) return {false, "nonfinite_command"};
  for (int j = 0; j < 5; ++j) {
    for (double v : {c.q[j], c.dq[j], c.kp[j], c.kd[j], c.tau_ff[j]})
      if (!std::isfinite(v)) return {false, "nonfinite_command"};
    if (c.kp[j] < 0 || c.kd[j] < 0) return {false, "negative_gain"};
    if (!caps.torque_feedforward && c.tau_ff[j] != 0) return {false, "unsupported_tau_ff"};
  }
  return {true, "validated"};
}

class RobotBackend {
 public:
  virtual ~RobotBackend() = default;
  virtual Capabilities capabilities() const = 0;
  virtual WriteResult Write(const Command&, Time now) = 0;
  virtual Observation Read(Time now) = 0;
};
}  // namespace sckd::io
