#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace sckd {

struct NativeMpcHitParams {
  double hit_speed_m_s = 0.0;
  double hit_yaw_deg = 0.0;
  double hit_pitch_deg = 0.0;
  double hit_vertical_speed_m_s = 0.0;
};

struct NativeMpcSample {
  std::array<double, 7> positions{};
  std::array<double, 7> velocities{};
  int index = -1;
};

struct NativeMpcPlan {
  static constexpr int kSteps = 3;
  static constexpr int kControls = 7;
  static constexpr double kTickSeconds = .001;
  static constexpr double kFollowThroughSeconds = .15;

  // Segment target velocities: measured velocity + cumulative increments.
  // Axis order: [vx, vy, dq1, dq2, dq3, dq4, dq5].
  std::array<std::array<double, kControls>, kSteps> velocities{};
  // Per-segment velocity increments (m/s for the base, rad/s for joints).
  std::array<std::array<double, kControls>, kSteps> velocity_increments{};
  bool solver_success = false;
  bool accuracy_success = false;
  double segment_dt_s = 0.0;
  int plan_ms = 0;
  std::vector<double> plan_pos;
  std::vector<double> plan_vel;
  double solve_ms = 0.0;
  bool success = false;
  std::string error;

  // Build the nmpc6-compatible 1 ms execution plan from measured state.
  void BuildTrajectory(const std::array<double, 14>& state);
  // Sample the dense plan; after its end, apply at most 150 ms of arm follow-through.
  NativeMpcSample Sample(double elapsed_s) const;
};

// Direct C++ bridge for the pinned X4 header/libnmpc.so.1 pair.
// No libmpc_c_api.so is loaded.
class NativeMpcBridge {
 public:
  static constexpr int kStateSize = 14;
  static constexpr int kTargetSize = 3;
  static constexpr int kPredictSteps = NativeMpcPlan::kSteps;

  explicit NativeMpcBridge(const std::string& libnmpc_path, int predict_steps = kPredictSteps);
  ~NativeMpcBridge();

  NativeMpcBridge(const NativeMpcBridge&) = delete;
  NativeMpcBridge& operator=(const NativeMpcBridge&) = delete;

  // This is deliberately explicit: missing support or rejected parameters are errors.
  void SetHitParams(const NativeMpcHitParams& params);
  void Reset();

  // State order: [x,y,q1..q5,vx,vy,dq1..dq5]. Target is [x,y,z].
  NativeMpcPlan Solve(const std::array<double, kStateSize>& state,
                      const std::array<double, kTargetSize>& target,
                      double remaining_s);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sckd
