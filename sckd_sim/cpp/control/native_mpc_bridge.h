#pragma once

#include <array>
#include <memory>
#include <string>

namespace sckd {

struct NativeMpcHitParams {
  double hit_speed_m_s = 0.0;
  double hit_yaw_deg = 0.0;
  double hit_pitch_deg = 0.0;
  double hit_vertical_speed_m_s = 0.0;
};

struct NativeMpcPlan {
  static constexpr int kSteps = 3;
  static constexpr int kControls = 7;

  // Segment-end velocities: [vx, vy, dq1, dq2, dq3, dq4, dq5].
  std::array<std::array<double, kControls>, kSteps> velocities{};
  // Supplied library controls are accelerations, not velocity increments.
  std::array<std::array<double, kControls>, kSteps> accelerations{};
  bool solver_success = false;
  bool accuracy_success = false;
  double segment_dt_s = 0.0;
  double solve_ms = 0.0;
  bool success = false;
  std::string error;

  // Continuous velocity along the piecewise-constant acceleration plan.
  std::array<double, kControls> VelocityAt(double elapsed_s) const;
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
