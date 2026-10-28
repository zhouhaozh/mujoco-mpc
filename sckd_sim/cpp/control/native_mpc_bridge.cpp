#include "control/native_mpc_bridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "nmpc.h"

// This header/library pair adds m_previous_acceleration to the old 832-byte ABI.
static_assert(sizeof(NMPC) == 856, "NMPC header ABI does not match the supplied X4 library");

namespace sckd {
namespace {
template <typename Values>
bool AllFinite(const Values& values) {
  return std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); });
}

double EpochMilliseconds() {
  return std::chrono::duration<double, std::milli>(
      std::chrono::system_clock::now().time_since_epoch()).count();
}
}  // namespace

std::array<double, NativeMpcPlan::kControls> NativeMpcPlan::VelocityAt(double elapsed_s) const {
  if (!success || segment_dt_s <= 0 || !std::isfinite(elapsed_s)) return {};
  const double elapsed = std::clamp(elapsed_s, 0.0, kSteps * segment_dt_s);
  const int segment = std::min(static_cast<int>(elapsed / segment_dt_s), kSteps - 1);
  const double remaining = (segment + 1) * segment_dt_s - elapsed;
  auto velocity = velocities[segment];
  for (int axis = 0; axis < kControls; ++axis)
    velocity[axis] -= accelerations[segment][axis] * remaining;
  return velocity;
}

struct NativeMpcBridge::Impl {
  std::unique_ptr<NMPC> controller;
  NativeMpcHitParams hit;
  double timestamp_ms = EpochMilliseconds();
};

NativeMpcBridge::NativeMpcBridge(const std::string& libnmpc_path, int predict_steps)
    : impl_(std::make_unique<Impl>()) {
  if (predict_steps != kPredictSteps)
    throw std::invalid_argument("native MPC bridge requires exactly 3 prediction steps");
  // Direct linking fixes the ABI at build time; reject configs selecting another binary.
  if (!std::filesystem::exists(libnmpc_path) ||
      !std::filesystem::equivalent(libnmpc_path, SCKD_NATIVE_MPC_LIBRARY))
    throw std::invalid_argument("native_mpc_library must select the directly linked X4 library: "
                                SCKD_NATIVE_MPC_LIBRARY);
  impl_->controller = std::make_unique<NMPC>(predict_steps);
}

NativeMpcBridge::~NativeMpcBridge() = default;

void NativeMpcBridge::SetHitParams(const NativeMpcHitParams& params) {
  if (!AllFinite(std::array<double, 4>{params.hit_speed_m_s, params.hit_yaw_deg,
                                      params.hit_pitch_deg, params.hit_vertical_speed_m_s}))
    throw std::invalid_argument("native MPC hit parameters must be finite");
  impl_->hit = params;
}

void NativeMpcBridge::Reset() {
  impl_->controller.reset();
  impl_->controller = std::make_unique<NMPC>(kPredictSteps);
  impl_->timestamp_ms = EpochMilliseconds();
}

NativeMpcPlan NativeMpcBridge::Solve(const std::array<double, kStateSize>& state,
                                     const std::array<double, kTargetSize>& target,
                                     double remaining_s) {
  NativeMpcPlan plan;
  if (!AllFinite(state) || !AllFinite(target) || !std::isfinite(remaining_s) || remaining_s <= 0) {
    plan.error = "native MPC solve input must be finite and remaining_s must be positive";
    return plan;
  }
  const double dt = remaining_s / kPredictSteps;
  if (dt <= 0) {
    plan.error = "native MPC segment duration underflow";
    return plan;
  }
  const auto start = std::chrono::steady_clock::now();
  try {
    if (!impl_->controller) throw std::runtime_error("native MPC controller is not initialized");
    bool success = false, accuracy_success = false;
    const auto& hit = impl_->hit;
    impl_->controller->get_controls();  // Consume any controls left by an earlier call.
    const auto solution = impl_->controller->opti_solution(
        std::vector<double>(state.begin(), state.end()),
        std::vector<double>(target.begin(), target.end()), std::vector<double>(kPredictSteps, dt),
        success, accuracy_success, hit.hit_speed_m_s, hit.hit_yaw_deg,
        hit.hit_pitch_deg, hit.hit_vertical_speed_m_s, impl_->timestamp_ms);
    plan.solver_success = success;
    plan.accuracy_success = accuracy_success;
    // get_controls can contain a previous plan: never accept it after a rejected solve.
    // The library also accepts a relaxed position gate with accuracy_success=false.
    // Its nonempty returned state distinguishes that case from a rejected solution.
    if (!success || solution.empty() || !AllFinite(solution))
      throw std::runtime_error("native MPC rejected solution (solver=" + std::to_string(success) +
                               ", accuracy=" + std::to_string(accuracy_success) + ")");
    const auto controls = impl_->controller->get_controls();
    if (controls.size() != kPredictSteps * NativeMpcPlan::kControls || !AllFinite(controls))
      throw std::runtime_error("native MPC requires 21 finite acceleration controls");
    std::array<double, NativeMpcPlan::kControls> velocity{};
    for (int axis = 0; axis < NativeMpcPlan::kControls; ++axis) velocity[axis] = state[7 + axis];
    for (int step = 0; step < kPredictSteps; ++step) {
      for (int axis = 0; axis < NativeMpcPlan::kControls; ++axis) {
        const double acceleration = controls[step * NativeMpcPlan::kControls + axis];
        plan.accelerations[step][axis] = acceleration;
        velocity[axis] += acceleration * dt;
        if (!std::isfinite(velocity[axis])) throw std::runtime_error("native MPC velocity overflow");
        plan.velocities[step][axis] = velocity[axis];
      }
    }
    plan.segment_dt_s = dt;
    plan.success = true;
  } catch (const std::exception& e) {
    plan.error = e.what();
  }
  plan.solve_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  return plan;
}

}  // namespace sckd
