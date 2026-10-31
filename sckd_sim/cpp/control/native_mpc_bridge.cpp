#include "control/native_mpc_bridge.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "nmpc.h"

static_assert(sizeof(NMPC) == 832, "NMPC header ABI does not match the supplied X4 library");

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

constexpr double kAccelerationInverse[NativeMpcPlan::kControls] = {
    1.0 / 8.0, 1.0 / 8.0, 1.0 / 30.0, 1.0 / 30.0,
    1.0 / 180.0, 1.0 / 400.0, 1.0 / 200.0};
}  // namespace

void NativeMpcPlan::BuildTrajectory(const std::array<double, 14>& state) {
  plan_ms = 0;
  plan_pos.clear();
  plan_vel.clear();
  if (!AllFinite(state) || !std::isfinite(segment_dt_s) || segment_dt_s <= 0)
    throw std::invalid_argument("native MPC trajectory requires finite state and positive segment duration");
  for (const auto& step : velocity_increments)
    if (!AllFinite(step)) throw std::invalid_argument("native MPC trajectory increments must be finite");

  std::array<double, kControls> target_velocity{};
  for (int axis = 0; axis < kControls; ++axis) target_velocity[axis] = state[7 + axis];
  for (int step = 0; step < kSteps; ++step) {
    for (int axis = 0; axis < kControls; ++axis) {
      target_velocity[axis] += velocity_increments[step][axis];
      if (!std::isfinite(target_velocity[axis]))
        throw std::runtime_error("native MPC target velocity overflow");
      velocities[step][axis] = target_velocity[axis];
    }
  }

  const double total_s = segment_dt_s + segment_dt_s + segment_dt_s;
  plan_ms = total_s >= 3.0 ? 3000 : static_cast<int>(total_s * 1000.0);
  const int width = plan_ms + 1;
  plan_pos.assign(kControls * width, 0.0);
  plan_vel.assign(kControls * width, 0.0);
  const int boundary1 = static_cast<int>(std::min(segment_dt_s, 3.0) * 1000.0);
  const int boundary2 = static_cast<int>(std::min(segment_dt_s + segment_dt_s, 3.0) * 1000.0);
  std::array<double, kControls> position{};
  std::array<double, kControls> velocity{};
  for (int axis = 0; axis < kControls; ++axis) {
    position[axis] = state[axis];
    velocity[axis] = state[7 + axis];
    plan_pos[axis * width] = position[axis];
    plan_vel[axis * width] = velocity[axis];
  }
  for (int i = 1; i <= plan_ms; ++i) {
    const int step = i <= boundary1 ? 0 : (i <= boundary2 ? 1 : 2);
    for (int axis = 0; axis < kControls; ++axis) {
      const double max_delta = (1.0 / kAccelerationInverse[axis]) * kTickSeconds;
      if (velocity[axis] < velocities[step][axis])
        velocity[axis] = std::min(velocity[axis] + max_delta, velocities[step][axis]);
      else
        velocity[axis] = std::max(velocity[axis] - max_delta, velocities[step][axis]);
      position[axis] += velocity[axis] * kTickSeconds;
      if (!std::isfinite(position[axis]) || !std::isfinite(velocity[axis]))
        throw std::runtime_error("native MPC dense trajectory overflow");
      plan_pos[axis * width + i] = position[axis];
      plan_vel[axis * width + i] = velocity[axis];
    }
  }
}

NativeMpcSample NativeMpcPlan::Sample(double elapsed_s) const {
  NativeMpcSample sample;
  const int width = plan_ms + 1;
  if (!success || plan_ms < 0 || !std::isfinite(elapsed_s) ||
      plan_pos.size() != static_cast<size_t>(kControls * width) ||
      plan_vel.size() != static_cast<size_t>(kControls * width))
    return sample;
  const double elapsed = std::max(elapsed_s, 0.0);
  const double plan_duration = plan_ms * kTickSeconds;
  sample.index = elapsed >= plan_duration
                     ? plan_ms
                     : static_cast<int>(std::lround(elapsed * 1000.0));
  for (int axis = 0; axis < kControls; ++axis) {
    sample.positions[axis] = plan_pos[axis * width + sample.index];
    sample.velocities[axis] = plan_vel[axis * width + sample.index];
  }
  const double over = elapsed - plan_duration;
  if (over > 0) {
    const double extension = std::min(over, kFollowThroughSeconds);
    for (int axis = 2; axis < kControls; ++axis)
      sample.positions[axis] += sample.velocities[axis] * extension;
    if (over > kFollowThroughSeconds) sample.velocities.fill(0.0);
  }
  return sample;
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
      throw std::runtime_error("native MPC requires 21 finite velocity increments");
    for (int step = 0; step < kPredictSteps; ++step) {
      for (int axis = 0; axis < NativeMpcPlan::kControls; ++axis) {
        const double increment = controls[step * NativeMpcPlan::kControls + axis];
        plan.velocity_increments[step][axis] = increment;
      }
    }
    plan.segment_dt_s = dt;
    plan.BuildTrajectory(state);
    plan.success = true;
  } catch (const std::exception& e) {
    plan.error = e.what();
  }
  plan.solve_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  return plan;
}

}  // namespace sckd
