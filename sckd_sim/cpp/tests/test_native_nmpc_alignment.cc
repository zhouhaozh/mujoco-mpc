#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "control/legacy_nmpc.h"
#include "control/native_mpc_bridge.h"

#ifndef SCKD_LEGACY_NMPC_LIBRARY
#error "SCKD_LEGACY_NMPC_LIBRARY must name the nmpc6 generated library"
#endif

namespace {

using sckd::LegacyNmpc;
using sckd::LegacyNmpcConfig;
using sckd::LegacyNmpcRequest;
using sckd::LegacyNmpcResult;
using sckd::NativeMpcPlan;

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void RequireNear(double actual, double expected, double tolerance, const char* message) {
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      std::abs(actual - expected) > tolerance) {
    std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
    throw std::runtime_error(message);
  }
}

std::array<double, 14> LoggedState() {
  return {2.78, 0.03, -0.0193708, 0.276508, -0.34669, -1.5647, 0.0161073,
          0.503533, -0.0818887, 0, 0, 0, 0, 0};
}

LegacyNmpcRequest LoggedRequest() {
  LegacyNmpcRequest request;
  const auto state = LoggedState();
  std::copy(state.begin(), state.end(), request.state);
  const double target[3] = {3.59, -1.26, .8};
  std::copy(target, target + 3, request.target);
  request.dt[0] = request.dt[1] = request.dt[2] = .158679;
  request.hit_spd = 5;
  request.hit_speed_z = 2.1;
  request.hit_pitch_deg = 61;
  request.hit_yaw_deg = 0;
  return request;
}

NativeMpcPlan SyntheticPlan(const std::array<double, 14>& state, double segment_dt) {
  NativeMpcPlan plan;
  plan.segment_dt_s = segment_dt;
  // Mixed signs force acceleration, deceleration and direction changes.
  plan.velocity_increments[0] = {1, -.5, -.4, .3, -.2, .1, -.6};
  plan.velocity_increments[1] = {-2, 1, .8, -.7, .6, -.5, 1.2};
  plan.velocity_increments[2] = {1, -.5, -.4, .4, -.4, .4, -.6};
  plan.BuildTrajectory(state);
  plan.success = true;
  return plan;
}

void CheckSyntheticTrajectory() {
  std::array<double, 14> state = {1, -2, .1, .2, .3, .4, .5,
                                  .5, -.25, -.02, .04, -.06, .08, -.1};
  auto plan = SyntheticPlan(state, .0017);  // 5 ms total; boundaries at 1 and 3 ms.
  Require(plan.plan_ms == 5, "non-integer segment duration was rounded incorrectly");
  Require(plan.plan_pos.size() == 42 && plan.plan_vel.size() == 42,
          "axis-major dense trajectory has the wrong size");

  const int width = plan.plan_ms + 1;
  const double expected_vx[] = {.5, .508, .5, .492, .5, .5};
  double expected_x = state[0];
  for (int i = 0; i <= plan.plan_ms; ++i) {
    RequireNear(plan.plan_vel[i], expected_vx[i], 1e-12,
                "base velocity does not follow the nmpc6 acceleration ramp");
    if (i) expected_x += expected_vx[i] * NativeMpcPlan::kTickSeconds;
    RequireNear(plan.plan_pos[i], expected_x, 1e-12,
                "base position is not integrated from the ramped velocity");
  }
  RequireNear(plan.plan_pos[2 * width], state[2], 1e-12,
              "joint trajectory origin does not use measured position");
  RequireNear(plan.plan_vel[2 * width], state[9], 1e-12,
              "joint trajectory origin does not use measured velocity");
  RequireNear(plan.plan_vel[2 * width + 1], -.05, 1e-12,
              "joint did not accelerate in the initial negative direction");
  RequireNear(plan.plan_vel[2 * width + 2], -.02, 1e-12,
              "joint reversal ramp did not approach zero from below");
  RequireNear(plan.plan_vel[2 * width + 3], .01, 1e-12,
              "joint reversal ramp did not cross zero");

  const auto origin = plan.Sample(-1);
  const auto next = plan.Sample(NativeMpcPlan::kTickSeconds);
  Require(origin.index == 0 && next.index == 1,
          "next-tick sampling does not advance past the measured origin");
  RequireNear(next.positions[0], plan.plan_pos[1], 0,
              "next-tick position does not match dense plan index 1");
  Require(plan.Sample(std::numeric_limits<double>::quiet_NaN()).index == -1,
          "nonfinite sampling time was accepted");
  plan.success = false;
  Require(plan.Sample(0).index == -1, "failed plan returned an execution sample");
  plan.success = true;

  const double end_s = plan.plan_ms * NativeMpcPlan::kTickSeconds;
  const auto follow = plan.Sample(end_s + .1);
  const auto stop = plan.Sample(end_s + NativeMpcPlan::kFollowThroughSeconds + .001);
  Require(follow.index == plan.plan_ms && stop.index == plan.plan_ms,
          "follow-through did not clamp the dense-plan index");
  for (int axis = 0; axis < 2; ++axis) {
    RequireNear(follow.positions[axis], plan.plan_pos[axis * width + plan.plan_ms], 0,
                "base position must remain at the dense-plan endpoint");
    RequireNear(follow.velocities[axis], plan.plan_vel[axis * width + plan.plan_ms], 0,
                "base velocity must continue during follow-through");
  }
  for (int axis = 2; axis < 7; ++axis) {
    const double endpoint_q = plan.plan_pos[axis * width + plan.plan_ms];
    const double endpoint_dq = plan.plan_vel[axis * width + plan.plan_ms];
    RequireNear(follow.positions[axis], endpoint_q + endpoint_dq * .1, 1e-12,
                "joint follow-through is not constant-velocity extrapolation");
    RequireNear(stop.positions[axis], endpoint_q + endpoint_dq * .15, 1e-12,
                "joint did not hold the 150 ms follow-through endpoint");
  }
  for (double velocity : stop.velocities)
    RequireNear(velocity, 0, 0, "velocity did not stop after follow-through");

  // A zero-length dense plan makes the plan endpoint exactly representable, so
  // these checks isolate the strict `over > 150 ms` boundary semantics.
  auto boundary = SyntheticPlan(state, .0001);
  Require(boundary.plan_ms == 0, "follow-through boundary fixture has a dense tick");
  const double before_150 = std::nextafter(NativeMpcPlan::kFollowThroughSeconds, 0.0);
  const double after_150 =
      std::nextafter(NativeMpcPlan::kFollowThroughSeconds,
                     std::numeric_limits<double>::infinity());
  const auto before_boundary = boundary.Sample(before_150);
  const auto at_boundary = boundary.Sample(NativeMpcPlan::kFollowThroughSeconds);
  const auto after_boundary = boundary.Sample(after_150);
  RequireNear(before_boundary.velocities[2], state[9], 0,
              "joint stopped before the 150 ms follow-through boundary");
  RequireNear(at_boundary.velocities[2], state[9], 0,
              "joint stopped at the inclusive 150 ms follow-through boundary");
  RequireNear(after_boundary.velocities[2], 0, 0,
              "joint continued after the 150 ms follow-through boundary");
  RequireNear(at_boundary.positions[2], state[2] + state[9] * .15, 1e-12,
              "joint position at the 150 ms boundary is wrong");
  RequireNear(after_boundary.positions[2], at_boundary.positions[2], 0,
              "joint position changed after the follow-through boundary");

  auto cap = SyntheticPlan(state, 2.0);
  Require(cap.plan_ms == 3000, "dense trajectory exceeded the nmpc6 3 s cap");

  auto replanned_state = state;
  replanned_state[0] = 4.5;
  replanned_state[7] = -1.25;
  auto replanned = SyntheticPlan(replanned_state, .0017);
  RequireNear(replanned.plan_pos[0], replanned_state[0], 0,
              "replan inherited the previous command position");
  RequireNear(replanned.plan_vel[0], replanned_state[7], 0,
              "replan inherited the previous command velocity");
  RequireNear(replanned.Sample(.001).velocities[0], -1.242, 1e-12,
              "replan next-tick command did not start from new measured velocity");

  // Model consecutive accepted solves: each new plan starts from the prior
  // 1 ms command, and execution must select index 1 rather than replay index 0.
  auto measured = state;
  for (int iteration = 0; iteration < 5; ++iteration) {
    const double previous_x = measured[0];
    const double previous_vx = measured[7];
    auto iteration_plan = SyntheticPlan(measured, .0017);
    const auto command = iteration_plan.Sample(NativeMpcPlan::kTickSeconds);
    Require(command.index == 1, "consecutive replan sampled measured index 0");
    RequireNear(command.velocities[0], previous_vx + .008, 1e-12,
                "consecutive replan velocity did not advance one tick");
    RequireNear(command.positions[0], previous_x + command.velocities[0] * .001,
                1e-12, "consecutive replan position did not advance one tick");
    for (int axis = 0; axis < 7; ++axis) {
      measured[axis] = command.positions[axis];
      measured[7 + axis] = command.velocities[axis];
    }
  }
}

void CheckLegacyDifferential() {
  LegacyNmpcConfig config;
  config.nlp_so = SCKD_LEGACY_NMPC_LIBRARY;
  LegacyNmpc legacy(config);
  const auto request = LoggedRequest();
  LegacyNmpcResult result;
  Require(legacy.Solve(request, &result), "nmpc6 reference solve was rejected");

  NativeMpcPlan native;
  native.segment_dt_s = request.dt[0];
  for (int step = 0; step < 3; ++step)
    for (int axis = 0; axis < 7; ++axis)
      native.velocity_increments[step][axis] = result.controls[axis * 3 + step];
  native.BuildTrajectory(LoggedState());
  native.success = true;

  Require(native.plan_ms == result.plan_ms, "native and nmpc6 plan lengths differ");
  Require(native.plan_pos.size() == result.plan_pos.size() &&
          native.plan_vel.size() == result.plan_vel.size(),
          "native and nmpc6 dense trajectory shapes differ");
  double max_position_error = 0, max_velocity_error = 0;
  for (size_t i = 0; i < native.plan_pos.size(); ++i) {
    max_position_error = std::max(max_position_error,
                                  std::abs(native.plan_pos[i] - result.plan_pos[i]));
    max_velocity_error = std::max(max_velocity_error,
                                  std::abs(native.plan_vel[i] - result.plan_vel[i]));
  }
  Require(max_position_error <= 1e-10 && max_velocity_error <= 1e-10,
          "native dense trajectory differs from nmpc6");
  std::cout << "native_nmpc6_alignment_passed=true plan_ms=" << native.plan_ms
            << " max_position_error=" << max_position_error
            << " max_velocity_error=" << max_velocity_error << '\n';
}

}  // namespace

int main() try {
  CheckSyntheticTrajectory();
  CheckLegacyDifferential();
  return 0;
} catch (const std::exception& error) {
  std::cerr << "native_nmpc6_alignment_failed: " << error.what() << '\n';
  return 1;
}
