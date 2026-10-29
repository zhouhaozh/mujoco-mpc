#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

namespace sckd {

struct HitPositionSample {
  double trajectory_time_s = 0;
  double simulation_time_s = 0;
  std::array<double, 3> ball_position_m{};
  std::array<double, 3> racket_center_m{};
};

struct HitPositionError {
  double hit_time_s = 0;
  double simulation_time_s = 0;
  std::array<double, 3> ball_position_m{};
  std::array<double, 3> racket_center_m{};
  std::array<double, 3> delta_m{};
  double distance_m = 0;
  double sampling_interval_s = 0;
};

class HitPositionErrorTracker {
 public:
  // A missing target preserves the latest valid future hit time.
  void UpdatePendingHit(std::optional<double> hit_time_s) {
    if (!captured_ && hit_time_s && std::isfinite(*hit_time_s)) pending_hit_time_s_ = *hit_time_s;
  }

  bool Observe(const HitPositionSample& current) {
    bool captured_now = false;
    if (!captured_ && previous_ && pending_hit_time_s_) {
      constexpr double kBoundaryTolerance = 1e-12;
      const double begin = previous_->trajectory_time_s;
      const double end = current.trajectory_time_s;
      if (end >= begin && *pending_hit_time_s_ >= begin - kBoundaryTolerance &&
          *pending_hit_time_s_ <= end + kBoundaryTolerance) {
        const double interval = end - begin;
        const double alpha = interval > kBoundaryTolerance
                                 ? std::clamp((*pending_hit_time_s_ - begin) / interval, 0.0, 1.0)
                                 : 0.0;
        HitPositionError result;
        result.hit_time_s = *pending_hit_time_s_;
        result.simulation_time_s = Interpolate(previous_->simulation_time_s,
                                                current.simulation_time_s, alpha);
        result.sampling_interval_s = current.simulation_time_s - previous_->simulation_time_s;
        for (int axis = 0; axis < 3; ++axis) {
          result.ball_position_m[axis] = Interpolate(previous_->ball_position_m[axis],
                                                     current.ball_position_m[axis], alpha);
          result.racket_center_m[axis] = Interpolate(previous_->racket_center_m[axis],
                                                     current.racket_center_m[axis], alpha);
          result.delta_m[axis] = result.ball_position_m[axis] - result.racket_center_m[axis];
          result.distance_m += result.delta_m[axis] * result.delta_m[axis];
        }
        result.distance_m = std::sqrt(result.distance_m);
        captured_ = result;
        captured_now = true;
      }
    }
    previous_ = current;
    return captured_now;
  }

  const std::optional<HitPositionError>& captured() const { return captured_; }

 private:
  static double Interpolate(double begin, double end, double alpha) {
    return begin + alpha * (end - begin);
  }

  std::optional<double> pending_hit_time_s_;
  std::optional<HitPositionSample> previous_;
  std::optional<HitPositionError> captured_;
};

}  // namespace sckd
