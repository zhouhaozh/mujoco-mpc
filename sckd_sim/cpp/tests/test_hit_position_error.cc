#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "sim/hit_position_error.h"

using namespace sckd;

namespace {
HitPositionSample Sample(double trajectory_time, double simulation_time,
                         std::array<double, 3> ball, std::array<double, 3> racket) {
  return {trajectory_time, simulation_time, ball, racket};
}
}  // namespace

TEST_CASE("fractional hit crossing interpolates ball and racket center") {
  HitPositionErrorTracker tracker;
  tracker.Observe(Sample(1.0, 2.0, {0, 2, 4}, {1, 1, 1}));
  tracker.UpdatePendingHit(1.25);
  CHECK(tracker.Observe(Sample(1.5, 2.5, {2, 4, 6}, {3, 1, -1})));
  const auto& result = *tracker.captured();
  CHECK(result.hit_time_s == doctest::Approx(1.25));
  CHECK(result.simulation_time_s == doctest::Approx(2.25));
  CHECK(result.ball_position_m == std::array<double, 3>{1, 3, 5});
  CHECK(result.racket_center_m == std::array<double, 3>{2, 1, 0});
  CHECK(result.delta_m == std::array<double, 3>{-1, 2, 5});
  CHECK(result.distance_m == doctest::Approx(std::sqrt(30.0)));
  CHECK(result.sampling_interval_s == doctest::Approx(.5));
}

TEST_CASE("newer valid target replaces pending time before crossing") {
  HitPositionErrorTracker tracker;
  tracker.Observe(Sample(0, 1, {0, 0, 0}, {0, 0, 0}));
  tracker.UpdatePendingHit(.6);
  tracker.Observe(Sample(.5, 1.5, {.5, 0, 0}, {0, 0, 0}));
  tracker.UpdatePendingHit(.9);
  CHECK(tracker.Observe(Sample(1, 2, {1, 0, 0}, {0, 0, 0})));
  CHECK(tracker.captured()->hit_time_s == doctest::Approx(.9));
  CHECK(tracker.captured()->ball_position_m[0] == doctest::Approx(.9));
}

TEST_CASE("invalid target retains last valid pending time") {
  HitPositionErrorTracker tracker;
  tracker.Observe(Sample(0, 1, {0, 0, 0}, {0, 0, 0}));
  tracker.UpdatePendingHit(.75);
  tracker.UpdatePendingHit(std::nullopt);
  CHECK(tracker.Observe(Sample(1, 2, {1, 0, 0}, {0, 0, 0})));
  CHECK(tracker.captured()->hit_time_s == doctest::Approx(.75));
}

TEST_CASE("first capture remains frozen") {
  HitPositionErrorTracker tracker;
  tracker.Observe(Sample(0, 0, {0, 0, 0}, {0, 0, 0}));
  tracker.UpdatePendingHit(.5);
  REQUIRE(tracker.Observe(Sample(1, 1, {1, 0, 0}, {0, 0, 0})));
  const HitPositionError first = *tracker.captured();
  tracker.UpdatePendingHit(1.5);
  CHECK_FALSE(tracker.Observe(Sample(2, 2, {20, 0, 0}, {10, 0, 0})));
  CHECK(tracker.captured()->hit_time_s == first.hit_time_s);
  CHECK(tracker.captured()->ball_position_m == first.ball_position_m);
}

TEST_CASE("no valid target never captures") {
  HitPositionErrorTracker tracker;
  tracker.Observe(Sample(0, 0, {0, 0, 0}, {0, 0, 0}));
  tracker.UpdatePendingHit(std::nullopt);
  CHECK_FALSE(tracker.Observe(Sample(1, 1, {1, 0, 0}, {0, 0, 0})));
  CHECK_FALSE(tracker.captured().has_value());
}

TEST_CASE("hit exactly on sampling boundary is captured") {
  HitPositionErrorTracker tracker;
  tracker.Observe(Sample(0, 2, {0, 0, 0}, {1, 0, 0}));
  tracker.UpdatePendingHit(1.0);
  CHECK(tracker.Observe(Sample(1, 3, {2, 0, 0}, {1, 0, 0})));
  CHECK(tracker.captured()->simulation_time_s == doctest::Approx(3));
  CHECK(tracker.captured()->delta_m[0] == doctest::Approx(1));
}
