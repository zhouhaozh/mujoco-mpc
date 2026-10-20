// 舵限率实验隔离测试:只观察控制器命令,不以假硬件模拟电机动力学。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "control/chassis_servo.h"

using namespace sckd;

namespace {

class FakeRobot final : public RobotInterface {
 public:
  bool enable(int, bool) override { return true; }
  void set_arm_hybrid(const ArmHybridCmd&) override { ++arm_writes; }
  void set_axis_cmd(int axis, const MotorCmd& cmd) override {
    REQUIRE(axis >= 0);
    REQUIRE(axis < kNumAxes);
    commands[axis] = cmd;
    ++writes[axis];
  }
  BaseState base_state() const override { return {}; }
  JointReport joint(int axis) const override { return measured[axis]; }
  JointReport joint_measured(int axis) override { return measured[axis]; }
  bool get_imu(ImuSample*) override { return false; }
  bool arm_gravity(const double*, double*) override { return false; }

  std::array<MotorCmd, kNumAxes> commands{};
  std::array<JointReport, kNumAxes> measured{};
  std::array<int, kNumAxes> writes{};
  int arm_writes = 0;
};

ChassisConfig MakeCfg() {
  ChassisConfig cfg;
  cfg.corner_x = {0.475, 0.475, -0.475, -0.475};
  cfg.corner_y = {0.475, -0.475, 0.475, -0.475};
  cfg.wheel_sign = {1, -1, 1, -1, 1, -1, 1, -1};
  return cfg;
}

void ReachForwardSpeed(ChassisServo& servo) {
  servo.set_twist(Twist{1.0, 0.0, 0.0});
  for (int ms = 0; ms <= 124; ++ms) servo.update(ms * 0.001);
  REQUIRE(servo.ramped().vx == doctest::Approx(1.0));
  REQUIRE(servo.ramped().vy == doctest::Approx(0.0));
}

}  // namespace

TEST_CASE("default steering limits preserve explicit enabled behavior") {
  const auto cfg = MakeCfg();
  FakeRobot default_hw, enabled_hw;
  ChassisServo original(cfg, &default_hw), enabled(cfg, &enabled_hw);
  enabled.set_steering_rate_limits(true, true);
  for (int ms = 0; ms <= 160; ++ms) {
    const Twist cmd = ms < 125 ? Twist{1, 0, 0} : Twist{0, 1, 1};
    original.set_twist(cmd);
    enabled.set_twist(cmd);
    original.update(ms * 0.001);
    enabled.update(ms * 0.001);
    CHECK(original.ramped().vx == enabled.ramped().vx);
    CHECK(original.ramped().vy == enabled.ramped().vy);
    CHECK(original.ramped().wz == enabled.ramped().wz);
    for (int axis = 0; axis < kNumAxes; ++axis) {
      CHECK(default_hw.commands[axis].q == enabled_hw.commands[axis].q);
      CHECK(default_hw.commands[axis].dq == enabled_hw.commands[axis].dq);
      CHECK(default_hw.commands[axis].tau_ff == enabled_hw.commands[axis].tau_ff);
    }
  }
}

TEST_CASE("moving ninety degree turn separates direction and feedforward limits") {
  const auto cfg = MakeCfg();
  const double half_pi = std::acos(-1.0) / 2.0;
  for (bool direction_limited : {false, true}) {
    for (bool feedforward_limited : {false, true}) {
      CAPTURE(direction_limited);
      CAPTURE(feedforward_limited);
      FakeRobot hw;
      ChassisServo servo(cfg, &hw);
      servo.set_steering_rate_limits(direction_limited, feedforward_limited);
      ReachForwardSpeed(servo);
      servo.set_twist(Twist{0, 1, 0});
      for (int ms = 125; ms <= 128; ++ms) servo.update(ms * 0.001);

      const double angle = direction_limited ? 3.0 * 0.004 : half_pi;
      const double raw_rate = angle / 0.004;
      CHECK(std::atan2(servo.ramped().vy, servo.ramped().vx) ==
            doctest::Approx(angle).epsilon(1e-10));
      CHECK(std::hypot(servo.ramped().vx, servo.ramped().vy) ==
            doctest::Approx(1.0));
      for (int corner = 0; corner < kNumCorners; ++corner) {
        const auto& cmd = hw.commands[AxisSteer(corner)];
        CHECK(cmd.q == doctest::Approx(angle).epsilon(1e-10));
        CHECK(cmd.dq == doctest::Approx(feedforward_limited
                          ? std::min(raw_rate, 8.0) : raw_rate).epsilon(1e-10));
        CHECK(cmd.kp == cfg.servo.steer_kp);
        CHECK(cmd.kd == cfg.servo.steer_kd);
      }
    }
  }
}

TEST_CASE("feedforward clip is independent of direction limit at startup") {
  const auto cfg = MakeCfg();
  const double quarter_pi = std::acos(-1.0) / 4.0;
  for (bool direction_limited : {false, true}) {
    for (bool feedforward_limited : {false, true}) {
      CAPTURE(direction_limited);
      CAPTURE(feedforward_limited);
      FakeRobot hw;
      ChassisServo servo(cfg, &hw);
      servo.set_steering_rate_limits(direction_limited, feedforward_limited);
      servo.set_twist(Twist{1, 1, 0});
      servo.update(0.0);
      for (int corner = 0; corner < kNumCorners; ++corner) {
        CHECK(hw.commands[AxisSteer(corner)].q == doctest::Approx(quarter_pi));
        CHECK(hw.commands[AxisSteer(corner)].dq == doctest::Approx(
            feedforward_limited ? 8.0 : quarter_pi / 0.004));
      }
    }
  }
}

TEST_CASE("steering switches preserve yaw ramp wheel acceleration and arm isolation") {
  const auto cfg = MakeCfg();
  for (bool direction_limited : {false, true}) {
    for (bool feedforward_limited : {false, true}) {
      CAPTURE(direction_limited);
      CAPTURE(feedforward_limited);
      FakeRobot hw;
      ChassisServo servo(cfg, &hw);
      servo.set_steering_rate_limits(direction_limited, feedforward_limited);
      servo.set_twist(Twist{1, 0, 1});
      std::array<double, kDriveWheels> previous{};
      bool wheel_ramp_active = false;
      for (int ms = 0; ms <= 80; ++ms) {
        servo.update(ms * 0.001);
        const int allocation_ticks = ms / 4 + 1;
        CHECK(servo.ramped().wz == doctest::Approx(std::min(
            1.0, allocation_ticks * cfg.ramp.ang_accel_limit_rad_s2 * 0.004)));
        CHECK(servo.ramped().vx == doctest::Approx(std::min(
            1.0, allocation_ticks * cfg.ramp.accel_limit_mps2 * 0.004)));
        CHECK(servo.ramped().vy == doctest::Approx(0.0));
        for (int corner = 0; corner < kNumCorners; ++corner) {
          for (int side = 0; side < kDrivesPerCorner; ++side) {
            const int axis = side == 0 ? AxisWheelL(corner) : AxisWheelR(corner);
            const int idx = 2 * corner + side;
            const double speed = cfg.wheel_sign[idx] * cfg.wheel_radius *
                                 servo.wheel_velocity_reference(axis);
            const double dv = std::fabs(speed - previous[idx]);
            CHECK(dv <= cfg.servo.wheel_target_accel_mps2 * 0.001 + 1e-12);
            if (dv > 0.99 * cfg.servo.wheel_target_accel_mps2 * 0.001)
              wheel_ramp_active = true;
            CHECK(std::fabs(hw.commands[axis].tau_ff) <= 36.0);
            previous[idx] = speed;
          }
        }
      }
      CHECK(wheel_ramp_active);
      CHECK(hw.arm_writes == 0);
      for (int axis = kNumChassis; axis < kNumAxes; ++axis)
        CHECK(hw.writes[axis] == 0);
    }
  }
}
