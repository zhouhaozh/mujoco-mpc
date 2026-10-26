// 仅在 Ubuntu 的 sckd_sim 工作目录运行；读取现有模型，不连接 SDK/真机。
// 对照原执行顺序，验证新语义适配器没有改变电机命令、动力学或测量噪声流。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <array>
#include <cmath>
#include <cstdint>

#include "control/chassis_servo.h"
#include "hal/sim_backend.h"
#include "runtime/mujoco_robot_backend.h"

using namespace sckd;
namespace io = sckd::io;

namespace {

struct Fixture {
  SimBackend simulation{"models/generated/x4_tennis_demo.mjcf",
                        "models/config/motors.json",
                        "models/config/x4_tennis.json",
                        "models/config/chassis_m1.json"};
  ChassisServo servo{simulation.chassis_config(), &simulation};
  io::MujocoRobotBackend backend{simulation, servo};

  Fixture() {
    REQUIRE_MESSAGE(simulation.ok(), "run from sckd_sim with its Ubuntu model resources");
    REQUIRE(simulation.reset(0));
    simulation.enable(-1, true);  // 返回的是先前使能状态，不是成功标记。
    simulation.hold_arm(60.0, 1.5);
    REQUIRE(simulation.model()->opt.timestep == doctest::Approx(.001));
  }
};

void SameValue(double actual, double expected) {
  CHECK(std::isfinite(actual));
  CHECK(std::isfinite(expected));
  CHECK(actual == doctest::Approx(expected).epsilon(1e-12).scale(1.0));
}

void SameMotorCommand(const MotorCmd& actual, const MotorCmd& expected) {
  // 命令映射是直接赋值，必须逐字段精确一致，不能靠动力学容差掩盖丢字段。
  CHECK(actual.q == expected.q);
  CHECK(actual.dq == expected.dq);
  CHECK(actual.kp == expected.kp);
  CHECK(actual.kd == expected.kd);
  CHECK(actual.tau_ff == expected.tau_ff);
}

void SameState(const SimBackend& actual, const SimBackend& expected) {
  CHECK(actual.tick() == expected.tick());
  SameValue(actual.data()->time, expected.data()->time);
  for (int axis = 0; axis < kNumAxes; ++axis) {
    CAPTURE(axis);
    const auto a = actual.joint(axis), e = expected.joint(axis);
    SameValue(a.q, e.q);
    SameValue(a.dq, e.dq);
    SameValue(a.tau_cmd, e.tau_cmd);
    CHECK(a.enabled == e.enabled);
    SameMotorCommand(actual.motor_issued(axis), expected.motor_issued(axis));
    SameMotorCommand(actual.motor_effective(axis), expected.motor_effective(axis));
  }
  const auto a = actual.base_state(), e = expected.base_state();
  SameValue(a.x, e.x); SameValue(a.y, e.y); SameValue(a.z, e.z);
  SameValue(a.yaw, e.yaw); SameValue(a.roll, e.roll); SameValue(a.pitch, e.pitch);
  SameValue(a.vx, e.vx); SameValue(a.vy, e.vy); SameValue(a.wz, e.wz);
}

std::array<double, kNumArm> ArmStart(const SimBackend& simulation) {
  std::array<double, kNumArm> q{};
  for (int j = 0; j < kNumArm; ++j) q[j] = simulation.joint(AxisArm(j)).q;
  return q;
}

io::Command CommandAt(std::uint64_t tick, const std::array<double, kNumArm>& q0) {
  io::Command command;
  command.sequence = tick;
  command.plan_id = 7;
  const double t = tick * .001;
  command.issued = {t, io::Clock::kSimulation};
  command.expires = {t + .004, io::Clock::kSimulation};
  command.body_twist = tick < 30 ? std::array<double, 3>{.35, -.1, .15} :
                       tick < 60 ? std::array<double, 3>{-.15, .3, -.1} :
                                   std::array<double, 3>{.05, -.25, .2};
  for (int j = 0; j < kNumArm; ++j) {
    const double phase = 50.0 * t + .3 * j;
    command.q[j] = q0[j] + .01 * std::sin(phase);
    command.dq[j] = .5 * std::cos(phase);
    command.kp[j] = 45.0 + 2.0 * j;
    command.kd[j] = 1.4 + .1 * j;
    command.tau_ff[j] = (.4 + .1 * j) * (1.0 + .2 * std::sin(phase));
  }
  return command;
}

void LegacySubmit(Fixture& fixture, const io::Command& command) {
  fixture.servo.set_twist({command.body_twist[0], command.body_twist[1], command.body_twist[2]});
  fixture.servo.update(command.issued.seconds);
  ArmHybridCmd arm;
  for (int j = 0; j < kNumArm; ++j) {
    arm.q[j] = command.q[j]; arm.dq[j] = command.dq[j];
    arm.kp[j] = command.kp[j]; arm.kd[j] = command.kd[j];
    arm.tau_ff[j] = command.tau_ff[j];
  }
  fixture.simulation.set_arm_hybrid(arm);
}

}  // namespace

TEST_CASE("MuJoCo adapter matches the legacy execution path with changing targets and feedforward") {
  Fixture legacy, adapted;
  // 非默认限率配置由调用者持有；适配器不能重置伺服状态或开关。
  legacy.servo.set_steering_rate_limits(false, true);
  adapted.servo.set_steering_rate_limits(false, true);
  const auto q0 = ArmStart(legacy.simulation);
  SameState(adapted.simulation, legacy.simulation);
  for (std::uint64_t tick = 1; tick <= 80; ++tick) {
    CAPTURE(tick);
    const auto command = CommandAt(tick, q0);
    LegacySubmit(legacy, command);
    REQUIRE(adapted.backend.Write(command, command.issued).submitted);
    adapted.backend.Tick(command.issued);
    // 先看延迟队列入口，再看已经推进的状态，覆盖非零力矩前馈与延迟语义。
    for (int axis = 0; axis < kNumAxes; ++axis)
      SameMotorCommand(adapted.simulation.motor_issued(axis), legacy.simulation.motor_issued(axis));
    REQUIRE(legacy.simulation.run(1));
    REQUIRE(adapted.simulation.run(1));
    SameState(adapted.simulation, legacy.simulation);
  }
}

TEST_CASE("rejected MuJoCo commands do not alter issued motors or the latched chassis target") {
  Fixture unchanged, rejected;
  const auto q0 = ArmStart(unchanged.simulation);
  const auto first = CommandAt(1, q0);
  for (auto* fixture : {&unchanged, &rejected}) {
    REQUIRE(fixture->backend.Write(first, first.issued).submitted);
    fixture->backend.Tick(first.issued);
    REQUIRE(fixture->simulation.run(1));
  }
  std::array<MotorCmd, kNumAxes> before{};
  for (int axis = 0; axis < kNumAxes; ++axis) before[axis] = rejected.simulation.motor_issued(axis);
  auto altered = CommandAt(2, q0);
  altered.body_twist = {-.8, .9, -.7};
  altered.q.fill(.3); altered.tau_ff.fill(8.0);
  const auto verify_unchanged = [&] {
    for (int axis = 0; axis < kNumAxes; ++axis)
      SameMotorCommand(rejected.simulation.motor_issued(axis), before[axis]);
    SameState(rejected.simulation, unchanged.simulation);
  };
  const auto expired = rejected.backend.Write(altered, altered.expires);
  CHECK_FALSE(expired.submitted); CHECK(expired.reason == "expired");
  verify_unchanged();
  altered.frame_id = "sdk_model";
  const auto wrong_frame = rejected.backend.Write(altered, altered.issued);
  CHECK_FALSE(wrong_frame.submitted); CHECK(wrong_frame.reason == "frame_mismatch");
  verify_unchanged();
  altered.frame_id = "mpc_model"; altered.sequence = first.sequence;
  const auto old_sequence = rejected.backend.Write(altered, altered.issued);
  CHECK_FALSE(old_sequence.submitted); CHECK(old_sequence.reason == "out_of_order_sequence");
  verify_unchanged();
  // set_twist 本身不会立即写电机；跨过4ms分配周期才能揭露误改底盘目标。
  for (int tick = 2; tick <= 6; ++tick) {
    for (auto* fixture : {&unchanged, &rejected}) {
      fixture->backend.Tick({tick * .001, io::Clock::kSimulation});
      REQUIRE(fixture->simulation.run(1));
    }
    SameState(rejected.simulation, unchanged.simulation);
  }
  auto next = CommandAt(7, q0);
  next.sequence = 2;  // 被拒绝的 sequence=2 不得占用序号。
  REQUIRE(rejected.backend.Write(next, next.issued).submitted);
}

TEST_CASE("MuJoCo Read leaves simulation state and subsequent noisy measurements unchanged") {
  Fixture unread, observed;
  REQUIRE_MESSAGE(observed.simulation.chassis_config().odom.wheel_speed_noise_rad_s > 0,
                  "this test requires the default nonzero wheel measurement noise");
  REQUIRE(unread.simulation.run(7));
  REQUIRE(observed.simulation.run(7));
  for (int repeat = 0; repeat < 12; ++repeat) {
    const auto feedback = observed.backend.Read({.008, io::Clock::kSimulation});
    CHECK(feedback.ready);
    CHECK(feedback.coherent_snapshot);
    CHECK(feedback.base_source == io::Source::kSimulationTruth);
    CHECK(feedback.base_stamp_known);
    SameValue(feedback.base_stamp.seconds, .007);
  }
  SameState(observed.simulation, unread.simulation);
  bool saw_wheel_noise = false;
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (int axis = 0; axis < kNumAxes; ++axis) {
      CAPTURE(repeat); CAPTURE(axis);
      const auto actual = observed.simulation.joint_measured(axis);
      const auto expected = unread.simulation.joint_measured(axis);
      CHECK(actual.q == expected.q);
      CHECK(actual.dq == expected.dq);
      if (axis < kNumChassis && axis % 3 != 0)
        saw_wheel_noise = saw_wheel_noise || expected.dq != unread.simulation.joint(axis).dq;
    }
  }
  CHECK(saw_wheel_noise);
  // 继续跨过下一个200Hz IMU采样点，检查共享RNG的后续消费者。
  REQUIRE(unread.simulation.run(4));
  REQUIRE(observed.simulation.run(4));
  SameState(observed.simulation, unread.simulation);
  ImuSample actual_imu, expected_imu;
  REQUIRE(observed.simulation.get_imu(&actual_imu));
  REQUIRE(unread.simulation.get_imu(&expected_imu));
  CHECK(actual_imu.t_us == expected_imu.t_us);
  for (int k = 0; k < 3; ++k) {
    CHECK(actual_imu.acc[k] == expected_imu.acc[k]);
    CHECK(actual_imu.gyr[k] == expected_imu.gyr[k]);
  }
}
