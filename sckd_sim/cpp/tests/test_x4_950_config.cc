// 静态接线检查：没有控制器实例、命令发送、仿真步进或求解器。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include "hal/sim_backend.h"
#include "runtime/runtime_config.h"
#include "../../third_party/nmpc6/x4_950_fk.h"

using namespace sckd;

TEST_CASE("X4 racket effective length and analytical FK match the generated model") {
  SimBackend be("models/generated/x4_950_scene.mjcf", "models/config/motors_x4_950.json",
                "models/config/x4_950.json", "models/config/chassis_x4_950.json");
  REQUIRE(be.ok());
  REQUIRE(be.reset());
  const auto* m = be.model();
  auto* d = be.data_mut();
  const int link5 = mj_name2id(m, mjOBJ_BODY, "Link5");
  const int link6 = mj_name2id(m, mjOBJ_BODY, "Link6");
  CHECK(m->body_pos[3*link5+1]+m->body_pos[3*link6+1] == doctest::Approx(.865));
  CHECK(m->body_pos[3*link6+1] == doctest::Approx(.930));
  const int site = mj_name2id(m, mjOBJ_SITE, "racket_center");
  const int face = mj_name2id(m, mjOBJ_GEOM, "racket_face");
  const double poses[][5] = {{0,0,0,0,0}, {0,-.5,-.5,0,0},
                             {.3,-1,-.2,.4,.7}, {0,0,-.5,0,0}};
  const double base[2] = {0,0};
  for (const auto& q : poses) {
    for (int j=0;j<5;++j) {
      const int joint = mj_name2id(m, mjOBJ_JOINT, ("Joint"+std::to_string(j+1)).c_str());
      d->qpos[m->jnt_qposadr[joint]] = q[j];
    }
    mj_forward(m,d);
    double expected[3];
    sckd_nmpc6::X4950ArmPos(base,q,expected);
    for (int axis=0;axis<3;++axis) {
      CHECK(d->site_xpos[3*site+axis] == doctest::Approx(expected[axis]).epsilon(1e-9));
      CHECK(d->geom_xpos[3*face+axis] == doctest::Approx(expected[axis]).epsilon(1e-9));
    }
  }
}

TEST_CASE("X4 runtime loads independent configs with truth feedback and dynamic caps") {
  const auto cfg = runtime::ResolvedConfig::Resolve(
      {"config_check", "--config", "models/config/runtime_x4_950.json"},
      [](const std::string&) -> std::optional<std::string> { return std::nullopt; });
  const auto& v = cfg.values();
  CHECK_FALSE(cfg.modes().inverse_ff);
  CHECK_FALSE(cfg.modes().identified_gains);
  CHECK(v.nmpc6_nlp_path == "third_party/nmpc6/x4_950_nlp.so");
  SimBackend be(v.scene_path, v.motors_path, v.robot_variant_path, v.chassis_path);
  REQUIRE(be.ok());
  REQUIRE(be.reset()); // 仅 mj_forward，时间保持零。
  CHECK(be.model()->opt.timestep == doctest::Approx(.001));
  CHECK(be.chassis_config().servo.period_ms == 1);
  CHECK(be.motor_config().command_delay_ms == 0);
  CHECK(be.motor_config().truth_feedback);
  REQUIRE(be.motor_config().arm_servo.enabled);
  CHECK(be.motor_config().arm_servo.kd[4] <= 5);
  CHECK(be.chassis_config().wheel_radius == doctest::Approx(.0825));
  CHECK(be.chassis_config().dual_offset == doctest::Approx(.024));

  const int axis = AxisArm(4);
  const int jid = mj_name2id(be.model(), mjOBJ_JOINT, "Joint5");
  be.data_mut()->qpos[be.model()->jnt_qposadr[jid]] = .123456789;
  CHECK(be.joint_measured(axis).q == .123456789);
  CHECK(be.quantize(.123456789) == .123456789);
  CHECK_FALSE(be.joint(axis).enabled);
  CHECK(be.motor_torque_cap(axis, 2, 0) == doctest::Approx(1.6));
  CHECK(be.motor_torque_cap(AxisWheelL(0), 20, 0) == doctest::Approx(11));
  CHECK(be.motor_torque_cap(AxisWheelL(0), 20, 67.187) == doctest::Approx(5));
  CHECK(be.motor_torque_cap(AxisWheelL(0), -20, 67.187) == doctest::Approx(11));
  CHECK(be.motor_torque_cap(AxisWheelL(0), 20, 76) == 0);
  CHECK(be.data()->time == 0);
  CHECK(be.tick() == 0);

  // 错配必须在进入控制器前失败。
  CHECK_THROWS(LoadMotorConfig("models/config/motors.json", v.robot_variant_path));
  SimBackend wrong_chassis(v.scene_path, v.motors_path, v.robot_variant_path,
                           "models/config/chassis_m1.json");
  CHECK_FALSE(wrong_chassis.ok());
}

TEST_CASE("tennis motor and servo defaults remain unchanged") {
  const auto cfg = LoadMotorConfig("models/config/motors.json", "models/config/x4_tennis.json");
  CHECK_FALSE(cfg.arm_servo.enabled);
  CHECK_FALSE(cfg.truth_feedback);
  CHECK(cfg.command_delay_ms == 9);
  CHECK(cfg.encoder_bits == 14);
  CHECK(cfg.profile_peak);
  CHECK(LoadChassisConfig("models/config/chassis_m1.json").servo.period_ms == 4);
}

TEST_CASE("X4 directory bundles load sibling motors and reject another bundle") {
  const std::string root = "models/config/measured_dynamics/";
  for (const std::string name : {"current", "baseline", "fitted"}) {
    const auto bundle = root + name + "/";
    const auto cfg = LoadMotorConfig(bundle + "motors.json", bundle + "robot.json");
    CHECK(cfg.robot_variant == "x4_950");
    CHECK(cfg.arm_servo.enabled);
  }
  CHECK_THROWS(LoadMotorConfig(root + "baseline/motors.json", root + "fitted/robot.json"));
}
