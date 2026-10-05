// 电机模型单元测试(doctest):限幅链、T-N 包络插值、方向判定、速度保护、
// 延迟管道(9 拍)、断流锁存、协议范围截断、使能语义。配置为内存构造(与
// models/config/motors.json 同值,来源:docs/motors 手册,calibrated=false)。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <cmath>

#include "robot/motor_model.h"

using namespace sckd;

namespace {

MotorSystemConfig MakeCfg() {
  MotorTypeConfig rs04;
  rs04.name = "RS04";
  rs04.peak_torque_nm = 120; rs04.continuous_torque_nm = 40; rs04.braking_torque_nm = 120;
  rs04.no_load_speed_rad_s = 20.944; rs04.hybrid_speed_limit_rad_s = 15;
  rs04.hybrid_position_limit_rad = 12.57;
  rs04.kp_min = 0; rs04.kp_max = 5000; rs04.kd_min = 0; rs04.kd_max = 100;
  rs04.motoring = {{0, 120}, {9.948, 120}, {11.519, 110}, {13.614, 95}, {15.708, 78},
                   {16.755, 60}, {17.802, 52}, {18.850, 44}, {19.583, 32}, {19.897, 10},
                   {20.944, 0}};

  MotorTypeConfig rs06;
  rs06.name = "RS06";
  rs06.peak_torque_nm = 36; rs06.continuous_torque_nm = 11; rs06.braking_torque_nm = 36;
  rs06.no_load_speed_rad_s = 50.265; rs06.hybrid_speed_limit_rad_s = 50;
  rs06.hybrid_position_limit_rad = 12.57;
  rs06.kp_min = 0; rs06.kp_max = 5000; rs06.kd_min = 0; rs06.kd_max = 100;
  rs06.motoring = {{0, 36}, {29.322, 36}, {31.416, 33}, {33.510, 30}, {36.652, 26},
                   {39.270, 22}, {41.888, 18}, {43.982, 12}, {44.820, 10}, {45.029, 5},
                   {50.265, 0}};

  MotorSystemConfig cfg;
  cfg.motors = {rs04, rs06};
  cfg.command_delay_ms = 9.0;  // -> delay_slots = 9
  cfg.control_dt_s = 0.001;
  cfg.profile_peak = true;
  for (int i = 0; i < kNumAxes; ++i) {
    JointConfig jc;
    jc.joint = kAxisJointName[i];
    jc.motor = "RS06";
    jc.mechanism_limit_nm = 36.0;
    if (i == AxisArm(0) || i == AxisArm(1)) {  // J1/J2 = RS04
      jc.motor = "RS04";
      jc.mechanism_limit_nm = 120.0;
    }
    cfg.joints.push_back(jc);
  }
  return cfg;
}

}  // namespace

TEST_CASE("disable -> zero output regardless of command") {
  auto cfg = MakeCfg();
  MotorModel mm(cfg);
  MotorCmd c; c.kp = 5000; c.q = 0.1;
  mm.set_cmd(0, c);
  for (int i = 0; i < 20; ++i) mm.tick();
  CHECK(mm.compute(0, 0.0, 0.0) == 0.0);
  mm.enable(0, true);
  CHECK(mm.compute(0, 0.0, 0.0) == doctest::Approx(36.0));  // 500 -> cap 36
}

TEST_CASE("clip chain: mechanism/profile cap at zero speed") {
  auto cfg = MakeCfg();
  MotorModel mm(cfg);
  mm.enable(-1, true);
  MotorCmd c; c.kp = 5000; c.q = 0.1;  // tau_req = 500
  mm.set_cmd(0, c);
  for (int i = 0; i < 20; ++i) mm.tick();
  CHECK(mm.compute(0, 0.0, 0.0) == doctest::Approx(36.0));   // RS06 + 机构 36
  mm.set_cmd(AxisArm(1), c);                                 // J2: RS04, 机构 120
  for (int i = 0; i < 20; ++i) mm.tick();
  CHECK(mm.compute(AxisArm(1), 0.0, 0.0) == doctest::Approx(120.0));
}

TEST_CASE("T-N envelope interpolation and saturation") {
  auto cfg = MakeCfg();
  MotorModel mm(cfg);
  mm.enable(-1, true);
  MotorCmd c; c.kp = 5000; c.q = 1.0;  // tau_req 始终远超包络
  mm.set_cmd(0, c);
  for (int i = 0; i < 20; ++i) mm.tick();
  SUBCASE("zero speed -> peak")      { CHECK(mm.compute(0, 0, 0) == doctest::Approx(36.0)); }
  SUBCASE("knot point 33.510")       { CHECK(mm.compute(0, 0, 33.510) == doctest::Approx(30.0)); }
  SUBCASE("between knots 34.0") {
    const double want = 30.0 + (34.0 - 33.510) / (36.652 - 33.510) * (26.0 - 30.0);
    CHECK(mm.compute(0, 0, 34.0) == doctest::Approx(want).epsilon(1e-9));
  }
  SUBCASE("beyond no-load -> 0")     { CHECK(mm.compute(0, 0, 50.265) == doctest::Approx(0.0)); }
  SUBCASE("speed protection >50 while motoring -> 0") {
    CHECK(mm.compute(0, 0, 51.0) == doctest::Approx(0.0));
  }
}

TEST_CASE("direction: braking uses independent cap") {
  auto cfg = MakeCfg();
  MotorModel mm(cfg);
  mm.enable(-1, true);
  MotorCmd c; c.kp = 5000; c.q = -1.0;  // tau_req = -500,与 dq=+45 反向 => 制动
  mm.set_cmd(0, c);
  for (int i = 0; i < 20; ++i) mm.tick();
  CHECK(mm.compute(0, 0.0, 45.0) == doctest::Approx(-36.0));  // 制动上限 36,不吃 T-N 降额
}

TEST_CASE("command delay: 9 ticks, then latched") {
  auto cfg = MakeCfg();
  MotorModel mm(cfg);
  mm.enable(-1, true);
  MotorCmd c; c.kp = 5000; c.q = 0.1;  // 生效后输出 36
  mm.set_cmd(0, c);
  CHECK(mm.compute(0, 0, 0) == doctest::Approx(0.0));  // 管道内仍是零命令
  for (int i = 0; i < 8; ++i) mm.tick();
  CHECK(mm.compute(0, 0, 0) == doctest::Approx(0.0));  // 第 8 拍仍未生效
  mm.tick();                                            // 第 9 拍生效
  CHECK(mm.compute(0, 0, 0) == doctest::Approx(36.0));
  for (int i = 0; i < 100; ++i) mm.tick();              // 断流:目标锁存
  CHECK(mm.compute(0, 0, 0) == doctest::Approx(36.0));
}

TEST_CASE("protocol range clamps") {
  auto cfg = MakeCfg();
  MotorModel mm(cfg);
  mm.enable(-1, true);
  MotorCmd c; c.kp = 9999; c.q = 0.001;  // kp 截到 5000 -> tau = 5
  mm.set_cmd(0, c);
  for (int i = 0; i < 20; ++i) mm.tick();
  CHECK(mm.compute(0, 0, 0) == doctest::Approx(5.0));
  MotorCmd big; big.kp = 1.0; big.q = 100.0;  // 位置截到 12.57 -> tau = 12.57
  mm.set_cmd(0, big);
  for (int i = 0; i < 20; ++i) mm.tick();
  CHECK(mm.compute(0, 0, 0) == doctest::Approx(12.57));
}

TEST_CASE("continuous profile derates cap") {
  auto cfg = MakeCfg();
  cfg.profile_peak = false;  // 连续档:RS06 = 11
  MotorModel mm(cfg);
  mm.enable(-1, true);
  MotorCmd c; c.kp = 5000; c.q = 0.1;
  mm.set_cmd(0, c);
  for (int i = 0; i < 20; ++i) mm.tick();
  CHECK(mm.compute(0, 0, 0) == doctest::Approx(11.0));
}
