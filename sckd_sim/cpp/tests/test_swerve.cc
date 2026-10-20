// 舵轮运动学单元测试:分配↔正解往返、纯平移/纯自旋、反转+迟滞、速度死区、
// 双轮差速一致性、WrapPi。几何取 tennis 实测(corner ±0.475, dual 0.055)。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <cmath>
#include <random>

#include "robot/swerve_kinematics.h"

using namespace sckd;

namespace {

SwerveKinematics MakeKin() {
  SwerveGeometry g;
  const double xy[4][2] = {{0.475, 0.475}, {0.475, -0.475}, {-0.475, 0.475}, {-0.475, -0.475}};
  for (int i = 0; i < 4; ++i) {
    g.corner_x[i] = xy[i][0];
    g.corner_y[i] = xy[i][1];
  }
  g.wheel_radius = 0.0828;
  g.dual_offset = 0.055;
  return SwerveKinematics(g, CoordinationParams{});
}

// 分配 -> 舵角到位 -> 二次分配(此时舵角误差=0,cos 投影不再降速) -> 双轮差速 -> 正解。
// 一次分配即正解会看到 cos(err) 瞬态降速(行为正确,不是稳态),见 cos-projection 测试。
Twist RoundTrip(const SwerveKinematics& k, const Twist& t) {
  double steer[4] = {0, 0, 0, 0};
  bool rev[4] = {false, false, false, false};
  CornerCmd cmd[4];
  k.Allocate(t, steer, rev, cmd);          // 1) 决定目标角与反转状态
  for (int i = 0; i < 4; ++i) steer[i] = cmd[i].steer_target;
  k.Allocate(t, steer, rev, cmd);          // 2) 舵到位:误差≈0,满速
  double mps[8] = {0};
  for (int i = 0; i < 4; ++i) {
    double dm[2];
    SwerveKinematics::DualWheel(cmd[i].drive_mps[0], t.wz, 0.0, 0.055, dm);
    mps[2 * i] = dm[0];
    mps[2 * i + 1] = dm[1];
  }
  return k.EstimateTwist(mps, steer);
}

}  // namespace

TEST_CASE("pure translation roundtrip") {
  auto k = MakeKin();
  const Twist t{1.2, -0.4, 0.0};
  const Twist back = RoundTrip(k, t);
  CHECK(back.vx == doctest::Approx(t.vx).scale(1.0).epsilon(1e-9));
  CHECK(back.vy == doctest::Approx(t.vy).scale(1.0).epsilon(1e-9));
  CHECK(back.wz == doctest::Approx(t.wz).scale(1.0).epsilon(1e-9));
}

TEST_CASE("pure spin roundtrip (module speeds from |corner|·wz)") {
  auto k = MakeKin();
  const Twist t{0.0, 0.0, 0.8};
  const Twist back = RoundTrip(k, t);
  CHECK(std::fabs(back.vx) < 1e-9);
  CHECK(std::fabs(back.vy) < 1e-9);
  CHECK(back.wz == doctest::Approx(t.wz).epsilon(1e-9));
}

TEST_CASE("general twist roundtrip over random samples") {
  auto k = MakeKin();
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  for (int s = 0; s < 50; ++s) {
    const Twist t{d(rng), d(rng), d(rng) * 1.5};
    const Twist back = RoundTrip(k, t);
    CHECK(back.vx == doctest::Approx(t.vx).epsilon(1e-9));
    CHECK(back.vy == doctest::Approx(t.vy).epsilon(1e-9));
    CHECK(back.wz == doctest::Approx(t.wz).epsilon(1e-9));
  }
}

TEST_CASE("dual wheel differential: rigid part exact, steer scrub bounded") {
  auto k = MakeKin();
  const Twist t{0.0, 0.0, 1.0};
  double steer[4] = {0, 0, 0, 0};
  bool rev[4] = {false, false, false, false};
  CornerCmd cmd[4];
  k.Allocate(t, steer, rev, cmd);
  for (int i = 0; i < 4; ++i) steer[i] = cmd[i].steer_target;
  k.Allocate(t, steer, rev, cmd);  // 舵到位,二次分配取满速
  // (a) steer_rate=0:差速只含 wz 项,与正解刚体模型严格一致 -> 精确恢复
  double mps[8];
  for (int i = 0; i < 4; ++i) {
    double dm[2];
    SwerveKinematics::DualWheel(cmd[i].drive_mps[0], t.wz, 0.0, 0.055, dm);
    mps[2 * i] = dm[0];
    mps[2 * i + 1] = dm[1];
    steer[i] = cmd[i].steer_target;
  }
  const Twist back0 = k.EstimateTwist(mps, steer);
  CHECK(std::fabs(back0.vx) < 1e-9);
  CHECK(std::fabs(back0.vy) < 1e-9);
  CHECK(back0.wz == doctest::Approx(t.wz).epsilon(1e-9).scale(1.0));
  // (b) steer_rate≠0:−d·θ̇ 是舵轴带动(刮擦),正解中表现为有界小残差
  //     (~θ̇·d/R_corner 量级),不破坏恢复
  const double steer_rate = 0.7;
  for (int i = 0; i < 4; ++i) {
    double dm[2];
    SwerveKinematics::DualWheel(cmd[i].drive_mps[0], t.wz, steer_rate, 0.055, dm);
    mps[2 * i] = dm[0];
    mps[2 * i + 1] = dm[1];
  }
  const Twist back1 = k.EstimateTwist(mps, steer);
  CHECK(std::fabs(back1.vx) < 5e-3);
  CHECK(std::fabs(back1.vy) < 5e-3);
  CHECK(std::fabs(back1.wz - t.wz) < 0.05);
}

TEST_CASE("reverse decision with hysteresis") {
  auto k = MakeKin();
  double steer[4] = {0, 0, 0, 0};
  bool rev[4] = {false, false, false, false};
  CornerCmd cmd[4];
  // 反向指令:目标角差 ~π -> 反转,轮速取负,目标角保持当前
  k.Allocate(Twist{-1.0, 0.0, 0.0}, steer, rev, cmd);
  CHECK(rev[0]);
  CHECK(cmd[0].drive_mps[0] < 0.0);
  CHECK(std::fabs(cmd[0].steer_target) < 1e-9);
  // 迟滞:轻微偏移不立即翻回
  double steer2[4] = {0.05, 0.05, 0.05, 0.05};
  bool rev2[4] = {true, true, true, true};
  k.Allocate(Twist{-1.0, 0.0, 0.0}, steer2, rev2, cmd);
  CHECK(rev2[0]);  // |err_rev| = 0.05 < threshold+hyst -> 保持反转
}

TEST_CASE("zero twist holds steer") {
  auto k = MakeKin();
  double steer[4] = {0.3, -1.2, 2.0, 0.0};
  bool rev[4] = {false, false, false, false};
  CornerCmd cmd[4];
  k.Allocate(Twist{0, 0, 0}, steer, rev, cmd);
  for (int i = 0; i < 4; ++i) {
    CHECK(cmd[i].steer_target == doctest::Approx(steer[i]));
    CHECK(cmd[i].drive_mps[0] == 0.0);
    CHECK(cmd[i].drive_mps[1] == 0.0);
  }
}

TEST_CASE("cos projection derates transient mismatch") {
  auto k = MakeKin();
  double steer[4] = {0, 0, 0, 0};
  bool rev[4] = {false, false, false, false};
  CornerCmd cmd[4];
  k.Allocate(Twist{1.2, -0.4, 0.0}, steer, rev, cmd);
  const double full = std::hypot(1.2, -0.4);          // 1.2649
  const double derate = std::cos(std::atan2(-0.4, 1.2));  // cos(18.43°)=0.94868
  CHECK(cmd[0].drive_mps[0] == doctest::Approx(full * derate).epsilon(1e-6));
}

TEST_CASE("WrapPi / AngleDiff") {
  CHECK(WrapPi(3.5) == doctest::Approx(3.5 - 2 * M_PI));
  CHECK(WrapPi(-3.5) == doctest::Approx(-3.5 + 2 * M_PI));
  CHECK(AngleDiff(0.1, 6.2) == doctest::Approx(0.1 - 6.2 + 2 * M_PI));
}
