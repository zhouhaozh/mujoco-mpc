// M1 验收 2/3:臂重力前馈静差对照(有/无 FF,多姿态)。
// 判据(设计 §11 M1):带重力前馈静差 < 2°(0.0349 rad)且显著优于无 FF。
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "hal/sim_backend.h"

using namespace sckd;

namespace {

struct PoseCase {
  const char* name;
  double q[kNumArm];
};

int Run(const std::string& scene, const std::string& mj, const std::string& vj,
        const std::string& cj) {
  SimBackend be(scene, mj, vj, cj);
  if (!be.ok() || !be.reset(0)) {
    std::printf("[FAIL] backend init\n");
    return 1;
  }
  be.enable(-1, true);
  be.hold_chassis(2.0);

  const std::vector<PoseCase> cases = {
      {"home", {0, 0, 0, 0, 0}},
      {"side", {0, -0.6, 0.3, 0, 0.5}},
      {"reach", {0.5, -1.2, 0.45, 0.3, 1.0}},  // J3 上限 0.5(URDF 限位),目标须在界内
      {"low", {0, -0.9, 0.45, 0, 0.8}},
  };

  auto settle_error = [&](const double q_target[kNumArm], bool use_ff) {
    ArmHybridCmd to;
    for (int j = 0; j < kNumArm; ++j) {
      to.q[j] = q_target[j];
      to.kp[j] = 200;
      to.kd[j] = (j == 4) ? 1.5 : 12.0;   // J5 滚转轴 kd 界 ~2.2(§12)
    }
    double ff[kNumArm];
    if (use_ff && be.arm_gravity(q_target, ff))
      for (int j = 0; j < kNumArm; ++j) to.tau_ff[j] = ff[j];
    be.set_arm_hybrid(to);
    be.run(3000);                          // 3 s 到位
    ArmHybridCmd hold = to;
    for (int j = 0; j < kNumArm; ++j) {
      hold.kp[j] = 60;
      hold.kd[j] = (j == 4) ? 1.5 : 6.0;
    }
    if (use_ff && be.arm_gravity(q_target, ff))
      for (int j = 0; j < kNumArm; ++j) hold.tau_ff[j] = ff[j];
    be.set_arm_hybrid(hold);
    be.run(2000);                          // 2 s 稳定
    double max_err = 0;
    int worst = -1;
    for (int j = 0; j < kNumArm; ++j) {
      const double e = std::fabs(be.joint(AxisArm(j)).q - q_target[j]);
      if (e > max_err) {
        max_err = e;
        worst = j;
      }
    }
    std::printf("    ff=%d  max_err=%.4f rad (%.2f deg) @ J%d\n", use_ff, max_err,
                max_err * 180.0 / M_PI, worst + 1);
    return max_err;
  };

  const double kPassRad = 0.0349;  // 2°
  bool pass = true;
  double worst_ff = 0;
  for (const auto& c : cases) {
    std::printf("pose %s:\n", c.name);
    const double e0 = settle_error(c.q, false);
    const double e1 = settle_error(c.q, true);
    worst_ff = std::fmax(worst_ff, e1);
    if (!(e1 < kPassRad && e1 < e0)) pass = false;
  }
  std::printf("worst(with FF) = %.2f deg(判据 < 2.00 deg 且优于无 FF)\n",
              worst_ff * 180.0 / M_PI);
  if (pass) {
    std::printf("[PASS] m1_arm_gravity\n");
    return 0;
  }
  std::printf("[FAIL] m1_arm_gravity\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  const std::string mj = argc > 2 ? argv[2] : "models/config/motors.json";
  const std::string vj = argc > 3 ? argv[3] : "models/config/x4_tennis.json";
  const std::string cj = argc > 4 ? argv[4] : "models/config/chassis_m1.json";
  return Run(scene, mj, vj, cj);
}
