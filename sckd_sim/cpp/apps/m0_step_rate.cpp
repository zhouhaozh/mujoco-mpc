// M0 验收 2/4:单环境吞吐 —— in-process 直连,≥10k steps/s(设计 §11 M0)。
#include <chrono>
#include <cstdio>
#include <string>

#include "hal/sim_backend.h"

using namespace sckd;

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  const std::string mj = argc > 2 ? argv[2] : "models/config/motors.json";
  const std::string vj = argc > 3 ? argv[3] : "models/config/x4_tennis.json";

  SimBackend be(scene, mj, vj);
  if (!be.ok() || !be.reset(0)) {
    std::printf("[FAIL] backend init\n");
    return 1;
  }
  be.enable(-1, true);
  be.hold_chassis(2.0);
  be.hold_arm(60.0, 1.5);  // J5 滚转轴 kd 界 ~2.2
  be.run(2000);  // 预热(接触稳定、缓存热)

  const int kSteps = 200000;
  const auto t0 = std::chrono::steady_clock::now();
  be.run(kSteps);
  const auto t1 = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count();
  const double sps = kSteps / sec;

  std::printf("model: nbody=%d nq=%d nv=%d nu=%d dt=%.4g s\n",
              static_cast<int>(be.model()->nbody), static_cast<int>(be.model()->nq),
              static_cast<int>(be.model()->nv), static_cast<int>(be.model()->nu),
              be.model()->opt.timestep);
  std::printf("%d steps in %.3f s -> %.0f steps/s (实时 %.1fx)\n", kSteps, sec, sps,
              sps * be.model()->opt.timestep);

  if (sps >= 10000.0) {
    std::printf("[PASS] step_rate >= 10000 steps/s\n");
    return 0;
  }
  std::printf("[FAIL] step_rate < 10000 steps/s\n");
  return 1;
}
