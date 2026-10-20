// M0 验收 1/4:确定性 —— 同种子(同初始 keyframe+同命令序列)两次完整运行,
// 全轨迹(qpos/qvel/ctrl)逐位一致。依据设计 §11 M0 与 §8.3。
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hal/sim_backend.h"

using namespace sckd;

namespace {

struct RunResult {
  uint64_t hash = 0;
  std::vector<double> qpos_final;
  bool ok = false;
};

RunResult RunOnce(const std::string& scene, const std::string& mj, const std::string& vj) {
  RunResult r;
  SimBackend be(scene, mj, vj);
  if (!be.ok() || !be.reset(0)) return r;
  be.enable(-1, true);
  be.hold_chassis(2.0);
  be.hold_arm(60.0, 1.5);  // J5 滚转轴 kd 界 ~2.2

  uint64_t h = 1469598103934665603ull;  // FNV-1a
  auto mix = [&h](const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 1099511628211ull;
    }
  };

  for (int t = 1; t <= 6000; ++t) {
    if (t == 1000 || t == 3000) {  // 脚本化命令(两次阶跃,覆盖 RS04/RS06)
      ArmHybridCmd c;
      c.kp[0] = 60; c.kd[0] = 6;
      c.kp[1] = 300; c.kd[1] = 20;  // J2 (RS04)
      c.kp[2] = 60; c.kd[2] = 6;
      c.kp[3] = 60; c.kd[3] = 6;
      c.kp[4] = 120; c.kd[4] = 8;   // J5 (RS06)
      c.q[1] = (t == 1000) ? -0.6 : -0.3;
      c.q[4] = (t == 1000) ? 0.5 : 1.0;
      be.set_arm_hybrid(c);
    }
    be.run(1);
    mix(be.data()->qpos, sizeof(double) * be.model()->nq);
    mix(be.data()->qvel, sizeof(double) * be.model()->nv);
    mix(be.data()->ctrl, sizeof(double) * be.model()->nu);
  }
  r.hash = h;
  r.qpos_final.assign(be.data()->qpos, be.data()->qpos + be.model()->nq);
  r.ok = true;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  const std::string mj = argc > 2 ? argv[2] : "models/config/motors.json";
  const std::string vj = argc > 3 ? argv[3] : "models/config/x4_tennis.json";

  const RunResult a = RunOnce(scene, mj, vj);
  const RunResult b = RunOnce(scene, mj, vj);
  if (!a.ok || !b.ok) {
    std::printf("[FAIL] backend init/reset failed\n");
    return 1;
  }

  const bool hash_eq = (a.hash == b.hash);
  const bool final_eq =
      (a.qpos_final.size() == b.qpos_final.size() &&
       std::memcmp(a.qpos_final.data(), b.qpos_final.data(),
                   sizeof(double) * a.qpos_final.size()) == 0);
  std::printf("run1 hash=%016llx  run2 hash=%016llx\n",
              static_cast<unsigned long long>(a.hash), static_cast<unsigned long long>(b.hash));
  std::printf("trajectory_hash_equal=%d  final_qpos_bit_equal=%d\n", hash_eq, final_eq);

  if (hash_eq && final_eq) {
    std::printf("[PASS] determinism: 6000 步全轨迹逐位一致\n");
    return 0;
  }
  std::printf("[FAIL] determinism\n");
  return 1;
}
