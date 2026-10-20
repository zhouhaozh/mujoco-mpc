// M0 验收 3/4:8 轮静止接触稳定 —— keyframe 落地、阻尼保持,量测 5 s:
// 速度/倾角/沉陷漂移有界、8 轮均有接触、无数值异常。依据设计 §11 M0。
#include <cmath>
#include <cstdio>
#include <string>

#include <mujoco/mujoco.h>

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
  be.hold_arm(60.0, 1.5);  // J5 滚转轴惯量小,kd 界 ~2.2(见 m0_arm_step_response 注)
  be.run(5000);  // 5 s 沉降

  double max_vx = 0, max_vy = 0, max_wz = 0, max_tilt = 0, z_min = 1e9, z_max = -1e9;
  bool all_finite = true;
  const int nv = be.model()->nv, nq = be.model()->nq;
  for (int t = 0; t < 5000; ++t) {  // 量测窗口 5 s
    be.run(1);
    const mjData* d = be.data();
    for (int i = 0; i < nv && all_finite; ++i)
      if (!std::isfinite(d->qvel[i])) all_finite = false;
    for (int i = 0; i < nq && all_finite; ++i)
      if (!std::isfinite(d->qpos[i])) all_finite = false;
    const BaseState s = be.base_state();
    max_vx = std::fmax(max_vx, std::fabs(s.vx));
    max_vy = std::fmax(max_vy, std::fabs(s.vy));
    max_wz = std::fmax(max_wz, std::fabs(s.wz));
    max_tilt = std::fmax(max_tilt, std::fmax(std::fabs(s.roll), std::fabs(s.pitch)));
    z_min = std::fmin(z_min, s.z);
    z_max = std::fmax(z_max, s.z);
  }

  // 接触统计:轮圆柱(col_*)与地面
  const mjModel* m = be.model();
  const mjData* d = be.data();
  int wheel_contacts = 0;
  double normal_sum = 0;
  for (int i = 0; i < d->ncon; ++i) {
    const char* n1 = mj_id2name(m, mjOBJ_GEOM, d->contact[i].geom1);
    const char* n2 = mj_id2name(m, mjOBJ_GEOM, d->contact[i].geom2);
    const bool w1 = n1 && std::string(n1).rfind("col_", 0) == 0;
    const bool w2 = n2 && std::string(n2).rfind("col_", 0) == 0;
    if (w1 || w2) {
      ++wheel_contacts;
      mjtNum f[6];
      mj_contactForce(m, const_cast<mjData*>(d), i, f);
      normal_sum += f[0];
    }
  }
  const double z_drift = z_max - z_min;
  const double weight_n = 51.26 * 9.81;

  std::printf("max|vx|=%.4f m/s  max|vy|=%.4f m/s  max|wz|=%.4f rad/s  max_tilt=%.4f rad (%.2f deg)\n",
              max_vx, max_vy, max_wz, max_tilt, max_tilt * 180.0 / M_PI);
  std::printf("z range=[%.4f,%.4f] drift=%.4f m  wheel_contacts=%d  normal_sum=%.1f N (整机重 %.1f N)\n",
              z_min, z_max, z_drift, wheel_contacts, normal_sum, weight_n);
  std::printf("all_finite=%d\n", all_finite);

  const bool pass = all_finite && wheel_contacts >= 8 && max_vx < 0.02 && max_vy < 0.02 &&
                    max_wz < 0.05 && max_tilt < 0.0175 && z_drift < 0.005 &&
                    normal_sum > 0.8 * weight_n;
  if (pass) {
    std::printf("[PASS] static_contact\n");
    return 0;
  }
  std::printf("[FAIL] static_contact\n");
  return 1;
}
