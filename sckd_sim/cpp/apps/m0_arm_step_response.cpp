// M0 验收 4/4:臂混合控制阶跃 —— 限幅语义(RS04 J2 ≤120,RS06 J5 ≤36 N·m)、
// 到达性与稳定性;输出 CSV 至 eval/reports/。依据设计 §11 M0 与 §3.3。
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "hal/sim_backend.h"

using namespace sckd;

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  const std::string mj = argc > 2 ? argv[2] : "models/config/motors.json";
  const std::string vj = argc > 3 ? argv[3] : "models/config/x4_tennis.json";
  const std::string csv = argc > 4 ? argv[4] : "eval/reports/arm_step_response.csv";

  SimBackend be(scene, mj, vj);
  if (!be.ok() || !be.reset(0)) {
    std::printf("[FAIL] backend init\n");
    return 1;
  }
  be.enable(-1, true);
  be.hold_chassis(2.0);

  // 阶段:0~2 s 保持;2 s J2 阶跃 -0.6 rad(RS04, kp300/kd20);
  //       5 s J5 阶跃 +1.0 rad(RS06, kp120/kd1.5);共 9 s。
  // J5 为绕拍杆滚转轴,等效惯量 ~0.0011 kg·m²(拍 ixx);armature=0(既定事实)
  // 下 1 kHz 离散稳定界 kd < 2I/dt ≈ 2.2 N·m·s/rad,kd>该值进入 ±cap 跳道极限环。
  // 实机由转子反射惯量(9:1)主导,该界远高——转子惯量辨识列入 F3。
  ArmHybridCmd hold, step2, step5;
  for (int j = 0; j < kNumArm; ++j) {
    hold.kp[j] = 60; hold.kd[j] = 1.5;
  }
  step2 = hold; step5 = hold;
  step2.kp[1] = 300; step2.kd[1] = 20; step2.q[1] = -0.6;  // J2: I~0.7 kg·m², kd 界 ~1400
  step5 = step2;
  step5.kp[4] = 120; step5.kd[4] = 1.5; step5.q[4] = 1.0;

  std::filesystem::create_directories(std::filesystem::path(csv).parent_path());
  std::ofstream out(csv);
  out << "t_s,q_J2,dq_J2,tau_J2,q_J5,dq_J5,tau_J5\n";

  double max_tau2 = 0, max_tau5 = 0;
  std::vector<double> q2_tail, q5_tail;
  for (int t = 1; t <= 9000; ++t) {
    if (t == 1)    be.set_arm_hybrid(hold);
    if (t == 2000) be.set_arm_hybrid(step2);
    if (t == 5000) be.set_arm_hybrid(step5);
    be.run(1);
    const JointReport j2 = be.joint(AxisArm(1));
    const JointReport j5 = be.joint(AxisArm(4));
    max_tau2 = std::fmax(max_tau2, std::fabs(j2.tau_cmd));
    max_tau5 = std::fmax(max_tau5, std::fabs(j5.tau_cmd));
    out << t * 0.001 << ',' << j2.q << ',' << j2.dq << ',' << j2.tau_cmd << ','
        << j5.q << ',' << j5.dq << ',' << j5.tau_cmd << '\n';
    if (t > 8000) {  // 末段 1 s 用于整定判定
      q2_tail.push_back(j2.q);
      q5_tail.push_back(j5.q);
    }
  }
  out.close();

  const double q2_final = q2_tail.back(), q5_final = q5_tail.back();
  auto stdev = [](const std::vector<double>& v) {
    double m = 0;
    for (double x : v) m += x;
    m /= v.size();
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
  };
  const double s2 = stdev(q2_tail), s5 = stdev(q5_tail);
  const bool finite = std::isfinite(q2_final) && std::isfinite(q5_final) &&
                      std::isfinite(max_tau2) && std::isfinite(max_tau5);
  const bool caps_ok = max_tau2 <= 120.0 * (1 + 1e-9) && max_tau5 <= 36.0 * (1 + 1e-9);
  const bool reach_ok = q2_final > -0.75 && q2_final < -0.45 && q5_final > 0.85 &&
                        q5_final < 1.15;
  const bool settle_ok = s2 < 0.01 && s5 < 0.01;

  std::printf("J2(RS04): max|tau|=%.2f N·m (cap 120)  final q=%.3f rad (target -0.6)  tail std=%.4f\n",
              max_tau2, q2_final, s2);
  std::printf("J5(RS06): max|tau|=%.2f N·m (cap 36)   final q=%.3f rad (target +1.0)  tail std=%.4f\n",
              max_tau5, q5_final, s5);
  std::printf("csv -> %s\n", csv.c_str());

  if (finite && caps_ok && reach_ok && settle_ok) {
    std::printf("[PASS] arm_step_response(限幅链/到达/整定)\n");
    return 0;
  }
  std::printf("[FAIL] arm_step_response: finite=%d caps=%d reach=%d settle=%d\n", finite,
              caps_ok, reach_ok, settle_ok);
  return 1;
}
