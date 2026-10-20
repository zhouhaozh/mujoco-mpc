#pragma once
// 网球空气动力(§3.5):T1 = 重力 + 二次阻力 + 马格努斯(升力随 spin 比)。
// 参数为文献名义值(calibrated=false);F3 标定后替换。
#include <array>

namespace sckd {

struct BallAeroParams {
  double mass = 0.057;       // kg
  double radius = 0.033;     // m
  double rho = 1.225;        // kg/m³
  double Cd = 0.55;
  double Cl_k = 0.30;        // Cl = Cl_k * S(S=ωr/|v|)
};

// 全加速度(m/s²):a = g + 阻力 + 马格努斯;v/ω 世界系
void BallAeroAccel(const BallAeroParams& p, const double v[3], const double w[3],
                   double a[3]);

// T1 无旋简化(规划/预测器内模型):仅阻力+重力
void BallAeroAccelNoSpin(const BallAeroParams& p, const double v[3], double a[3]);

}  // namespace sckd
