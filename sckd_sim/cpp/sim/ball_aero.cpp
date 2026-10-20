#include "sim/ball_aero.h"

#include <cmath>

namespace sckd {

void BallAeroAccel(const BallAeroParams& p, const double v[3], const double w[3],
                   double a[3]) {
  const double vmag = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const double area = M_PI * p.radius * p.radius;
  const double k = 0.5 * p.rho * area / p.mass;
  for (int i = 0; i < 3; ++i) a[i] = 0.0;
  a[2] -= 9.81;
  if (vmag < 1e-9) return;
  const double kd = k * p.Cd * vmag;                 // 阻力:a_d = -k·Cd·|v|·v
  for (int i = 0; i < 3; ++i) a[i] -= kd * v[i];
  // 马格努斯:F = ½ρClA|v|²(ω̂×v̂),Cl=Cl_k·ωr/|v| → a = k·Cl_k·r·(ω×v)
  const double cr[3] = {w[1] * v[2] - w[2] * v[1], w[2] * v[0] - w[0] * v[2],
                        w[0] * v[1] - w[1] * v[0]};
  for (int i = 0; i < 3; ++i) a[i] += k * p.Cl_k * p.radius * cr[i];
}

void BallAeroAccelNoSpin(const BallAeroParams& p, const double v[3], double a[3]) {
  const double w0[3] = {0, 0, 0};
  BallAeroAccel(p, v, w0, a);
}

}  // namespace sckd
