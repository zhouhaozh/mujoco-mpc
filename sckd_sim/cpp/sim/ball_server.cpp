#include "sim/ball_server.h"

#include <cmath>

namespace sckd {

BallServer::BallServer(const ServerParams& p) : p_(p) {}

void BallServer::Spawn(double qpos[7], double v[3], double w[3]) const {
  const double el = p_.elevation_deg * M_PI / 180.0;
  const double az = p_.azimuth_deg * M_PI / 180.0;
  const double ch = std::cos(el), sh = std::sin(el);
  v[0] = p_.speed * ch * std::cos(az);
  v[1] = p_.speed * ch * std::sin(az);
  v[2] = p_.speed * sh;
  // 上旋轴:垂直于速度的水平方向(ω×v 升力向上)
  const double wx = std::cos(az), wy = std::sin(az);
  const double spin = p_.spin_rpm * 2.0 * M_PI / 60.0;
  w[0] = -wy * spin;
  w[1] = wx * spin;
  w[2] = 0.0;
  for (int i = 0; i < 3; ++i) qpos[i] = p_.spawn[i];
  qpos[3] = 1.0;
  qpos[4] = qpos[5] = qpos[6] = 0.0;
}

}  // namespace sckd
