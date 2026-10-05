#pragma once
// 参数化发球器(§8.1 Server):按速度/仰角/方位/转速生成来球初态。
#include "sim/ball_aero.h"

namespace sckd {

struct ServerParams {
  BallAeroParams aero;
  double speed = 12.0;         // m/s
  double elevation_deg = 12.0;
  double azimuth_deg = 180.0;  // 180° = 朝 −x(向机器人)
  double spin_rpm = 2000.0;    // 上旋
  double spawn[3] = {8.0, 0.0, 2.0};
};

class BallServer {
 public:
  explicit BallServer(const ServerParams& p);
  // 初态:qpos[3+4](位置+单位四元数)、速度 v[3]、角速度 w[3]
  void Spawn(double qpos[7], double v[3], double w[3]) const;
  void set_params(const ServerParams& p) { p_ = p; }

 private:
  ServerParams p_;
};

}  // namespace sckd
