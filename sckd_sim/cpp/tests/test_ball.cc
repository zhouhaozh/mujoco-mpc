// 球栈单元测试:气动(T1 降速/终末趋势)、物理 EKF vs 逐轴 CA 基线(A/B 前置校验)、
// 反弹分段、拦截采样。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>
#include <cmath>
#include <random>
#include "estimation/ball_predictor.h"
#include "sim/ball_aero.h"
#include "sim/ball_server.h"

using namespace sckd;

TEST_CASE("aero: drag decelerates, gravity pulls") {
  BallAeroParams p;
  double v[3] = {10, 0, 0}, a[3];
  BallAeroAccelNoSpin(p, v, a);
  CHECK(a[0] < -1.5);        // 10 m/s 时阻力减速显著
  CHECK(a[2] == doctest::Approx(-9.81));
}

TEST_CASE("EKF converges on synthetic T1 flight (no bounce)") {
  BallAeroParams p;
  BounceParams bp;
  BallPredictor ekf(p, 0.030, bp);
  // 真值:12 m/s 自 (8,0,2) 朝 −x 仰角 12°
  double x[6] = {8, 0, 2, -11.7, 0, 2.5};
  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0, 0.030);
  double t = 0;
  bool first = true;
  for (int i = 0; i < 300; ++i) {  // 0.3 s @ 1kHz obs(仅测试)
    t += 0.001;
    double v3[3] = {x[3], x[4], x[5]}, a3[3];
    BallAeroAccelNoSpin(p, v3, a3);
    for (int j = 0; j < 3; ++j) { x[j] += 0.001 * x[3 + j]; x[3 + j] += 0.001 * a3[j]; }
    if (i % 5 == 0) {  // 200Hz 观测
      BallObs o{t, {x[0] + noise(rng), x[1] + noise(rng), x[2] + noise(rng)}};
      if (first) { ekf.Reset(o); first = false; } else { ekf.Update(o); }
    }
  }
  double pv[3];
  ekf.Predict(t, pv);
  const double err = std::hypot(pv[0] - x[0], pv[1] - x[1], pv[2] - x[2]);
  CHECK(err < 0.15);  // 当前时刻预测 <15cm
  // 0.3s 外推
  double pf[3];
  ekf.Predict(t + 0.3, pf);
  // 真值外推
  double xt[6] = {x[0], x[1], x[2], x[3], x[4], x[5]};
  for (int i = 0; i < 300; ++i) {
    double v3[3] = {xt[3], xt[4], xt[5]}, a3[3];
    BallAeroAccelNoSpin(p, v3, a3);
    for (int j = 0; j < 3; ++j) { xt[j] += 0.001 * xt[3 + j]; xt[3 + j] += 0.001 * a3[j]; }
  }
  const double err_f = std::hypot(pf[0] - xt[0], pf[1] - xt[1], pf[2] - xt[2]);
  CHECK(err_f < 0.30);  // 0.3s 前视 <30cm
  MESSAGE("EKF cur_err=" << err << " fwd_err=" << err_f);
}

TEST_CASE("A/B: physics EKF beats CA on long-horizon (0.5s+)") {
  BallAeroParams p;
  BounceParams bp;
  BallPredictor ekf(p, 0.030, bp);
  CaAxisFilter ca(0.030, 4.0e6);
  double x[6] = {8, 0, 2, -11.7, 0, 2.5};
  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0, 0.030);
  double t = 0;
  bool first = true;
  for (int i = 0; i < 500; ++i) {  // 0.5 s
    t += 0.001;
    double v3[3] = {x[3], x[4], x[5]}, a3[3];
    BallAeroAccelNoSpin(p, v3, a3);
    for (int j = 0; j < 3; ++j) { x[j] += 0.001 * x[3 + j]; x[3 + j] += 0.001 * a3[j]; }
    if (i % 5 == 0) {
      BallObs o{t, {x[0] + noise(rng), x[1] + noise(rng), x[2] + noise(rng)}};
      if (first) { ekf.Reset(o); ca.Reset(o); first = false; }
      else { ekf.Update(o); ca.Update(o); }
    }
  }
  // 真值 0.5s 外推
  double xt[6] = {x[0], x[1], x[2], x[3], x[4], x[5]};
  for (int i = 0; i < 500; ++i) {
    double v3[3] = {xt[3], xt[4], xt[5]}, a3[3];
    BallAeroAccelNoSpin(p, v3, a3);
    for (int j = 0; j < 3; ++j) { xt[j] += 0.001 * xt[3 + j]; xt[3 + j] += 0.001 * a3[j]; }
  }
  double pe[3], pc[3];
  ekf.Predict(t + 0.5, pe);
  ca.Predict(t + 0.5, pc);
  const double ee = std::hypot(pe[0] - xt[0], pe[1] - xt[1], pe[2] - xt[2]);
  const double ec = std::hypot(pc[0] - xt[0], pc[1] - xt[1], pc[2] - xt[2]);
  MESSAGE("EKF=" << ee << " CA=" << ec);
  CHECK(ee < ec);  // 物理路线在长前视上应不差于运动学基线
}

TEST_CASE("bounce: predictor segments at ground contact") {
  BallAeroParams p;
  BounceParams bp;
  BallPredictor ekf(p, 0.030, bp);
  // 状态:下落,即将反弹
  double x[6] = {0, 0, 0.5, 0, 0, -5};
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0, 0.030);
  double t = 0;
  bool first = true;
  for (int i = 0; i < 3000; ++i) {  // 3 s 含反弹
    t += 0.001;
    double v3[3] = {x[3], x[4], x[5]}, a3[3];
    BallAeroAccelNoSpin(p, v3, a3);
    for (int j = 0; j < 3; ++j) { x[j] += 0.001 * x[3 + j]; x[3 + j] += 0.001 * a3[j]; }
    if (x[2] < 0.033 && x[5] < 0) {  // 真值生成 = 已知碰撞模型(与预测器同式)
      x[2] = 0.033;
      const double vz_in = x[5];
      x[5] = -vz_in * bp.restitution_z;
      const double vt = std::hypot(x[3], x[4]);
      if (vt > 1e-9) {
        const double jn = (1.0 + bp.restitution_z) * (-vz_in);
        const double pr = std::min((2.0 / 7.0) * vt, bp.mu * jn);
        x[3] -= pr * x[3] / vt;
        x[4] -= pr * x[4] / vt;
      }
    }
    if (i % 5 == 0) {
      BallObs o{t, {x[0] + noise(rng), x[1] + noise(rng), x[2] + noise(rng)}};
      if (first) { ekf.Reset(o); first = false; } else { ekf.Update(o); }
    }
  }
  // 反弹后 0.3s:预测不应偏离太远(z 应为正)
  double pv[3];
  ekf.Predict(t, pv);
  CHECK(pv[2] > -0.2);
  CHECK(std::hypot(pv[0] - x[0], pv[1] - x[1], pv[2] - x[2]) < 0.3);
}

TEST_CASE("server spawn: velocity direction") {
  ServerParams sp;
  BallServer srv(sp);
  double q[7], v[3], w[3];
  srv.Spawn(q, v, w);
  CHECK(v[0] < 0);                    // 朝 −x(机器人方向)
  CHECK(v[2] > 0);                    // 仰角向上
  CHECK(w[1] < 0);                    // 上旋(绕 +y 负向)
  CHECK(q[0] == doctest::Approx(8.0));
}
