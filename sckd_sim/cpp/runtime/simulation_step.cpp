#include "runtime/intercept_modules.h"
#include "runtime/controller_adapters.h"
#include "runtime/plan_supervisor.h"
#include "runtime/synchronous_schedule.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
namespace sckd::runtime {

void ApplyFlightForces(AeroContext c) {
  auto& measured = c.measured;
  auto& d = c.d;
  auto& aero = c.aero;
  auto& ball_bid = c.ball_bid;
  auto& ball_vadr = c.ball_vadr;
  // 球气动(真值侧;无旋简化,与预测器 T1 模型一致)
  // 单位:xfrc_applied 为力(N)→ 系数 0.5ρCdA(勿除质量,见 m2_predictor 注)
  if(!measured) {
    const double* bv = d->qvel + ball_vadr;
    const double vm = std::sqrt(bv[0]*bv[0]+bv[1]*bv[1]+bv[2]*bv[2]);
    const double area = M_PI * aero.radius * aero.radius;
    const double kk = 0.5 * aero.rho * area * aero.Cd;   // F = -kk·|v|·v [N]
    for (int i = 0; i < 3; ++i)
      d->xfrc_applied[ball_bid*6+i] = -kk*vm*bv[i];
    d->xfrc_applied[ball_bid*6+3] = d->xfrc_applied[ball_bid*6+4]
        = d->xfrc_applied[ball_bid*6+5] = 0;
  }

}
}
