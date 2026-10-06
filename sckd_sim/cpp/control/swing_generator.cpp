#include "control/swing_generator.h"
#include "control/arm_ik.h"
#include <cmath>
#include <algorithm>

namespace sckd {

namespace {
// 后引偏置:J1 后旋 + J2 抬 + J3 收(与穿击方向相反)
constexpr double kBackDelta[5] = {-0.45, 0.25, 0.10, 0.0, 0.0};

// 五次多项式插值 a→b(静止到静止),t 裁剪到 [t0, t0+T]
void QuinticProfile(double t0, double T, const double a[5], const double b[5],
                    double t, SwingCmd& cmd) {
  double s = (t - t0) / T;
  s = std::clamp(s, 0.0, 1.0);
  const double p  = 10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s;
  const double dp = (30*s*s - 60*s*s*s + 30*s*s*s*s) / T;
  for (int j = 0; j < 5; ++j) {
    cmd.q[j]  = a[j] + (b[j] - a[j]) * p;
    cmd.dq[j] = (b[j] - a[j]) * dp;
  }
}
}  // namespace

void SwingGenerator::Init(const HitPlan& plan, const double q_current[5], ArmIk& ik) {
  IkResult res = ik.Solve(plan.p_hit, plan.base_target[0], plan.base_target[1],
                          plan.base_target[2], q_current);
  if (!res.converged) { active_ = false; return; }
  for (int j = 0; j < 5; ++j) {
    q0_[j] = q_current[j];
    q_hit_[j] = res.q[j];
    q_back_[j] = q_hit_[j] + kBackDelta[j];
    q_follow_[j] = 2.0 * q_hit_[j] - q_back_[j];  // 对称:穿击段中点恰过 q_hit
  }
  t_hit_ = plan.t_hit;
  active_ = true;
  locked_ = false;
}

bool SwingGenerator::UpdatePlan(const HitPlan& plan, ArmIk& ik) {
  if (!active_ || locked_ || !plan.feasible) return false;
  IkResult res = ik.Solve(plan.p_hit, plan.base_target[0], plan.base_target[1],
                          plan.base_target[2], q_hit_);  // 上一解作种子,保证连续
  if (!res.converged) return false;
  for (int j = 0; j < 5; ++j) {
    q_hit_[j] = res.q[j];
    q_back_[j] = q_hit_[j] + kBackDelta[j];
    q_follow_[j] = 2.0 * q_hit_[j] - q_back_[j];
  }
  t_hit_ = plan.t_hit;
  return true;
}

bool SwingGenerator::Update(double t_abs, SwingCmd& cmd) {
  if (!active_) return false;
  const double h = 0.5 * t_swing_;          // 穿击半长
  const double t_swing0 = t_hit_ - h;       // 穿击段起点
  const double t_back0 = t_swing0 - t_back_;// 后引段起点
  if (!locked_ && t_abs >= t_back0) locked_ = true;

  if (t_abs < t_back0) {
    // 准备段:保持起始构型(重力前馈由调用方叠加)
    for (int j = 0; j < 5; ++j) { cmd.q[j] = q0_[j]; cmd.dq[j] = 0; }
  } else if (t_abs < t_swing0) {
    QuinticProfile(t_back0, t_back_, q0_, q_back_, t_abs, cmd);
  } else if (t_abs < t_hit_ + h) {
    QuinticProfile(t_swing0, t_swing_, q_back_, q_follow_, t_abs, cmd);
    for (int j = 0; j < 5; ++j) { cmd.kp[j] = 200; cmd.kd[j] = (j == 4 ? 1.5 : 10); }
  } else {
    // 随拍后保持
    for (int j = 0; j < 5; ++j) { cmd.q[j] = q_follow_[j]; cmd.dq[j] = 0; }
  }
  return true;
}

}  // namespace sckd
