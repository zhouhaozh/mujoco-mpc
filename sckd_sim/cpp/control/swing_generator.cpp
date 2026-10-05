#include "control/swing_generator.h"
#include "control/arm_ik.h"
#include <cmath>
#include <algorithm>

namespace sckd {

void SwingGenerator::Init(const HitPlan& plan, const double q_current[5], ArmIk& ik) {
  IkResult res = ik.Solve(plan.p_hit, plan.base_target[0], plan.base_target[1],
                          plan.base_target[2], q_current);
  if (!res.converged) { active_ = false; return; }
  for (int j = 0; j < 5; ++j) q_hit_[j] = res.q[j];
  // 后引拍:J1 后旋 + J2 抬 + J3 收
  for (int j = 0; j < 5; ++j) q_back_[j] = q_hit_[j];
  q_back_[0] -= 0.55; q_back_[1] += 0.30; q_back_[2] += 0.12;
  t_hit_rel_ = plan.t_hit - plan.time_margin - t_swing_ - 0;  // 近似;实际由调用方传
  active_ = true;
}

bool SwingGenerator::Update(double t_rel, SwingCmd& cmd) {
  if (!active_) return false;
  if (t_rel < t_hit_rel_ - t_swing_) {
    for (int j=0;j<5;++j) { cmd.q[j]=q_back_[j]; cmd.dq[j]=0; }
  } else if (t_rel < t_hit_rel_) {
    const double s = (t_rel - (t_hit_rel_-t_swing_)) / t_swing_;
    const double p = 10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s;
    const double dp = (30*s*s - 60*s*s*s + 30*s*s*s*s) / t_swing_;
    for (int j=0;j<5;++j) {
      cmd.q[j] = q_back_[j] + (q_hit_[j]-q_back_[j])*p;
      cmd.dq[j] = (q_hit_[j]-q_back_[j])*dp;
    }
    for (int j=0;j<5;++j) { cmd.kp[j]=200; cmd.kd[j]=(j==4?1.5:10); }
  } else if (t_rel < t_hit_rel_ + 0.25) {
    for (int j=0;j<5;++j) { cmd.q[j]=q_hit_[j]+0.10*(q_hit_[j]-q_back_[j]); cmd.dq[j]=0; }
  } else {
    for (int j=0;j<5;++j) { cmd.q[j]*=0.9995; cmd.dq[j]=0; }
  }
  return true;
}

}  // namespace sckd
