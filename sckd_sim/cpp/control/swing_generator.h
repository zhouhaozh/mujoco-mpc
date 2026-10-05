#pragma once
// 挥拍轨迹生成(基线版:关节空间五次多项式;§6.4 算法中立接口的 IK/PVT 实现)。
// 臂 NLP(M2 后续)替换本层时接口不变。
#include "planning/hit_planner.h"

namespace sckd {

struct SwingCmd {
  double q[5] = {0}, dq[5] = {0};
  double kp[5] = {120,120,120,120,60};
  double kd[5] = {8,8,8,8,1.5};
  double tau_ff[5] = {0};
};

class SwingGenerator {
 public:
  // 从 HitPlan 生成挥拍:先 IK 得 q_hit,再后引 + 五次插值
  void Init(const HitPlan& plan, const double q_current[5], class ArmIk& ik);
  // 每步更新:t 相对于 Init 时刻
  bool Update(double t_rel, SwingCmd& cmd);
  bool active() const { return active_; }
  double t_hit() const { return t_hit_rel_; }
 private:
  double q_back_[5] = {0}, q_hit_[5] = {0};
  double t_hit_rel_ = 0, t_swing_ = 0.35;
  bool active_ = false;
  bool use_gravity_ff_ = true;
};

}  // namespace sckd
