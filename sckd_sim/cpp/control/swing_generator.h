#pragma once
// 挥拍轨迹生成(基线版:关节空间五次多项式;§6.4 算法中立接口的 IK/PVT 实现)。
// 臂 NLP(M2 后续)替换本层时接口不变。
// 时序契约:全部时刻为【绝对仿真时间】,与 HitPlan.t_hit 同基准(M2 闭环根因之二:
// 旧版用相对时刻,重规划后 t_hit 漂移导致挥拍与到达时刻脱节)。
//   准备段(保持 q0) → 后引段(t_back_) → 穿击段(t_swing_,对称分布于 t_hit 两侧,
//   五次多项式 q_back→q_follow,中点恰过 q_hit 且速度近峰值 → 接触即有效击球)。
#include "planning/hit_planner.h"

namespace sckd {

struct SwingCmd {
  double q[5] = {0}, dq[5] = {0};
  double ddp[5] = {0};        // 关节加速度(供惯性前馈 τ_ff += Î·q̈)
  double kp[5] = {120,120,120,120,60};
  double kd[5] = {8,8,8,8,1.5};
  double tau_ff[5] = {0};
};

class SwingGenerator {
 public:
  // 从 HitPlan 生成挥拍:IK 解 q_hit,构造后引/随拍构型;t_hit 取 plan.t_hit(绝对)
  void Init(const HitPlan& plan, const double q_current[5], class ArmIk& ik);
  // 重规划:仅后引段开始前允许(此后 locked_);更新 q_hit/t_hit,不重启已开始的挥拍
  bool UpdatePlan(const HitPlan& plan, class ArmIk& ik);
  // 每步更新:t_abs 为绝对仿真时刻
  bool Update(double t_abs, SwingCmd& cmd);
  bool active() const { return active_; }
  bool locked() const { return locked_; }   // 已进入后引段:时序锁止,禁止重规划
  double t_hit() const { return t_hit_; }
 private:
  double q0_[5] = {0};        // 起始构型(保持段目标)
  double q_back_[5] = {0};    // 后引构型
  double q_hit_[5] = {0};     // 击球构型(穿击段中点)
  double q_follow_[5] = {0};  // 随拍构型(= 2*q_hit - q_back,保证中点过 q_hit)
  double t_hit_ = 0;
  double t_swing_ = 0.45;     // 穿击段全长(0.35→0.45:压 J1/J2 加速度需求)
  double t_back_ = 0.30;      // 后引段时长
  bool active_ = false;
  bool locked_ = false;
};

}  // namespace sckd
