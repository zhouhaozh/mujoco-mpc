#pragma once
// 臂挥拍(轨迹 QP 版,§6.4):SwingGenerator(五次多项式基线)的同位替代。
// 两阶段(python/tools/gen_arm_nlp.py 契约):
//   阶段1 ArmIk 解 q_hit(拍心=p*)+ J5 对准 n* + dq_hit=J⁺v*(拍速=反弹逆解)
//   阶段2 codegen QP(arm_nlp.c):Chebyshev 样条 C(5×9),边界/终值内点/末端
//          静止/72V 速度限位/系数盒,全线性约束(与底盘 MPC 同模式)
// 播放:t∈[t0, t0+T] 映射 s∈[0,1](s_hit=0.72 为触球内点,其后随挥收势)。
#include "control/swing_generator.h"   // SwingCmd + HitPlan(经 hit_planner.h)

namespace sckd {

class ArmSwing {
 public:
  static constexpr int kM = 9;             // 与 gen_arm_nlp.py META.M 一致
  static constexpr double kSHit = 0.72;    // 终值内点
  static constexpr double kLead = 0.060;   // 终值提前(伺服滞后补偿,扫 0.03~0.09 最优)

  // 阶段1+2 一次性求解(t_now=当前绝对仿真时刻;dq=当前关节速度)
  void Init(const HitPlan& plan, const double q_current[5], const double dq_current[5],
            class ArmIk& ik, double t_now);
  // 重规划(锁止前):更新终值并重解 QP(~1ms)
  bool UpdatePlan(const HitPlan& plan, class ArmIk& ik, double t_now);
  // 播放:绝对时刻 → q/dq/ddp + 增益(SwingCmd 契约,ddp 供惯性前馈)
  bool Update(double t_abs, SwingCmd& cmd);
  bool active() const { return active_; }
  bool locked() const;                    // 接近触球(t_hit−t < 0.15)后锁止
  double t_hit() const { return t_hit_; }
  double solve_ms() const { return solve_ms_; }   // 诊断

 private:
  // retime=true:初解,定 t0/T 锚;false:精化,仅更新终值目标
  bool Solve(const HitPlan& plan, class ArmIk& ik, double t_now, bool retime);
  double t0_ = 0, t_hit_ = 0, T_ = 0.65, t_last_solve_ = 0;
  double q0_[5] = {0}, dq0_[5] = {0};
  double coeff_[45] = {0};                // x[j*M+a] = C[j][a]
  bool active_ = false;
  double solve_ms_ = 0.0;
  static constexpr double kLockMargin = 0.15;
};

}  // namespace sckd
