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
  static constexpr double kLead = 0.020;   // 终值提前基值(闭环 τ̂ 补偿主滞后)

  // 阶段1+2 一次性求解(t_now=当前绝对仿真时刻;dq=当前关节速度)
  void Configure(bool validate,bool coherent_clock,double transport_s=.009) {validate_=validate;coherent_clock_=coherent_clock;transport_s_=transport_s;}
  int plan_attempts=0,plan_accepts=0;
  double last_constraint_violation=0;
  const HitPlan& accepted_plan()const{return accepted_plan_;}
  double accepted_time()const{return accepted_time_;}
  void Init(const HitPlan& plan, const double q_current[5], const double dq_current[5],
            class ArmIk& ik, double t_now);
  // 重规划:锁止前自由更新终值;锁止后按 §6.5"冻结大改、仅微调"——终值构型
  // 变化逐关节 ≤ kMicroAdj 才重解(等价拍心 ±cm 级),否则保持(挥拍中途大改必失稳)
  bool UpdatePlan(const HitPlan& plan, class ArmIk& ik, double t_now);
  // 播放:绝对时刻 → q/dq/ddp + 增益(SwingCmd 契约,ddp 供惯性前馈)。
  // q_meas(可空):实测关节角,用于闭环滞后补偿 —— τ̂=e·dq/|dq|²(指令轨迹的
  // 最小二乘时间移位),播放时钟前移 τ̂:拍在 t_hit 实际抵达(替代常数提前量,
  // 高位族轨迹激进滞后大,常数覆盖不了,2026-10-06)
  bool Update(double t_abs, const double* q_meas, SwingCmd& cmd);
  bool active() const { return active_; }
  bool locked() const;                    // 接近触球(t_hit−t < 0.15)后锁止
  double t_hit() const { return t_hit_; }
  double solve_ms() const { return solve_ms_; }   // 诊断

 private:
  // retime=true:初解,定 t0/T 锚;false:精化,仅更新终值目标
  bool Solve(const HitPlan& plan, class ArmIk& ik, double t_now, bool retime);
  HitPlan accepted_plan_;double accepted_time_=-1;
  bool validate_=false,coherent_clock_=false;double transport_s_=.009;
  double t0_ = 0, t_hit_ = 0, T_ = 0.65, t_last_solve_ = 0;
  double q0_[5] = {0}, dq0_[5] = {0}, q_hit_[5] = {0}, q_lock_[5] = {0};
  double coeff_[45] = {0};                // x[j*M+a] = C[j][a]
  bool active_ = false, lock_snap_ = false;
  double solve_ms_ = 0.0;
  static constexpr double kLockMargin = 0.15;
  static constexpr double kMicroAdj = 0.05;  // 锁止后终值微调界(rad/关节,对锁止快照)
};

}  // namespace sckd
