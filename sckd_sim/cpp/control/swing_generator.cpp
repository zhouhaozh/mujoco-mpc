#include "control/swing_generator.h"
#include "control/arm_ik.h"
#include <cmath>
#include <algorithm>

namespace sckd {

namespace {
// 后引偏置:J1 后旋 + J2 抬 + J3 收(与穿击方向相反)
constexpr double kBackDelta[5] = {-0.35, 0.25, 0.10, 0.0, 0.0};
// 穿击提前量:补偿臂跟踪系统滞后(面内 ~0.12m@t_hit,J1 残余滞后主导,
// 2026-10-06 拍面系分解定位)。0 = 不提前
constexpr double kStrikeLead = 0.050;

// 五次多项式插值 a→b(静止到静止),t 裁剪到 [t0, t0+T]
void QuinticProfile(double t0, double T, const double a[5], const double b[5],
                    double t, SwingCmd& cmd) {
  double s = (t - t0) / T;
  s = std::clamp(s, 0.0, 1.0);
  const double p   = 10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s;
  const double dp  = (30*s*s - 60*s*s*s + 30*s*s*s*s) / T;
  const double ddp = (60*s - 180*s*s + 120*s*s*s) / (T * T);
  for (int j = 0; j < 5; ++j) {
    cmd.q[j]   = a[j] + (b[j] - a[j]) * p;
    cmd.dq[j]  = (b[j] - a[j]) * dp;
    cmd.ddp[j] = (b[j] - a[j]) * ddp;
  }
}

// J5 姿态搜索:J5 为过拍心的纯滚转(不改变拍心位置),在 ±π 内取 24 个候选,
// 选拍面法向与目标方向 n*(意图出球方向,近似垂直墙+仰角)最对齐者。
// 位置 IK 不出姿态 —— 2026-10-06 接触诊断:面法向与反弹方向差 ~130°,
// 球贴面滑行 10+ms 不反弹;对齐后仍受软接触限制,配合拍面 solref 调硬。
double AlignFaceJ5(ArmIk& ik, const double q_ik[5], double bx, double by,
                  const double n_star[3]) {
  const double nm = std::sqrt(n_star[0]*n_star[0] + n_star[1]*n_star[1] + n_star[2]*n_star[2]);
  if (nm < 0.5) return q_ik[4];
  double q_best = q_ik[4], best = -2.0;
  for (int k = 0; k < 24; ++k) {
    double q[5];
    for (int j = 0; j < 5; ++j) q[j] = q_ik[j];
    q[4] = q_ik[4] + (k - 12) * (M_PI / 12.0);
    double n[3];
    ik.FaceNormal(q, bx, by, n);
    const double s = (n[0]*n_star[0] + n[1]*n_star[1] + n[2]*n_star[2]) / nm;
    if (s > best) { best = s; q_best = q[4]; }
  }
  return q_best;
}
}  // namespace

void SwingGenerator::Init(const HitPlan& plan, const double q_current[5], ArmIk& ik) {
  // M2 策略:底盘航向保持 0(不转向,挥拍旋转由 J1 提供),IK 按 yaw=0 解
  // (按 base_target[2] 的指向角解会在真实航向 0 下拍心偏 ~0.7m —— 2026-10-06 定位;
  //  航向窗优化随 M3 底盘 MPC 一并处理)
  IkResult res = ik.Solve(plan.p_hit, plan.base_target[0], plan.base_target[1],
                          0.0, q_current);
  if (!res.converged) { active_ = false; return; }
  res.q[4] = AlignFaceJ5(ik, res.q, plan.base_target[0], plan.base_target[1], plan.n_face);
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
                          0.0, q_hit_);  // 上一解作种子,保证连续;yaw=0 同 Init
  if (!res.converged) return false;
  res.q[4] = AlignFaceJ5(ik, res.q, plan.base_target[0], plan.base_target[1], plan.n_face);
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
  const double tc = t_hit_ - kStrikeLead;   // 实际穿击中心(提前补偿跟踪滞后)
  const double h = 0.5 * t_swing_;          // 穿击半长
  const double t_swing0 = tc - h;           // 穿击段起点
  const double t_back0 = t_swing0 - t_back_;// 后引段起点
  // 锁止点 = 穿击段开始(后引段仍允许 RefineAt 微调 q_hit/q_back:
  // 末期预测误差已 ~cm 级,重定向跳变有界)
  if (!locked_ && t_abs >= t_swing0) locked_ = true;

  if (t_abs < t_back0) {
    // 准备段:保持起始构型(重力前馈由调用方叠加)
    for (int j = 0; j < 5; ++j) { cmd.q[j] = q0_[j]; cmd.dq[j] = 0; }
  } else if (t_abs < t_swing0) {
    QuinticProfile(t_back0, t_back_, q0_, q_back_, t_abs, cmd);
  } else if (t_abs < t_hit_ + h) {
    QuinticProfile(t_swing0, t_swing_, q_back_, q_follow_, t_abs, cmd);
    // 增益按轴惯量整定:J1 惯量小,kp=200/kd=10 在 9ms 命令延迟下抖振失稳
    // (速度抖 ±6 rad/s、均值掉半 —— 2026-10-06 定位;J2 同时 120 N·m 饱和,
    //  靠惯性前馈与放长的挥拍段共同缓解)
    for (int j = 0; j < 5; ++j) {
      cmd.kp[j] = (j == 0) ? 250 : 120;
      cmd.kd[j] = (j == 0) ? 3.0 : (j == 4 ? 1.5 : 8.0);
    }
  } else {
    // 随拍后保持
    for (int j = 0; j < 5; ++j) { cmd.q[j] = q_follow_[j]; cmd.dq[j] = 0; }
  }
  return true;
}

}  // namespace sckd
