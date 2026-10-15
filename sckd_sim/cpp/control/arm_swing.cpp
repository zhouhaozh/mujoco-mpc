#include "control/arm_swing.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

#include <osqp/osqp.h>

#include "control/arm_ik.h"
#include "control/gen/arm_qp_data.h"

namespace sckd {
namespace {
constexpr int kM = ArmSwing::kM;
constexpr int kNx = arm_qp_data::kNx;
constexpr int kNg = arm_qp_data::kNg;
constexpr int kNeq = arm_qp_data::kNeq;
constexpr int kNp = 21;

// vendored OSQP 直接驱动(casadi qpsol codegen 包装层互操作失败,改数据头+直调,
// 2026-10-06;H/A 常数仅 l/u 随参数变 → setup 一次 + update_data_vec)
struct OsqpCtx {
  OSQPSolver* solver = nullptr;
  std::vector<OSQPInt> Pp, Pi, Ap, Ai;
  std::vector<OSQPFloat> Px, Ax;
  OSQPCscMatrix Pc, Ac;
  OSQPSettings st{};
  std::vector<OSQPFloat> l, u;

  OsqpCtx() {
    // P = H 上三角 CSC
    for (int c = 0; c < kNx; ++c) {
      Pp.push_back((OSQPInt)Px.size());
      for (int r = 0; r <= c; ++r) {
        const double v = arm_qp_data::H[r][c];
        if (std::fabs(v) > 1e-14) { Pi.push_back(r); Px.push_back(v); }
      }
    }
    Pp.push_back((OSQPInt)Px.size());
    for (int c = 0; c < kNx; ++c) {
      Ap.push_back((OSQPInt)Ax.size());
      for (int r = 0; r < kNg; ++r) {
        const double v = arm_qp_data::A[r][c];
        if (std::fabs(v) > 1e-14) { Ai.push_back(r); Ax.push_back(v); }
      }
    }
    Ap.push_back((OSQPInt)Ax.size());
    Pc.m = Pc.n = kNx; Pc.p = Pp.data(); Pc.i = Pi.data(); Pc.x = Px.data();
    Pc.nzmax = (OSQPInt)Px.size(); Pc.nz = -1;
    Ac.m = kNg; Ac.n = kNx; Ac.p = Ap.data(); Ac.i = Ai.data(); Ac.x = Ax.data();
    Ac.nzmax = (OSQPInt)Ax.size(); Ac.nz = -1;
    l.resize(kNg); u.resize(kNg);
    osqp_set_default_settings(&st);
    st.verbose = 0;
    st.warm_starting = 1;
    st.polishing = 1;
    st.eps_abs = 1e-6;
    st.eps_rel = 1e-6;
    st.max_iter = 8000;
    std::vector<OSQPFloat> q(kNx, 0.0), l0(kNg, 0.0), u0(kNg, 0.0);
    if (osqp_setup(&solver, &Pc, q.data(), &Ac, l0.data(), u0.data(),
                   kNg, kNx, &st) != 0) { solver = nullptr; }
  }
};

// 进程级单例(线程不安全:与既有仿真单线程口径一致)
OsqpCtx& Qp() {
  static OsqpCtx ctx;
  return ctx;
}

// 求解:等式行 l=u=B_eq·p;不等式行 l=−INFTY,u=u_row
bool SolveQp(const double p[kNp], double x[kNx],bool validate,double* violation) {
  OsqpCtx& c = Qp();
  if (!c.solver) return false;
  for (int r = 0; r < kNg; ++r) {
    if (r < kNeq) {
      double b = 0;
      for (int k = 0; k < kNp; ++k) b += arm_qp_data::B_eq[r][k] * p[k];
      c.l[r] = c.u[r] = b;
    } else {
      c.l[r] = -OSQP_INFTY;
      c.u[r] = arm_qp_data::u_row[r];
    }
  }
  if (osqp_update_data_vec(c.solver, nullptr, c.l.data(), c.u.data()) != 0)
    return false;
  if (osqp_solve(c.solver) != 0) return false;
  const OSQPInt st = c.solver->info->status_val;
  if (st != OSQP_SOLVED && st != OSQP_SOLVED_INACCURATE) return false;
  *violation=0;
  for (int i = 0; i < kNx; ++i) {x[i]=c.solver->solution->x[i];if(!std::isfinite(x[i]))*violation=1e9;}
  for(int r=0;r<kNg;++r){double v=0;for(int i=0;i<kNx;++i)v+=arm_qp_data::A[r][i]*x[i];*violation=std::max(*violation,std::max(c.l[r]-v,v-c.u[r]));}
  if(validate&&(*violation>1e-4||st!=OSQP_SOLVED))return false;
  return true;
}

// Chebyshev 基及 s 域导数(与 gen_arm_nlp.py basis() 同递推;ds=2·du 已乘)
void Basis(double s, double* tk, double* d1, double* d2, double* d3) {
  const double u = 2 * s - 1;
  tk[0] = 1.0; tk[1] = u; d1[1] = 1.0;
  d1[0] = d2[0] = d2[1] = d3[0] = d3[1] = 0.0;
  for (int k = 2; k < kM; ++k) {
    tk[k] = 2 * u * tk[k - 1] - tk[k - 2];
    d1[k] = 2 * tk[k - 1] + 2 * u * d1[k - 1] - d1[k - 2];
    d2[k] = 4 * d1[k - 1] + 2 * u * d2[k - 1] - d2[k - 2];
    d3[k] = 6 * d2[k - 1] + 2 * u * d3[k - 1] - d3[k - 2];
  }
  for (int k = 0; k < kM; ++k) { d1[k] *= 2; d2[k] *= 4; d3[k] *= 8; }
}

// J5 姿态对准(与 swing_generator.cpp 同法):过拍心纯滚转,24 候选
double AlignFaceJ5(ArmIk& ik, const double q_ik[5], double bx, double by,
                   const double n_star[3]) {
  const double nm = std::sqrt(n_star[0] * n_star[0] + n_star[1] * n_star[1] +
                              n_star[2] * n_star[2]);
  if (nm < 0.5) return q_ik[4];
  double q_best = q_ik[4], best = -2.0;
  for (int k = 0; k < 24; ++k) {
    double q[5];
    for (int j = 0; j < 5; ++j) q[j] = q_ik[j];
    q[4] = q_ik[4] + (k - 12) * (M_PI / 12.0);
    double n[3];
    ik.FaceNormal(q, bx, by, n);
    const double s = (n[0] * n_star[0] + n[1] * n_star[1] + n[2] * n_star[2]) / nm;
    if (s > best) { best = s; q_best = q[4]; }
  }
  return q_best;
}

// dq = Jᵀ(JJᵀ+λI)⁻¹ v(阻尼最小范数;3×3 解)
void JPinvV(const double J[15], const double v[3], double dq[5]) {
  double A[3][3] = {{0}};
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k)
      for (int c = 0; c < 5; ++c) A[i][k] += J[i * 5 + c] * J[k * 5 + c];
  const double lam = 0.05;
  for (int i = 0; i < 3; ++i) A[i][i] += lam * lam;
  const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                   - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                   + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  if (std::fabs(det) < 1e-12) {
    for (int j = 0; j < 5; ++j) dq[j] = 0;
    return;
  }
  double Ai[3][3];
  Ai[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det;
  Ai[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) / det;
  Ai[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det;
  Ai[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) / det;
  Ai[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det;
  Ai[1][2] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]) / det;
  Ai[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det;
  Ai[2][1] = -(A[0][0] * A[2][1] - A[0][1] * A[2][0]) / det;
  Ai[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det;
  double y[3];
  for (int i = 0; i < 3; ++i)
    y[i] = Ai[i][0] * v[0] + Ai[i][1] * v[1] + Ai[i][2] * v[2];
  for (int c = 0; c < 5; ++c)
    dq[c] = J[0 * 5 + c] * y[0] + J[1 * 5 + c] * y[1] + J[2 * 5 + c] * y[2];
}
}  // namespace

void ArmSwing::Init(const HitPlan& plan, const double q_current[5],
                    const double dq_current[5], ArmIk& ik, double t_now) {
  for (int j = 0; j < 5; ++j) { q0_[j] = q_current[j]; dq0_[j] = dq_current[j]; }
  lock_snap_ = false;
  // 初解失败不得置活(2026-10-07 随机族定位):active_=true 而解失败时 t_hit_/T_
  // 为陈旧零值 → locked() 恒真 → 上层停止重规划、也不再重 Init(臂瘫痪必 miss)。
  // 失败返回非活,上层下一拍以精化后的计划重试 Init(自我恢复)
  active_ = Solve(plan, ik, t_now, /*retime=*/true);
}

bool ArmSwing::UpdatePlan(const HitPlan& plan, ArmIk& ik, double t_now) {
  if (!active_) return false;
  const double bx = plan.base_pred_ok ? plan.base_pred[0] : plan.base_target[0];
  const double by = plan.base_pred_ok ? plan.base_pred[1] : plan.base_target[1];
  if (locked()) {
    // 锁止后微调(§6.5:冻结大改、仅 ±cm 级微调;2026-10-07 实现):完全冻结会
    // 丢弃 lock→t_hit 间 0.15-0.3s 的预测收敛,随机族实测陈旧终值面内未中
    // 0.14-0.24m。微调界对【锁止快照】计(逐步界会使漂移累积成大改,反例
    // plan_err 0.94/1.05,2026-10-07):新终值与快照差逐关节 ≤kMicroAdj 才重解
    if (!lock_snap_) {
      for (int j = 0; j < 5; ++j) q_lock_[j] = q_hit_[j];
      lock_snap_ = true;
    }
    IkResult res = ik.Solve(plan.p_hit, bx, by,
                            0.0, q_lock_);   // 播种锁止快照(同支微调)
    if (res.err > 0.03) return false;
    for (int j = 0; j < 5; ++j)
      if (std::fabs(res.q[j] - q_lock_[j]) > kMicroAdj) return false;
  }
  // 精化:时间锚定(t0/T/q0/dq0 不变,仅终值目标更新)—— 若每次重置 t0,
  // 臂永远处于新轨迹起点,不会真正进入挥拍(滚动重规划病,2026-10-06 定位)
  return Solve(plan, ik, t_now, /*retime=*/false);
}

bool ArmSwing::locked() const {
  // 锁止:距终值点(t_hit−kLead)的剩余时间 < 裕量(以最近一次求解时刻计)
  return (t_hit_ - (coherent_clock_?0:ArmSwing::kLead)) - t_last_solve_ < 0.15;
}

bool ArmSwing::Solve(const HitPlan& plan, ArmIk& ik, double t_now, bool retime) {
  ++plan_attempts;
  // 终值锚定基座:v2 triage(2026-10-08)——优先预测实际底盘位姿(app 外推填充),
  // 拍终值随"底盘真正会在的位置"解;未填时退回规划 base_target(旧行为)
  const double bx = plan.base_pred_ok ? plan.base_pred[0] : plan.base_target[0];
  const double by = plan.base_pred_ok ? plan.base_pred[1] : plan.base_target[1];
  // ---- 阶段1:终值构型/速度 ----
  // 解支搜索(2026-10-08,抽球族定位):位置 IK 存在解流形,拍面朝向随解支
  // 漂移——低位平击(z 0.6-0.8)实测触球法向偏 25-60°(n=(0.41,−0.64,0.65)
  // 反例:球被弹向侧上,vx 不反转),J5 纯滚转只能补绕柄轴 1 维。J1/J2 扰动
  // 种子各解位置 IK + J5 对准,取法向对准最优解支;首种子 = 原行为,仅当
  // 对准显著更优(>+0.02)才改换 —— 固定/随机族既有解不受扰
  // 播种:Init 从当前臂构型 q0_;重解从上次终值 q_hit_(解支连续性——Init 选定
  // 的支不因重解落回默认盆)
  double seed5[5];
  for (int j = 0; j < 5; ++j) seed5[j] = retime ? q0_[j] : q_hit_[j];
  IkResult res = ik.Solve(plan.p_hit, bx, by, 0.0, seed5);
  // 终值 IK 容差放宽到 3cm(非 converged 的 1cm):工作空间边缘(低位平击 z~0.5)
  // 收敛残差 1-2cm 是常态,拍面半宽 11cm 下无损;按 1cm 拒解曾使随机族低位发球
  // 的起挥 Init 连败数百次 → 臂瘫痪(2026-10-07 定位)。真不可达(>3cm)仍拒。
  if (res.err > 0.03) return false;
  res.q[4] = AlignFaceJ5(ik, res.q, bx, by, plan.n_face);
  if (retime) {   // 仅 Init 定支:重解期换支会打断轨迹连续性(面内 0.89 反例)
    double n0[3];
    ik.FaceNormal(res.q, bx, by, n0);
    double best_al = n0[0]*plan.n_face[0] + n0[1]*plan.n_face[1] +
                     n0[2]*plan.n_face[2];
    static const double kSeed[7][2] = {{0.35,0},{-0.35,0},{0,0.25},{0,-0.25},
                                       {0.6,0},{-0.6,0},{0,0.45}};
    for (const auto& sd : kSeed) {
      double q_seed[5];
      for (int j = 0; j < 5; ++j) q_seed[j] = q0_[j];
      q_seed[0] += sd[0]; q_seed[1] += sd[1];
      IkResult r = ik.Solve(plan.p_hit, bx, by, 0.0, q_seed);
      if (r.err > 0.03) continue;
      r.q[4] = AlignFaceJ5(ik, r.q, bx, by, plan.n_face);
      double n[3];
      ik.FaceNormal(r.q, bx, by, n);
      const double al = n[0]*plan.n_face[0] + n[1]*plan.n_face[1] +
                        n[2]*plan.n_face[2];
      if (al > best_al + 0.02) { best_al = al; res = r; }
    }
  }
  double q_hit[5];
  for (int j = 0; j < 5; ++j) q_hit[j] = res.q[j];
  for (int j = 0; j < 5; ++j) q_hit_[j] = q_hit[j];   // 锁止后微调的比对基准
  double J[15], dq_hit[5];
  ik.SiteJacobian(q_hit, bx, by, J);
  // 方案 B′ 终端交接(2026-10-07):拍速 = v_base + J·dq → 关节终值速度改为差项
  // dq_hit = J⁺(v* − v_base*)——底盘 fly-through 先贡献,臂只补余量(高位族
  // 关节速度需求随之下降)。J 按 yaw=0 解(hit_planner 契约),v_base* 世界系
  // 可直接减;v_base*=0(定点模式)时退化为原 JPinvV(J, v*)。
  double v_term[3];
  for (int i = 0; i < 3; ++i)
    v_term[i] = plan.v_racket[i] - ((i < 2) ? plan.v_base_star[i] : 0.0);
  JPinvV(J, v_term, dq_hit);

  // ---- 时间窗:仅初解定锚;终值(=触球内点)落在 t_hit − kLead ----
  t_last_solve_ = t_now;
  t_hit_ = plan.t_hit;
  double T = T_;
  if (retime) {
    const double t_term = t_hit_ - (coherent_clock_?0:ArmSwing::kLead);
    T = (t_term - t_now) / kSHit;
    if (T < 0.30 || T > 1.8) return false;   // 时窗异常(过近/过远)拒解
    T_ = T;
    t0_ = t_now;
  }

  // ---- 阶段2:轨迹 QP(vendored OSQP 直调;H/A 常数,仅 l/u 随参数更新) ----
  double p[kNp];
  int o = 0;
  for (int j = 0; j < 5; ++j) p[o++] = q0_[j];
  for (int j = 0; j < 5; ++j) p[o++] = dq0_[j] * T;      // s 域(×T 契约)
  for (int j = 0; j < 5; ++j) p[o++] = q_hit[j];
  for (int j = 0; j < 5; ++j) p[o++] = dq_hit[j] * T;
  p[o++] = T;

  double z[kNx];
  const auto tc0 = std::chrono::steady_clock::now();
  const bool ok = SolveQp(p, z,validate_,&last_constraint_violation);
  solve_ms_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count() * 1e3;
  if (!ok) return false;
  if(validate_)for(int k=0;k<=100;++k){double tk[kM],d1[kM],d2[kM],d3[kM],q[5]={0};Basis(k/100.0,tk,d1,d2,d3);for(int j=0;j<5;++j)for(int a=0;a<kM;++a)q[j]+=z[j*kM+a]*tk[a];if(!ik.WithinLimits(q))return false;double pos[3];ik.SitePos(q,bx,by,pos);if(pos[2]<.35)return false;}
  for (int i = 0; i < kNx; ++i) coeff_[i] = z[i];
  accepted_plan_=plan;accepted_time_=t_now;++plan_accepts;
  return true;
}

bool ArmSwing::Update(double t_abs, const double* q_meas, SwingCmd& cmd) {
  if (!active_) return false;
  double tau = 0.0;   // 闭环滞后:指令轨迹最小二乘时间移位(s 域)
  if (q_meas && !coherent_clock_) {
    double snom = std::clamp((t_abs - t0_) / T_, 0.0, 1.0);
    double tk[kM], d1[kM], d2[kM], d3[kM];
    Basis(snom, tk, d1, d2, d3);
    double num = 0, den = 0;
    for (int j = 0; j < 5; ++j) {
      double q = 0, dq = 0;
      for (int a = 0; a < kM; ++a) {
        const double c = coeff_[j * kM + a];
        q += c * tk[a]; dq += c * d1[a];
      }
      const double dqs = dq / T_;
      num += (q_meas[j] - q) * dqs;    // 负滞后(臂落后)→ τ>0
      den += dqs * dqs;
    }
    if (den > 1e-6) tau = std::clamp(-num / den, 0.0, 0.10);
  }
  if(coherent_clock_)tau=transport_s_;
  const double s = std::clamp((t_abs + tau - t0_) / T_, 0.0, 1.0);
  double tk[kM], d1[kM], d2[kM], d3[kM];
  Basis(s, tk, d1, d2, d3);
  for (int j = 0; j < 5; ++j) {
    double q = 0, dq = 0, ddq = 0;
    for (int a = 0; a < kM; ++a) {
      const double c = coeff_[j * kM + a];
      q += c * tk[a]; dq += c * d1[a]; ddq += c * d2[a];
    }
    cmd.q[j] = q;
    cmd.dq[j] = dq / T_;
    cmd.ddp[j] = ddq / (T_ * T_);
    cmd.kp[j] = (j == 0) ? 250 : 120;
    cmd.kd[j] = (j == 0) ? 3.0 : (j == 4 ? 1.5 : 8.0);
  }
  return true;
}

}  // namespace sckd
