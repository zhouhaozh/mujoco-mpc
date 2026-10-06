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
bool SolveQp(const double p[kNp], double x[kNx]) {
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
  for (int i = 0; i < kNx; ++i) x[i] = c.solver->solution->x[i];
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
  active_ = true;
  Solve(plan, ik, t_now, /*retime=*/true);
}

bool ArmSwing::UpdatePlan(const HitPlan& plan, ArmIk& ik, double t_now) {
  if (!active_ || locked()) return false;
  // 精化:时间锚定(t0/T/q0/dq0 不变,仅终值目标更新)—— 若每次重置 t0,
  // 臂永远处于新轨迹起点,不会真正进入挥拍(滚动重规划病,2026-10-06 定位)
  return Solve(plan, ik, t_now, /*retime=*/false);
}

bool ArmSwing::locked() const {
  // 锁止:距终值点(t_hit−kLead)的剩余时间 < 裕量(以最近一次求解时刻计)
  return (t_hit_ - ArmSwing::kLead) - t_last_solve_ < 0.15;
}

bool ArmSwing::Solve(const HitPlan& plan, ArmIk& ik, double t_now, bool retime) {
  // ---- 阶段1:终值构型/速度 ----
  IkResult res = ik.Solve(plan.p_hit, plan.base_target[0], plan.base_target[1],
                          0.0, q0_);
  if (!res.converged) return false;
  double q_hit[5];
  for (int j = 0; j < 5; ++j) q_hit[j] = res.q[j];
  q_hit[4] = AlignFaceJ5(ik, q_hit, plan.base_target[0], plan.base_target[1],
                         plan.n_face);
  double J[15], dq_hit[5];
  ik.SiteJacobian(q_hit, plan.base_target[0], plan.base_target[1], J);
  JPinvV(J, plan.v_racket, dq_hit);

  // ---- 时间窗:仅初解定锚;终值(=触球内点)落在 t_hit − kLead ----
  t_last_solve_ = t_now;
  t_hit_ = plan.t_hit;
  double T = T_;
  if (retime) {
    const double t_term = t_hit_ - ArmSwing::kLead;
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
  const bool ok = SolveQp(p, z);
  solve_ms_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count() * 1e3;
  if (!ok) return false;
  for (int i = 0; i < kNx; ++i) coeff_[i] = z[i];
  return true;
}

bool ArmSwing::Update(double t_abs, SwingCmd& cmd) {
  if (!active_) return false;
  const double s = std::clamp((t_abs - t0_) / T_, 0.0, 1.0);
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
