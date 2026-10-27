#include "control/wholebody_mpc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include <osqp/osqp.h>

namespace sckd {
namespace {

constexpr int kNXs = 14;   // [x,y,vx,vy,q1..5,dq1..5]
constexpr int kNUs = 7;    // [ax,ay,ddq1..5]
constexpr int kNSs = 6;    // [sp3, sv3](无界软松弛)
constexpr int kN = WholebodyMpc::kN;
constexpr int kNVar = kNXs * (kN + 1) + kNUs * kN + kNSs;      // 272
constexpr int kRowX0 = 0;
constexpr int kRowDyn = kNXs;                                  // 14·N 行
// 接触流形行:每节点 6 行(k=1..N 并集枚举,模式固定;激活节点由行界决定,
// 未激活行 ±inf 失效)——支持内点接触(k_contact,2026-10-09)
constexpr int kNTermRows = 6 * kN;
constexpr int kRowTerm = kNXs + kNXs * kN;
constexpr int kRowBox = kRowTerm + kNTermRows;
constexpr int kMRows = kRowBox + kNVar;
inline int iTerm(int k, int i) { return kRowTerm + (k - 1) * 6 + i; }

inline int iX(int k, int j) { return k * kNXs + j; }
inline int iU(int k, int j) { return kNXs * (kN + 1) + k * kNUs + j; }
inline int iS(int j) { return kNXs * (kN + 1) + kNUs * kN + j; }

// 终端几何在 q̄ 处的值与一阶导(FK/J/n + 中心差分 D_lJ、D_ln)
void EvalTerm(ArmIk& ik, const double qbar[5], double fk[3], double Jb[15],
              double dJ[5][15], double nbar[3], double dn[5][3]) {
  ik.SitePos(qbar, 0, 0, fk);
  ik.SiteJacobian(qbar, 0, 0, Jb);
  ik.FaceNormal(qbar, 0, 0, nbar);
  const double eps = 1e-4;
  for (int l = 0; l < 5; ++l) {
    double qp[5], qm[5], Jp[15], Jm[15], np[3], nm[3];
    for (int j = 0; j < 5; ++j) { qp[j] = qbar[j]; qm[j] = qbar[j]; }
    qp[l] += eps; qm[l] -= eps;
    ik.SiteJacobian(qp, 0, 0, Jp);
    ik.SiteJacobian(qm, 0, 0, Jm);
    ik.FaceNormal(qp, 0, 0, np);
    ik.FaceNormal(qm, 0, 0, nm);
    for (int i = 0; i < 15; ++i) dJ[l][i] = (Jp[i] - Jm[i]) / (2 * eps);
    for (int i = 0; i < 3; ++i) dn[l][i] = (np[i] - nm[i]) / (2 * eps);
  }
}

// A 条目枚举(模式与值同源生成;emit(row,col,val))。
// 终端(线性化):pos_i: base_i + fk̄ + J̄(q_N−q̄) = p* + sp
//               vel_i: v_base_i + J̄·dq_N + Σ_l δq_l·Cv[i][l] = v* + sv
// Cv[i][l]=Σ_m (D_lJ)_im·dq̄_m(J(q)dq 精确一阶展开交叉项)
template <typename Emit>
void EnumerateA(double h, const double Jb[15], const double Cv[3][5], Emit&& emit) {
  for (int j = 0; j < kNXs; ++j) emit(kRowX0 + j, iX(0, j), 1.0);
  for (int k = 0; k < kN; ++k) {
    const int r = kRowDyn + k * kNXs;
    emit(r + 0, iX(k + 1, 0), 1.0); emit(r + 0, iX(k, 0), -1.0);
    emit(r + 0, iX(k, 2), -h);      emit(r + 0, iU(k, 0), -0.5 * h * h);
    emit(r + 1, iX(k + 1, 1), 1.0); emit(r + 1, iX(k, 1), -1.0);
    emit(r + 1, iX(k, 3), -h);      emit(r + 1, iU(k, 1), -0.5 * h * h);
    emit(r + 2, iX(k + 1, 2), 1.0); emit(r + 2, iX(k, 2), -1.0);
    emit(r + 2, iU(k, 0), -h);
    emit(r + 3, iX(k + 1, 3), 1.0); emit(r + 3, iX(k, 3), -1.0);
    emit(r + 3, iU(k, 1), -h);
    for (int j = 0; j < 5; ++j) {
      emit(r + 4 + j, iX(k + 1, 4 + j), 1.0); emit(r + 4 + j, iX(k, 4 + j), -1.0);
      emit(r + 4 + j, iX(k, 9 + j), -h);      emit(r + 4 + j, iU(k, 2 + j), -0.5 * h * h);
      emit(r + 9 + j, iX(k + 1, 9 + j), 1.0); emit(r + 9 + j, iX(k, 9 + j), -1.0);
      emit(r + 9 + j, iU(k, 2 + j), -h);
    }
  }
  for (int k = 1; k <= kN; ++k) {   // 接触流形:并集枚举(值各节点同源)
    for (int i = 0; i < 3; ++i) {
      const int r = iTerm(k, i);
      if (i < 2) emit(r, iX(k, i), 1.0);
      for (int j = 0; j < 5; ++j) emit(r, iX(k, 4 + j), Jb[i * 5 + j]);
      emit(r, iS(i), -1.0);
    }
    for (int i = 0; i < 3; ++i) {
      const int r = iTerm(k, 3 + i);
      if (i < 2) emit(r, iX(k, 2 + i), 1.0);
      for (int l = 0; l < 5; ++l) emit(r, iX(k, 4 + l), Cv[i][l]);
      for (int j = 0; j < 5; ++j) emit(r, iX(k, 9 + j), Jb[i * 5 + j]);
      emit(r, iS(3 + i), -1.0);
    }
  }
  for (int c = 0; c < kNVar; ++c) emit(kRowBox + c, c, 1.0);
}

// P(上三角)条目:控制能量 + 无界松弛罚(对角)+ 法向软对准接触节点 q 块
// (g=∂(n·n*)/∂q;并集枚举,激活节点 kc 外值 0)+ 底盘走廊/锚吸引对角
// (线性项在 q_lin,调用方填)。kc=kN = legacy 终端接触
template <typename Emit>
void EnumerateP(double h, double w_sp, double w_sv, const double g[5],
                double w_n, double w_anchor, double w_corr, int kc, double a2,
                const double ddq_lim[5], Emit&& emit) {
  for (int k = 0; k < kN; ++k) {
    emit(iU(k, 0), iU(k, 0), 2.0 * h / a2);
    emit(iU(k, 1), iU(k, 1), 2.0 * h / a2);
    for (int j = 0; j < 5; ++j)
      emit(iU(k, 2 + j), iU(k, 2 + j), 2.0 * h / (ddq_lim[j] * ddq_lim[j]));
  }
  for (int j = 0; j < 3; ++j) emit(iS(j), iS(j), 2.0 * w_sp);
  for (int j = 0; j < 3; ++j) emit(iS(3 + j), iS(3 + j), 2.0 * w_sv);
  for (int k = 0; k <= kN; ++k) {   // 走廊节点对角;接触节点叠加锚吸引
    const double w = w_corr + (k == kc ? w_anchor : 0.0);
    emit(iX(k, 0), iX(k, 0), 2.0 * w);   // (单条发射:重复 (row,col) 会造出
    emit(iX(k, 1), iX(k, 1), 2.0 * w);   //  非法 CSC,OSQP update 恒败)
  }
  for (int k = 1; k <= kN; ++k)     // 法向块(激活节点外值 0,模式并集)
    for (int c = 0; c < 5; ++c)
      for (int r = 0; r <= c; ++r)
        emit(iX(k, 4 + r), iX(k, 4 + c),
             (k == kc) ? 2.0 * w_n * g[r] * g[c] : 0.0);
}

// 模式排序序(与 wholebody_ocp.cpp 同法:Ax[s]=tmp[aord[s]])
std::vector<int> BuildOrder(bool p_mat) {
  std::vector<std::pair<int, int>> e;
  double Jb0[15] = {0}, Cv0[3][5] = {{0}}, g0[5] = {0}, ddq0[5] = {1, 1, 1, 1, 1};
  if (p_mat)
    EnumerateP(0.05, 1, 1, g0, 1, 1, 1, kN, 1, ddq0, [&](int r, int c, double) { e.emplace_back(c, r); });
  else
    EnumerateA(0.05, Jb0, Cv0, [&](int r, int c, double) { e.emplace_back(c, r); });
  std::vector<int> ord(e.size());
  for (int i = 0; i < (int)ord.size(); ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return e[a] < e[b]; });
  return ord;
}

// vendored OSQP 上下文(模式一次 setup,值全量替换;进程级单例,单线程口径
// 与 arm_swing Qp() 一致)
struct WmQp {
  OSQPSolver* solver = nullptr;
  std::vector<OSQPInt> Ap, Ai, Pp, Pi;
  std::vector<OSQPFloat> Ax, Px, q_lin, l, u;
  std::vector<int> aord, pord;
  bool ok = false;

  WmQp() {
    aord = BuildOrder(false);
    pord = BuildOrder(true);
    std::vector<std::pair<int, int>> ae, pe;
    double Jb0[15] = {0}, Cv0[3][5] = {{0}}, g0[5] = {0}, ddq0[5] = {1, 1, 1, 1, 1};
    EnumerateA(0.05, Jb0, Cv0, [&](int r, int c, double) { ae.emplace_back(c, r); });
    EnumerateP(0.05, 1, 1, g0, 1, 1, 1, kN, 1, ddq0, [&](int r, int c, double) { pe.emplace_back(c, r); });

    Ax.assign(ae.size(), 0.0);
    Ai.resize(ae.size());
    Ap.assign(kNVar + 1, 0);
    {
      int idx = 0;
      for (int c = 0; c < kNVar; ++c) {
        Ap[c] = idx;
        while (idx < (int)aord.size() && ae[aord[idx]].first == c) {
          Ai[idx] = ae[aord[idx]].second;
          ++idx;
        }
      }
      Ap[kNVar] = (OSQPInt)aord.size();
    }
    Px.assign(pe.size(), 0.0);
    Pi.resize(pe.size());
    Pp.assign(kNVar + 1, 0);
    {
      int idx = 0;
      for (int c = 0; c < kNVar; ++c) {
        Pp[c] = idx;
        while (idx < (int)pord.size() && pe[pord[idx]].first == c) {
          Pi[idx] = pe[pord[idx]].second;
          ++idx;
        }
      }
      Pp[kNVar] = (OSQPInt)pord.size();
    }
    q_lin.assign(kNVar, 0.0);
    l.assign(kMRows, -OSQP_INFTY);
    u.assign(kMRows, OSQP_INFTY);

    OSQPCscMatrix Pc, Ac;
    Pc.m = Pc.n = kNVar; Pc.p = Pp.data(); Pc.i = Pi.data(); Pc.x = Px.data();
    Pc.nzmax = (OSQPInt)Px.size(); Pc.nz = -1;
    Ac.m = kMRows; Ac.n = kNVar; Ac.p = Ap.data(); Ac.i = Ai.data(); Ac.x = Ax.data();
    Ac.nzmax = (OSQPInt)Ax.size(); Ac.nz = -1;
    OSQPSettings st{};
    osqp_set_default_settings(&st);
    st.verbose = 0;
    st.warm_starting = 1;
    st.eps_abs = 1e-4;   // 闭环(锚盒+大 dq̄ 线性化)下 1e-5 触发 MAX_ITER
    st.eps_rel = 1e-4;   // (12000 迭代,42 次/2trial);外层 SQP+质量门槛吸收
    st.max_iter = 12000;
    st.polishing = 1;
    if (osqp_setup(&solver, &Pc, q_lin.data(), &Ac, l.data(), u.data(),
                   kMRows, kNVar, &st) != 0) {
      solver = nullptr;
      return;
    }
    ok = true;
  }
};

WmQp& Qp() {
  static WmQp ctx;
  return ctx;
}

// 非线性终端残差:pos = ‖base+FK(q)−p*‖,vel = ‖[vx,vy,0]+J(q)·dq−v*‖
void TermResidual(ArmIk& ik, const double base[2], const double vbase[2],
                  const double q[5], const double dq[5], const double p_hit[3],
                  const double v_star[3], double* pos_r, double* vel_r) {
  double fk[3], J[15];
  ik.SitePos(q, 0, 0, fk);
  ik.SiteJacobian(q, 0, 0, J);
  const double pe[3] = {base[0] + fk[0] - p_hit[0], base[1] + fk[1] - p_hit[1],
                        fk[2] - p_hit[2]};
  double ve[3] = {vbase[0], vbase[1], 0.0};
  for (int j = 0; j < 5; ++j) {
    ve[0] += J[j] * dq[j];
    ve[1] += J[5 + j] * dq[j];
    ve[2] += J[10 + j] * dq[j];
  }
  for (int i = 0; i < 3; ++i) ve[i] -= v_star[i];
  *pos_r = std::hypot(pe[0], pe[1], pe[2]);
  *vel_r = std::hypot(ve[0], ve[1], ve[2]);
}

}  // namespace

WholebodyMpc::WholebodyMpc(ArmIk& ik, const mjModel* m, const WmLimits& lim)
    : ik_(ik), lim_(lim) {
  // 按控制轴名取活动关节,不把固定 Link6 当成第六个关节。
  for (int i = 0; i < 5; ++i) {
    const int j = mj_name2id(m, mjOBJ_JOINT, kAxisJointName[AxisArm(i)]);
    if (j < 0 || m->jnt_type[j] != mjJNT_HINGE)
      throw std::invalid_argument("WholebodyMpc requires Joint1..Joint5 hinges");
    if ((i == 1 || i == 2) && !m->jnt_limited[j])
      throw std::invalid_argument("WholebodyMpc requires confirmed Joint2/Joint3 position limits");
    if (m->jnt_limited[j]) {
      lim_.q_lo[i] = m->jnt_range[2 * j];
      lim_.q_hi[i] = m->jnt_range[2 * j + 1];
    } else {
      lim_.q_lo[i] = -3.5; lim_.q_hi[i] = 3.5;
    }
  }
}

bool WholebodyMpc::Solve(const WmRequest& req, WmPlan* out) {
  last_reason="time_domain";last_ms=0;last_pos_res=1e9;
  const double T = req.t_hit - req.t_now;
  if (T < 0.15 || T > 3.0) return false;
  const double h = T / kN;
  // 接触节点:内点(k_contact)或终端(legacy);行/信任域/法向块都挂它
  const int kc = (req.k_contact > 0 && req.k_contact <= kN) ? req.k_contact : kN;
  WmQp& qp = Qp();
  if (!qp.ok) {last_reason="qp_setup";return false;}
  const auto tc0 = std::chrono::steady_clock::now();

  double x0r[kNXs];
  x0r[0] = req.base0[0]; x0r[1] = req.base0[1];
  x0r[2] = std::clamp(req.vbase0[0], -lim_.v_axis, lim_.v_axis);
  x0r[3] = std::clamp(req.vbase0[1], -lim_.v_axis, lim_.v_axis);
  for (int j = 0; j < 5; ++j) {
    x0r[4 + j] = std::clamp(req.q0[j], lim_.q_lo[j], lim_.q_hi[j]);
    x0r[9 + j] = std::clamp(req.dq0[j], -lim_.dq_lim[j], lim_.dq_lim[j]);
  }

  // 行界:X0/动力学/接触节点等式(未激活流形行 ±inf 失效)+ 变量界
  // (松弛无界——子问题恒可行的关键)
  auto fill_rows = [&](const double term_rhs[6], const double qtr[5], double trust) {
    for (int j = 0; j < kNXs; ++j) qp.l[kRowX0 + j] = qp.u[kRowX0 + j] = x0r[j];
    for (int r = kRowDyn; r < kRowTerm; ++r) qp.l[r] = qp.u[r] = 0.0;
    for (int r = kRowTerm; r < kRowBox; ++r) {
      qp.l[r] = -OSQP_INFTY;
      qp.u[r] = OSQP_INFTY;
    }
    for (int i = 0; i < 6; ++i)
      qp.l[iTerm(kc, i)] = qp.u[iTerm(kc, i)] = term_rhs[i];
    for (int k = 0; k <= kN; ++k) {
      qp.l[kRowBox + iX(k, 0)] = -OSQP_INFTY; qp.u[kRowBox + iX(k, 0)] = OSQP_INFTY;
      qp.l[kRowBox + iX(k, 1)] = -OSQP_INFTY; qp.u[kRowBox + iX(k, 1)] = OSQP_INFTY;
      if (k == kc && req.base_anchor_ok) {   // 接触节点底盘锚邻域(重解稳定)
        qp.l[kRowBox + iX(k, 0)] = req.base_anchor[0] - req.base_anchor_r;
        qp.u[kRowBox + iX(k, 0)] = req.base_anchor[0] + req.base_anchor_r;
        qp.l[kRowBox + iX(k, 1)] = req.base_anchor[1] - req.base_anchor_r;
        qp.u[kRowBox + iX(k, 1)] = req.base_anchor[1] + req.base_anchor_r;
      }
      qp.l[kRowBox + iX(k, 2)] = -lim_.v_axis; qp.u[kRowBox + iX(k, 2)] = lim_.v_axis;
      qp.l[kRowBox + iX(k, 3)] = -lim_.v_axis; qp.u[kRowBox + iX(k, 3)] = lim_.v_axis;
      for (int j = 0; j < 5; ++j) {
        double lo = lim_.q_lo[j], hi = lim_.q_hi[j];
        if (k == kc) {   // 接触节点信任域(阻尼;松弛无界后仅防解支跳)
          lo = std::max(lo, qtr[j] - trust);
          hi = std::min(hi, qtr[j] + trust);
        }
        qp.l[kRowBox + iX(k, 4 + j)] = lo;
        qp.u[kRowBox + iX(k, 4 + j)] = hi;
        qp.l[kRowBox + iX(k, 9 + j)] = -lim_.dq_lim[j];
        qp.u[kRowBox + iX(k, 9 + j)] = lim_.dq_lim[j];
      }
    }
    for (int k = 0; k < kN; ++k)
      for (int j = 0; j < 7; ++j) {
        const double limj = (j < 2) ? lim_.a_max : lim_.ddq_lim[j - 2];
        qp.l[kRowBox + iU(k, j)] = -limj; qp.u[kRowBox + iU(k, j)] = limj;
      }
    for (int j = 0; j < 6; ++j) {   // 松弛无界(罚在 P)
      qp.l[kRowBox + iS(j)] = -OSQP_INFTY;
      qp.u[kRowBox + iS(j)] = OSQP_INFTY;
    }
  };

  // SQP 迭代点热启动(上一拍接受解终端;首拍/参考跳变=IK 播种,缺省实测)
  const bool reseed =
      has_prev_ && (std::hypot(req.p_hit[0] - prev_p_hit_[0],
                               req.p_hit[1] - prev_p_hit_[1],
                               req.p_hit[2] - prev_p_hit_[2]) > lim_.reseed_dp);
  double qbar[5], dqbar[5];
  for (int j = 0; j < 5; ++j) {
    qbar[j] = (has_prev_ && !reseed)
                  ? q_prev_[j]
                  : (req.q_seed_ok ? req.q_seed[j] : x0r[4 + j]);
    dqbar[j] = (has_prev_ && !reseed) ? dq_prev_[j] : x0r[9 + j];
  }

  bool solved = false;
  double trust = (trust_warm_ > 0.0 && !reseed)
                     ? std::clamp(trust_warm_, lim_.trust_min, lim_.trust_max)
                     : lim_.trust0;
  double best_res = 1e9, cost = 0;
  double z_term[kNXs] = {0}, best_term[kNXs] = {0};
  std::vector<double> zbest(kNVar, 0.0), zfull(kNVar, 0.0);
  int iters = 0;
  for (int it = 0; it < kMaxSqpIters; ++it) {
    ++iters;
    double fk[3], Jb[15], nbar[3], dJ[5][15], dn[5][3];
    EvalTerm(ik_, qbar, fk, Jb, dJ, nbar, dn);
    double Cv[3][5] = {{0}};
    for (int i = 0; i < 3; ++i)
      for (int l = 0; l < 5; ++l) {
        double s = 0;
        for (int mm = 0; mm < 5; ++mm) s += dJ[l][i * 5 + mm] * dqbar[mm];
        Cv[i][l] = s;
      }
    double g[5];
    for (int j = 0; j < 5; ++j)
      g[j] = dn[j][0] * req.n_star[0] + dn[j][1] * req.n_star[1] +
             dn[j][2] * req.n_star[2];
    double cprime;
    {
      const double nn = nbar[0] * req.n_star[0] + nbar[1] * req.n_star[1] +
                        nbar[2] * req.n_star[2];
      double gq = 0;
      for (int j = 0; j < 5; ++j) gq += g[j] * qbar[j];
      cprime = (nn - 1.0) - gq;
    }
    double rhs[6];
    for (int i = 0; i < 3; ++i) {
      double jq = 0, cvq = 0;
      for (int j = 0; j < 5; ++j) {
        jq += Jb[i * 5 + j] * qbar[j];
        cvq += Cv[i][j] * qbar[j];   // 速度行常数:Σ_l Cv_il·q̄_l(展开
      }                              // Cv·(q_N−q̄) 时的平移项,漏掉=线性模型
      rhs[i] = req.p_hit[i] - fk[i] + jq;          // 与真值差 m/s 级,2026-10-09 定位)
      rhs[3 + i] = req.v_star[i] + cvq;
    }
    fill_rows(rhs, qbar, trust);
    {
      static std::vector<double> tmp;
      tmp.resize(qp.aord.size());
      int idx = 0;
      EnumerateA(h, Jb, Cv, [&](int, int, double v) { tmp[idx++] = v; });
      for (int s = 0; s < (int)qp.aord.size(); ++s) qp.Ax[s] = tmp[qp.aord[s]];
    }
    {
      static std::vector<double> tmp;
      tmp.resize(qp.pord.size());
      int idx = 0;
      const double w_a = req.base_anchor_ok ? lim_.w_anchor : 0.0;
      const double w_c = (req.corr_ok && (int)req.corr_x.size() == kN + 1)
                             ? lim_.w_corr : 0.0;
      EnumerateP(h, lim_.w_sp, lim_.w_sv, g, lim_.w_normal, w_a, w_c, kc,
                 lim_.a_max * lim_.a_max, lim_.ddq_lim,
                 [&](int, int, double v) { tmp[idx++] = v; });
      for (int s = 0; s < (int)qp.pord.size(); ++s) qp.Px[s] = tmp[qp.pord[s]];
      std::fill(qp.q_lin.begin(), qp.q_lin.end(), 0.0);
      for (int j = 0; j < 5; ++j)
        qp.q_lin[iX(kc, 4 + j)] = 2.0 * lim_.w_normal * cprime * g[j];
      if (req.base_anchor_ok) {   // 锚吸引线性项(对角在 P)
        qp.q_lin[iX(kc, 0)] = -2.0 * w_a * req.base_anchor[0];
        qp.q_lin[iX(kc, 1)] = -2.0 * w_a * req.base_anchor[1];
      }
      if (w_c > 0.0) {           // 走廊线性项:节点拉向绝对时刻标称位置
        for (int k = 0; k <= kN; ++k) {
          qp.q_lin[iX(k, 0)] = -2.0 * w_c * req.corr_x[k];
          qp.q_lin[iX(k, 1)] = -2.0 * w_c * req.corr_y[k];
        }
      }
    }
    if (osqp_update_data_mat(qp.solver, qp.Px.data(), OSQP_NULL, (OSQPInt)qp.Px.size(),
                             qp.Ax.data(), OSQP_NULL, (OSQPInt)qp.Ax.size()) != 0 ||
        osqp_update_data_vec(qp.solver, qp.q_lin.data(), qp.l.data(), qp.u.data()) != 0)
      break;
    if (osqp_solve(qp.solver) != 0) break;
    const OSQPInt stv = qp.solver->info->status_val;
    if (stv != OSQP_SOLVED && stv != OSQP_SOLVED_INACCURATE) {
      if (getenv("WM_DEBUG"))
        std::fprintf(stderr, "[wm] osqp st=%d it=%d\n", (int)stv,
                     (int)qp.solver->info->iter);
      break;
    }
    const OSQPFloat* z = qp.solver->solution->x;
    for (int i = 0; i < kNVar; ++i) zfull[i] = z[i];
    for (int j = 0; j < kNXs; ++j) z_term[j] = z[iX(kc, j)];
    solved = true;
    cost = qp.solver->info->obj_val;

    // 非线性残差(真值)驱动自适应信任域:改善→扩域继续;变差→回退迭代点缩域
    double pos_r, vel_r;
    TermResidual(ik_, z_term, z_term + 2, z_term + 4, z_term + 9,
                 req.p_hit, req.v_star, &pos_r, &vel_r);
    if (getenv("WM_DEBUG")) {
      // 线性化行残差(QP 口径):vbase + Cv·q_N + J̄·dq_N − sv − v*
      double linv[3] = {0, 0, 0};
      for (int i = 0; i < 3; ++i) {
        linv[i] = (i < 2 ? z_term[2 + i] : 0.0) - req.v_star[i];
        for (int l = 0; l < 5; ++l) linv[i] += Cv[i][l] * z_term[4 + l];
        for (int j = 0; j < 5; ++j) linv[i] += Jb[i * 5 + j] * z_term[9 + j];
        linv[i] -= zfull[iS(3 + i)];
      }
      double dqm = 0, dqm_bar = 0, dqjump = 0;
      for (int j = 0; j < 5; ++j) {
        dqm = std::max(dqm, std::fabs(z_term[9 + j]));
        dqm_bar = std::max(dqm_bar, std::fabs(dqbar[j]));
        dqjump = std::max(dqjump, std::fabs(z_term[9 + j] - dqbar[j]));
      }
      std::fprintf(stderr, "[wm] it=%d trust=%.2f pos_r=%.4f vel_r=%.4f linv=(%.3f,%.3f,%.3f) |dq|=%.1f dqjump=%.2f sv=(%.3f,%.3f,%.3f) cost=%.4g\n",
                   it, trust, pos_r, vel_r, linv[0], linv[1], linv[2], dqm, dqjump,
                   zfull[iS(3)], zfull[iS(4)], zfull[iS(5)], cost);
    }
    if (pos_r + 0.2 * vel_r < best_res) {
      best_res = pos_r + 0.2 * vel_r;
      std::copy(zfull.begin(), zfull.end(), zbest.begin());
      std::copy(z_term, z_term + kNXs, best_term);
      for (int j = 0; j < 5; ++j) { qbar[j] = z_term[4 + j]; dqbar[j] = z_term[9 + j]; }
      trust = std::min(trust * 1.4, lim_.trust_max);
      if (pos_r < lim_.converge_pos) break;   // 收敛
    } else {
      trust *= 0.5;
      if (trust < lim_.trust_min) break;   // 无法再改善,取 best
      for (int j = 0; j < 5; ++j) {   // 迭代点回退 best,小步重试
        qbar[j] = best_term[4 + j];
        dqbar[j] = best_term[9 + j];
      }
    }
  }
  const double ms = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - tc0).count() * 1e3;
  last_ms=ms;
  if (!solved) { last_reason="qp_or_sqp_failed";++fails; return false; }
  // 质量门槛:best 迭代对真值流形的残差必须过线,否则按拒收(调用方保持
  // 上一计划)。best-effort 解的终端可在流形外任意远(松弛罚有限),执行它
  // = 底盘奔向臂不可达点(闭环 1m 级 miss 定位,2026-10-09)
  {
    double pr0, vr0;
    TermResidual(ik_, best_term, best_term + 2, best_term + 4, best_term + 9,
                 req.p_hit, req.v_star, &pr0, &vr0);
    last_pos_res=pr0;
    if (pr0 > lim_.accept_pos) {
      last_reason="position_residual";
      ++rejects;
      if (getenv("WM_DEBUG"))
        std::fprintf(stderr, "[wm] reject: pos_res=%.3f\n", pr0);
      return false;
    }
  }
  if (ms > (has_prev_ ? lim_.ms_budget : lim_.ms_budget_first)) {
    last_reason="solve_time_budget";
    ++timeouts; ++rejects;
    if (getenv("WM_DEBUG")) std::fprintf(stderr, "[wm] reject: ms=%.1f\n", ms);
    return false;
  }
  if (has_prev_) {
    const double dp = std::hypot(req.p_hit[0] - prev_p_hit_[0],
                                 req.p_hit[1] - prev_p_hit_[1],
                                 req.p_hit[2] - prev_p_hit_[2]);
    const double jump = std::hypot(best_term[0] - prev_term_[0],
                                   best_term[1] - prev_term_[1]);
    if (dp < 0.05 && jump > lim_.jump_reject) {
      last_reason="base_jump";
      ++rejects;
      if (getenv("WM_DEBUG")) std::fprintf(stderr, "[wm] reject: jump=%.2f\n", jump);
      return false;
    }
  }

  // 输出(best 迭代点)
  last_reason="accepted";
  out->ok = true;
  out->contact_time=req.t_now+kc*h;
  out->t0 = req.t_now;
  out->h = h;
  out->term_base[0] = best_term[0]; out->term_base[1] = best_term[1];
  out->term_vbase[0] = best_term[2]; out->term_vbase[1] = best_term[3];
  for (int j = 0; j < 5; ++j) {
    out->term_q[j] = best_term[4 + j];
    out->term_dq[j] = best_term[9 + j];
  }
  for (int i = 0; i < 3; ++i) {
    out->slack_p[i] = zbest[iS(i)];
    out->slack_v[i] = zbest[iS(3 + i)];
  }
  double pr, vr;
  TermResidual(ik_, out->term_base, out->term_vbase, out->term_q, out->term_dq,
               req.p_hit, req.v_star, &pr, &vr);
  out->pos_res = pr;
  out->vel_res = vr;
  {
    double nbar[3];
    ik_.FaceNormal(out->term_q, 0, 0, nbar);
    out->n_dot = nbar[0] * req.n_star[0] + nbar[1] * req.n_star[1] +
                 nbar[2] * req.n_star[2];
  }
  out->cost = cost;
  out->ms = ms;
  out->sqp_iters = iters;
  out->rx.assign(kN + 1, 0.0); out->ry.assign(kN + 1, 0.0);
  out->rvx.assign(kN + 1, 0.0); out->rvy.assign(kN + 1, 0.0);
  out->rq.assign(5 * (kN + 1), 0.0);
  out->rdq.assign(5 * (kN + 1), 0.0);
  out->rddq.assign(5 * kN, 0.0);
  for (int k = 0; k <= kN; ++k) {
    out->rx[k] = zbest[iX(k, 0)];
    out->ry[k] = zbest[iX(k, 1)];
    out->rvx[k] = zbest[iX(k, 2)];
    out->rvy[k] = zbest[iX(k, 3)];
    for (int j = 0; j < 5; ++j) {
      out->rq[j * (kN + 1) + k] = zbest[iX(k, 4 + j)];
      out->rdq[j * (kN + 1) + k] = zbest[iX(k, 9 + j)];
    }
  }
  for (int k = 0; k < kN; ++k)
    for (int j = 0; j < 5; ++j)
      out->rddq[j * kN + k] = zbest[iU(k, 2 + j)];
  for (int j = 0; j < 5; ++j) {
    q_prev_[j] = out->term_q[j];
    dq_prev_[j] = out->term_dq[j];
  }
  trust_warm_ = trust;   // 下一拍从本拍终值信任域起步(跳变守门在下游)
  prev_term_[0] = out->term_base[0];
  prev_term_[1] = out->term_base[1];
  for (int i = 0; i < 3; ++i) prev_p_hit_[i] = req.p_hit[i];
  has_prev_ = true;
  ++solves;
  ms_sum += ms;
  ms_max = std::max(ms_max, ms);
  return true;
}

}  // namespace sckd
