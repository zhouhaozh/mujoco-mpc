#include "control/wholebody_ocp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include <osqp/osqp.h>

namespace sckd {
namespace {

constexpr int kNXs = 4;    // [x,y,vx,vy]
constexpr int kNUs = 2;    // [ax,ay]
constexpr int kN = WholebodyOcp::kN;
constexpr int kNVar = kNXs * (kN + 1) + kNUs * kN;       // 76
constexpr int kRowX0 = 0;
constexpr int kRowDyn = kNXs;                            // 4·N 行
constexpr int kRowBox = kNXs + kNXs * kN;                // 76 行(单位块)
constexpr int kMRows = kRowBox + kNVar;

inline int iX(int k, int j) { return k * kNXs + j; }
inline int iU(int k, int j) { return kNXs * (kN + 1) + k * kNUs + j; }

// A 条目枚举(模式与值同源生成,杜绝双维护漂移;emit(row,col,val))
template <typename Emit>
void EnumerateA(double h, Emit&& emit) {
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
  }
  for (int c = 0; c < kNVar; ++c) emit(kRowBox + c, c, 1.0);
}

// P(上三角)条目:控制能量 + 终端位置/速度代价(全对角,全凸)
template <typename Emit>
void EnumerateP(double h, double a2, double w_anchor, double w_vel, Emit&& emit) {
  for (int k = 0; k < kN; ++k) {
    emit(iU(k, 0), iU(k, 0), 2.0 * h / a2);
    emit(iU(k, 1), iU(k, 1), 2.0 * h / a2);
  }
  emit(iX(kN, 0), iX(kN, 0), 2.0 * w_anchor);
  emit(iX(kN, 1), iX(kN, 1), 2.0 * w_anchor);
  emit(iX(kN, 2), iX(kN, 2), 2.0 * w_vel);
  emit(iX(kN, 3), iX(kN, 3), 2.0 * w_vel);
}

// 模式排序序:逻辑条目按 (col,row) 字典序 → CSC;值填充序与此一致
// (Ax[s] = tmp[aord[s]],aord[s] = 第 s 小条目的逻辑号)
std::vector<int> BuildOrder(bool p_mat) {
  std::vector<std::pair<int, int>> e;
  if (p_mat)
    EnumerateP(0.05, 1, 1, 1, [&](int r, int c, double) { e.emplace_back(c, r); });
  else
    EnumerateA(0.05, [&](int r, int c, double) { e.emplace_back(c, r); });
  std::vector<int> ord(e.size());
  for (int i = 0; i < (int)ord.size(); ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) { return e[a] < e[b]; });
  return ord;
}

// vendored OSQP 上下文(模式一次 setup,值全量替换;进程级单例,单线程口径
// 与 arm_swing Qp() 一致)
struct WboQp {
  OSQPSolver* solver = nullptr;
  std::vector<OSQPInt> Ap, Ai, Pp, Pi;
  std::vector<OSQPFloat> Ax, Px, q_lin, l, u;
  std::vector<int> aord, pord;
  bool ok = false;

  WboQp() {
    aord = BuildOrder(false);
    pord = BuildOrder(true);
    std::vector<std::pair<int, int>> ae, pe;
    EnumerateA(0.05, [&](int r, int c, double) { ae.emplace_back(c, r); });
    EnumerateP(0.05, 1, 1, 1, [&](int r, int c, double) { pe.emplace_back(c, r); });

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
    st.eps_abs = 1e-5;
    st.eps_rel = 1e-5;
    st.max_iter = 8000;
    st.polishing = 1;
    if (osqp_setup(&solver, &Pc, q_lin.data(), &Ac, l.data(), u.data(),
                   kMRows, kNVar, &st) != 0) {
      solver = nullptr;
      return;
    }
    ok = true;
  }
};

WboQp& Qp() {
  static WboQp ctx;
  return ctx;
}

}  // namespace

bool WholebodyOcp::Solve(const WboRequest& req, WboResult* out) {
  const double T = req.t_hit - req.t_now;
  if (T < 0.15 || T > 3.0) return false;   // 末段窗冻结 / 异常时窗拒收
  const double h = T / kN;
  WboQp& qp = Qp();
  if (!qp.ok) return false;
  const auto tc0 = std::chrono::steady_clock::now();

  // anchor*:离线联合 OCP 最优后撤,按击球高插值(低位 0.45 ← v2sweep 联合
  // 最优;高位 0.75 ← 钉 0.45 在 z≥1.1 的拍心松弛反比 0.8 差 5cm,臂不在
  // 本优化内的补偿)。距离过近(D≤0.3)无后撤语义,锚=当前位置
  const double back = lim_.back_star +
      (lim_.back_star_hi - lim_.back_star) *
          std::clamp((req.p_hit[2] - lim_.z_lo) / (lim_.z_hi - lim_.z_lo), 0.0, 1.0);
  const double dx = req.p_hit[0] - req.base0[0], dy = req.p_hit[1] - req.base0[1];
  const double D = std::hypot(dx, dy);
  const double anchor[2] = {
      (D > 0.30) ? req.p_hit[0] - back * dx / D : req.base0[0],
      (D > 0.30) ? req.p_hit[1] - back * dy / D : req.base0[1]};

  // X0 rhs(实测,钳入界)
  const double x0r[kNXs] = {
      req.base0[0], req.base0[1],
      std::clamp(req.vbase0[0], -lim_.v_axis, lim_.v_axis),
      std::clamp(req.vbase0[1], -lim_.v_axis, lim_.v_axis)};

  // 行界:X0/动力学等式 + 变量界
  for (int j = 0; j < kNXs; ++j) qp.l[kRowX0 + j] = qp.u[kRowX0 + j] = x0r[j];
  for (int r = kRowDyn; r < kRowBox; ++r) qp.l[r] = qp.u[r] = 0.0;
  for (int k = 0; k <= kN; ++k) {
    qp.l[kRowBox + iX(k, 0)] = -OSQP_INFTY; qp.u[kRowBox + iX(k, 0)] = OSQP_INFTY;
    qp.l[kRowBox + iX(k, 1)] = -OSQP_INFTY; qp.u[kRowBox + iX(k, 1)] = OSQP_INFTY;
    qp.l[kRowBox + iX(k, 2)] = -lim_.v_axis; qp.u[kRowBox + iX(k, 2)] = lim_.v_axis;
    qp.l[kRowBox + iX(k, 3)] = -lim_.v_axis; qp.u[kRowBox + iX(k, 3)] = lim_.v_axis;
  }
  for (int k = 0; k < kN; ++k)
    for (int j = 0; j < 2; ++j) {
      qp.l[kRowBox + iU(k, j)] = -lim_.a_max;
      qp.u[kRowBox + iU(k, j)] = lim_.a_max;
    }

  // 值:A(动力学 h)+ P + 线性项(终端代价交叉项 −2w·target)
  {
    static std::vector<double> tmp;
    tmp.resize(qp.aord.size());
    int idx = 0;
    EnumerateA(h, [&](int, int, double v) { tmp[idx++] = v; });
    for (int s = 0; s < (int)qp.aord.size(); ++s) qp.Ax[s] = tmp[qp.aord[s]];
  }
  {
    static std::vector<double> tmp;
    tmp.resize(qp.pord.size());
    int idx = 0;
    EnumerateP(h, lim_.a_max * lim_.a_max, lim_.w_anchor, lim_.w_vel,
               [&](int, int, double v) { tmp[idx++] = v; });
    for (int s = 0; s < (int)qp.pord.size(); ++s) qp.Px[s] = tmp[qp.pord[s]];
    std::fill(qp.q_lin.begin(), qp.q_lin.end(), 0.0);
    qp.q_lin[iX(kN, 0)] = -2.0 * lim_.w_anchor * anchor[0];
    qp.q_lin[iX(kN, 1)] = -2.0 * lim_.w_anchor * anchor[1];
    qp.q_lin[iX(kN, 2)] = -2.0 * lim_.w_vel * req.v_star[0];
    qp.q_lin[iX(kN, 3)] = -2.0 * lim_.w_vel * req.v_star[1];
  }
  if (osqp_update_data_mat(qp.solver, qp.Px.data(), OSQP_NULL, (OSQPInt)qp.Px.size(),
                           qp.Ax.data(), OSQP_NULL, (OSQPInt)qp.Ax.size()) != 0 ||
      osqp_update_data_vec(qp.solver, qp.q_lin.data(), qp.l.data(), qp.u.data()) != 0) {
    ++fails;
    return false;
  }
  if (osqp_solve(qp.solver) != 0) { ++fails; return false; }
  const OSQPInt stv = qp.solver->info->status_val;
  if (stv != OSQP_SOLVED && stv != OSQP_SOLVED_INACCURATE) {
    ++fails;
    if (getenv("WBO_DEBUG")) std::fprintf(stderr, "[wbo] status=%d\n", (int)stv);
    return false;
  }
  const OSQPFloat* z = qp.solver->solution->x;
  const double ms = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - tc0).count() * 1e3;

  const double tb[2] = {z[iX(kN, 0)], z[iX(kN, 1)]};
  // 守门:解时窗(首拍冷解另给预算)
  if (ms > (has_prev_ ? lim_.ms_budget : lim_.ms_budget_first)) {
    ++timeouts; ++rejects;
    if (getenv("WBO_DEBUG")) std::fprintf(stderr, "[wbo] reject: ms=%.2f\n", ms);
    return false;
  }
  // 守门:终端跳变拒收(p_hit 未动而终端底盘跳 > 阈;p_hit 精化 >5cm 放行)
  if (has_prev_) {
    const double dp = std::hypot(req.p_hit[0] - prev_p_hit_[0],
                                 req.p_hit[1] - prev_p_hit_[1],
                                 req.p_hit[2] - prev_p_hit_[2]);
    const double jump = std::hypot(tb[0] - prev_term_[0], tb[1] - prev_term_[1]);
    if (dp < 0.05 && jump > lim_.jump_reject) {
      ++rejects;
      if (getenv("WBO_DEBUG"))
        std::fprintf(stderr, "[wbo] reject: jump=%.3f\n", jump);
      return false;
    }
  }

  out->ok = true;
  out->term_base[0] = tb[0];
  out->term_base[1] = tb[1];
  out->term_vbase[0] = z[iX(kN, 2)];
  out->term_vbase[1] = z[iX(kN, 3)];
  out->cost = qp.solver->info->obj_val;
  out->ms = ms;
  out->t0 = req.t_now;
  out->h = h;
  out->rx.assign(kN + 1, 0.0);
  out->ry.assign(kN + 1, 0.0);
  out->rvx.assign(kN + 1, 0.0);
  out->rvy.assign(kN + 1, 0.0);
  for (int k = 0; k <= kN; ++k) {
    out->rx[k] = z[iX(k, 0)];
    out->ry[k] = z[iX(k, 1)];
    out->rvx[k] = z[iX(k, 2)];
    out->rvy[k] = z[iX(k, 3)];
  }
  prev_term_[0] = tb[0];
  prev_term_[1] = tb[1];
  for (int i = 0; i < 3; ++i) prev_p_hit_[i] = req.p_hit[i];
  has_prev_ = true;
  ++solves;
  ms_sum += ms;
  ms_max = std::max(ms_max, ms);
  return true;
}

}  // namespace sckd
