#include "estimation/ball_predictor.h"

#include <cmath>

namespace sckd {

// ---------- BallPredictor:物理 EKF ----------

BallPredictor::BallPredictor(const BallAeroParams& a, double obs_noise_std,
                             const BounceParams& bp)
    : a_(a), bp_(bp), r_std_(obs_noise_std) {}

void BallPredictor::Reset(const BallObs& o) {
  for (int i = 0; i < 3; ++i) {
    x_[i] = o.p[i];
    x_[3 + i] = 0.0;
  }
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) P_[i][j] = 0.0;
  for (int i = 0; i < 3; ++i) P_[i][i] = 1e-4;
  for (int i = 3; i < 6; ++i) P_[i][i] = 100.0;  // v 未知(10 m/s std,过大致增益发散)
  t_last_ = o.t;
  t_last_bounce_ = -10.0;
  init_ = true;
}

namespace {
// 单步传播(共享给 Propagate 与 Predict):含反弹事件;返回是否发生反弹
bool PropagateState(const BallAeroParams& a, const BounceParams& bp,
                    double x[6], double t_now, double h, double* t_bounce) {
  double v[3] = {x[3], x[4], x[5]}, acc[3];
  BallAeroAccelNoSpin(a, v, acc);
  for (int i = 0; i < 3; ++i) {
    x[i] += h * x[3 + i];
    x[3 + i] += h * acc[i];
  }
  if (x[2] < a.radius && x[5] < 0 && (t_now - *t_bounce) > bp.min_interval_s) {
    x[2] = a.radius;
    x[5] = -x[5] * bp.restitution_z;
    x[3] *= bp.friction_xy;
    x[4] *= bp.friction_xy;
    *t_bounce = t_now;
    return true;
  }
  return false;
}
}  // namespace

void BallPredictor::Propagate(double dt) {
  if (dt <= 0) return;
  const int n = std::max(1, static_cast<int>(std::ceil(dt / 0.002)));
  const double h = dt / n;
  const double area = M_PI * a_.radius * a_.radius;
  const double kd = 0.5 * a_.rho * area * a_.Cd / a_.mass;
  for (int s = 0; s < n; ++s) {
    const double t_step = t_last_ + (s + 1) * h;
    const bool bounced = PropagateState(a_, bp_, x_, t_step, h, &t_last_bounce_);
    // 雅可比(F = I + h·A)传播协方差
    double v[3] = {x_[3], x_[4], x_[5]};
    const double vm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    double A[6][6] = {{0}};
    A[0][3] = A[1][4] = A[2][5] = 1.0;
    if (vm > 1e-6) {
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          A[3 + i][3 + j] = -kd * ((i == j ? vm : 0.0) + v[i] * v[j] / vm);
    }
    double F[6][6], tmp[6][6];
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) F[i][j] = (i == j ? 1.0 : 0.0) + h * A[i][j];
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) {
        double s2 = 0;
        for (int k = 0; k < 6; ++k) s2 += F[i][k] * P_[k][j];
        tmp[i][j] = s2;
      }
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) {
        double s2 = 0;
        for (int k = 0; k < 6; ++k) s2 += tmp[i][k] * F[j][k];
        P_[i][j] = s2;
      }
    // 过程噪声(物理模型主导,取小);反弹时放大
    const double qv = 2.0, qa = 100.0;
    for (int i = 0; i < 3; ++i) {
      P_[i][i] += qv * h;
      P_[3 + i][3 + i] += qa * h * (bounced ? 20.0 : 1.0);
    }
  }
}

void BallPredictor::Update(const BallObs& o) {
  if (!init_) {
    Reset(o);
    return;
  }
  Propagate(o.t - t_last_);
  t_last_ = o.t;
  for (int i = 0; i < 3; ++i) {   // 位置观测,H=[I 0]
    const double s = P_[i][i] + r_std_ * r_std_;
    double K[6];
    for (int r = 0; r < 6; ++r) K[r] = P_[r][i] / s;
    const double y = o.p[i] - x_[i];
    for (int r = 0; r < 6; ++r) x_[r] += K[r] * y;
    double prow[6];  // 原始行 i(防原地覆盖)
    for (int c = 0; c < 6; ++c) prow[c] = P_[i][c];
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c) P_[r][c] -= K[r] * prow[c];
  }
}

bool BallPredictor::PredictFull(double t_query, double p_out[3], double v_out[3]) const {
  if (!init_ || t_query < t_last_) return false;
  double x[6] = {x_[0], x_[1], x_[2], x_[3], x_[4], x_[5]};
  const double dt = t_query - t_last_;
  const int n = std::max(1, static_cast<int>(std::ceil(dt / 0.002)));
  const double h = dt / n;
  double tb = t_last_bounce_;
  for (int s = 0; s < n; ++s)
    PropagateState(a_, bp_, x, t_last_ + (s + 1) * h, h, &tb);
  for (int i = 0; i < 3; ++i) {
    p_out[i] = x[i];
    v_out[i] = x[3 + i];
  }
  return true;
}

bool BallPredictor::Predict(double t_query, double p_out[3]) const {
  double v[3];
  return PredictFull(t_query, p_out, v);
}

bool BallPredictor::Intercept(const double base[3], double t_now, double zlo, double zhi,
                              double reach, double p_hit[3], double* t_hit) const {
  for (double t = t_now + 0.15; t < t_now + 2.0; t += 0.02) {
    double p[3];
    if (!Predict(t, p)) return false;
    if (p[2] >= zlo && p[2] <= zhi) {
      const double dx = p[0] - base[0], dy = p[1] - base[1];
      if (dx * dx + dy * dy <= reach * reach) {
        for (int i = 0; i < 3; ++i) p_hit[i] = p[i];
        *t_hit = t;
        return true;
      }
    }
  }
  return false;
}

// ---------- CaAxisFilter:逐轴 CA(trobot 基线)----------

CaAxisFilter::CaAxisFilter(double meas_sigma, double jerk_q)
    : meas_sigma_(meas_sigma), jerk_q_(jerk_q) {}

void CaAxisFilter::Reset(const BallObs& o) {
  for (int i = 0; i < 3; ++i) {
    ax_[i] = Axis{};
    ax_[i].p = o.p[i];
    ax_[i].P[0][0] = 1e-4;
    ax_[i].P[1][1] = 100.0;
    ax_[i].P[2][2] = 100.0;
  }
  t_last_ = o.t;
  init_ = true;
}

void CaAxisFilter::PredictAxis(Axis& ax, double dt) const {
  const double q = jerk_q_ * 1e-6;  // (mm/s³)² -> m 制
  const double dt2 = dt * dt;
  const double p0 = ax.p, v0 = ax.v;
  ax.p = p0 + v0 * dt + 0.5 * ax.acc * dt2;
  ax.v = v0 + ax.acc * dt;
  const double Q[3][3] = {{q * dt2 * dt2 * dt / 4.0, q * dt2 * dt / 2.0, q * dt2 / 2.0},
                          {q * dt2 * dt / 2.0, q * dt2, q * dt},
                          {q * dt2 / 2.0, q * dt, q}};
  const double F[3][3] = {{1, dt, 0.5 * dt2}, {0, 1, dt}, {0, 0, 1}};
  double tmp[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += F[i][k] * ax.P[k][j];
      tmp[i][j] = s;
    }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double s = 0;
      for (int k = 0; k < 3; ++k) s += tmp[i][k] * F[j][k];
      ax.P[i][j] = s + Q[i][j];
    }
}

void CaAxisFilter::Update(const BallObs& o) {
  if (!init_) {
    Reset(o);
    return;
  }
  const double dt = o.t - t_last_;
  for (int i = 0; i < 3; ++i) PredictAxis(ax_[i], dt);
  t_last_ = o.t;
  const double r2 = meas_sigma_ * meas_sigma_;
  for (int a = 0; a < 3; ++a) {
    Axis& st = ax_[a];
    const double s = st.P[0][0] + r2;
    const double K[3] = {st.P[0][0] / s, st.P[1][0] / s, st.P[2][0] / s};
    const double y = o.p[a] - st.p;
    st.p += K[0] * y;
    st.v += K[1] * y;
    st.acc += K[2] * y;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) st.P[i][j] -= K[i] * st.P[0][j];
  }
}

bool CaAxisFilter::Predict(double t_query, double p_out[3]) const {
  if (!init_ || t_query < t_last_) return false;
  const double dt = t_query - t_last_;
  for (int i = 0; i < 3; ++i)
    p_out[i] = ax_[i].p + ax_[i].v * dt + 0.5 * ax_[i].acc * dt * dt;
  return true;
}

}  // namespace sckd
