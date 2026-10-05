#include "control/arm_ik.h"
#include <cmath>
#include <algorithm>

namespace sckd {

ArmIk::ArmIk(mjModel* m, mjData* d) : m_(m), d_(d) {
  site_id_ = mj_name2id(m, mjOBJ_SITE, "racket_center");
  for (int j = 0; j < 5; ++j) {
    qadr_[j] = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, kAxisJointName[AxisArm(j)])];
    vadr_[j] = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, kAxisJointName[AxisArm(j)])];
  }
}

IkResult ArmIk::Solve(const double target[3], double base_x, double base_y,
                      double base_yaw, const double q_seed[5], int max_iter) {
  IkResult r;
  for (int j = 0; j < 5; ++j) {
    r.q[j] = q_seed[j];
    d_->qpos[qadr_[j]] = q_seed[j];
  }
  d_->qpos[0] = base_x; d_->qpos[1] = base_y; d_->qpos[2] = 0.209;
  d_->qpos[3] = std::cos(base_yaw*0.5); d_->qpos[4] = std::sin(base_yaw*0.5);
  d_->qpos[5] = 0; d_->qpos[6] = 0;
  mj_forward(m_, d_);
  double best_err = 1e9; double best_q[5];
  std::vector<double> jacp(3 * m_->nv), jacr(3 * m_->nv);
  for (int it = 0; it < max_iter; ++it) {
    mj_kinematics(m_, d_);
    double x[3] = {0,0,0};
    mju_rotVecQuat(x, m_->site_pos + site_id_*3, d_->xquat + m_->site_bodyid[site_id_]*4);
    const double* sp = d_->xpos + m_->site_bodyid[site_id_]*3;
    double err[3] = {target[0]-sp[0]-x[0], target[1]-sp[1]-x[1], target[2]-sp[2]-x[2]};
    double J[3][5] = {{0}};
    mj_jacSite(m_, d_, jacp.data(), jacr.data(), site_id_);
    for (int c = 0; c < 5; ++c) {
      J[0][c] = jacp[0*m_->nv + vadr_[c]];
      J[1][c] = jacp[1*m_->nv + vadr_[c]];
      J[2][c] = jacp[2*m_->nv + vadr_[c]];
    }
    double A[3][3] = {{0}}; const double lam = 0.05;
    for (int i=0;i<3;++i) for (int k=0;k<3;++k) for (int c=0;c<5;++c) A[i][k] += J[i][c]*J[k][c];
    for (int i=0;i<3;++i) A[i][i] += lam*lam;
    double det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
               - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
               + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    if (std::fabs(det) < 1e-12) break;
    double Ai[3][3];
    Ai[0][0]=(A[1][1]*A[2][2]-A[1][2]*A[2][1])/det; Ai[0][1]=-(A[0][1]*A[2][2]-A[0][2]*A[2][1])/det; Ai[0][2]=(A[0][1]*A[1][2]-A[0][2]*A[1][1])/det;
    Ai[1][0]=-(A[1][0]*A[2][2]-A[1][2]*A[2][0])/det; Ai[1][1]=(A[0][0]*A[2][2]-A[0][2]*A[2][0])/det; Ai[1][2]=-(A[0][0]*A[1][2]-A[0][2]*A[1][0])/det;
    Ai[2][0]=(A[1][0]*A[2][1]-A[1][1]*A[2][0])/det; Ai[2][1]=-(A[0][0]*A[2][1]-A[0][1]*A[2][0])/det; Ai[2][2]=(A[0][0]*A[1][1]-A[0][1]*A[1][0])/det;
    double dq[5] = {0};
    for (int c=0;c<5;++c) for (int i=0;i<3;++i) for (int k=0;k<3;++k) dq[c] += J[k][c]*Ai[k][i]*err[i];
    for (int j=0;j<5;++j) dq[j] = std::clamp(dq[j], -0.05, 0.05);
    for (int j=0;j<5;++j) r.q[j] += dq[j];
    r.q[1]=std::clamp(r.q[1],-3.14,0.5); r.q[2]=std::clamp(r.q[2],-3.14,0.5);
    const double e2 = std::sqrt(err[0]*err[0]+err[1]*err[1]+err[2]*err[2]);
    if (e2 < best_err) { best_err=e2; for(int j=0;j<5;++j) best_q[j]=r.q[j]; }
    if (best_err < 0.003) break;
    for (int j=0;j<5;++j) d_->qpos[qadr_[j]] = r.q[j];
    mj_forward(m_, d_);
  }
  for (int j=0;j<5;++j) r.q[j] = best_q[j];
  r.err = best_err;
  r.converged = best_err < 0.01;
  return r;
}

}  // namespace sckd
