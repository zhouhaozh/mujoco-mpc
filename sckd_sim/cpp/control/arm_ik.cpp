#include "control/arm_ik.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace sckd {

ArmIk::ArmIk(mjModel* m, mjData* d) : m_(m), d_(d) {
  site_id_ = mj_name2id(m, mjOBJ_SITE, "racket_center");
  face_id_ = mj_name2id(m, mjOBJ_GEOM, "racket_face");
  const int base = mj_name2id(m, mjOBJ_BODY, "base_link");
  if (site_id_ < 0 || face_id_ < 0 || base < 0 ||
      m->geom_type[face_id_] != mjGEOM_BOX || m->body_jntnum[base] != 1 ||
      m->jnt_type[m->body_jntadr[base]] != mjJNT_FREE)
    throw std::invalid_argument("ArmIk requires base_link free joint, racket_center and racket_face box");
  base_qadr_ = m->jnt_qposadr[m->body_jntadr[base]];
  const int home = mj_name2id(m, mjOBJ_KEY, "home");
  const mjtNum* qhome = home >= 0 ? m->key_qpos + home * m->nq : m->qpos0;
  base_height_ = qhome[base_qadr_ + 2];
  const mjtNum* size = m->geom_size + 3 * face_id_;
  face_axis_ = std::min_element(size, size + 3) - size;
  for (int j = 0; j < 5; ++j) {
    const int id = mj_name2id(m, mjOBJ_JOINT, kAxisJointName[AxisArm(j)]);
    if (id < 0 || m->jnt_type[id] != mjJNT_HINGE)
      throw std::invalid_argument("ArmIk requires Joint1..Joint5 hinges");
    joint_id_[j] = id;
    qadr_[j] = m->jnt_qposadr[id];
    vadr_[j] = m->jnt_dofadr[id];
  }
}

void ArmIk::SetBasePose(double x, double y, double yaw) {
  mjtNum* pose = d_->qpos + base_qadr_;
  pose[0] = x; pose[1] = y; pose[2] = base_height_;
  pose[3] = std::cos(yaw * 0.5); pose[4] = 0;
  pose[5] = 0; pose[6] = std::sin(yaw * 0.5);
}

IkResult ArmIk::Solve(const double target[3], double base_x, double base_y,
                      double base_yaw, const double q_seed[5], int max_iter) {
  // X4_950 的 J2/J3 源限位缺失;FK 可用,但不能据此做有界 IK。
  if (!m_->jnt_limited[joint_id_[1]] || !m_->jnt_limited[joint_id_[2]])
    throw std::invalid_argument("ArmIk::Solve requires confirmed Joint2/Joint3 position limits");
  IkResult r;
  for (int j = 0; j < 5; ++j) {
    r.q[j] = q_seed[j];
    d_->qpos[qadr_[j]] = q_seed[j];
  }
  SetBasePose(base_x, base_y, base_yaw);
  mj_forward(m_, d_);
  double best_err = 1e9; double best_q[5];
  std::vector<double> jacp(3 * m_->nv), jacr(3 * m_->nv);
  for (int it = 0; it < max_iter; ++it) {
    mj_kinematics(m_, d_);
    const double* sp = d_->site_xpos + site_id_ * 3;
    double err[3] = {target[0]-sp[0], target[1]-sp[1], target[2]-sp[2]};
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
    for (int j = 0; j < 5; ++j) {
      const int id = joint_id_[j];
      if (m_->jnt_limited[id])
        r.q[j] = std::clamp(r.q[j], m_->jnt_range[2*id], m_->jnt_range[2*id+1]);
    }
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

void ArmIk::FaceNormal(const double q[5], double base_x, double base_y, double n_out[3]) {
  for (int j = 0; j < 5; ++j) d_->qpos[qadr_[j]] = q[j];
  SetBasePose(base_x, base_y, 0);
  mj_kinematics(m_, d_);
  for (int i = 0; i < 3; ++i)
    n_out[i] = d_->geom_xmat[face_id_ * 9 + i * 3 + face_axis_];
}

void ArmIk::SitePos(const double q[5], double base_x, double base_y, double p_out[3]) {
  for (int j = 0; j < 5; ++j) d_->qpos[qadr_[j]] = q[j];
  SetBasePose(base_x, base_y, 0);
  mj_kinematics(m_, d_);
  for (int i = 0; i < 3; ++i) p_out[i] = d_->site_xpos[site_id_ * 3 + i];
}

void ArmIk::SiteJacobian(const double q[5], double base_x, double base_y, double J[15]) {
  for (int j = 0; j < 5; ++j) d_->qpos[qadr_[j]] = q[j];
  SetBasePose(base_x, base_y, 0);
  mj_kinematics(m_, d_);
  mj_comPos(m_, d_);   // mj_jacSite 读 d->cdof(此前依赖调用方跑过 mj_forward
                       // 留下的 cdof,scratch 数据未跑过 IK 时全零 —— 2026-10-08
                       // wholebody_ocp 单测定位;qpos 相同则 cdof 恒等,值不变)
  std::vector<double> jacp(3 * m_->nv), jacr(3 * m_->nv);
  mj_jacSite(m_, d_, jacp.data(), jacr.data(), site_id_);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 5; ++c)
      J[r * 5 + c] = jacp[r * m_->nv + vadr_[c]];
}

bool ArmIk::WithinLimits(const double q[5])const {
  if (!m_->jnt_limited[joint_id_[1]] || !m_->jnt_limited[joint_id_[2]]) return false;
  for (int j = 0; j < 5; ++j) {
    const int id = joint_id_[j];
    if (!std::isfinite(q[j]) || (m_->jnt_limited[id] &&
        (q[j] < m_->jnt_range[2*id]-1e-5 || q[j] > m_->jnt_range[2*id+1]+1e-5)))
      return false;
  }
  return true;
}

}  // namespace sckd
