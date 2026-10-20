#include "control/chassis_mpc.h"

namespace sckd {

ChassisMpc::ChassisMpc() : mem_(chassis_mpc_checkout()) {
  // 约束行边界(g 的语义:表达式 ≤ 0 或 = 0;边界经 lbg/ubg 传入):
  //   行 0..6(N+1)-1   动力学等式        -> lb=ub=0
  //   行 之后全部       不等式(expr≤0)   -> lb=-inf, ub=0
  // 变量界 lbx/ubx = ±inf(无)。此前误把不等式行也设 ±inf -> 约束全部失效(wz 冲到 6)
  for (int i = 0; i < 6 * (kN + 1); ++i) lbg_[i] = ubg_[i] = 0.0;
  for (int i = 6 * (kN + 1); i < kNg; ++i) {
    lbg_[i] = -1e20;
    ubg_[i] = 0.0;
  }
}

ChassisMpc::~ChassisMpc() {
  if (mem_ >= 0) chassis_mpc_release(mem_);
}

void ChassisMpc::PackParams(double* p, const double x0[kNx],
                            const std::array<double, kNx * (kN + 1)>& ref,
                            const double* cossin_2N, const Limits& lim) {
  int o = 0;
  for (int i = 0; i < kNx; ++i) p[o++] = x0[i];
  for (int i = 0; i < kNx * (kN + 1); ++i) p[o++] = ref[i];
  for (int i = 0; i < 2 * kN; ++i) p[o++] = cossin_2N[i];
  p[o++] = lim.a_max;
  p[o++] = lim.v_max;
  p[o++] = lim.alpha_max;
  p[o++] = lim.wz_max;
  p[o++] = lim.theta_max;
}

bool ChassisMpc::Solve(const double* p) {
  for (int i = 0; i < kNp; ++i) p_[i] = p[i];
  // qpsol 接口:n_in=8(x0,p,lbx,ubx,lbg,ubg,lam_x0,lam_g0)/n_out=6(x,f,g,lam_x,lam_g,lam_p);
  // 但 codegen 工作区要求 arg=20/res=12 槽(含内部暂存),按 work_bytes 分配
  const double* arg[kArg] = {};
  double* res[kRes] = {};
  arg[0] = z_.data();      // x0(warm start:上一次解)
  arg[1] = p_.data();      // 参数
  arg[2] = lbx_.data();    // lbx/ubx/lbg/ubg 显式(casadi 视 nullptr 为缺输入)
  arg[3] = ubx_.data();
  arg[4] = lbg_.data();
  arg[5] = ubg_.data();
  double cost = 0.0;
  res[0] = z_.data();      // 解 x
  res[1] = &cost;          // 目标值
  const int flag = chassis_mpc(arg, res, nullptr, w_.data(), mem_);
  return flag == 0;
}

}  // namespace sckd
