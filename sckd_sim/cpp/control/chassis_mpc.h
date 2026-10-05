#pragma once
// 底盘 MPC:封装 python/tools/gen_chassis_mpc.py 生成的 CasADi qpsol(OSQP) C 代码。
// 运行时零 casadi/python 依赖;依赖 vendored OSQP 1.0.0。
// 问题:N=100 步(dt=20ms,视界 2s),x=[px,py,ψ,vx,vy,wz],u=[ax,ay,α];
//      摩擦圆多边形约束(§6.2)+ 航向窗;参数 p=[x0(6), ref(606), (c0,s0), lim(5)]。
#include <array>
#include <vector>

extern "C" {
int chassis_mpc(const double** arg, double** res, long long* iw, double* w, int mem);
int chassis_mpc_checkout();
int chassis_mpc_release(int mem);
}

namespace sckd {

class ChassisMpc {
 public:
  static constexpr int kNx = 6;          // [px,py,ψ,vx,vy,wz]
  static constexpr int kNu = 3;          // [ax,ay,α]
  static constexpr int kN = 100;         // 视界步数
  static constexpr int kNz = kNx * (kN + 1) + kNu * kN;     // 906
  static constexpr int kNp = kNx + kNx * (kN + 1) + 2 * kN + 5;  // 817(含逐步 cos/sin)
  static constexpr int kW = 26963;       // 生成代码工作区(double,随重生成更新)
  static constexpr int kArg = 20;        // codegen work_bytes:arg/res 槽位数
  static constexpr int kRes = 12;        //(n_in/n_out 之外含内部暂存槽,勿按 8/6 分配)
  static constexpr int kNg = 6 * (kN + 1) + kN * (8 + 1) + (kN + 1) * (8 + 2);  // 2516 约束行

  struct Limits {
    double a_max = 8.0;        // 摩擦圆(μ·g×0.8 工程裕度,§6.2)
    double v_max = 4.0;
    double alpha_max = 20.0;   // rad/s²
    double wz_max = 1.5708;
    double theta_max = 0.5236; // 航向窗 ±30°(QR 可见性占位,§14-8)
  };

  ChassisMpc();
  ~ChassisMpc();
  ChassisMpc(const ChassisMpc&) = delete;
  ChassisMpc& operator=(const ChassisMpc&) = delete;

  // 求解;p 布局 [x0(6) | ref((N+1)×6) | c0,s0 | lim(5)];返回是否得到解
  bool Solve(const double* p);

  // 解访问:状态 k(0..N)/输入 k(0..N-1)
  const double* x_at(int k) const { return z_.data() + k * kNx; }
  const double* u_at(int k) const { return z_.data() + kNx * (kN + 1) + k * kNu; }
  // 下发伺服的 twist = 状态 k=1 的速度分量
  void NextTwist(double* vx, double* vy, double* wz) const {
    *vx = x_at(1)[3];
    *vy = x_at(1)[4];
    *wz = x_at(1)[5];
  }

  // 参数打包辅助
  // cos/sin 序列(2N)按参考航向逐步填充(调用方从 ref[k].ψ 计算)
  static void PackParams(double* p, const double x0[kNx],
                         const std::array<double, kNx * (kN + 1)>& ref,
                         const double* cossin_2N, const Limits& lim);

 private:
  int mem_ = -1;
  std::vector<double> z_{std::vector<double>(kNz, 0.0)};   // 解兼 warm start 初值
  std::vector<double> p_{std::vector<double>(kNp, 0.0)};
  std::vector<double> w_{std::vector<double>(kW, 0.0)};
  std::vector<double> lbx_{std::vector<double>(kNz, -1e20)};
  std::vector<double> ubx_{std::vector<double>(kNz, 1e20)};
  std::vector<double> lbg_{std::vector<double>(kNg, -1e20)};
  std::vector<double> ubg_{std::vector<double>(kNg, 1e20)};
};

}  // namespace sckd
