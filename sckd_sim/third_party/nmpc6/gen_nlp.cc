// nlp.c/nlp.so 生成器(一次性离线工具,2026-10-09):逐式复刻旧仓库 HEAD
// (c0d98ae,网球参数版)set_my_nmpc_solver() 的符号构型,generate_dependencies
// 产出与加载侧 ABI 严格同版的生成码。
//
// 背景:旧仓库根目录残留的 nlp.so(Sep 8)是 26 参数/45 约束版,与 git 里
// 任何版本的 nmpc.cc 都不匹配(09-03 起源码均为 27 参数/87 约束,实机
// Sep 18/19 会话在 c0d98ae 之后运行 → 机上 nlp.so 必为重生成版,本地残留
// 是陈旧副本)。本工具按 HEAD 源码重生成。
//
// 编译运行(在 sckd_sim/ 下):
//   g++ -O2 -std=c++17 -I /home/ubuntu/deps/casadi/include \
//       -L /home/ubuntu/deps/casadi/lib -Wl,-rpath,/home/ubuntu/deps/casadi/lib \
//       third_party/nmpc6/gen_nlp.cc -o /tmp/gen_nlp -lcasadi && /tmp/gen_nlp
//   gcc -fPIC -shared -Ofast third_party/nmpc6/nlp.c -o third_party/nmpc6/nlp.so -lm
// 生成后跑 ctest -R legacy_nmpc(实机日志参数回归)。
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <casadi/casadi.hpp>

using casadi::SX;
using casadi::Slice;

int main() {
  const int n_states = 14, n_controls = 7, m_predict_step = 3;
  const double racket_len = 0.85;

  // ---- 解析 FK(旧 GetEndPos 逐式) ----
  auto GetEndPos = [&](SX thetas, SX add_base_vx) {
    SX d1 = 0.685, a2 = 0.705, d4 = -0.1705, d5 = -racket_len;
    SX c1 = cos(thetas(2, 0)), s1 = sin(thetas(2, 0));
    SX c2 = cos(thetas(3, 0)), s2 = sin(thetas(3, 0));
    SX c23 = cos(thetas(3, 0) + thetas(4, 0)), s23 = sin(thetas(3, 0) + thetas(4, 0));
    SX c4 = cos(thetas(5, 0)), s4 = sin(thetas(5, 0));
    SX c5 = cos(thetas(6, 0)), s5 = sin(thetas(6, 0));
    SX omega1 = thetas(9, 0), omega3 = thetas(11, 0), omega4 = thetas(12, 0);
    SX nx = c1*c23*c4*c5 + s1*s4*c5 - c1*s23*s5;
    SX ny = s1*c23*c4*c5 - c1*s4*c5 - s1*s23*s5;
    SX nz = s23*c4*c5 + c23*s5;
    SX ox = -c1*c23*c4*s5 - s1*s4*s5 - c1*s23*c5;
    SX oy = -s1*c23*c4*s5 + c1*s4*s5 - s1*s23*c5;
    SX oz = -s23*c4*s5 + c23*c5;
    SX ax = -c1*c23*s4 + s1*c4;
    SX ay = -s1*c23*s4 - c1*c4;
    SX az = -s23*s4;
    SX px = -d5*c1*c23*s4 + d5*s1*c4 + d4*c1*s23 + a2*c1*c2;
    SX py = -d5*s1*c23*s4 - d5*c1*c4 + d4*s1*s23 + a2*s1*c2;
    SX pz = -d5*s23*s4 - d4*c23 + a2*s2 + d1;
    SX vx = (d5*s1*c23*s4 + d5*c1*c4 - d4*s1*s23 - a2*s1*c2)*omega1
          + (d5*c1*s23*s4 + d4*c1*c23)*omega3
          + (-d5*c1*c23*c4 - d5*s1*s4)*omega4 + add_base_vx*thetas(7, 0);
    SX vy = (-d5*s1*c23*s4 + d5*s1*c4 + d4*c1*s23 + a2*c1*c2)*omega1
          + (d5*s1*s23*s4 + d4*s1*c23)*omega3
          + (-d5*s1*c23*c4 + d5*c1*s4)*omega4;
    SX vz = (-d5*c23*s4 + d4*s23)*omega3 + (-d5*s23*c4)*omega4;
    SX M = SX::sym("endPosMat", 4, 4);
    M(0,0)=nx; M(0,1)=ox; M(0,2)=ax; M(0,3)=px+thetas(0,0);
    M(1,0)=ny; M(1,1)=oy; M(1,2)=ay; M(1,3)=py+thetas(1,0);
    M(2,0)=nz; M(2,1)=oz; M(2,2)=az; M(2,3)=pz;
    M(3,0)=vx; M(3,1)=vy; M(3,2)=vz; M(3,3)=1;
    return M;
  };

  SX thetas = SX::sym("thetas", n_states, 1);
  SX add_base_vx = SX::sym("add_base_vx", 1, 1);
  SX endpos = GetEndPos(thetas, add_base_vx);
  casadi::Function fk_solver("fk_solver", {thetas, add_base_vx}, {endpos},
                             {"thetas", "add_base_vx"}, {"endpos"});

  SX ACC = SX::sym("ACC", 7, 1);
  ACC(0,0)=1.0/8.0;   // vx_acc
  ACC(1,0)=1.0/8.0;   // vy_acc
  ACC(2,0)=1.0/30.0;  // J1
  ACC(3,0)=1.0/30.0;  // J2
  ACC(4,0)=1.0/180.0; // J3
  ACC(5,0)=1.0/400.0; // J4
  ACC(6,0)=1.0/200.0; // J5

  SX U = SX::sym("U", n_controls, m_predict_step);
  SX X = SX::sym("X", n_states, m_predict_step + 1);
  SX PPPPP = SX::sym("PPPPP", n_states + 13);       // 27(HEAD 版)
  SX opt_var = SX::reshape(U.T(), -1, 1);
  X(Slice(), 0) = PPPPP(Slice(0, n_states, 1));

  const double eps = 1e-4, k = 30.0;
  auto smooth_sign = [&](const SX& x) { return tanh(k * x); };
  auto smooth_fmax = [&](const SX& a, const SX& b) {
    return 0.5 * (a + b + SX::sqrt(SX::pow(a - b, 2) + eps));
  };
  auto smooth_fmin = [&](const SX& a, const SX& b) {
    return 0.5 * (a + b - SX::sqrt(SX::pow(a - b, 2) + eps));
  };

  for (int i = 0; i < m_predict_step; ++i) {
    SX dt = PPPPP(n_states + 3);                     // 动力学:单一 dt(原样)
    X(0, i+1) = (X(7,i)+U(0,i))*dt - (U(0,i)*sqrt(U(0,i)*U(0,i)+eps))/2.0*ACC(0,0) + X(0,i);
    X(1, i+1) = (X(8,i)+U(1,i))*dt - (U(1,i)*sqrt(U(1,i)*U(1,i)+eps))/2.0*ACC(1,0) + X(1,i);
    X(2, i+1) = (X(9,i)+U(2,i))*dt - (U(2,i)*sqrt(U(2,i)*U(2,i)+eps))/2.0*ACC(2,0) + X(2,i);
    X(3, i+1) = (X(10,i)+U(3,i))*dt - (U(3,i)*sqrt(U(3,i)*U(3,i)+eps))/2.0*ACC(3,0) + X(3,i);
    X(4, i+1) = (X(11,i)+U(4,i))*dt - (U(4,i)*sqrt(U(4,i)*U(4,i)+eps))/2.0*ACC(4,0) + X(4,i);
    X(5, i+1) = (X(12,i)+U(5,i))*dt - (U(5,i)*sqrt(U(5,i)*U(5,i)+eps))/2.0*ACC(5,0) + X(5,i);
    X(6, i+1) = (X(13,i)+U(6,i))*dt - (U(6,i)*sqrt(U(6,i)*U(6,i)+eps))/2.0*ACC(6,0) + X(6,i);
    X(7, i+1) = X(7, i) + U(0, i);
    X(8, i+1) = X(8, i) + U(1, i);
    X(9, i+1) = X(9, i) + U(2, i);
    X(10,i+1) = X(10,i) + U(3, i);
    X(11,i+1) = X(11,i) + U(4, i);
    X(12,i+1) = X(12,i) + U(5, i);
    X(13,i+1) = X(13,i) + U(6, i);
  }

  // ---- 代价(终端唯一) ----
  SX cost_fun = SX::sym("cost_fun");
  cost_fun = 0;
  SX final_state = X(Slice(), m_predict_step);
  SX end_pos = fk_solver(std::vector<SX>{final_state, PPPPP(n_states+11)}).at(0);
  SX hitangles = SX::sym("hitangles", 3, 1);
  SX pitch = PPPPP(n_states+9)*M_PI/180.0;
  SX yaw = PPPPP(n_states+10)*M_PI/180.0;
  hitangles(0,0) = cos(pitch)*cos(yaw);
  hitangles(1,0) = cos(pitch)*sin(yaw);
  hitangles(2,0) = sin(pitch);
  SX vy_ref = tan(yaw)*PPPPP(n_states+7) + PPPPP(n_states+12);
  SX cosf_p1 = 1 - fabs(end_pos(0,0)*hitangles(0,0) + end_pos(1,0)*hitangles(1,0)
                        + end_pos(2,0)*hitangles(2,0));
  SX cosf_p2 = (end_pos(0,3)-PPPPP(n_states))*(end_pos(0,3)-PPPPP(n_states))
             + (end_pos(1,3)-PPPPP(n_states+1))*(end_pos(1,3)-PPPPP(n_states+1))
             + (end_pos(2,3)-PPPPP(n_states+2))*(end_pos(2,3)-PPPPP(n_states+2));
  SX cosf_p3 = 8.0*(PPPPP(n_states+7)-end_pos(3,0))*(PPPPP(n_states+7)-end_pos(3,0))
             + 0.5*(vy_ref - end_pos(3,1))*(vy_ref - end_pos(3,1))
             + 2.0*(PPPPP(n_states+8)-end_pos(3,2))*(PPPPP(n_states+8)-end_pos(3,2));
  for (int kk = 0; kk < m_predict_step; ++kk) {
    cost_fun = cost_fun + 0.00001*U(0,kk)*U(0,kk) + PPPPP(n_states+6)*U(1,kk)*U(1,kk)
             + 0.1*U(2,kk)*U(2,kk) + 0.5*U(3,kk)*U(3,kk) + 0.1*U(4,kk)*U(4,kk)
             + 0.1*U(5,kk)*U(5,kk) + 0.5*U(6,kk)*U(6,kk);
  }
  SX cosf_p4 = 1.7*X(7, m_predict_step)*X(7, m_predict_step)
             + 1.7*X(8, m_predict_step)*X(8, m_predict_step);
  cost_fun = cost_fun + 1000.0*cosf_p1 + 2500*cosf_p2 + cosf_p3 + cosf_p4;

  // ---- 约束 G(87 行:刹车位置 3×7 + 位置 3×7 + 底盘速 3×2 + 关节速 3×5
  //      + 加速度 3×7 + 自碰 z 3×1) ----
  SX G;
  for (int i = 0; i < m_predict_step; ++i) {
    SX A = smooth_sign(X(Slice(7,14,1), i) * (X(Slice(7,14,1), i) + U(Slice(0,7,1), i)));
    SX M1 = X(Slice(0,7,1), i+1) - X(Slice(0,7,1), i);
    SX Fv = smooth_sign(X(Slice(7,14,1), i));
    SX M2 = 0.5*ACC*X(Slice(7,14,1), i)*X(Slice(7,14,1), i);
    G = SX::vertcat({G, X(Slice(0,7,1), i)
                        + smooth_fmin(SX(1)+A, SX(1))*M1
                        + smooth_fmax(-A, SX(0))*Fv*M2});
  }
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, X(Slice(0,7,1), i+1)});
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, X(Slice(7,9,1), i+1)});
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, X(Slice(9,14,1), i+1)});
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, 1.0/ACC*PPPPP(n_states+3+i)
                       - sqrt(U(Slice(0,7,1), i)*U(Slice(0,7,1), i) + eps)});
  for (int i = 0; i < m_predict_step; ++i) {
    SX state_i = X(Slice(), i);
    SX end_pos_i = fk_solver(std::vector<SX>{state_i, PPPPP(n_states+11)}).at(0);
    G = SX::vertcat({G, end_pos_i(2,3) - 0.35});
  }

  casadi::SXDict nlp_prob = {{"f", cost_fun}, {"x", opt_var}, {"p", PPPPP}, {"g", G}};
  casadi::Dict opts;
  opts["ipopt.linear_solver"] = "mumps";   // 重生成版默认 mumps(免 HSL)
  opts["ipopt.max_iter"] = 270;
  opts["ipopt.print_level"] = 0;
  opts["print_time"] = 0;
  casadi::Function solver = casadi::nlpsol("nlpsol", "ipopt", nlp_prob, opts);
  // 生成文件名不得含目录成分(CodeGenerator 以其为 C 入口名,斜杠非法)
  solver.generate_dependencies("nlp.c");
  std::printf("nlp.c generated: nx=%lld np=%lld ng=%lld\n",
              solver.size_in(0).first, solver.size_in(1).first,
              solver.size_out(1).first);
  return 0;
}
