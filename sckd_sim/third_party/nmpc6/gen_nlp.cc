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
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include <casadi/casadi.hpp>

#include "sim_fk.h"

using casadi::SX;
using casadi::Slice;

int main(int argc,char**argv) {
  const bool servo=argc>1;
  nlohmann::json identification;if(servo){std::ifstream f(argv[1]);f>>identification;}
  const int n_states = 14, n_controls = 7, m_predict_step = 3;

  // ---- sim(x4_tennis)精确 FK(2026-10-09 替换旧灵足 X1 解析式:旧式拍长
  // d5=0.85 为羽毛球拍,本机腕滚半径 0.584)。位置/法向出自 sim_fk.h;
  // 速度 = jacobian(p,q)·ω + plat·(vx,vy,0) —— 符号精确导数,消掉旧手写
  // J 丢 ω2/ω5 项的近似;plat 开关下 vx,vy 都计入(旧版漏 vy,留档修正)
  auto GetEndPos = [&](SX thetas, SX add_base_vx) {
    SX qv[5] = {thetas(2, 0), thetas(3, 0), thetas(4, 0), thetas(5, 0),
                thetas(6, 0)};
    SX base[2] = {thetas(0, 0), thetas(1, 0)};
    SX p[3], n[3];
    sckd_nmpc6::SimArmPos(base, qv, p);
    sckd_nmpc6::SimArmNormal(qv, n);
    SX pvec = SX::vertcat({p[0], p[1], p[2]});
    SX qvec = SX::vertcat({qv[0], qv[1], qv[2], qv[3], qv[4]});
    SX wvec = SX::vertcat({thetas(9, 0), thetas(10, 0), thetas(11, 0),
                           thetas(12, 0), thetas(13, 0)});
    SX J = SX::jacobian(pvec, qvec);                  // 3×5 精确
    SX v3[3] = {SX(0), SX(0), SX(0)};
    for (int r = 0; r < 3; ++r)
      for (int j = 0; j < 5; ++j) v3[r] += J(r, j) * wvec(j);
    SX M = SX::zeros(4, 4);
    M(0, 0) = n[0]; M(1, 0) = n[1]; M(2, 0) = n[2];
    M(0, 3) = p[0]; M(1, 3) = p[1]; M(2, 3) = p[2];
    M(3, 0) = v3[0] + add_base_vx * thetas(7, 0);
    M(3, 1) = v3[1] + add_base_vx * thetas(8, 0);
    M(3, 2) = v3[2];
    M(3, 3) = 1;
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
  SX PPPPP = SX::sym("PPPPP", n_states + 13+(servo?10:0));       // servo adds effective command q/dq
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

  // 执行器状态预测：固定姿态辨识的二阶PD误差模型 + 9ms传输。
  // 全逆动力学前馈匹配局部I/B时，e_dot=w,w_dot=-kp/I*e-(kd+B)/I*w。
  // 保留实际q/dq作为状态；旧有效命令填充最初9ms，不把目标误差伪装成时间移位。
  SX Y=X;
  if(servo) {
    auto evolve=[&](int j,const SX&e,const SX&w,const SX&t) {
      double I=identification["axes"][j]["inertia_kg_m2"],B=identification["axes"][j]["damping_Nms_rad"];
      double kp=identification["kp"][j],kd=identification["kd"][j];
      double a=(kd+B)/(2*I),om2=kp/I,disc=a*a-om2;
      SX c,ss;
      if(disc<0){double om=std::sqrt(-disc);c=cos(om*t);ss=sin(om*t)/om;}
      else {double om=std::sqrt(std::max(disc,1e-12));c=cosh(om*t);ss=sinh(om*t)/om;}
      return SX::vertcat({exp(-a*t)*((c+a*ss)*e+ss*w),exp(-a*t)*(-om2*ss*e+(c-a*ss)*w)});
    };
    for(int j=0;j<5;++j) {
      SX oldq=PPPPP(27+j),oldv=PPPPP(32+j),q0=X(2+j,0),v0=X(9+j,0);
      SX eq0=q0-oldq,ev0=v0-oldv;
      const double delay=.009;
      SX ed=evolve(j,eq0,ev0,SX(delay));
      SX qd=oldq+oldv*delay+ed(0),vd=oldv+ed(1);
      for(int k=0;k<3;++k){
        SX Tk=(k+1)*PPPPP(17),query=fmax(Tk-delay,SX(0)),qref=q0,vref=v0;
        for(int seg=0;seg<3;++seg){
          SX h=PPPPP(17),u=U(2+j,seg),tt=fmin(fmax(query-seg*h,SX(0)),h);
          double amax=1.0/static_cast<double>(ACC(2+j,0));
          SX duration=sqrt(u*u+eps)/amax,on=fmin(tt,duration),acc=u/duration;
          SX qq=X(2+j,seg)+X(9+j,seg)*tt+.5*acc*on*on+u*fmax(tt-duration,SX(0));
          SX vv=X(9+j,seg)+acc*on;
          qref=if_else(query>=seg*h,qq,qref);vref=if_else(query>=seg*h,vv,vref);
        }
        SX ee=evolve(j,qd-q0,vd-v0,fmax(Tk-delay,SX(0)));
        SX early=evolve(j,eq0,ev0,Tk);
        Y(2+j,k+1)=if_else(Tk>=delay,qref+ee(0),oldq+oldv*Tk+early(0));
        Y(9+j,k+1)=if_else(Tk>=delay,vref+ee(1),oldv+early(1));
      }
    }
    casadi::Function state_rollout("actuator_rollout",{opt_var,PPPPP},{Y(Slice(),3)});
    state_rollout.generate("actuator_rollout.c");
  }

  // ---- 代价(终端唯一) ----
  SX cost_fun = SX::sym("cost_fun");
  cost_fun = 0;
  SX final_state = Y(Slice(), m_predict_step);
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
    G = SX::vertcat({G, Y(Slice(0,7,1), i+1)});
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, X(Slice(7,9,1), i+1)});
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, Y(Slice(9,14,1), i+1)});
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, 1.0/ACC*PPPPP(n_states+3+i)
                       - sqrt(U(Slice(0,7,1), i)*U(Slice(0,7,1), i) + eps)});
  for (int i = 0; i < m_predict_step; ++i) {
    SX state_i = servo?Y(Slice(),i+1):X(Slice(),i);
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
  solver.generate_dependencies(servo?"nlp_servo.c":"nlp.c");
  std::printf("generated: nx=%lld np=%lld ng=%lld\n",
              solver.size_in(0).first, solver.size_in(1).first,
              solver.size_out("g").first);
  return 0;
}
