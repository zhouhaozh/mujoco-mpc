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
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <vector>

#include <casadi/casadi.hpp>

#include "robot_fk.h"

using casadi::SX;
using casadi::Slice;

int main(int argc,char**argv) {
  // 默认保留旧库单dt动力学。只有独立时间搜索显式开启逐节点dt，
  // 不允许误把均匀子步的servo/chassis预测库标作支持非均匀。
  bool nonuniform_time = false;
  bool geometry_only = false;
  auto robot = sckd_nmpc6::RobotGeometry::kTennis;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--nonuniform-time") nonuniform_time = true;
    else if (std::string(argv[i]) == "--geometry-only") geometry_only = true;
    else if (std::string(argv[i]) == "--robot") {
      if (++i >= argc) throw std::invalid_argument("--robot requires x4_tennis or x4_950");
      robot = sckd_nmpc6::ParseRobotGeometry(argv[i]);
    }
    else positional.emplace_back(argv[i]);
  }
  if (robot == sckd_nmpc6::RobotGeometry::kX4950 && !geometry_only) {
    std::fprintf(stderr, "X4_950 geometry only: joint/chassis limits and execution parameters are unconfirmed; NLP generation disabled\n");
    return 2;
  }
  if (geometry_only && !positional.empty())
    throw std::invalid_argument("geometry-only must not load servo/chassis identification");
  const bool chassis=positional.size()>1 && positional[1]=="chassis";
  const bool servo=!positional.empty() && !chassis;
  if (nonuniform_time && (servo || chassis)) {
    std::fprintf(stderr, "--nonuniform-time supports only baseline generation without identification JSON\n");
    return 2;
  }
  nlohmann::json identification;if(servo||chassis){std::ifstream f(positional[0]);f>>identification;}
  const int n_states = 14, n_controls = 7, m_predict_step = 3;

  // ---- 位置/法向出自 robot_fk.h 选定分支;默认网球公式保持不变。
  // 速度 = jacobian(p,q)·ω + plat·(vx,vy,0) —— 符号精确导数,消掉旧手写
  // J 丢 ω2/ω5 项的近似;plat 开关下 vx,vy 都计入(旧版漏 vy,留档修正)
  auto GetEndPos = [&](SX thetas, SX add_base_vx) {
    SX qv[5] = {thetas(2, 0), thetas(3, 0), thetas(4, 0), thetas(5, 0),
                thetas(6, 0)};
    SX base[2] = {thetas(0, 0), thetas(1, 0)};
    SX p[3], n[3];
    sckd_nmpc6::RobotArmPos(robot, base, qv, p);
    sckd_nmpc6::RobotArmNormal(robot, qv, n);
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

  if (geometry_only) {
    const SX J = SX::jacobian(endpos(Slice(0,3), 3), thetas);
    casadi::Function geometry("robot_geometry", {thetas, add_base_vx},
                             {endpos, J(Slice(), Slice(2,7))});
    geometry.generate(robot == sckd_nmpc6::RobotGeometry::kX4950
                      ? "x4_950_geometry.c" : "x4_tennis_geometry.c");
    return 0;  // 不构造 nlpsol,不读取或生成硬件约束。
  }

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
  SX PPPPP = SX::sym("PPPPP", n_states + 13+(servo?10:chassis?11:0));       // servo adds effective command q/dq
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
    SX dt = PPPPP(n_states + 3 + (nonuniform_time ? i : 0));
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

  if(chassis) {
    // 执行端降阶模型：同方向率/幅值斜坡、舵角投影、轮目标斜坡，
    // 实际平移响应由独立激励拟合；不声称复制轮胎/电机全动力学。
    const double tau_v=identification["motion_tau_s"],tau_s=identification["steer_tau_s"];
    const double accel=identification["config"]["ramp"]["accel_limit_mps2"];
    const double turn=identification["config"]["ramp"]["dir_rate_limit_rad_s"];
    const double wheel_acc=identification["config"]["servo"]["wheel_accel_mps2"];
    auto clip=[](const SX&x,const SX&lim){return fmin(fmax(x,-lim),lim);};
    auto angle=[](const SX&x){return atan2(sin(x),cos(x));};
    SX rx=PPPPP(27),ry=PPPPP(28),theta[4],wheel[4],psi=PPPPP(37);
    for(int j=0;j<4;++j){theta[j]=PPPPP(29+j);wheel[j]=PPPPP(33+j);}
    SX bvx=cos(psi)*X(7,0)+sin(psi)*X(8,0),bvy=-sin(psi)*X(7,0)+cos(psi)*X(8,0),px=X(0,0),py=X(1,0);
    SX trace=SX::zeros(4,37);trace(0,0)=px;trace(1,0)=py;trace(2,0)=X(7,0);trace(3,0)=X(8,0);
    SX h=PPPPP(17)/12;std::vector<SX> wxhist,wyhist;
    SX wx0=0,wy0=0;for(int j=0;j<4;++j){wx0+=wheel[j]*cos(theta[j])*.25;wy0+=wheel[j]*sin(theta[j])*.25;}
    wxhist.push_back(wx0);wyhist.push_back(wy0);
    for(int seg=0;seg<3;++seg){
      for(int step=1;step<=12;++step){
        SX local=step*h,vcx=X(7,seg)+clip(U(0,seg),8*local),vcy=X(8,seg)+clip(U(1,seg),8*local);
        SX cx=cos(psi)*vcx+sin(psi)*vcy,cy=-sin(psi)*vcx+cos(psi)*vcy;
        SX rm=sqrt(rx*rx+ry*ry+1e-12),cm=sqrt(cx*cx+cy*cy+1e-12);
        SX a0=atan2(ry,rx+1e-9),a1=atan2(cy,cx+1e-9),da=clip(angle(a1-a0),turn*h),mag=rm+clip(cm-rm,accel*h);
        SX nx=mag*cos(a0+da),ny=mag*sin(a0+da);
        rx=if_else(rm>1e-5 && cm>1e-5,nx,rx+clip(cx-rx,accel*h));
        ry=if_else(rm>1e-5 && cm>1e-5,ny,ry+clip(cy-ry,accel*h));
        SX direction=atan2(ry,rx+1e-9),speed=sqrt(rx*rx+ry*ry+1e-12),wx=0,wy=0;
        for(int j=0;j<4;++j){
          SX err=angle(direction-theta[j]),reverse=fabs(err)>M_PI/2;
          SX folded=if_else(reverse,angle(err+M_PI),err);
          SX target=if_else(speed>.001,if_else(reverse,-speed,speed)*fmax(cos(folded),SX(0)),SX(0));
          wheel[j]+=clip(target-wheel[j],wheel_acc*h);
          theta[j]+=if_else(speed>.001,clip(folded*(1-exp(-h/tau_s)),8*h),SX(0));
          wx+=wheel[j]*cos(theta[j])*.25;wy+=wheel[j]*sin(theta[j])*.25;
        }
        wxhist.push_back(wx);wyhist.push_back(wy);
        int index=seg*12+step;SX query=index*h-.009,delaywx=wx0,delaywy=wy0;
        for(int n=1;n<=index;++n){SX alpha=fmin(fmax((query-(n-1)*h)/h,SX(0)),SX(1));delaywx+=(wxhist[n]-wxhist[n-1])*alpha;delaywy+=(wyhist[n]-wyhist[n-1])*alpha;}
        SX oldx=bvx,oldy=bvy;double gain=identification.value("motion_gain",1.0);
        bvx+=(gain*delaywx-bvx)*(1-exp(-h/tau_v));bvy+=(gain*delaywy-bvy)*(1-exp(-h/tau_v));
        SX vx=cos(psi)*bvx-sin(psi)*bvy,vy=sin(psi)*bvx+cos(psi)*bvy;
        px+=.5*h*(vx+cos(psi)*oldx-sin(psi)*oldy);py+=.5*h*(vy+sin(psi)*oldx+cos(psi)*oldy);
        trace(0,index)=px;trace(1,index)=py;trace(2,index)=vx;trace(3,index)=vy;
      }
      Y(0,seg+1)=px;Y(1,seg+1)=py;Y(7,seg+1)=trace(2,(seg+1)*12);Y(8,seg+1)=trace(3,(seg+1)*12);
    }
    casadi::Function rollout("chassis_rollout",{opt_var,PPPPP},{Y,trace});rollout.generate("chassis_rollout.c");
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
  SX Z=chassis?Y:X;
  for (int i = 0; i < m_predict_step; ++i) {
    SX A = smooth_sign(Z(Slice(7,14,1), i) * (Z(Slice(7,14,1), i) + U(Slice(0,7,1), i)));
    SX M1 = Z(Slice(0,7,1), i+1) - Z(Slice(0,7,1), i);
    SX Fv = smooth_sign(Z(Slice(7,14,1), i));
    SX M2 = 0.5*ACC*Z(Slice(7,14,1), i)*Z(Slice(7,14,1), i);
    G = SX::vertcat({G, Z(Slice(0,7,1), i)
                        + smooth_fmin(SX(1)+A, SX(1))*M1
                        + smooth_fmax(-A, SX(0))*Fv*M2});
  }
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, Y(Slice(0,7,1), i+1)});
  for (int i = 0; i < m_predict_step; ++i)
    G = SX::vertcat({G, (chassis?Y:X)(Slice(7,9,1), i+1)});
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

  if(chassis)for(int i=0;i<3;++i)G=SX::vertcat({G,X(Slice(7,9,1),i+1)}); // 命令速度也保持原逐轴上限
  if (!servo && !chassis) {
    // 独立验收入口：不用求解器便可对照任意控制/时间下的状态、约束和代价。
    // 与nlp.c分开编译保存，可同时比较默认旧公式和逐段公式，禁止覆盖默认库。
    casadi::Function probe("time_grid_probe", {opt_var, PPPPP},
                           {X, G, cost_fun, SX(nonuniform_time ? 1 : 0)},
                           {"controls", "parameters"},
                           {"states", "constraints", "cost", "per_stage_time"});
    probe.generate("time_grid_probe.c");
  }
  casadi::SXDict nlp_prob = {{"f", cost_fun}, {"x", opt_var}, {"p", PPPPP}, {"g", G}};
  casadi::Dict opts;
  opts["ipopt.linear_solver"] = "mumps";   // 重生成版默认 mumps(免 HSL)
  opts["ipopt.max_iter"] = 270;
  opts["ipopt.print_level"] = 0;
  opts["print_time"] = 0;
  casadi::Function solver = casadi::nlpsol("nlpsol", "ipopt", nlp_prob, opts);
  // 生成文件名不得含目录成分(CodeGenerator 以其为 C 入口名,斜杠非法)
  const char* generated_path = chassis ? "nlp_chassis.c" : servo ? "nlp_servo.c" : "nlp.c";
  solver.generate_dependencies(generated_path);
  // 编入生成库的能力标记；运行驱动校验实际加载的.so，不信任配置布尔值。
  std::ofstream generated(generated_path, std::ios::app);
  if (!generated) throw std::runtime_error("cannot append time-grid schema to generated C");
  generated << "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
            << "int sckd_time_grid_schema(void) { return " << (nonuniform_time ? 1 : 0) << "; }\n"
            << "#ifdef __cplusplus\n}\n#endif\n";
  generated.close();
  if (!generated) throw std::runtime_error("cannot save time-grid schema to generated C");
  std::printf("generated: nx=%lld np=%lld ng=%lld nonuniform_time=%d\n",
              solver.size_in(0).first, solver.size_in(1).first,
              solver.size_out("g").first, static_cast<int>(nonuniform_time));
  return 0;
}
