// 旧仓库 NMPC 驱动层实现(见 legacy_nmpc.h 头注)。对应原文件:
//   set_my_nmpc_solver_dll() → 构造(IPOPT 选项;MA57→mumps 默认)
//   opti_solution()          → Solve(守门链逐式照搬,拒绝显式化)
//   GetEndPos2()             → EndPos(解析 FK,数值 rollout/精度门共用)
//   SetGuss/setInitGuss_Order→ Reset/SetGuess
// 原文件 1989 行 → 本文件 ~460 行:日志/ONNX/boost/OpenCV/timestamp 全剥离,
// 回合状态收敛进 Impl(原为类成员+文件级全局混排)。
#include "control/legacy_nmpc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

#include <casadi/casadi.hpp>

namespace sckd {
namespace {

constexpr int kNs = 14;   // 状态 [x,y,θ1..5,vx,vy,ω1..5]
constexpr int kNu = 7;    // 控制 [dvx,dvy,dω1..5]
constexpr int kNp = 3;    // 预测步数(与生成码 nlp.so 绑定)

// 加速度上限倒数(ACC=1/a_max;原文件三处同表,此处唯一真源)
constexpr double kAccInv[kNu] = {1.0 / 8.0,   1.0 / 8.0,   1.0 / 30.0,
                                 1.0 / 30.0,  1.0 / 180.0, 1.0 / 400.0,
                                 1.0 / 200.0};

// 关节速度窗(前场球 / 正常球;rad/s —— 原表注释误标 m/s,数值原样)
constexpr double kSpdFrontLo[5] = {-10.0, -10.0, -40.0, -48.0, -15.0};
constexpr double kSpdFrontHi[5] = {10.0, 10.0, 40.0, 48.0, 15.0};
constexpr double kSpdNormLo[5] = {-10.0, -10.0, -20.0, -48.0, -15.0};
constexpr double kSpdNormHi[5] = {10.0, 10.0, 20.0, 48.0, 15.0};

// hit_spd≤此值时末端 vx 计入底盘 vx(低球速挥拍,底盘速度是拍速的一部分)
constexpr double kAddBaseVxThreshold = 28.0;

// 冻结窗:剩余击球时间>1s 且被冻关节静止(|v|<0.05)→ 首步控制块 {5,6}
// (dω4,dω5)零增量。速度门必须:lbx=ubx=0 的语义是"速度指令增量为 0"=
// 维持当前速度,只有当前速度本来就 0 才等于"不动"(原注,带速度冻会漂)
constexpr double kFreezeWindowSec = 1.0;
constexpr int kMaxFreezeSteps = 1;
constexpr int kFreezeCtrl[] = {5, 6};
constexpr int kFreezeCtrlCount = 2;
constexpr double kFreezeVelEps = 0.05;

// 将任意角度按 180° 周期回绕到 [-90°, 90°)(原 WrapYaw90Deg)
double WrapYaw90(double deg) {
  double w = std::fmod(deg + 90.0, 180.0);
  if (w < 0.0) w += 180.0;
  return w - 90.0;
}

// 解析 FK(原 GetEndPos2 逐式照搬;灵足 X1:d1=0.685, a2=0.705, d4=−0.1705,
// d5=−racket_len)。out=[px,py,pz,pitch°,vx,vy,vz];add_base_vx=true 时 vx
// 计入底盘 vx。速度雅可比按原样省略 ω2/ω5 项(与生成码一致,误差网口径)。
void EndPos(const double th[14], double racket_len, bool add_base_vx,
            double out[7]) {
  const double d1 = 0.685, a2 = 0.705, d4 = -0.1705, d5 = -racket_len;
  const double c1 = std::cos(th[2]), s1 = std::sin(th[2]);
  const double c2 = std::cos(th[3]), s2 = std::sin(th[3]);
  const double c23 = std::cos(th[3] + th[4]), s23 = std::sin(th[3] + th[4]);
  const double c4 = std::cos(th[5]), s4 = std::sin(th[5]);
  const double c5 = std::cos(th[6]), s5 = std::sin(th[6]);
  const double w1 = th[9], w3 = th[11], w4 = th[12];
  const double nz = s23 * c4 * c5 + c23 * s5;
  out[0] = -d5 * c1 * c23 * s4 + d5 * s1 * c4 + d4 * c1 * s23 +
           a2 * c1 * c2 + th[0];
  out[1] = -d5 * s1 * c23 * s4 - d5 * c1 * c4 + d4 * s1 * s23 +
           a2 * s1 * c2 + th[1];
  out[2] = -d5 * s23 * s4 - d4 * c23 + a2 * s2 + d1;
  out[3] = std::fabs(std::asin(std::max(-1.0, std::min(1.0, nz))) * 180.0 /
                     M_PI);
  out[4] = (d5 * s1 * c23 * s4 + d5 * c1 * c4 - d4 * s1 * s23 -
            a2 * s1 * c2) * w1 +
           (d5 * c1 * s23 * s4 + d4 * c1 * c23) * w3 +
           (-d5 * c1 * c23 * c4 - d5 * s1 * s4) * w4 +
           (add_base_vx ? th[7] : 0.0);
  out[5] = (-d5 * s1 * c23 * s4 + d5 * s1 * c4 + d4 * c1 * s23 +
            a2 * c1 * c2) * w1 +
           (d5 * s1 * s23 * s4 + d4 * s1 * c23) * w3 +
           (-d5 * s1 * c23 * c4 + d5 * c1 * s4) * w4;
  out[6] = (-d5 * c23 * s4 + d4 * s23) * w3 + (-d5 * s23 * c4) * w4;
}

}  // namespace

struct LegacyNmpc::Impl {
  casadi::Function solver;
  double racket_len = 0.85;

  // ---- 回合/拍间状态(原类成员+文件级全局) ----
  double guess[kNp * kNu] = {0};   // 初值,块序 [dvx×3, dvy×3, …, dω5×3]
  double last_term[kNs] = {0};     // 上一接受解 rollout 终态(跳变守门锚)
  bool have_last = false;
  int hit_type = 0;                // 0 未定 / 1 前场 / 2 正常
  bool lock_hit_speed = false;
  double initial_dt = -1.0;        // 新回合首个剩余时间
  double m1_a1 = -90, m1_a2 = 90;  // 动态角窗(°)
  double m4_a1 = -270, m4_a2 = 90;
  double m2_a1 = 10, m2_a2 = 60;
  double m3_a1 = -144, m3_a2 = 0;
  double m5_lo = -120, m5_hi = 120;
  double acc_thr = 0.05;
  int solve_number = 0;            // 严格档+求解成功的累计(首解收紧 m5 窗)
  int inference_count = 0;
  double comp[5] = {0};            // 误差补偿 [Δx,Δy,Δz,Δpitch°,Δyaw°]
  bool comp_locked = false;        // 剩余≤50ms 锁存(后续拍不更新)
  double min_dt_ms = -1.0;
  bool pred_valid = false;
  double pred_vx = 0, pred_vy = 0; // 首解捕获的触球底盘速度

  void ResetRound() {
    // 原"全零 guess"重置块:回合级状态归零(guess 由 Reset 再清)
    hit_type = 0;
    lock_hit_speed = false;
    m1_a1 = -90; m1_a2 = 90;
    m4_a1 = -270; m4_a2 = 90;
    m5_lo = -170; m5_hi = 170;
    solve_number = 0;
    inference_count = 0;
    comp_locked = false;
    min_dt_ms = -1.0;
    pred_valid = false;
    pred_vx = pred_vy = 0;
    for (int i = 0; i < 5; ++i) comp[i] = 0.0;
    have_last = false;
  }
};

LegacyNmpc::LegacyNmpc(const LegacyNmpcConfig& cfg) {
  impl_ = new Impl;
  impl_->racket_len = cfg.racket_len;
  casadi::Dict opts;
  opts["ipopt.linear_solver"] = cfg.linear_solver;
  opts["ipopt.max_iter"] = cfg.max_iter;
  opts["ipopt.print_level"] = 0;
  opts["print_time"] = 0;
  opts["ipopt.tol"] = 1e-6;
  opts["ipopt.acceptable_tol"] = 1e-6;
  opts["ipopt.acceptable_obj_change_tol"] = 1e-8;
  opts["ipopt.warm_start_init_point"] = "no";
  opts["ipopt.warm_start_entire_iterate"] = "no";
  opts["ipopt.mu_strategy"] = "adaptive";
  opts["ipopt.nlp_scaling_method"] = "gradient-based";
  opts["ipopt.mu_init"] = 1e-3;
  opts["ipopt.nlp_lower_bound_inf"] = -1e19;
  opts["ipopt.nlp_upper_bound_inf"] = 1e19;
  opts["ipopt.obj_scaling_factor"] = 1.0;
  opts["ipopt.dual_inf_tol"] = 1e-6;
  opts["ipopt.constr_viol_tol"] = 1e-6;
  opts["ipopt.compl_inf_tol"] = 1e-6;
  opts["ipopt.mu_target"] = 1e-12;
  opts["ipopt.bound_relax_factor"] = 1e-8;
  // nlpsol 从 .so 动态加载生成码(等价原 set_my_nmpc_solver_dll 的
  // casadi::nlpsol("nlpsol","ipopt","nlp.so",opts))
  impl_->solver = casadi::nlpsol("nmpc6", "ipopt", cfg.nlp_so, opts);
}

LegacyNmpc::~LegacyNmpc() { delete impl_; }

void LegacyNmpc::Reset() {
  std::fill(impl_->guess, impl_->guess + kNp * kNu, 0.0);
  impl_->ResetRound();
}

void LegacyNmpc::SetGuess(const double guess_block[21]) {
  std::memcpy(impl_->guess, guess_block, sizeof(double) * kNp * kNu);
}

bool LegacyNmpc::Solve(const LegacyNmpcRequest& req, LegacyNmpcResult* out) {
  Impl& im = *impl_;
  const auto t0 = std::chrono::steady_clock::now();
  *out = LegacyNmpcResult{};

  // 状态本地拷贝(回合重置时对 ω4 清零会改它,后续全部用这份 —— 原语义)
  double cs_in[kNs];
  std::memcpy(cs_in, req.state, sizeof(double) * kNs);

  // ---- 新回合检测(全零 guess)→ 回合重置 + ω4 小值清零 ----
  bool all_zero = true;
  for (int i = 0; i < kNp * kNu; ++i)
    if (std::fabs(im.guess[i]) >= 0.001) { all_zero = false; break; }
  if (all_zero) {
    im.ResetRound();
    im.initial_dt = req.dt[0] + req.dt[1] + req.dt[2];
    // 击球前 4 号电机微动视为静止(其冻结速度门实际为 0.5)
    if (std::fabs(cs_in[12]) < 0.5) cs_in[12] = 0.0;
    // 角窗基线(正手/反手按目标 y 侧选 m4 窗)
    if (req.target[1] >= 0.0) { im.m4_a1 = -180; im.m4_a2 = 90; }
    else                      { im.m4_a1 = -270; im.m4_a2 = 0; }
  }

  // ---- 误差补偿注入(上一拍 comp;Δpitch 原实现未启用,Δyaw 走回绕) ----
  double tgt[3] = {req.target[0] + im.comp[0], req.target[1] + im.comp[1],
                   req.target[2] + im.comp[2]};
  double hit_yaw = WrapYaw90(req.hit_yaw_deg + 0.0);   // +comp[4] 未启用
  double hit_spd = std::max(2.0, std::min(req.hit_spd, 53.0));
  double hit_pitch = std::max(-30.0, std::min(req.hit_pitch_deg, 90.0));
  hit_yaw = std::max(-25.0, std::min(hit_yaw, 30.0));
  double hit_z = std::max(-5.0, std::min(req.hit_speed_z, 20.0));
  const bool use_plat_vx = hit_spd <= kAddBaseVxThreshold;

  // ---- 击球分型 + 动态角窗/精度阈(前场 vs 正常;lock_hit_speed 一次性) ----
  if (im.hit_type == 1 ||
      (im.hit_type == 0 && hit_pitch > 50.0 && tgt[2] < 0.9)) {
    im.hit_type = 1;                                   // 前场球
    im.m1_a1 = -60;   im.m1_a2 = 60;
    im.m4_a1 = -360;  im.m4_a2 = 180;
    im.m2_a1 = 10;    im.m2_a2 = 45;
    im.m3_a1 = -100;  im.m3_a2 = -10;
    im.acc_thr = 0.1;
  } else {
    im.hit_type = 2;                                   // 正常球
    if (tgt[2] >= 0.9 && !im.lock_hit_speed) {
      if (tgt[0] > 2.5) { im.m1_a1 = -90;  im.m1_a2 = 90; }
      else              { im.m1_a1 = -130; im.m1_a2 = 130; }
      im.lock_hit_speed = true;
    }
    im.m2_a1 = 10;
    im.m2_a2 = (tgt[2] <= 1.2) ? 60 : 70;
    im.m3_a1 = -144;  im.m3_a2 = 0;   // 挥拍避机器中心外壳
    im.m4_a1 = -360;  im.m4_a2 = 180;
    im.acc_thr = 0.05;
  }

  // ---- 决策变量界(块序;±[4,4,5,3,40,50,10]) ----
  std::vector<double> lbx, ubx;
  lbx.reserve(kNp * kNu); ubx.reserve(kNp * kNu);
  const double ulo[kNu] = {-4, -4, -5, -3, -40, -50, -10};
  const double uhi[kNu] = {4, 4, 5, 3, 40, 50, 10};
  for (int c = 0; c < kNu; ++c)
    for (int k = 0; k < kNp; ++k) { lbx.push_back(ulo[c]); ubx.push_back(uhi[c]); }

  // ---- 冻结窗:剩余>1s 且被冻关节静止 → 首步 dω4/dω5 零增量 ----
  // (速度门必须:lbx=ubx=0 是"增量 0"=维持当前速度,非"不动" —— 原注)
  {
    bool freeze_allowed = true;
    for (int ci = 0; ci < kFreezeCtrlCount; ++ci)
      if (std::fabs(cs_in[7 + kFreezeCtrl[ci]]) >= kFreezeVelEps) {
        freeze_allowed = false;
        break;
      }
    if (freeze_allowed) {
      double t_remain = 0.0;
      for (int k = 0; k < kNp && k < 3; ++k) t_remain += req.dt[k];
      for (int k = 0; k < kNp; ++k) {
        const bool frozen = (k < kMaxFreezeSteps && t_remain > kFreezeWindowSec);
        if (k < 3) t_remain -= req.dt[k];
        if (!frozen) continue;
        for (int ci = 0; ci < kFreezeCtrlCount; ++ci) {
          const int idx = kFreezeCtrl[ci] * kNp + k;   // 块布局
          lbx[idx] = 0.0;
          ubx[idx] = 0.0;
          im.guess[idx] = 0.0;
        }
      }
    }
  }

  // ---- 约束界(与生成码 G 布局逐块对应,共 3×29=87 行) ----
  std::vector<double> lbg, ubg;
  lbg.reserve(87); ubg.reserve(87);
  const double D2R = M_PI / 180.0;
  auto push_state_bounds = [&]() {          // 位置块 ×2(刹车行 + 步末行)
    for (int k = 0; k < kNp; ++k) {
      lbg.push_back(-1.0);  ubg.push_back(5.7);      // x∈[−1,5.7](半场)
      lbg.push_back(-2.0);  ubg.push_back(2.0);      // y(左边界弃右手挥拍)
      lbg.push_back(im.m1_a1 * D2R); ubg.push_back(im.m1_a2 * D2R);
      lbg.push_back(im.m2_a1 * D2R); ubg.push_back(im.m2_a2 * D2R);
      lbg.push_back(im.m3_a1 * D2R); ubg.push_back(im.m3_a2 * D2R);
      lbg.push_back(im.m4_a1 * D2R); ubg.push_back(im.m4_a2 * D2R);
      lbg.push_back(im.m5_lo * D2R); ubg.push_back(im.m5_hi * D2R);
    }
  };
  push_state_bounds();
  push_state_bounds();
  for (int k = 0; k < kNp; ++k) {           // 底盘速度
    lbg.push_back(-4.0); ubg.push_back(4.0);
    lbg.push_back(-4.0); ubg.push_back(4.0);
  }
  {                                          // 关节速度(分型选表;口径=目标
    const bool front =                      // x>5.0,与角窗的 pitch>50 不同,
        im.hit_type == 1 ||                  // 原实现如此,原样保留)
        (im.hit_type == 0 && req.target[0] > 5.0 && req.target[2] < 0.9);
    for (int k = 0; k < kNp; ++k)
      for (int j = 0; j < 5; ++j) {
        lbg.push_back(front ? kSpdFrontLo[j] : kSpdNormLo[j]);
        ubg.push_back(front ? kSpdFrontHi[j] : kSpdNormHi[j]);
      }
  }
  for (int k = 0; k < kNp; ++k)             // 加速度 |U|≤a_max·dt
    for (int c = 0; c < kNu; ++c) { lbg.push_back(0.0); ubg.push_back(casadi::inf); }
  for (int k = 0; k < kNp; ++k) {           // 末端 z≥0.35(自碰)
    lbg.push_back(0.0); ubg.push_back(casadi::inf);
  }

  // ---- 参数向量(27 元,布局与生成码 PPPPP 一致;nlp.so 由
  // third_party/nmpc6/gen_nlp.cc 从 HEAD 源码重生成,2026-10-09 工程师
  // 确认源码即真源) ----
  std::vector<double> p;
  p.reserve(kNs + 13);
  for (int i = 0; i < kNs; ++i) p.push_back(cs_in[i]);
  for (int i = 0; i < 3; ++i) p.push_back(tgt[i]);
  for (int i = 0; i < 3; ++i) p.push_back(req.dt[i]);
  p.push_back(10.0);                        // w_y(原 cost_para 算后弃用,10.0)
  p.push_back(hit_spd);
  p.push_back(hit_z);
  p.push_back(hit_pitch);
  p.push_back(hit_yaw);
  p.push_back(use_plat_vx ? 1.0 : 0.0);     // 末端vx是否计入底盘vx
  p.push_back(tgt[2] < 0.99 ? 10.0 : 0.0);  // 低球 vy 目标偏置(速度误差权重)

  // ---- 求解 ----
  bool solver_ok = false;
  std::vector<double> res_control_all;
  try {
    // map<string,DM> 调用形态(本版 casadi 3.7 头无 Dict(GenericType)
    // operator() 重载;旧机编译头版本不同 —— 接口语义等价)
    std::map<std::string, casadi::DM> args;
    args["lbx"] = casadi::DM(lbx);
    args["ubx"] = casadi::DM(ubx);
    args["lbg"] = casadi::DM(lbg);
    args["ubg"] = casadi::DM(ubg);
    args["x0"] = casadi::DM(std::vector<double>(im.guess, im.guess + kNp * kNu));
    args["p"] = casadi::DM(p);
    std::map<std::string, casadi::DM> res = im.solver(args);
    res_control_all = std::vector<double>(res.at("x"));
    casadi::Dict stats = im.solver.stats();
    const std::string st = static_cast<std::string>(stats.at("return_status"));
    solver_ok = (st == "Solve_Succeeded" || st == "Solved_To_Acceptable_Level");
  } catch (const std::exception& e) {
    out->status = LegacyNmpcStatus::kSolverFail;
    out->solve_ms = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count() * 1e3;
    ++rejects;
    if (getenv("NMPC6_DEBUG"))
      std::fprintf(stderr, "[nmpc6] solver exception: %s\n", e.what());
    return false;
  }
  // 控制量块切片(块序直接对应)
  const double* dvx = res_control_all.data() + 0 * kNp;
  const double* dvy = res_control_all.data() + 1 * kNp;
  const double* do1 = res_control_all.data() + 2 * kNp;
  const double* do2 = res_control_all.data() + 3 * kNp;
  const double* do3 = res_control_all.data() + 4 * kNp;
  const double* do4 = res_control_all.data() + 5 * kNp;
  const double* do5 = res_control_all.data() + 6 * kNp;

  // ---- 数值 rollout(梯形积分;符号式用 sqrt(U²+ε) 平滑,数值用 fabs,
  // 原两处口径本就不同,逐式照搬) ----
  double cs[kNs];
  std::memcpy(cs, cs_in, sizeof(double) * kNs);
  for (int j = 0; j < kNp; ++j) {
    const double dt = req.dt[j];
    const double u[kNu] = {dvx[j], dvy[j], do1[j], do2[j], do3[j], do4[j],
                           do5[j]};
    for (int c = 0; c < kNu; ++c) {
      const double a_max = 1.0 / kAccInv[c];
      cs[c] = (cs[7 + c] + u[c]) * dt -
              (u[c] * std::fabs(u[c])) / 2.0 / a_max + cs[c];
      cs[7 + c] += u[c];
    }
  }

  // ---- 1ms 复检:加速度受限速度斜坡下逐毫秒 FK,z<0.35 = 挥拍打机器/地面 ----
  {
    double st[kNs];
    std::memcpy(st, cs_in, sizeof(double) * kNs);
    bool is_hit = false;
    for (int j = 0; j < kNp && !is_hit; ++j) {
      const double dt = req.dt[j];
      const double u[kNu] = {dvx[j], dvy[j], do1[j], do2[j], do3[j], do4[j],
                             do5[j]};
      double pos[7] = {st[0], st[1], st[2], st[3], st[4], st[5], st[6]};
      double vel[7] = {st[7], st[8], st[9], st[10], st[11], st[12], st[13]};
      for (int c = 0; c < kNu; ++c) {
        const double a_max = 1.0 / kAccInv[c];
        st[c] = (st[7 + c] + u[c]) * dt -
                (u[c] * std::fabs(u[c])) / 2.0 / a_max + st[c];
        st[7 + c] += u[c];
      }
      const double cmd[7] = {st[7], st[8], st[9], st[10], st[11], st[12], st[13]};
      const int n_ms = (int)(dt * 1000.0);
      for (int t = 1; t < n_ms; ++t) {
        for (int c = 0; c < kNu; ++c) {
          const double a_max = 1.0 / kAccInv[c];
          const double dv = a_max * 0.001;
          if (vel[c] < cmd[c]) {
            vel[c] = std::min(vel[c] + dv, cmd[c]);
          } else {
            vel[c] = std::max(vel[c] - dv, cmd[c]);
          }
          pos[c] += vel[c] * 0.001;
        }
        double th1ms[kNs] = {0};
        for (int c = 0; c < 7; ++c) { th1ms[c] = pos[c]; th1ms[7 + c] = vel[c]; }
        double ep[7];
        EndPos(th1ms, im.racket_len, use_plat_vx, ep);
        if (ep[2] < 0.35) { is_hit = true; break; }
      }
    }
    if (is_hit) {
      out->status = LegacyNmpcStatus::kSelfCollision;
      out->solver_ok = solver_ok;
      out->solve_ms = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count() * 1e3;
      ++rejects;
      return false;
    }
  }

  // ---- 首解捕获触球底盘速度(后续重解不覆盖;下游调 hit_spd 用) ----
  if (!im.pred_valid) {
    im.pred_vx = cs[7];
    im.pred_vy = cs[8];
    im.pred_valid = true;
  }
  out->pred_vxy_at_hit[0] = im.pred_vx;
  out->pred_vxy_at_hit[1] = im.pred_vy;

  // ---- 末端评价 + 误差补偿推理(下一拍生效;剩余≤50ms 锁存) ----
  double ep[7];
  EndPos(cs, im.racket_len, use_plat_vx, ep);
  for (int i = 0; i < 7; ++i) out->end_pos[i] = ep[i];
  {
    const double dt_ms = (req.dt[0] + req.dt[1] + req.dt[2]) * 1000.0;
    if (im.min_dt_ms < 0.0 || dt_ms < im.min_dt_ms) im.min_dt_ms = dt_ms;
    if (!im.comp_locked && im.min_dt_ms <= 50.0) im.comp_locked = true;
    if (error_comp) {
      double raw[5] = {0};
      error_comp(ep, raw);
      if (!im.comp_locked) {
        im.comp[0] = std::max(-0.02, std::min(raw[0], 0.16));
        im.comp[1] = std::max(-0.10, std::min(raw[1], 0.10));
        im.comp[2] = std::max(-0.05, std::min(raw[2], 0.10));
        im.comp[3] = std::max(-15.0, std::min(raw[3], 15.0));
        double dy = WrapYaw90(raw[4]);
        if (!std::isfinite(dy) || std::fabs(dy) > 30.0) dy = 0.0;
        im.comp[4] = std::max(-30.0, std::min(dy, 30.0));
      }
    }
  }

  // ---- 解跳变守门(vs 上一接受解:m1>60° / m3>50° / m4>90°) ----
  if (im.have_last) {
    const double d1 = std::fabs(cs[2] - im.last_term[2]) / M_PI * 180.0;
    const double d3 = std::fabs(cs[4] - im.last_term[4]) / M_PI * 180.0;
    const double d4 = std::fabs(cs[5] - im.last_term[5]) / M_PI * 180.0;
    if (d1 > 60.0 || d3 > 50.0 || d4 > 90.0) {
      out->status = LegacyNmpcStatus::kJump;
      out->solver_ok = solver_ok;
      out->solve_ms = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0).count() * 1e3;
      ++rejects;
      if (getenv("NMPC6_DEBUG"))
        std::fprintf(stderr, "[nmpc6] jump: %.1f/%.1f/%.1f deg\n", d1, d3, d4);
      return false;
    }
  }

  // ---- 角度窗守门(终态角须在动态窗+10° 内;m5 不查,原样) ----
  if (cs[2] > (im.m1_a2 + 10.0) * D2R || cs[2] < (im.m1_a1 - 10.0) * D2R ||
      cs[3] > (im.m2_a2 + 10.0) * D2R || cs[3] < (im.m2_a1 - 10.0) * D2R ||
      cs[4] > (im.m3_a2 + 10.0) * D2R || cs[4] < (im.m3_a1 - 10.0) * D2R ||
      cs[5] > (im.m4_a2 + 10.0) * D2R || cs[5] < (im.m4_a1 - 10.0) * D2R) {
    out->status = LegacyNmpcStatus::kAngleLimit;
    out->solver_ok = solver_ok;
    out->solve_ms = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count() * 1e3;
    ++rejects;
    return false;
  }

  // ---- 精度门(双档):严格 = 三轴<acc_thr 且 |vx|>0.6·hit_spd;
  //      宽松 = 三轴<0.20 且 IPOPT 成功。严格档不要求 solver_ok(原语义),
  //      但 solve_number/跳变锚/m5 收紧只在 严格+solver_ok 时推进 ----
  const double ex = std::fabs(ep[0] - tgt[0]);
  const double ey = std::fabs(ep[1] - tgt[1]);
  const double ez = std::fabs(ep[2] - tgt[2]);
  bool strict = ex < im.acc_thr && ey < im.acc_thr && ez < im.acc_thr &&
                std::fabs(ep[4]) > hit_spd * 0.6;
  bool loose = ex < 0.20 && ey < 0.20 && ez < 0.20 && solver_ok;
  if (!strict && !loose) {
    out->status = LegacyNmpcStatus::kAccFail;
    out->solver_ok = solver_ok;
    out->solve_ms = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count() * 1e3;
    ++rejects;
    if (getenv("NMPC6_DEBUG"))
      std::fprintf(stderr, "[nmpc6] acc-fail: e=(%.3f,%.3f,%.3f) vx=%.2f/%.2f\n",
                   ex, ey, ez, ep[4], hit_spd * 0.6);
    return false;
  }

  // ---- 接受:下发目标 = rollout 终态 + 末端位置误差反馈(x/y);
  //      首个 严格+成功 解收紧 m5 角窗(沿挥拍方向 ±50°) ----
  if (strict && solver_ok) {
    ++im.solve_number;
    std::memcpy(im.last_term, cs, sizeof(double) * kNs);
    im.have_last = true;
    if (im.solve_number == 1) {
      const double tgt5 = cs[6] * 180.0 / M_PI;
      const double cur5 = cs_in[6] * 180.0 / M_PI;
      if (tgt5 > cur5) { im.m5_lo = cur5 - 10; im.m5_hi = tgt5 + 50; }
      else             { im.m5_lo = tgt5 - 50; im.m5_hi = cur5 + 10; }
    }
  }
  for (int i = 0; i < kNs; ++i) out->term_state[i] = cs[i];
  out->term_state[0] += (tgt[0] - ep[0]);   // 末端误差反馈(拍心差补到底盘)
  out->term_state[1] += (tgt[1] - ep[1]);
  for (int i = 0; i < kNp * kNu; ++i)
    out->controls[i] = res_control_all[i];
  std::memcpy(im.guess, res_control_all.data(), sizeof(double) * kNp * kNu);
  out->status = strict ? LegacyNmpcStatus::kOk : LegacyNmpcStatus::kLoose;
  out->solver_ok = solver_ok;
  out->solve_ms = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count() * 1e3;
  ++solves;
  ms_sum += out->solve_ms;
  ms_max = std::max(ms_max, out->solve_ms);
  if (getenv("NMPC6_DEBUG"))
    std::fprintf(stderr, "[nmpc6] %s e=(%.3f,%.3f,%.3f) vx=%.2f ms=%.1f\n",
                 strict ? "ok" : "loose", ex, ey, ez, ep[4], out->solve_ms);
  return true;
}

}  // namespace sckd
