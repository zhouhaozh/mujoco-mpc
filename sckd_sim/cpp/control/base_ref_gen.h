#pragma once
// 底盘 MPC 参考轨迹生成(MPC 上场,2026-10-07):击球计划 → 2s 视界到位参考。
// 【绝对时间锚定】参考在初规划时刻起锚(位置/时刻),之后每次重解按绝对时刻取值
// r(t_now + k·dt)—— 若每拍从当前位置重建,smoothstep 起点速度为 0,参考每拍
// "重新开始且不动",MPC 追着不动参考爬行(实测锁止偏差 0.5m,2026-10-07);
// 绝对时刻参考落后时会拉着机器人赶上(m1_mpc_tracking 同哲学)。
// 位置:smoothstep(锚点 → base_target,T_arrive 内),到位后保持;
//      终点 tx,ty 可随 RefineAt 精化更新(小跳变,起锚位置/时刻不变)。
// 速度:smoothstep 析出导数 6s(1−s)/T·(target−anchor),世界系 → 车体系
//      (MPC 模型 vx,vy 为车体系,m1_mpc_tracking 契约)。
// 航向:目标 0(臂 IK 按 yaw=0 解,hit_planner 契约);k=0 钳 ±0.2 防初始违反航向窗。
#include <algorithm>
#include <array>
#include <cmath>

#include "control/chassis_mpc.h"

namespace sckd {

class BaseRefGen {
 public:
  using Ref = std::array<double, ChassisMpc::kNx * (ChassisMpc::kN + 1)>;
  static constexpr double kDt = 0.020;   // 与 MPC 视界步长一致

  struct Schedule {
    double t0 = 0;          // 起锚时刻(初规划时刻,绝对仿真时刻)
    double ax = 0, ay = 0;  // 起锚位置(世界系)
    double tx = 0, ty = 0;  // 终点(RefineAt 可更新)
    double t_arrive = 0.8;  // 到位时长(自 t0 起)
  };

  // x0=[px,py,ψ,vx,vy,wz](vx,vy 车体系;t0 已按绝对时刻取值)
  // 速度剖面:梯形(0.15s 爬升 / 巡航 / 0.25s 收敛)—— smoothstep 起步慢,早期
  // 位置误差≈0 无驱动力,MPC 系统性落后参考速度 30-40%(2026-10-07 实测);
  // 梯形早发力,到位即停。
  static void Build(const Schedule& sc, double t_now, const double x0[6],
                    Ref& ref, double cs2[2 * ChassisMpc::kN]) {
    constexpr int N = ChassisMpc::kN, NX = ChassisMpc::kNx;
    const double dx = sc.tx - sc.ax, dy = sc.ty - sc.ay;
    const double Ta = std::max(sc.t_arrive, 1e-3);
    // 归一化梯形位移剖面 p(τ),τ∈[0,1]:爬升 a=0.15/Ta、收敛 b=0.25/Ta(单位 τ)
    const double a = std::min(0.15 / Ta, 0.45), b = std::min(0.25 / Ta, 0.45);
    // 面积归一:∫v dτ=1,v_peak = 1/(1-(a+b)/2)
    const double vp = 1.0 / (1.0 - (a + b) / 2.0);
    auto prof = [&](double tau, double* p, double* dp) {
      tau = std::clamp(tau, 0.0, 1.0);
      if (tau < a) { *p = 0.5 * vp * tau * tau / a; *dp = vp * tau / a; }
      else if (tau <= 1.0 - b) { *p = vp * (tau - 0.5 * a); *dp = vp; }
      else {
        const double s = (1.0 - tau) / b;
        *p = 1.0 - 0.5 * vp * b * s * s;
        *dp = vp * s;
      }
    };
    for (int k = 0; k <= N; ++k) {
      double p, dp;
      prof((t_now + k * kDt - sc.t0) / Ta, &p, &dp);
      dp /= Ta;                                          // dτ/dt = 1/Ta
      double* r = &ref[k * NX];
      double psi = 0.0;
      if (k == 0) psi = std::clamp(psi, x0[2] - 0.2, x0[2] + 0.2);
      r[0] = sc.ax + dx * p;
      r[1] = sc.ay + dy * p;
      r[2] = psi;
      const double vwx = dx * dp, vwy = dy * dp;         // 世界系参考速度
      const double c = std::cos(psi), sn = std::sin(psi);
      r[3] =  c * vwx + sn * vwy;
      r[4] = -sn * vwx + c * vwy;
      r[5] = 0.0;
    }
    for (int k = 0; k < N; ++k) {
      cs2[2 * k] = std::cos(ref[(k + 1) * NX + 2]);
      cs2[2 * k + 1] = std::sin(ref[(k + 1) * NX + 2]);
    }
  }
};

}  // namespace sckd
