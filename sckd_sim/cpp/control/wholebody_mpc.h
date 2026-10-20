#pragma once
// 方案 B 全身 MPC(v3,2026-10-09 立项):联合(底盘+臂)OCP 滚动重解,臂轨迹
// 直接出自优化 —— 与 ocpfly(只供锚,挥拍层原地不动)的本质区别 = 留档重开
// 前置条件"挥拍层与锚几何联合重整定"的正面实现。
//
// 【问题】N=12 步,x=[x,y,vx,vy,q1..5,dq1..5](14),u=[ax,ay,ddq1..5](7);
// 双积分器动力学;硬界:每轴速度钳 1.5/a_max 8/q 限位/dq/ddq 界;
// 终端流形(t_hit,全部【无界软松弛】+大二次罚):
//   拍心 = p* + sp(base+FK(q_N),线性化)
//   拍速 = v* + sv(v_base + J(q_N)·dq_N,含 D_lJ 交叉项的精确一阶展开)
//   法向 (n·n*)² 软代价
// 非线性只在终端(FK/J/n(q_N))→ SQP 每轮线性化(OSQP 值更新,模式一次 setup)。
//
// 【与上一版联合 SQP 的关键差异(2026-10-09 失败复盘)】slack 有界(±0.05)
// 曾使线性化漂移时子问题不可行(实测残余不可行 0.013)→ 迭代在解支间打飞。
// 无界松弛+罚 → 子问题恒可行(软约束 MPC 标准形态),信任域仅作步长阻尼。
//
// 【不爬行的原因】(对照 ocpfly 参考重启病):终端松弛罚(1e4/m²)远大于
// 控制能量(~1e-2),计划对"按时到达"有硬压力,早期段不会偷懒;执行按
// 绝对时刻采样最后接受计划(非每拍从当前位置重启参考)。
//
// 守门:解时窗(首拍 100ms/热解 40ms——研究口径,真机实时需 acados/降 N,
// 诚实标注)+终端底盘跳变拒收+末段窗(t_hit−t<0.15 冻结)。失败=调用方
// 保持上一计划,连续失败回退 B′ 由调用方管。
#include <vector>
#include <string>

#include "control/arm_ik.h"

namespace sckd {

struct WmLimits {
  double v_axis = 1.5;                       // 底盘每轴速度钳(servo clamp 同源)
  double a_max = 8.0;
  double dq_lim[5] = {18, 18, 25, 25, 30};   // wholebody_ocp.py DQ_LIM 同源
  double ddq_lim[5] = {80, 80, 120, 120, 200};
  double w_sp = 2e3;                         // 终端拍心松弛罚(/m²,无界;1e4 与
                                             // 控制权 1e-6 差 10 个量级,OSQP 相对
                                             // 口径下解但原始量纲违约 m/s 级)
  double w_sv = 2e3;                         // 终端拍速松弛罚(/(m/s)²,无界)
  double w_normal = 50.0;                    // (n·n*)² 软代价
  double w_anchor = 50.0;                    // 终端底盘锚吸引 w·|base_N−anchor|²
                                             // (闭环 2026-10-09 定位:无此代价
                                             // 时"原地停车+臂全伸展凑流形"比
                                             // 走位省能量,重解每拍续 park、末
                                             // 段 0.3s 冲刺 0.5m 跟不上——
                                             // ocpfly 爬行病的变体。层级:
                                             // 流形(2e3/m²)≫锚≫控制能量(~1),
                                             // 终端被拉回锚邻域中心、走位摊全程)
  double w_corr = 10.0;                      // 底盘走廊节点位置跟踪代价(/m²/节点,
                                             // 配 WmRequest::corr_*;锚吸引解决
                                             // "去哪",走廊解决"何时走"——最小
                                             // 能量解无走廊时走位摊给永远被重
                                             // 解取消的"以后")
  double trust0 = 0.60;                      // q_N 信任域初值(rad)
  double trust_min = 0.03, trust_max = 1.50;
  double converge_pos = 0.008;               // 非线性拍心残差收敛阈(m,甜区
                                             // 容差内;0.003 曾致多数解烧满
                                             // 8 迭代被解时窗拒收,2026-10-09)
  double accept_pos = 0.05;                  // 接受门槛:best 迭代 pos_res 超
                                             // 此值返回 false(按拒收计)。
                                             // 闭环定位(2026-10-09):无此门槛
                                             // 时 best-effort 解(残差 0.1-1m)
                                             // 也被接受,终端底盘跑到臂不可达
                                             // 处,底盘忠实跟踪 → 触球偏 1m 级
  double ms_budget_first = 100.0;            // 首拍冷解
  double ms_budget = 60.0;                   // 热解(20Hz 拍 50ms,研究口径
                                             // 容忍至 60——闭环实测 SQP 信任域
                                             // 每拍从 0.60 重走扩张链,40ms 拒
                                             // 收占 55%,计划过早冻结)
  double jump_reject = 0.35;                 // 终端底盘跳变拒收(m)
  double reseed_dp = 0.10;                   // 参考跳变重播种阈(m):p_hit 距
                                             // 上拍 >此值弃 SQP 热启动,改用
                                             // q_seed 线性化(闭环定位:预测器
                                             // 收敛期 p_hit 漂 0.6m,热启动终
                                             // 端锚死旧流形,8 迭代追不上→
                                             // 质量门槛连拒→计划冻结陈旧点)
  double q_lo[5] = {-3.5, -3.5, -3.5, -3.5, -3.5};
  double q_hi[5] = {3.5, 3.5, 3.5, 3.5, 3.5};
};

struct WmRequest {
  double t_now = 0, t_hit = 0;
  double p_hit[3] = {0}, v_star[3] = {0}, n_star[3] = {0};
  double base0[2] = {0}, vbase0[2] = {0};    // 实测(世界系,yaw=0 契约)
  double q0[5] = {0}, dq0[5] = {0};          // 实测臂状态
  double q_seed[5] = {0};                    // 首拍线性化播种(调用方 IK@规划锚)
  bool q_seed_ok = false;
  // 终端底盘锚邻域(可选):终端 base 限在 base_anchor±base_anchor_r。
  // 动机(闭环 2026-10-09):终端自由时重解间漂移无界(实测 1.1m 跳变,臂全
  // 伸展凑流形),浪费底盘行程且物理不可达;锚邻域恰覆盖 v2 量化的耦合红利
  // 区(back 0.45-0.75 ⇔ 锚 ±0.35m),B′ 启发式锚的稳定性红利保留
  double base_anchor[2] = {0};
  double base_anchor_r = 0.35;
  bool base_anchor_ok = false;
  // 底盘走廊(可选,k=0..N 节点参考,世界系):节点位置软跟踪绝对时刻标称。
  // 动机(2026-10-09 闭环定位,ocpfly 爬行病在联合 OCP 内的变体):仅端点
  // 约束下,最小能量解把走位摊给"计划内的以后",每拍重解 X0=实测(静止)
  // → 执行只采到各计划 k≈0(速度≈0),"以后"被新计划永远重置,底盘 park 到
  // 末段才冲刺。走廊给出即时位置梯度,走位从当前拍开始
  std::vector<double> corr_x, corr_y;
  bool corr_ok = false;
  // 接触节点(2026-10-09,镜像 ArmSwing s_hit=0.72 内点接触):>0 时命中流形
  // (pos/vel/法向/锚/信任域)挂在内节点 k 而非终端节点 N,其后节点=自由随挥
  // (OCP 自己成形的减速段)。动机:峰值拍速贴着终端节点 + PD 伺服带宽有限
  // (τ̂ 只补纯延迟不补幅值衰减)→ 触球总抓到爬坡段拍速/面内偏 0.1-0.3m;
  // 内点接触让峰速落在触球带内、且随挥不是恒速外推而是规划出来的。
  // ≤0 或 >kN = 流形在终端(legacy,单测口径)
  int k_contact = -1;
};

struct WmPlan {
  bool ok = false;
  double contact_time = 0;                 // 诊断：离散接触节点绝对时刻
  double t0 = 0, h = 0;                      // 计划锚(绝对时刻;t0=t_now)
  // 逐节点轨迹(k=0..N;执行层绝对时刻线性插值采样)
  std::vector<double> rx, ry, rvx, rvy;              // 底盘
  std::vector<double> rq, rdq, rddq;                 // 臂(5×(N+1) 行主序)
  double term_base[2] = {0}, term_vbase[2] = {0};
  double term_q[5] = {0}, term_dq[5] = {0};
  double slack_p[3] = {0}, slack_v[3] = {0};
  double pos_res = 1e9;                      // 非线性拍心残差(收敛度)
  double vel_res = 1e9;
  double n_dot = -1;
  double cost = 0, ms = 0;
  int sqp_iters = 0;
};

class WholebodyMpc {
 public:
  static constexpr int kN = 12;
  static constexpr int kMaxSqpIters = 8;

  // lim.q_lo/q_hi 从 mjModel 覆盖(racket site→base_link 链,无限位取 ±3.5)
  WholebodyMpc(ArmIk& ik, const mjModel* m, const WmLimits& lim);

  // 求解(20Hz 调用);false=拒收(守门/求解失败),调用方保持上一计划
  bool Solve(const WmRequest& req, WmPlan* out);

  // 诊断统计(实例级)
  std::string last_reason; double last_ms=0, last_pos_res=1e9;
  int solves = 0, fails = 0, rejects = 0, timeouts = 0;
  double ms_max = 0, ms_sum = 0;

 private:
  ArmIk& ik_;
  WmLimits lim_;
  bool has_prev_ = false;
  double prev_term_[2] = {0}, prev_p_hit_[3] = {0};
  double q_prev_[5] = {0}, dq_prev_[5] = {0};   // SQP 热启动(上一拍终端)
  double trust_warm_ = -1.0;   // 信任域跨解热启动(上一接受解终值;省每拍
                               // 重走 ×1.4 扩张链的 3 个迭代,闭环超时定位)
};

}  // namespace sckd
