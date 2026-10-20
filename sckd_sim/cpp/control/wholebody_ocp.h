#pragma once
// 方案 B v2 在线 OCP 协调器 v1(2026-10-08,PROGRESS 待办 #1)—— 底盘 OCP。
//
// 【问题】底盘平面双积分器 (x,y,vx,vy),输入 (ax,ay);硬约束:每轴速度钳
// 1.5(伺服 clamp 同源)/ a_max 8(摩擦圆×0.8)。终端(t_hit)自由度:
//   - 位置:代价拉向 anchor* = p*_xy − back*·d̂(back*≈0.45 取自离线联合
//     OCP 扫描 wholebody_ocp.py v2sweep:0.36-0.47 近常值,back=0.8 钉死即
//     三轮蒸馏失败的那一档);
//   - 速度:代价拉向 v*_xy(拍速需求的 xy,离线联合最优的终端底盘速度沿
//     v* 方向顶钳,非 B′ 的 d̂ 投影)。
// 全凸、单 QP、无 SQP —— 联合(含臂)SQP 原型已验证可行但收敛调优未闭合
// (信任域与线性化互追,留档 git),v1 先回答立项核心问题:
//
//   【可达性在优化内】三轮蒸馏(v_base 规则/信念对齐/back(z))闭环全败的
//   共同根因 = 几何理想目标超出闭环可达集(back(z) 实测:init-fail 51 次、
//   锁止底盘误差 0.93-1.57m)。本求解器从实测状态出发,终端只能落在双积分
//   可达集内 —— anchor* 不可达时沿可达集滑动,而非命令不可达目标。
//
// 【交接契约(设计文档 §6.2 B′)】输出终端底盘位姿/速度:ArmSwing 终值按
// base_pred=term_base 解(替代 triage 的 P 控误差外推),差项 v_base*=
// term_vbase;底盘层 P+ff 直接跟踪本解轨迹(替代 BaseRefGen Hermite——
// 其只能表沿 d̂ 的终端速度)。
//
// 守门(旧版 nmpc.cc 适配):解时窗(热解=20Hz 拍周期,首拍冷解 80ms)、
// 终端跳变拒收(p_hit 未动而终端底盘跳 >0.35m)、末段窗(t_hit−t<0.15 冻结)。
// 失败语义:Solve 返回 false,调用方保持上一参考;连续失败回退 B′ 由调用方管。
#include <vector>

namespace sckd {

struct WboLimits {
  double v_axis = 1.5;        // 底盘每轴速度钳(m2_intercept servo clamp 同源)
  double a_max = 8.0;         // 底盘加速度界(§6.2 摩擦圆×0.8 工程裕度)
  double back_star = 0.45;    // 终端后撤距下限(离线 v2sweep:低位联合最优
                               // ≈0.36-0.47,即 B′ 0.8 的一半)
  double back_star_hi = 0.75; // 高位后撤距(v2sweep:z≥1.1 钉 0.45 的拍心松弛
  double z_lo = 0.75, z_hi = 1.05;  // 反而比 0.8 差 5cm —— 低位 0.45/高位
                               // 0.75 线性插值;臂不在优化内的补偿,v1 已知边界)
  double w_anchor = 4.0;      // 终端位置代价(对 anchor*)
  double w_vel = 0.5;         // 终端速度代价(对 v*_xy;能量项天然反对过飞)
  double ms_budget = 20.0;    // 热解时窗 = 20Hz 拍周期(超周期拒收)
  double ms_budget_first = 80.0;
  double jump_reject = 0.35;  // 终端底盘跳变拒收阈(m)
};

struct WboRequest {
  double t_now = 0, t_hit = 0;
  double p_hit[3] = {0};        // 击球点(取 xy 构造 anchor*)
  double v_star[3] = {0};       // 拍速需求(ReboundInverse;取 xy 作终端速度目标)
  double base0[2] = {0}, vbase0[2] = {0};   // 实测(世界系,yaw=0 契约)
};

struct WboResult {
  bool ok = false;
  double term_base[2] = {0};    // 终端底盘位置(=ArmSwing 换锚 base_pred)
  double term_vbase[2] = {0};   // 终端底盘速度(=差项 v_base*)
  double cost = 0, ms = 0;
  // 底盘参考轨迹(绝对时刻;t0=t_now,h=(t_hit−t_now)/N):逐节点位置/速度,
  // P+ff 线性插值跟踪 —— 轨迹动态可行(约束内),区别于 BaseRefGen 的
  // 可达性事后近似
  double t0 = 0, h = 0;
  std::vector<double> rx, ry, rvx, rvy;   // size N+1
};

class WholebodyOcp {
 public:
  static constexpr int kN = 12;   // 视界步数(~70ms 粒度,P+ff 插值跟踪)

  explicit WholebodyOcp(const WboLimits& lim = WboLimits()) : lim_(lim) {}

  // 求解(20Hz 调用);false = 拒收(守门或 QP 失败),调用方保持上一参考
  bool Solve(const WboRequest& req, WboResult* out);

  // 诊断统计(实例级累计)
  int solves = 0, fails = 0, rejects = 0, timeouts = 0;
  double ms_max = 0, ms_sum = 0;

 private:
  WboLimits lim_;
  bool has_prev_ = false;
  double prev_term_[2] = {0}, prev_p_hit_[3] = {0};
};

}  // namespace sckd
