#include "runtime/controller_adapters.h"
#include "robot_fk.h"
namespace sckd::runtime {
using sckd_nmpc6::RobotArmPos;
// v2 triage(2026-10-08):预测实际底盘位姿@t_hit,ArmSwing 终值 IK/J/J5 对准
// 按它解——fly-through 底盘滞后 0.1-0.2m,按规划 base_target 解则拍随基座
// 平移、无人补偿(面内 miss 0.16-0.47,抽球族定位)。
// 模型 = P 控误差动力学:pred = base_target − lag·exp(−dtg/τ),τ=1/kP=0.4s
// (t_hit 时参考恰为 base_target,当前位置误差按闭环时间常数衰减)。比线性
// 外推稳:加速段外推会把锚拖离目标 0.4-0.7m(fixed 族实测 100%→30-40%);
// 误差衰减模型对 lag≈0 的族自然回目标(无阈值跳变)。仅 fly 模式调用
void FillBasePred(HitPlan& plan, SimBackend& be, double t_now,
                         double ref_x, double ref_y) {
  if (!plan.feasible) return;
  const double dtg = plan.t_hit - t_now;
  if (dtg <= 0) { plan.base_pred_ok = false; return; }
  const BaseState b = be.base_state();
  const double lag_x = ref_x - b.x, lag_y = ref_y - b.y;   // 参考−实际(滞后>0)
  const double decay = std::exp(-dtg / 0.40);
  double px = plan.base_target[0] - lag_x * decay;
  double py = plan.base_target[1] - lag_y * decay;
  const double ddx = px - plan.base_target[0], ddy = py - plan.base_target[1];
  const double dd = std::hypot(ddx, ddy);
  if (dd > 0.40) {   // 限幅(异常 lag 不把终值拖出工作空间)
    px = plan.base_target[0] + ddx / dd * 0.40;
    py = plan.base_target[1] + ddy / dd * 0.40;
  }
  plan.base_pred[0] = px;
  plan.base_pred[1] = py;
  plan.base_pred_ok = true;
}

// ocpfly 拍内求解(2026-10-09):填实测状态(世界系;BaseState vx/vy 为车体系
// 需旋转),成功则写交接契约(plan.base_pred=OCP 终端底盘、v_base*=OCP 终端
// 速度)并返回 true。p_hit/v*(=v_racket) 取自当前 plan(RefineAt 精化随读)
bool RunOcp(WholebodyOcp& wbo, HitPlan& plan, SimBackend& be, double ts,
                   WboResult* out) {
  WboRequest req;
  req.t_now = ts;
  req.t_hit = plan.t_hit;
  for (int i = 0; i < 3; ++i) {
    req.p_hit[i] = plan.p_hit[i];
    req.v_star[i] = plan.v_racket[i];
  }
  const BaseState b = be.base_state();
  req.base0[0] = b.x;
  req.base0[1] = b.y;
  const double c = std::cos(b.yaw), sn = std::sin(b.yaw);
  req.vbase0[0] = c * b.vx - sn * b.vy;
  req.vbase0[1] = sn * b.vx + c * b.vy;
  if (!wbo.Solve(req, out)) return false;
  // 臂锚投影:OCP 可达集滑动的终端(远侧发球滑到离 p_hit >1m)不在臂工作
  // 环带内 → 终值 IK 必败(init-fail 连败,gs 侧发 70→40 回归定位,2026-10-09)。
  // 分工:调度终点 = OCP 原始终端(可达性感知参考);臂锚 = 投影到 p_hit 的
  // 臂可达环带 [0.30,1.00]m(几何一致性优先,底盘到不了的位置误差由拍面
  // 容差吸收 —— 与 B′/triage 的 base_target 锚同一性质,只是更近/更优)
  {
    constexpr double kRMin = 0.30, kRMax = 1.00;
    double ax = out->term_base[0], ay = out->term_base[1];
    const double ddx = ax - plan.p_hit[0], ddy = ay - plan.p_hit[1];
    const double dp = std::hypot(ddx, ddy);
    if (dp > kRMax) {
      ax = plan.p_hit[0] + ddx * kRMax / dp;
      ay = plan.p_hit[1] + ddy * kRMax / dp;
    } else if (dp < kRMin && dp > 1e-6) {
      ax = plan.p_hit[0] + ddx * kRMin / dp;
      ay = plan.p_hit[1] + ddy * kRMin / dp;
    }
    plan.base_pred[0] = ax;
    plan.base_pred[1] = ay;
  }
  plan.base_pred_ok = true;
  plan.v_base_star[0] = out->term_vbase[0];
  plan.v_base_star[1] = out->term_vbase[1];
  return true;
}

// wbmpc 拍内求解(2026-10-09,方案 B v3 联合 MPC):填实测状态(底盘速度车体系
// →世界系旋转),成功即由调用方保存 WmPlan——臂/底盘执行全部直接采样它。
// q_seed 每拍都算(IK@当前 base_target,RefineAt 随读更新,~1ms):求解器在
// p_hit 参考跳变(>0.10m)时用它重播种,否则走 SQP 热启动
bool RunWbmpc(WholebodyMpc& wb, const HitPlan& plan, SimBackend& be,
                     ArmIk& ik, double ts, const double q_cur[5],
                     const double dq_cur[5], bool have_prev,
                     const BaseRefGen::Schedule& sched, WmPlan* out, ControlTrace& trace) {
  StageProfile::Scope profile("wbmpc_with_ik");
  // 内点接触(镜像 ArmSwing s_hit=0.72):接触流形挂内节点 k_c,其后 0.25s
  // = OCP 自己成形的随挥(不是恒速外推);接触时刻提前 30ms 让峰速落在触球
  // 带内——终端接触版实测触球抓到爬坡段拍速(1.5-2.5 vs 需求 3.3 m/s,
  // 出球 6.5 vs 基线 8.2+),2026-10-09 闭环定位
  constexpr double kWbLead = 0.030;
  constexpr double kWbTail = 0.25;
  WmRequest req;
  req.t_now = ts;
  const double t_c = plan.t_hit - kWbLead;    // 接触时刻(绝对)
  req.t_hit = t_c + kWbTail;                  // 视界终端(随挥段末)
  for (int i = 0; i < 3; ++i) {
    req.p_hit[i] = plan.p_hit[i];
    req.v_star[i] = plan.v_racket[i];
    req.n_star[i] = plan.n_face[i];
  }
  const BaseState b = be.base_state();
  req.base0[0] = b.x;
  req.base0[1] = b.y;
  const double c = std::cos(b.yaw), sn = std::sin(b.yaw);
  req.vbase0[0] = c * b.vx - sn * b.vy;
  req.vbase0[1] = sn * b.vx + c * b.vy;
  for (int j = 0; j < 5; ++j) { req.q0[j] = q_cur[j]; req.dq0[j] = dq_cur[j]; }
  {   // 播种:IK@当前规划锚(RefineAt 随读更新;have_prev 仅留诊断)
    (void)have_prev;
    const double q_ready[5] = {0.0, -1.0, 0.5, 0.0, 0.8};
    const IkResult seed = ik.Solve(plan.p_hit, plan.base_target[0],
                                   plan.base_target[1], 0.0, q_ready);
    if (seed.err < 0.03) {
      for (int j = 0; j < 5; ++j) req.q_seed[j] = seed.q[j];
      req.q_seed_ok = true;
    }
  }
  // 终端底盘锚邻域:±0.35m 围绕规划锚(覆盖 v2 红利区 back 0.45-0.75,
  // 杀死重解间终端 1m 级漂移/臂全伸展凑流形,2026-10-09 闭环定位)
  req.base_anchor[0] = plan.base_target[0];
  req.base_anchor[1] = plan.base_target[1];
  req.base_anchor_r = 0.35;
  req.base_anchor_ok = true;
  // 底盘走廊:节点位置拉向 BaseRefGen 绝对时刻标称(起锚于初规划时刻,不随
  // 重解重置)——否则最小能量解把走位摊给"计划内的以后",执行只采到各计划
  // k≈0(速度≈0),底盘 park 到末段才冲刺(ocpfly 爬行病的联合 OCP 变体)
  req.corr_x.resize(WholebodyMpc::kN + 1);
  req.corr_y.resize(WholebodyMpc::kN + 1);
  for (int k = 0; k <= WholebodyMpc::kN; ++k) {
    double vx, vy;
    BaseRefGen::EvalAt(sched, req.t_now + k * ((req.t_hit - req.t_now) / WholebodyMpc::kN),
                       &req.corr_x[k], &req.corr_y[k], &vx, &vy);
  }
  req.corr_ok = true;
  // 接触节点:t_c 对应的节点(留 ≥3 个随挥节点,≤N−3)
  req.k_contact = std::clamp(
      (int)std::lround((t_c - ts) / (req.t_hit - ts) * WholebodyMpc::kN),
      1, WholebodyMpc::kN - 3);
  const bool ok=StageProfile::Measure("wbmpc_solver", [&]{return wb.Solve(req, out);});
  if(trace.enabled()){
    nlohmann::json e={{"kind","solve"},{"mode","wbmpc"},{"t",ts},{"target",Vec(req.p_hit,3)},{"target_time",plan.t_hit},{"contact_time",req.t_now+req.k_contact*((req.t_hit-req.t_now)/WholebodyMpc::kN)},{"accepted",ok},{"reason",wb.last_reason},{"ms",wb.last_ms},{"pos_res",wb.last_pos_res}};
    if(ok){double p[3];RobotArmPos(sckd_nmpc6::ParseRobotGeometry(be.model()->names),out->term_base,out->term_q,p);e["endpoint_fk"]=Vec(p,3);e["n_dot"]=out->n_dot;}
    trace.Emit(e);
  }
  return ok;
}

}
