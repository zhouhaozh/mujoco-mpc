#include "runtime/intercept_modules.h"
#include "runtime/controller_adapters.h"
#include "runtime/plan_supervisor.h"
#include "runtime/synchronous_schedule.h"
#include "robot_fk.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
namespace sckd::runtime {
using sckd_nmpc6::RobotArmPos;

void UpdatePlanning(PlanningContext c, int t, double ts) {
  auto& be = c.be;
  auto& servo = c.servo;
  auto& ik = c.ik;
  auto& predictor = c.predictor;
  auto& planner = c.planner;
  auto& intent = c.intent;
  auto& plan = c.plan;
  auto& locked_plan = c.locked_plan;
  auto& wbo = c.wbo;
  auto& ocp_res = c.ocp_res;
  auto& wb = c.wb;
  auto& wb_plan = c.wb_plan;
  auto& n6 = c.n6;
  auto& aswing = c.aswing;
  auto& swing = c.swing;
  auto& mpc_sched = c.mpc_sched;
  auto& mpc_ref = c.mpc_ref;
  auto& trace = c.trace;
  auto& cfg = c.cfg;
  auto& n6_pos = c.n6_pos;
  auto& n6_vel = c.n6_vel;
  auto& m = c.m;
  auto& d = c.d;
  auto& steer_rate_mode = c.steer_rate_mode;
  auto& fly = c.fly;
  auto& n6_mode = c.n6_mode;
  auto& wb_mode = c.wb_mode;
  auto& ocp_mode = c.ocp_mode;
  auto& use_qp = c.use_qp;
  auto& corrected = c.corrected;
  auto& coherent = c.coherent;
  auto& chassis_model = c.chassis_model;
  auto& chassis_recovery = c.chassis_recovery;
  auto& chassis_guard = c.chassis_guard;
  auto& planned = c.planned;
  auto& ref_built = c.ref_built;
  auto& ocp_have = c.ocp_have;
  auto& ocp_dead = c.ocp_dead;
  auto& wb_have = c.wb_have;
  auto& wb_dead = c.wb_dead;
  auto& n6_have = c.n6_have;
  auto& n6_dead = c.n6_dead;
  auto& diagnostic_previous_request = c.diagnostic_previous_request;
  auto& locked_saved = c.locked_saved;
  auto& replans = c.replans;
  auto& ocp_fails_streak = c.ocp_fails_streak;
  auto& wb_fails_streak = c.wb_fails_streak;
  auto& n6_fails_streak = c.n6_fails_streak;
  auto& n6_recent_rejects = c.n6_recent_rejects;
  auto& n6_attempts = c.n6_attempts;
  auto& n6_fallbacks_t = c.n6_fallbacks_t;
  auto& wb_fallbacks = c.wb_fallbacks;
  auto& n6_ms = c.n6_ms;
  auto& trial = c.trial;
  auto& t_plan0 = c.t_plan0;
  auto& t_bounce_plan = c.t_bounce_plan;
  auto& n6_t0 = c.n6_t0;
  auto& n6_attempt_ms = c.n6_attempt_ms;
  auto& n6_attempt_max_ms = c.n6_attempt_max_ms;
  auto& diagnostic_goal_time = c.diagnostic_goal_time;
  auto& diagnostic_plan_t = c.diagnostic_plan_t;
  auto& diagnostic_contact_time = c.diagnostic_contact_time;
  auto& diagnostic_previous_target_time = c.diagnostic_previous_target_time;
  auto& lock_t = c.lock_t;
  auto& kInitDelay = c.kInitDelay;
  auto& q_cur = c.q_cur;
  auto& dq_cur = c.dq_cur;
  auto& diagnostic_goal = c.diagnostic_goal;
  auto& diagnostic_previous_target = c.diagnostic_previous_target;
  auto& lock_p = c.lock_p;
  // 初次规划:预测器就绪即规划(越早裕度越大)
  if (!planned && predictor.initialized()) {
    const BaseState bs = be.base_state();
    plan = StageProfile::Measure("initial_planning", [&]{return planner.Plan(predictor, bs, ts, intent);});
    if (plan.feasible) {
      planned = true;
      t_plan0 = ts;
      t_bounce_plan = predictor.NextBounceTime();
      // 定点模式清零 fly 意图:底盘不飞就不会有那份拍速,臂不得预支
      // (RefineAt 不改 v_base_star,一次清零覆盖全程)
      if (!fly) { plan.v_base_star[0] = plan.v_base_star[1] = 0; }
      // MPC 参考起锚:当前位置 → base_target。fly-through 恰在 t_hit 通过
      // 目标(触球时刻底盘在动,拍速含底盘贡献);定点提前 0.15s 到达停稳
      mpc_sched.t0 = ts;
      mpc_sched.ax = bs.x; mpc_sched.ay = bs.y;
      mpc_sched.tx = plan.base_target[0];
      mpc_sched.ty = plan.base_target[1];
      if (fly) {
        mpc_sched.t_arrive = std::clamp(plan.t_hit - ts, 0.35, 1.5);
        mpc_sched.v_fly = std::hypot(plan.v_base_star[0], plan.v_base_star[1]);
      } else {
        mpc_sched.t_arrive = std::clamp(plan.t_hit - ts - 0.15, 0.30, 1.2);
      }
      // ocpfly:首拍解(规划期一次性,冷解预算 80ms);成功即写交接契约,
      // 之后 +0.25s 的延迟起挥 Init 直接读到 base_pred
      if (ocp_mode && RunOcp(wbo, plan, be, ts, &ocp_res)) {
        ocp_have = true;
        // 底盘跟踪走 BaseRefGen 绝对时刻参考(OCP 参考每拍从当前位置重启
        // 会"参考回退爬行",首测底盘 1s 内位移 <0.2m,2026-10-09 定位);
        // OCP 供可达性感知终点:tx/ty/v_fly 每拍随重解更新(小跳变,起锚不变)
        mpc_sched.tx = ocp_res.term_base[0];
        mpc_sched.ty = ocp_res.term_base[1];
        mpc_sched.v_fly = std::min(1.2, std::hypot(ocp_res.term_vbase[0],
                                                   ocp_res.term_vbase[1]));
        std::printf("  [ocp] t=%.2f 终端base=(%.2f,%.2f) v=(%.2f,%.2f) %.1fms\n",
                    ts, ocp_res.term_base[0], ocp_res.term_base[1],
                    ocp_res.term_vbase[0], ocp_res.term_vbase[1], ocp_res.ms);
      }
      if (wb_mode) {
        // wbmpc 首拍解:OCP 轨迹从当前状态直达 t_hit(无独立"起挥"概念,
        // 延迟起挥不适用——联合解自己分配全程时间);失败则 20Hz 块重试,
        // 连败回退 B′
        if (RunWbmpc(wb, plan, be, ik, ts, q_cur, dq_cur, wb_have, mpc_sched, &wb_plan, trace)) {
          wb_have = true;
          std::printf("  [wb] t=%.2f 首解 term=(%.2f,%.2f) v=(%.2f,%.2f) pos_res=%.4f n·n*=%.2f %.1fms\n",
                      ts, wb_plan.term_base[0], wb_plan.term_base[1],
                      wb_plan.term_vbase[0], wb_plan.term_vbase[1],
                      wb_plan.pos_res, wb_plan.n_dot, wb_plan.ms);
        }
      } else if (use_qp && kInitDelay > 0) {
        // 延迟起挥:此处不 Init,由下方触发块在 t_plan0+kInitDelay 起挥
        // (等待期臂保持就位构型,RefineAt 持续精化计划)
      } else if (use_qp && !n6_mode) {
        if (fly && !ocp_mode && ref_built)
          FillBasePred(plan, be, ts, mpc_ref[0], mpc_ref[1]);
        StageProfile::Measure("arm_qp_init", [&]{aswing.Init(plan, q_cur, dq_cur, ik, ts);});
      }
      else if (!n6_mode) swing.Init(plan, q_cur, ik);
      // nmpc6:挥拍层不 Init(臂由 NMPC 轨迹接管;回退时才 Init)
      std::printf("  [plan] t=%.2f hit=(%.2f,%.2f,%.2f) t_hit=%.2f margin=%.2f base=(%.2f,%.2f)\n",
                  ts, plan.p_hit[0], plan.p_hit[1], plan.p_hit[2],
                  plan.t_hit, plan.time_margin, plan.base_target[0], plan.base_target[1]);
      if (fly)
        std::printf("  [fly] v_base*=(%.2f,%.2f) %.2f m/s,过靶时刻=t_hit=%.2f(跑程 %.2fm)\n",
                    plan.v_base_star[0], plan.v_base_star[1],
                    std::hypot(plan.v_base_star[0], plan.v_base_star[1]), plan.t_hit,
                    std::hypot(plan.base_target[0]-bs.x, plan.base_target[1]-bs.y));
    }
  }
  // QP 延迟起挥触发(wbmpc 不适用:臂由 OCP 解直接接管)
  if (use_qp && !wb_mode && !n6_mode && planned && kInitDelay > 0 && !aswing.active() && ts >= t_plan0 + kInitDelay) {
    if (fly && !ocp_mode && ref_built)
      FillBasePred(plan, be, ts, mpc_ref[0], mpc_ref[1]);
    StageProfile::Measure("arm_qp_init", [&]{aswing.Init(plan, q_cur, dq_cur, ik, ts);});
    if (!aswing.active() && ts - t_plan0 < kInitDelay + 0.60)   // 初解失败诊断(限窗打印)
      std::printf("  [init-fail] t=%.2f p=(%.2f,%.2f,%.2f) base_t=(%.2f,%.2f) t_hit=%.2f\n",
                  ts, plan.p_hit[0], plan.p_hit[1], plan.p_hit[2],
                  plan.base_target[0], plan.base_target[1], plan.t_hit);
  }
  // 重规划(50Hz):t_hit 锚定,RefineAt 仅精化 p_hit/base_target
  // (评分比较会拒绝精化:高度偏好偏爱陈旧高 z 计划 —— 见 hit_planner 注)
  // QP 路径锁止后仍调用(UpdatePlan 内部按 §6.5 限微调幅度,大改自拒);
  // 五次基线保持锁止即冻结。aswing 未激活时 locked() 恒真,须判活
  if (planned && (use_qp ? (aswing.active() || wb_mode || n6_mode || cfg.time_windows.refine_before_arm_active)
                         : (!swing.locked() || wb_mode || n6_mode)) &&
      SynchronousSchedule::Replan(t)) {
    const BaseState bs2 = be.base_state();
    StageProfile::Measure("target_refinement", [&]{planner.RefineAt(predictor, bs2, &plan, ts);});
    ++replans;
    // 底盘参考终点:锁止后冻结(fly-through 末段 chasing 收敛中的目标会
    // 拉偏触球几何,随机族 plan_err 0.94/1.05 反例,2026-10-07);臂仍可微调
    // ocpfly:终点=OCP 终端(下方 RunOcp 后更新),非 ocp 模式随精化走 base_target
    if (n6_mode && !n6_dead) {
      // nmpc6 50Hz 重解(旧机 MpcThread 同口径:状态=实测,dt=剩余/3,
      // 目标=p_hit/法向=plan.n_face/拍速=|plan.v_racket|)。末段 <100ms
      // 仅接受 严格+成功+<10ms 新解,否则保持旧轨迹(原语义);从未有解
      // 连败 >15 拍(0.75s)→ 本 trial 永久回退 B′
      // 目标锁止:lock 后(t_hit−0.15,与 B′/wbmpc 同窗)目标/时刻用快照,
      // 不再吃 RefineAt 精化 —— 末段 p_hit 随下坠球快速漂移,NMPC 终端在
      // 速度/角窗约束下追不上(末段 e 0.005→0.225 解退化 loose/拒收,计划
      // 老化拍滞后 0.3,2026-10-09 定位)。旧机 hit_pos 来自慢策略层,
      // 天然准静态,等价
      const bool tgt_locked = locked_saved;
      const double T6 = (tgt_locked ? lock_t : plan.t_hit) - ts;
      // 默认继续解到12ms。历史50ms停解曾造成旧轨迹陈旧；独立实验可调。
      if (T6 > cfg.time_windows.nmpc6.stop_s) {
        LegacyNmpcRequest rq;
        const bool uniform_time = cfg.time_windows.nmpc6.IsUniform();
        const auto stage_dt = cfg.time_windows.nmpc6.Durations(T6);
        std::copy(stage_dt.begin(), stage_dt.end(), rq.dt);
        for(int j=0;j<5;++j){auto c=be.motor_effective(AxisArm(j));rq.effective_q[j]=c.q;rq.effective_dq[j]=c.dq;}
        const BaseState bsn = be.base_state();
        if(chassis_model){
          const auto ramp=servo.ramped();rq.chassis_state[0]=ramp.vx;rq.chassis_state[1]=ramp.vy;
          for(int j=0;j<4;++j){rq.chassis_state[2+j]=be.joint(AxisSteer(j)).q;double w=0;
            for(int h=0;h<2;++h){int axis=h?AxisWheelR(j):AxisWheelL(j);w+=be.chassis_config().wheel_sign[2*j+h]*servo.wheel_velocity_reference(axis)*be.chassis_config().wheel_radius*.5;}
            rq.chassis_state[6+j]=w;
          }rq.chassis_state[10]=bsn.yaw;
        }
        rq.state[0] = bsn.x;
        rq.state[1] = bsn.y;
        for (int j = 0; j < 5; ++j) rq.state[2 + j] = q_cur[j];
        {  // 底盘速度:BaseState.vx/vy 为车体系 → 世界系
          const double c = std::cos(bsn.yaw), sn = std::sin(bsn.yaw);
          rq.state[7] = c * bsn.vx - sn * bsn.vy;
          rq.state[8] = sn * bsn.vx + c * bsn.vy;
        }
        for (int j = 0; j < 5; ++j) rq.state[9 + j] = dq_cur[j];
        for (int i = 0; i < 3; ++i)
          rq.target[i] = tgt_locked ? lock_p[i] : plan.p_hit[i];
        // 底盘锚:规划器可达环带位置(= 旧机上游走位策略的 sim 同源物;
        // 不加则 NMPC 自由落位停 1.3m 外臂全伸展凑目标,末段全拒收)
        if(coherent&&tgt_locked) {for(int i=0;i<3;++i){plan.v_racket[i]=locked_plan.v_racket[i];plan.n_face[i]=locked_plan.n_face[i];plan.base_target[i]=locked_plan.base_target[i];}}
        rq.base_anchor_ok = true;
        rq.base_anchor[0] = plan.base_target[0];
        rq.base_anchor[1] = plan.base_target[1];
        rq.base_anchor_r = 0.40;
        if (corrected) {
          rq.corridor_ok = true;
          double elapsed = 0;
          for (int k = 0; k < 3; ++k) {
            elapsed += rq.dt[k];
            double vx, vy;
            const double end_s = uniform_time ? (k+1)*T6/3 : elapsed;
            BaseRefGen::EvalAt(mpc_sched, ts+end_s, &rq.corridor_xy[k][0],
                              &rq.corridor_xy[k][1], &vx, &vy);
          }
        }
        if(chassis_recovery){
          double x,y,vx,vy;BaseRefGen::EvalAt(mpc_sched,ts,&x,&y,&vx,&vy);
          double elapsed=0;
          for(int k=0;k<2;++k){elapsed+=rq.dt[k];double z=uniform_time?(k+1)/3.0:elapsed/T6,w=1-3*z*z+2*z*z*z,dw=z-2*z*z+z*z*z;
            rq.corridor_xy[k][0]+=w*(bsn.x-x)+dw*T6*(rq.state[7]-vx);
            rq.corridor_xy[k][1]+=w*(bsn.y-y)+dw*T6*(rq.state[8]-vy);
          }
        }
        rq.hit_spd = std::hypot(plan.v_racket[0], plan.v_racket[1],
                                plan.v_racket[2]);
        {   // 末段速度-位置权衡标定(2026-10-09):OCP 速度项权重 8 在末
            // 段压过位置 2500·e²,挥扫线让位拍速 → 系统性短左;0.8=本机
            // 实测最优(1.0/0.9/0.8/0.65 → 60/60/80/60),旧机全速是直驱
            // 腕+羽毛球几何,sim 臂网球拍降两成。N6_SPD 可覆写扫参
          const double kSpdFac =
              cfg.nmpc6_speed_factor;
          rq.hit_spd *= kSpdFac;
        }
        rq.hit_pitch_deg =
            std::asin(std::clamp(plan.n_face[2], -1.0, 1.0)) * 180.0 / M_PI;
        rq.hit_yaw_deg =
            std::atan2(plan.n_face[1], plan.n_face[0]) * 180.0 / M_PI;
        rq.hit_speed_z = plan.v_racket[2];
        LegacyNmpcResult r6;
        if (cfg.nmpc6_trace && n6_have && n6_ms > 0) {
          // 末段跟踪 trace:实测 vs 当前计划@ts(定位执行亏欠滚雪球的层)
          const int W = n6_ms + 1;
          const int idx = std::clamp(
              (int)std::lround((ts - n6_t0) * 1000.0), 0, n6_ms);
          double dbx = n6_pos[idx] - rq.state[0],
                 dby = n6_pos[W + idx] - rq.state[1], dqm = 0;
          for (int j = 0; j < 5; ++j)
            dqm = std::max(dqm, std::fabs(
                n6_pos[(2 + j) * W + idx] - rq.state[2 + j]));
          std::printf("  [n6T] t=%.3f T=%.3f base=(%.3f,%.3f) dq=%.3f\n",
                      ts, T6, dbx, dby, dqm);
        }
        const bool n6_accepted = StageProfile::Measure("nmpc6_solver", [&]{return n6.Solve(rq, &r6);});
        n6_recent_rejects=n6_accepted?0:n6_recent_rejects+1;
        bool stale_plan=false;double stale_tracking=0,stale_goal=0;
        if(chassis_guard&&n6_have&&!n6_accepted){int w=n6_ms+1,idx=std::clamp(static_cast<int>((ts-n6_t0)*1000),0,n6_ms);
          stale_tracking=std::hypot(n6_pos[idx]-bsn.x,n6_pos[w+idx]-bsn.y);
          stale_goal=std::hypot(rq.target[0]-diagnostic_goal[0],rq.target[1]-diagnostic_goal[1],rq.target[2]-diagnostic_goal[2]);
          stale_plan=n6_recent_rejects>=5&&ts-n6_t0>.2&&T6<.5&&(stale_tracking>.25||stale_goal>.15);
        }
        ++n6_attempts; n6_attempt_ms += r6.solve_ms;
        n6_attempt_max_ms = std::max(n6_attempt_max_ms, r6.solve_ms);
        const bool diag_near_ok=n6_accepted&&PlanSupervisor::CanApplyNmpc(r6,coherent,T6,cfg.time_windows.nmpc6.strict_window_s);
        if(trace.enabled()){
          nlohmann::json e={{"kind","solve"},{"mode","nmpc6"},{"t",ts},{"target",Vec(rq.target,3)},{"target_time",ts+T6},{"accepted",n6_accepted},{"applied",diag_near_ok},{"reason",static_cast<int>(r6.status)},{"constraint_violation",r6.max_constraint_violation},{"target_version",plan.version},{"capture_time",plan.capture_time},{"base_anchor",Vec(rq.base_anchor,2)},{"solver_ok",r6.solver_ok},{"ms",r6.solve_ms},{"endpoint_fk",Vec(r6.end_pos,3)},{"state",Vec(rq.state,14)}};
          e["steer_rate_mode"]=steer_rate_mode;
          e["dt"]=Vec(rq.dt,3);e["freeze_speed_gate"]=r6.freeze_speed_gate;e["freeze_active"]=r6.freeze_active;
          e["freeze_window_s"]=cfg.time_windows.nmpc6.freeze_window_s;
          e["chassis_state"]=Vec(rq.chassis_state,11);e["corridor_xy"]={{rq.corridor_xy[0][0],rq.corridor_xy[0][1]},{rq.corridor_xy[1][0],rq.corridor_xy[1][1]}};
          e["stale_plan"]=stale_plan;e["recent_rejects"]=n6_recent_rejects;
          e["ipopt_status"]=r6.ipopt_status;e["ipopt_iterations"]=r6.ipopt_iterations;e["solver_exception"]=r6.solver_exception;
          e["constraint_values"]=r6.diagnostic_g;e["constraint_lower"]=r6.diagnostic_lbg;e["constraint_upper"]=r6.diagnostic_ubg;
          e["control_values"]=r6.diagnostic_u;e["control_lower"]=r6.diagnostic_lbx;e["control_upper"]=r6.diagnostic_ubx;
          if(diagnostic_previous_request){e["target_step_m"]=std::hypot(rq.target[0]-diagnostic_previous_target[0],rq.target[1]-diagnostic_previous_target[1],rq.target[2]-diagnostic_previous_target[2]);e["target_time_step_s"]=ts+T6-diagnostic_previous_target_time;}
          diagnostic_previous_request=true;for(int j=0;j<3;++j)diagnostic_previous_target[j]=rq.target[j];diagnostic_previous_target_time=ts+T6;
          e["remaining_s"]=T6;e["locked"]=tgt_locked;
          e["worst_constraint"]={{"name",r6.worst_constraint},{"index",r6.worst_index},{"side",r6.worst_side},{"value",r6.worst_value},{"lower",r6.worst_lower},{"upper",r6.worst_upper},{"raw_violation",r6.max_constraint_violation}};
          e["tracking_valid"]=n6_have && ts-n6_t0<=n6_ms*.001;
          if(n6_have){
            const int w=n6_ms+1,idx=std::clamp(static_cast<int>(std::lround((ts-n6_t0)*1000)),0,n6_ms);
            double br[2]={n6_pos[idx],n6_pos[w+idx]},ba[2]={rq.state[0],rq.state[1]},qr[5],qa[5],pr[3],pb[3],pa[3];
            for(int j=0;j<5;++j){qr[j]=n6_pos[(2+j)*w+idx];qa[j]=rq.state[2+j];}
            RobotArmPos(sckd_nmpc6::ParseRobotGeometry(m->names),br,qr,pr);RobotArmPos(sckd_nmpc6::ParseRobotGeometry(m->names),ba,qr,pb);RobotArmPos(sckd_nmpc6::ParseRobotGeometry(m->names),ba,qa,pa);
            e["base_xy_tracking_m"]=std::hypot(ba[0]-br[0],ba[1]-br[1]);
            e["joint_tracking_rad"]=nlohmann::json::array();for(int j=0;j<5;++j)e["joint_tracking_rad"].push_back(qa[j]-qr[j]);
            e["base_tracking_vector"]=Vec(pb,3);e["arm_tracking_vector"]=Vec(pa,3);
            for(int j=0;j<3;++j){e["base_tracking_vector"][j]=pb[j]-pr[j];e["arm_tracking_vector"][j]=pa[j]-pb[j];}
            e["target_vs_accepted_m"]=std::hypot(rq.target[0]-diagnostic_goal[0],rq.target[1]-diagnostic_goal[1],rq.target[2]-diagnostic_goal[2]);
            e["target_time_vs_accepted_s"]=ts+T6-diagnostic_goal_time;
            e["plan_age_s"]=ts-n6_t0;
          }
          if(n6_accepted && r6.plan_ms>0){int w=r6.plan_ms+1;double b[2]={r6.plan_pos[r6.plan_ms],r6.plan_pos[w+r6.plan_ms]},q[5],ep[3];for(int j=0;j<5;++j)q[j]=r6.plan_pos[(2+j)*w+r6.plan_ms];RobotArmPos(sckd_nmpc6::ParseRobotGeometry(m->names),b,q,ep);e["dense_endpoint_fk"]=Vec(ep,3);}
          trace.Emit(e);
        }
        if (n6_accepted) {
          if (cfg.nmpc6_trace)
            std::printf("  [n6+] t=%.3f %s e=(%.3f,%.3f,%.3f) end=(%.2f,%.2f,%.2f) "
                        "q°=(%.0f,%.0f,%.0f,%.0f,%.0f)\n",
                        ts, r6.status == LegacyNmpcStatus::kOk ? "ok" : "lo",
                        std::fabs(r6.end_pos[0] - rq.target[0]),
                        std::fabs(r6.end_pos[1] - rq.target[1]),
                        std::fabs(r6.end_pos[2] - rq.target[2]),
                        r6.end_pos[0], r6.end_pos[1], r6.end_pos[2],
                        r6.term_state[2] * 57.3, r6.term_state[3] * 57.3,
                        r6.term_state[4] * 57.3, r6.term_state[5] * 57.3,
                        r6.term_state[6] * 57.3);
          const bool near_ok =
              PlanSupervisor::CanApplyNmpc(r6,coherent,T6,cfg.time_windows.nmpc6.strict_window_s);
          if (near_ok) {
            if (!n6_have)
              std::printf("  [n6] t=%.2f 首解 end=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f) %.1fms\n",
                          ts, r6.end_pos[0], r6.end_pos[1], r6.end_pos[2],
                          r6.end_pos[4], r6.end_pos[5], r6.end_pos[6],
                          r6.solve_ms);
            for(int j=0;j<3;++j)diagnostic_goal[j]=rq.target[j];
            diagnostic_goal_time=ts+T6;diagnostic_plan_t=ts;diagnostic_contact_time=ts+T6;
            n6_pos = r6.plan_pos;
            n6_vel = r6.plan_vel;
            n6_ms = r6.plan_ms;
            n6_t0 = ts;
            n6_have = true;
            n6_fails_streak = 0;
          }
        } else if (PlanSupervisor::NmpcFallback(n6_have,n6_fails_streak,stale_plan)) {
          // 回退仅在"从未有解"(0.75s 内一个接受解都没有):一旦有解,
          // 末段拒收=保持旧 Plans 继续执行(旧机 MpcThread 语义,无回退);
          // 首版把末段严格门拒收计入连败 → 100Hz 下 150ms 假性回退,6/10
          // trial 被拖回 B′(2026-10-09 定位)
          if(stale_plan){trace.Emit({{"kind","stale_plan_fallback"},{"t",ts},{"tracking_m",stale_tracking},{"goal_change_m",stale_goal},{"remaining_s",T6},{"rejects",n6_recent_rejects}});if(locked_saved)plan=locked_plan;}
          n6_dead = true;
          ++n6_fallbacks_t;
          const BaseState bsf = be.base_state();
          mpc_sched.t0 = ts;                     // pfly 参考此刻重新起锚(防爬行)
          mpc_sched.ax = bsf.x; mpc_sched.ay = bsf.y;
          mpc_sched.tx = plan.base_target[0];
          mpc_sched.ty = plan.base_target[1];
          mpc_sched.t_arrive = std::clamp(plan.t_hit - ts, 0.35, 1.5);
          mpc_sched.v_fly = std::hypot(plan.v_base_star[0], plan.v_base_star[1]);
          if (use_qp) StageProfile::Measure("arm_qp_init", [&]{aswing.Init(plan, q_cur, dq_cur, ik, ts);});
          else swing.Init(plan, q_cur, ik);
          std::printf("  [n6-fallback] t=%.2f 连败 %d 拍 → B′(pfly+挥拍层)\n",
                      ts, n6_fails_streak);
        }
      }
    } else if (wb_mode && !wb_dead) {
      // 走廊终点随精化更新(起锚位置/时刻不变——BaseRefGen 绝对时刻语义)
      mpc_sched.tx = plan.base_target[0];
      mpc_sched.ty = plan.base_target[1];
      // wbmpc 20Hz 重解(末段窗 0.15 外;RefineAt 精化的 p_hit 随读进 OCP)。
      // 已有解而本拍拒收 = 信任域/跳变守门语义,保持上一计划继续执行;
      // 从未有解连败 >15 拍(0.75s)→ 本 trial 永久回退 B′(pfly+挥拍层)
      // (0.20 = 求解器末段窗 0.15 + 终端提前 0.03 + 余量)
      if (plan.t_hit - ts > 0.20) {
        WmPlan cand;
        if (RunWbmpc(wb, plan, be, ik, ts, q_cur, dq_cur, wb_have, mpc_sched, &cand, trace)) {
          if (cand.n_dot < 0.75) {
            // 法向质量门槛:n·n*<0.75 = 拍面 >41° 歪(实测 0.45 解被执行
            // →球从拍背过,min_dist 0.14 却无接触,2026-10-09)——弃新解
            // 保旧计划;持续歪→连败计数→回退 B′
            ++wb_fails_streak;
          } else {
            if (!wb_have)
              std::printf("  [wb] t=%.2f 迟到首解 term=(%.2f,%.2f) pos_res=%.4f n·n*=%.2f %.1fms\n",
                          ts, cand.term_base[0], cand.term_base[1],
                          cand.pos_res, cand.n_dot, cand.ms);
            else if (ts > t_plan0 + 0.6)   // 末段接受(冻结计划年龄/终端)
              std::printf("  [wb+] t=%.2f term=(%.2f,%.2f) v=(%.2f,%.2f) pr=%.3f n·n*=%.2f %.0fms\n",
                          ts, cand.term_base[0], cand.term_base[1],
                          cand.term_vbase[0], cand.term_vbase[1],
                          cand.pos_res, cand.n_dot, cand.ms);
            for(int j=0;j<3;++j)diagnostic_goal[j]=plan.p_hit[j];
            diagnostic_goal_time=plan.t_hit;diagnostic_plan_t=ts;diagnostic_contact_time=cand.contact_time;
            trace.Emit({{"kind","wb_apply"},{"t",ts},{"target",Vec(plan.p_hit,3)},{"contact_time",cand.contact_time},{"target_time",plan.t_hit}});
            wb_plan = cand;
            wb_have = true;
            wb_fails_streak = 0;
          }
        } else if (PlanSupervisor::WbmpcFallback(wb_have,wb_fails_streak)) {   // 陈旧守卫:已有解但连拒 1s
                                     // (中途回退 B′ 有时救场:挥拍层窗口
                                     // 尚可时其成熟度胜过冻结计划,7/10
                                     // vs 5/10 单种子对比,2026-10-09)
          wb_dead = true;
          ++wb_fallbacks;
          const BaseState bsf = be.base_state();
          mpc_sched.t0 = ts;                     // pfly 参考此刻重新起锚(防爬行)
          mpc_sched.ax = bsf.x; mpc_sched.ay = bsf.y;
          mpc_sched.tx = plan.base_target[0];
          mpc_sched.ty = plan.base_target[1];
          mpc_sched.t_arrive = std::clamp(plan.t_hit - ts, 0.35, 1.5);
          mpc_sched.v_fly = std::hypot(plan.v_base_star[0], plan.v_base_star[1]);
          if (use_qp) StageProfile::Measure("arm_qp_init", [&]{aswing.Init(plan, q_cur, dq_cur, ik, ts);});
          else swing.Init(plan, q_cur, ik);
          std::printf("  [wb-fallback] t=%.2f 连败 %d 拍 → B′(pfly+挥拍层)\n",
                      ts, wb_fails_streak);
        }
      }
    } else if (!(use_qp && aswing.locked()) && !ocp_mode) {
      mpc_sched.tx = plan.base_target[0];   // 终点随精化更新(起锚不变)
      mpc_sched.ty = plan.base_target[1];
    }
    if (use_qp) {
      if (ocp_mode && !ocp_dead && !aswing.locked()) {
        // ocpfly 20Hz 重解:锁止后冻结终端(§6.5);连败 ~0.5s 永久回退 pfly
        if (RunOcp(wbo, plan, be, ts, &ocp_res)) {
          ocp_have = true;
          ocp_fails_streak = 0;
          if (!(use_qp && aswing.locked())) {
            mpc_sched.tx = ocp_res.term_base[0];
            mpc_sched.ty = ocp_res.term_base[1];
            mpc_sched.v_fly = std::min(1.2, std::hypot(ocp_res.term_vbase[0],
                                                       ocp_res.term_vbase[1]));
          }
        } else if (++ocp_fails_streak > 25) {
          ocp_dead = true;
          plan.base_pred_ok = false;   // 锚退回规划 base_target(B′)
        }
      } else if (fly && !ocp_mode && ref_built) {
        // v2 triage:臂终值按预测实际底盘解(P 控误差外推)
        FillBasePred(plan, be, ts, mpc_ref[0], mpc_ref[1]);
      }
      if (aswing.active()) StageProfile::Measure("arm_qp_replan", [&]{aswing.UpdatePlan(plan, ik, ts);});
    }
    else if (swing.active()) swing.UpdatePlan(plan, ik);
    else swing.Init(plan, q_cur, ik);
  }

}
}
