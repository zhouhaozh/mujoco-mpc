#include "runtime/intercept_modules.h"
#include "runtime/controller_adapters.h"
#include "runtime/plan_supervisor.h"
#include "runtime/synchronous_schedule.h"
#include "sim_fk.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
namespace sckd::runtime {
using sckd_nmpc6::SimArmPos;

void RecordControlState(StateTraceContext c, const ExecutionFrame& frame, int t, double ts) {
  const auto& wb_exec=frame.wb_exec;
  const auto& n6_exec=frame.n6_exec;
  const auto& wb_base=frame.wb_base;
  const auto& wb_vbase=frame.wb_vbase;
  const auto& n6_base=frame.n6_base;
  const auto& n6_vbase=frame.n6_vbase;
  const auto& ac=frame.arm;
  auto& be = c.be;
  auto& servo = c.servo;
  auto& planner = c.planner;
  auto& plan = c.plan;
  auto& aswing = c.aswing;
  auto& mpc_ref = c.mpc_ref;
  auto& trace = c.trace;
  auto& m = c.m;
  auto& d = c.d;
  auto& variant = c.variant;
  auto& base_mode = c.base_mode;
  auto& steer_rate_mode = c.steer_rate_mode;
  auto& n6_mode = c.n6_mode;
  auto& wb_mode = c.wb_mode;
  auto& coherent = c.coherent;
  auto& planned = c.planned;
  auto& ref_built = c.ref_built;
  auto& wb_dead = c.wb_dead;
  auto& n6_dead = c.n6_dead;
  auto& locked_saved = c.locked_saved;
  auto& site_id = c.site_id;
  auto& n6_tau = c.n6_tau;
  auto& wb_tau = c.wb_tau;
  auto& diagnostic_goal_time = c.diagnostic_goal_time;
  auto& diagnostic_plan_t = c.diagnostic_plan_t;
  auto& diagnostic_contact_time = c.diagnostic_contact_time;
  auto& lock_t = c.lock_t;
  auto& diagnostic_goal = c.diagnostic_goal;
  auto& lock_p = c.lock_p;
  if(trace.enabled() && SynchronousSchedule::StateTrace(t)){
    if(!n6_exec&&!wb_exec&&aswing.active()) {
      const HitPlan& ap=aswing.accepted_plan();
      for(int j=0;j<3;++j)diagnostic_goal[j]=ap.p_hit[j];
      diagnostic_goal_time=ap.t_hit;diagnostic_contact_time=ap.t_hit-(coherent?0:ArmSwing::kLead);diagnostic_plan_t=aswing.accepted_time();
    }
    const BaseState ds=be.base_state();
    double actual_q[5],actual_dq[5],raw[5],applied[5],cap[5];
    for(int j=0;j<5;++j){auto r=be.joint(AxisArm(j));actual_q[j]=r.q;actual_dq[j]=r.dq;}
    be.arm_effort(raw,applied,cap);
    double ref_b[2]={ds.x,ds.y},ref_v[2]={0,0};
    if(n6_exec){for(int j=0;j<2;++j){ref_b[j]=n6_base[j];ref_v[j]=n6_vbase[j];}}
    else if(wb_exec){for(int j=0;j<2;++j){ref_b[j]=wb_base[j];ref_v[j]=wb_vbase[j];}}
    else if(planned&&ref_built){ref_b[0]=mpc_ref[0];ref_b[1]=mpc_ref[1];double c=cos(ds.yaw),sn=sin(ds.yaw);ref_v[0]=c*mpc_ref[3]-sn*mpc_ref[4];ref_v[1]=sn*mpc_ref[3]+c*mpc_ref[4];}
    double p_ref[3],p_base_actual[3],p_nominal_actual[3],actual_b[2]={ds.x,ds.y};
    SimArmPos(ref_b,ac.q,p_ref);SimArmPos(actual_b,ac.q,p_base_actual);SimArmPos(actual_b,actual_q,p_nominal_actual);
    mj_kinematics(m,d);double p_world[3],rot[3]={0,0,0};mju_rotVecQuat(rot,m->site_pos+3*site_id,d->xquat+4*m->site_bodyid[site_id]);for(int j=0;j<3;++j)p_world[j]=d->xpos[3*m->site_bodyid[site_id]+j]+rot[j];
    nlohmann::json motors=nlohmann::json::array();
    for(int a=0;a<kNumAxes;++a){auto r=be.joint(a);auto issued=be.motor_issued(a),eff=be.motor_effective(a);const bool wheel=a<12&&a%3!=0;motors.push_back({{"axis",a},{"name",kAxisJointName[a]},{"q",r.q},{"dq",r.dq},{"q_target",issued.q},{"wire_dq_target",issued.dq},{"dq_target",wheel?servo.wheel_velocity_reference(a):issued.dq},{"wheel_external_pi",wheel},{"kp",issued.kp},{"kd",issued.kd},{"tau_ff",issued.tau_ff},{"q_effective",eff.q},{"dq_effective",eff.dq},{"kp_effective",eff.kp},{"kd_effective",eff.kd},{"tau_ff_effective",eff.tau_ff},{"raw",be.motor_raw(a)},{"applied",r.tau_cmd},{"cap",be.motor_cap(a)}});}
    trace.Emit({{"qp_attempts",aswing.plan_attempts},{"qp_accepts",aswing.plan_accepts},{"target_snapshot",{{"version",plan.version},{"capture_time",plan.capture_time},{"position",Vec(plan.p_hit,3)},{"racket_velocity",Vec(plan.v_racket,3)},{"normal",Vec(plan.n_face,3)},{"arrival_time",plan.t_hit},{"base_anchor",Vec(plan.base_target,2)}}},{"arm_qp_active",aswing.active()},{"variant",variant},{"motors",motors},{"qpos",Vec(d->qpos,m->nq)},{"qvel",Vec(d->qvel,m->nv)},{"planner_checked",planner.checked},{"planner_rejected",planner.rejected},{"planner_last_reason",planner.last_reject},{"target_version",plan.version},{"capture_time",plan.capture_time},{"qp_constraint_violation",aswing.last_constraint_violation},{"kind","state"},{"steer_rate_mode",steer_rate_mode},{"t",ts-.001},{"mode",base_mode},{"active",n6_exec?"nmpc6":wb_exec?"wbmpc":planned?"pfly_or_wait":"standby"},{"fallback",(n6_mode&&n6_dead)||(wb_mode&&wb_dead)},{"planned",planned},{"locked",locked_saved},{"goal",Vec(locked_saved?lock_p:plan.p_hit,3)},{"planner_goal",Vec(plan.p_hit,3)},{"goal_time",locked_saved?lock_t:plan.t_hit},{"accepted_goal",Vec(diagnostic_goal,3)},{"accepted_goal_time",diagnostic_goal_time},{"plan_issued_t",diagnostic_plan_t},{"contact_time",diagnostic_contact_time},{"p_actual",Vec(p_world,3)},{"p_reference",Vec(p_ref,3)},{"p_command_actualbase",Vec(p_base_actual,3)},{"p_nominal_actual",Vec(p_nominal_actual,3)},{"base",{ds.x,ds.y,ds.z,ds.yaw}},{"v_base_world",{cos(ds.yaw)*ds.vx-sin(ds.yaw)*ds.vy,sin(ds.yaw)*ds.vx+cos(ds.yaw)*ds.vy}},{"base_ref",Vec(ref_b,2)},{"v_base_ref",Vec(ref_v,2)},{"q",Vec(actual_q,5)},{"dq",Vec(actual_dq,5)},{"q_cmd",Vec(ac.q,5)},{"dq_cmd",Vec(ac.dq,5)},{"tau_raw",Vec(raw,5)},{"tau",Vec(applied,5)},{"tau_cap",Vec(cap,5)},{"tau_lead",n6_exec?n6_tau:wb_exec?wb_tau:0}});
  }

}
}
