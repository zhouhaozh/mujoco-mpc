#pragma once
// 同步运行域的显式适配上下文。引用绑定在trial装配时完成；不拥有模型/资源。
#include <functional>
#include <memory>
#include <mujoco/mujoco.h>
#include "runtime/runtime_config.h"
#include "runtime/contracts.h"
#include "runtime/control_trace.h"
#include "core/stage_profile.h"
#include "control/arm_ik.h"
#include "control/arm_swing.h"
#include "control/base_ref_gen.h"
#include "control/chassis_servo.h"
#include "control/legacy_nmpc.h"
#include "control/swing_generator.h"
#include "control/wholebody_ocp.h"
#include "control/wholebody_mpc.h"
#include "estimation/ball_predictor.h"
#include "estimation/measured_ball_replay.h"
#include "estimation/socket_ball_predictor.h"
#include "hal/sim_backend.h"
#include "planning/hit_planner.h"
#include "sim/analytic_contact.h"
#include "sim/ball_aero.h"
namespace sckd::runtime {
struct ExecutionFrame {
  bool wb_exec=false, n6_exec=false;
  double wb_q[5]={},wb_dq[5]={},wb_ddq[5]={},wb_base[2]={},wb_vbase[2]={};
  double n6_q[5]={},n6_dq[5]={},n6_base[2]={},n6_vbase[2]={};
  ArmHybridCmd arm;
};
struct InputSnapshot {
  BallObservation ball;
  RobotEstimate robot;
  bool have_ball=false;
};
struct PlanningContext {
  SimBackend& be;
  ChassisServo& servo;
  ArmIk& ik;
  BallPredictionInterface& predictor;
  HitPlanner& planner;
  HitIntent& intent;
  HitPlan& plan;
  HitPlan& locked_plan;
  WholebodyOcp& wbo;
  WboResult& ocp_res;
  WholebodyMpc& wb;
  WmPlan& wb_plan;
  LegacyNmpc& n6;
  ArmSwing& aswing;
  SwingGenerator& swing;
  BaseRefGen::Schedule& mpc_sched;
  BaseRefGen::Ref& mpc_ref;
  ControlTrace& trace;
  const RuntimeValues& cfg;
  std::vector<double>& n6_pos;
  std::vector<double>& n6_vel;
  const mjModel* m;
  mjData* d;
  const std::string& steer_rate_mode;
  const bool& fly;
  const bool& n6_mode;
  const bool& wb_mode;
  const bool& ocp_mode;
  const bool& use_qp;
  const bool& corrected;
  const bool& coherent;
  const bool& chassis_model;
  const bool& chassis_recovery;
  const bool& chassis_guard;
  bool& planned;
  bool& ref_built;
  bool& ocp_have;
  bool& ocp_dead;
  bool& wb_have;
  bool& wb_dead;
  bool& n6_have;
  bool& n6_dead;
  bool& diagnostic_previous_request;
  bool& locked_saved;
  int& replans;
  int& ocp_fails_streak;
  int& wb_fails_streak;
  int& n6_fails_streak;
  int& n6_recent_rejects;
  int& n6_attempts;
  int& n6_fallbacks_t;
  int& wb_fallbacks;
  int& n6_ms;
  const int& trial;
  double& t_plan0;
  double& t_bounce_plan;
  double& n6_t0;
  double& n6_attempt_ms;
  double& n6_attempt_max_ms;
  double& diagnostic_goal_time;
  double& diagnostic_plan_t;
  double& diagnostic_contact_time;
  double& diagnostic_previous_target_time;
  double& lock_t;
  const double& kInitDelay;
  double (&q_cur)[5];
  double (&dq_cur)[5];
  double (&diagnostic_goal)[3];
  double (&diagnostic_previous_target)[3];
  double (&lock_p)[3];
};
struct ExecutionContext {
  SimBackend& be;
  ChassisServo& servo;
  HitPlan& plan;
  WmPlan& wb_plan;
  ArmSwing& aswing;
  SwingGenerator& swing;
  ChassisMpc& mpc;
  ChassisMpc::Limits& mpc_lim;
  BaseRefGen::Schedule& mpc_sched;
  BaseRefGen::Ref& mpc_ref;
  const RuntimeValues& cfg;
  const nlohmann::json& identified_gains_config;
  std::vector<double>& n6_pos;
  std::vector<double>& n6_vel;
  const mjModel* m;
  const std::string& variant;
  const bool& fly;
  const bool& use_mpc;
  const bool& n6_mode;
  const bool& wb_mode;
  const bool& use_qp;
  const bool& coherent;
  const bool& inverse_ff;
  const bool& identified_gains;
  bool& planned;
  bool& ref_built;
  bool& wb_have;
  bool& wb_dead;
  bool& n6_have;
  bool& n6_dead;
  bool& have_dq_command;
  int& n6_ms;
  int& mpc_solves;
  int& mpc_fails;
  double& t_plan0;
  double& n6_t0;
  double& n6_tau;
  double& wb_tau;
  double& mpc_ms_max;
  double (&q_cur)[5];
  double (&mpc_cs2)[2 * ChassisMpc::kN];
  double (&previous_dq_command)[5];
};
struct StateTraceContext {
  SimBackend& be;
  ChassisServo& servo;
  HitPlanner& planner;
  HitPlan& plan;
  ArmSwing& aswing;
  BaseRefGen::Ref& mpc_ref;
  ControlTrace& trace;
  const mjModel* m;
  mjData* d;
  const std::string& variant;
  const std::string& base_mode;
  const std::string& steer_rate_mode;
  const bool& n6_mode;
  const bool& wb_mode;
  const bool& coherent;
  bool& planned;
  bool& ref_built;
  bool& wb_dead;
  bool& n6_dead;
  bool& locked_saved;
  const int& site_id;
  double& n6_tau;
  double& wb_tau;
  double& diagnostic_goal_time;
  double& diagnostic_plan_t;
  double& diagnostic_contact_time;
  double& lock_t;
  double (&diagnostic_goal)[3];
  double (&lock_p)[3];
};
struct InputContext {
  SimBackend& be;
  BallPredictionInterface& predictor;
  std::unique_ptr<MeasuredBallReplay>& measured;
  const bool& deduplicate_obs;
  InputSnapshot& input_snapshot;
  double& last_obs_t;
  double (&q_cur)[5];
  double (&dq_cur)[5];
};
struct AeroContext {
  std::unique_ptr<MeasuredBallReplay>& measured;
  mjData* d;
  const BallAeroParams& aero;
  const int& ball_bid;
  const int& ball_vadr;
};
struct PhysicsContext {
  SimBackend& be;
  std::unique_ptr<MeasuredBallReplay>& measured;
  std::function<void(double)>& set_measured_reference;
  const mjModel* m;
  mjData* d;
  const RacketImpulseParams& ric_pp;
  const GroundImpulseParams& gnd_pp;
  bool& calib_saved;
  int& video_ground_contacts;
  const int& ball_qadr;
  const int& ball_vadr;
  const int& face_gid;
  const int& site_id;
  double& t_ground;
  double (&calib_vin)[3];
  double (&calib_win)[3];
  double (&calib_n)[3];
  double (&calib_bp)[3];
  double (&calib_sp)[3];
  double (&calib_rv)[3];
  char (&calib_other)[32];
};
struct VideoTraceContext {
  HitPlan& plan;
  std::unique_ptr<MeasuredBallReplay>& measured;
  std::unique_ptr<SocketBallPredictor>& remote_predictor;
  const mjModel* m;
  mjData* d;
  FILE* video_trace;
  const std::string& base_mode;
  const bool& n6_mode;
  const bool& wb_mode;
  const bool& measured_replay;
  bool& wb_dead;
  bool& n6_dead;
  bool& ball_hit;
  int& video_ground_contacts;
  const int& ball_qadr;
  const int& trial;
};
void UpdatePlanning(PlanningContext c, int t, double ts);
ExecutionFrame ExecuteTrajectory(ExecutionContext c, int t, double ts);
void RecordControlState(StateTraceContext c, const ExecutionFrame& frame, int t, double ts);
void UpdateInputs(InputContext c, double ts);
void ApplyFlightForces(AeroContext c);
void AdvanceSimulation(PhysicsContext c, double ts);
void RecordVideoTrace(VideoTraceContext c, double time_s);
}
