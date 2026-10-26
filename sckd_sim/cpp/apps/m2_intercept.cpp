#include "runtime/intercept_modules.h"
#include "runtime/controller_adapters.h"
#include "runtime/synchronous_schedule.h"
#include "runtime/mujoco_robot_backend.h"
#include "runtime/robot_io_log.h"
#include "core/stage_profile.h"
#include "estimation/socket_ball_predictor.h"
#include "estimation/measured_ball_replay.h"
#include "sim_fk.h"
#include <memory>
#include <fstream>
// M2 验收:定点喂球拦截闭环(集成验收,≥80% 命中)。
// 链路:发球 → 观测(300Hz/σ=30mm) → 预测器(EKF,差分预热) → 命中点规划
//       → 底盘就位(P 控) → IK → 挥拍(绝对时序) → 触球 → 统计
// 2026-10-06 改版(闭环 0/10 三根因修复):
//   ① 预测器差分预热初始化(库内);② 挥拍绝对时刻 + 穿击段对称过击球点;
//   ③ 场景可命中性:喂球 4.5 m/s / 仰角 40° / spawn(4.8,-0.5,3.0),
//      甜区(高 0.7~1.3m、距机器人 0.4~1.1m)出现在 t≈0.95~1.1s,
//      与管线最小提前量 prep(0.55)+swing(0.35)=0.9s 兼容。
// 分步诊断:②锁止击球点 vs 真值偏差(<0.2m)③拍-球最小距离(<0.15m)④命中(方向改变)。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <stdexcept>
#include <vector>

#include <signal.h>
#include <unistd.h>

#include <mujoco/mujoco.h>

#include "control/arm_ik.h"
#include "control/arm_swing.h"
#include "control/base_ref_gen.h"
#include "control/chassis_mpc.h"
#include "control/chassis_servo.h"
#include "control/legacy_nmpc.h"
#include "control/swing_generator.h"
#include "control/wholebody_ocp.h"
#include "control/wholebody_mpc.h"
#include "estimation/ball_predictor.h"
#include "hal/sim_backend.h"
#include "planning/hit_planner.h"
#include "sim/ball_aero.h"
#include "sim/ball_server.h"
#include "sim/analytic_contact.h"

using namespace sckd;
using namespace sckd::runtime;
using sckd_nmpc6::SimArmPos;


int main(int argc, char** argv) {
  const auto resolved=runtime::ResolvedConfig::Load(argc,argv);
  const auto& cfg=resolved.values(); const auto& modes=resolved.modes();
  const std::string io_log_prefix = cfg.robot_io_log_prefix.value_or("");
  const auto& arm_feedforward = cfg.arm_feedforward;
  const bool zero_arm_feedforward = arm_feedforward == "zero";
  if (!io_log_prefix.empty() || zero_arm_feedforward)
    std::printf("[robot IO] arm_feedforward=%s log_prefix=%s (simulation only)\n",
                arm_feedforward.c_str(), io_log_prefix.c_str());
  StageProfile::Get().Configure(cfg.timing_path.value_or(""));
  ControlTrace trace(cfg.control_diagnostics_path);
  trace.RecordConfig(resolved.ToJson());
  std::printf("[resolved config] %s\n",resolved.ToJson().dump().c_str());
  const auto& scene=cfg.scene_path;const int kTrials=cfg.trials;
  const char* replay_path=cfg.measured_replay_path?cfg.measured_replay_path->c_str():nullptr;
  const double kInitDelay=cfg.time_windows.pfly_init_delay_s.value_or(cfg.init_delay_s);
  const auto& base_mode=cfg.base_mode;const auto& variant=cfg.control_variant;
  const auto& steer_rate_mode=cfg.steer_rate_mode;
  const bool measured_replay=modes.measured_replay;
  const bool use_qp=modes.use_qp;
  const bool wb_mode=modes.wb_mode;
  const bool n6_mode=modes.n6_mode;
  const bool corrected=modes.corrected;
  const bool coherent=modes.coherent;
  const bool inverse_ff=modes.inverse_ff;
  const bool chassis_model=modes.chassis_model;
  const bool chassis_recovery=modes.chassis_recovery;
  const bool chassis_guard=modes.chassis_guard;
  const bool identified_gains=modes.identified_gains;
  const bool direction_limited=modes.direction_limited;
  const bool feedforward_limited=modes.feedforward_limited;
  const bool fly=modes.fly;
  const bool ocp_mode=modes.ocp_mode;
  const bool use_mpc=modes.use_mpc;
  const bool rand_serve=modes.rand_serve;
  const bool gs_serve=modes.gs_serve;
  const bool cross_bounce=modes.cross_bounce;
  const int fly_rule=modes.fly_rule;
  std::printf("[control variant] %s\n",variant.c_str());
  std::printf("[steer rate mode] %s; motor torque/speed envelope unchanged\n",steer_rate_mode.c_str());
  nlohmann::json identified_gains_config;
  if(identified_gains){std::ifstream f(cfg.servo_identification_path);if(!f)throw std::runtime_error("missing servo identification");f>>identified_gains_config;}
  SimBackend be(scene,cfg.motors_path,cfg.robot_variant_path,cfg.chassis_path);
  if (!be.ok()) { std::printf("[FAIL] init\n"); return 1; }
  const mjModel* m = be.model();
  mjData* d = be.data_mut();
  const int ball_qadr = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_vadr = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_bid = mj_name2id(m, mjOBJ_BODY, "ball");
  const int ball_gid = mj_name2id(m, mjOBJ_GEOM, "ball_geom");
  if(measured_replay) {
    // 外源轨迹不应对机器人施加伪造碰撞冲量。
    auto* visual_model=const_cast<mjModel*>(m);
    visual_model->geom_contype[ball_gid]=0;visual_model->geom_conaffinity[ball_gid]=0;
  }
  const int face_gid = mj_name2id(m, mjOBJ_GEOM, "racket_face");
  const RacketImpulseParams ric_pp;   // 拍-球解析冲量(§3.5;k=0.5 待实机替换)
  const GroundImpulseParams gnd_pp;   // 地面解析冲量(§3.5 地面支;模型已排除对)
  const int site_id = mj_name2id(m, mjOBJ_SITE, "racket_center");

  BallAeroParams aero;
  BounceParams bp;
  PlannerParams pp;
  pp.fly_rule = fly_rule;
  pp.atomic_targets=coherent;
  pp.min_lead_s=cfg.time_windows.planner_min_lead_s;
  pp.max_horizon_s=cfg.time_windows.planner_max_horizon_s;
  pp.target_lock_s=cfg.time_windows.target_lock_s;
  pp.cross_bounce = cross_bounce ? 1 : 0;
  ChassisServo servo(be.chassis_config(), &be);
  servo.set_steering_rate_limits(direction_limited,feedforward_limited);
  // IK 必须用独立 scratch mjData:ArmIk::Solve 直接写 qpos(base+臂),
  // 传活体 d 会每次重规划把整车瞬移到 IK 姿态(假接触/物理污染,2026-10-06 定位)
  mjData* d_ik = mj_makeData(const_cast<mjModel*>(m));
  mj_resetDataKeyframe(m, d_ik, 0);
  ArmIk ik(const_cast<mjModel*>(m), d_ik);
  std::mt19937 rng(cfg.seed);   // 种子( argv[5],鲁棒性验证 )
  // 底盘 MPC(跨 trial 复用:解兼 warm start;统计全程累计)
  ChassisMpc mpc;
  ChassisMpc::Limits mpc_lim;
  BaseRefGen::Ref mpc_ref{};
  bool ref_built = false;       // pfly 参考已构建(首次 Build 前 base_pred 不填)
  double mpc_cs2[2 * ChassisMpc::kN] = {0};
  int mpc_solves = 0, mpc_fails = 0;
  double mpc_ms_max = 0;
  // ocpfly 诊断(跨 trial 累计)
  int ocp_solves = 0, ocp_rejects = 0, ocp_fails = 0;
  double ocp_ms_max = 0;
  // wbmpc 诊断(跨 trial 累计)
  int wb_solves = 0, wb_rejects = 0, wb_fails = 0, wb_fallbacks = 0;
  double wb_ms_max = 0;
  // nmpc6 诊断(跨 trial 累计)
  int n6_attempts = 0; double n6_attempt_ms = 0, n6_attempt_max_ms = 0;
  int n6_solves_t = 0, n6_rejects_t = 0, n6_fallbacks_t = 0;
  double n6_ms_max_t = 0, n6_ms_sum_t = 0;
  // 注:不加误差补偿模型(用户定向 2026-10-09:sim 跑 NMPC 本体真实行为,
  // 补偿归真机误差网/实机标定)。已实验量化:在线 comp 学习可把 fixed 从
  // 20% 拉到 60-80%(系统性短左 0.15-0.3 与旧 driver comp[0] 钳位
  // [−0.02,+0.16] 不对称正界互证),结果留档 PROGRESS,机制按定向移除

  int hits = 0, contacts = 0;
  int feasible_serves = 0;   // 随机族:规划器找到可行计划的发球数(可达域分母)
  // GUI 同步(可选,M2_VIEW=1,2026-10-09):60Hz 状态写 /tmp/sckd_demo_state.bin,
  // python/tools/demo_viewer.py 渲染(与 m2_demo 同协议);附带实时节拍。
  // 写者互斥(pid 锁):状态文件是"最后写者赢",双实例同写 = viewer 看到两套
  // 指令交错打架(用户实测 2026-10-09);陈旧锁(进程已死)自动抢占;退出删
  // 状态文件+锁(viewer 侧另有 mtime 新鲜度防护,双保险)
  FILE* sh = nullptr;
  if (cfg.viewer_enabled) {
    const char* kLock=cfg.viewer_lock_path.c_str();
    bool take = true;
    if (FILE* lf = fopen(kLock, "r")) {
      int pid = 0;
      if (fscanf(lf, "%d", &pid) == 1 && pid > 0 && kill(pid, 0) == 0) {
        std::printf("[M2_VIEW] 已有写者 pid=%d 在用状态文件,本实例视图禁用"
                    "(双写者=两套指令交错;同时只开一个仿真实例)\n", pid);
        take = false;
      }
      fclose(lf);
    }
    if (take) {
      if (FILE* lf = fopen(kLock, "w")) {
        std::fprintf(lf, "%d", (int)getpid());
        fclose(lf);
      }
      sh = fopen(cfg.viewer_state_path.c_str(), "wb");
    }
  }
  // 可选离线录像证据；读取真实状态用于画面，预测只读取已有缓存。
  std::unique_ptr<FILE, decltype(&fclose)> video_trace(nullptr, &fclose);
  if (cfg.video_trace_path) {
    const char* path=cfg.video_trace_path->c_str();
    video_trace.reset(fopen(path, "wx"));
    if (!video_trace) throw std::runtime_error("cannot create new video trace");
  }
  double plan_err_sum = 0; int plan_err_n = 0;
  // M3 移动击球指标:触球底盘速度 / 拍速底盘贡献占比 / 出球速度
  double base_spd_sum = 0; int base_spd_n = 0;
  double share_sum = 0; int share_n = 0;
  double out_spd_sum = 0; int out_spd_n = 0;
  int fly_ok = 0;   // 命中且触球底盘速度>0.5 m/s
  for (int trial = 0; trial < kTrials; ++trial) {
    trace.SetTrial(trial);
    be.reset(0);
    const auto wall0 = std::chrono::steady_clock::now();   // M2_VIEW 实时节拍锚
    be.enable(-1, true);
    be.hold_arm(60.0, 1.5);
    servo = ChassisServo(be.chassis_config(), &be);
    servo.set_steering_rate_limits(direction_limited,feedforward_limited);
    Twist issued_body_twist{};
    io::MujocoRobotBackend recording_backend(be, servo);  // 只Read，原执行路径不重复Write。
    std::unique_ptr<io::CommandLog> robot_io_log;
    std::unique_ptr<MeasuredBallReplay> measured;
    if(measured_replay)measured=std::make_unique<MeasuredBallReplay>(replay_path);
    BallPredictor legacy_predictor(aero, 0.030, bp);
    std::unique_ptr<SocketBallPredictor> remote_predictor;
    const char* predictor_socket=cfg.predictor_socket_path?cfg.predictor_socket_path->c_str():nullptr;
    if (predictor_socket && *predictor_socket)
      remote_predictor = std::make_unique<SocketBallPredictor>(predictor_socket);
    BallPredictionInterface& predictor = remote_predictor
      ? static_cast<BallPredictionInterface&>(*remote_predictor)
      : static_cast<BallPredictionInterface&>(legacy_predictor);
    const bool deduplicate_obs = remote_predictor || cfg.deduplicate_ball_observations;
    double diagnostic_goal[3]={0};double diagnostic_goal_time=-1,diagnostic_plan_t=-1,diagnostic_contact_time=-1;
    double last_obs_t = -1e30;
    std::printf("[predictor] %s dedup=%d\n", remote_predictor ? "external" : "legacy", deduplicate_obs);
    HitPlanner planner(pp, aero);
    SwingGenerator swing;
    ArmSwing aswing;aswing.Configure(corrected,coherent,be.motor_config().command_delay_ms*.001);
    aswing.ConfigureTimeWindows(cfg.time_windows.target_lock_s,cfg.time_windows.pfly_legacy_lead_s,
      cfg.time_windows.pfly_duration_min_s,cfg.time_windows.pfly_duration_max_s);
    double dq_cur[5] = {0};
    HitPlan plan; plan.feasible = false;
    bool planned = false;
    double q_cur[5] = {0, -1.0, 0.5, 0, 0.8};
    double min_ball_dist = 1e9;
    bool ball_hit = false;
    // 诊断②:锁止时刻的击球点/时刻,与真值到时对比
    bool locked_saved = false, hit_eval_done = false;
    double lock_p[3] = {0}, lock_t = 0, plan_err = -1;
    double lock_base[2] = {0};   // 锁止时刻的臂锚底盘(wbmpc=OCP 终端,余=规划锚)
    int replans = 0;
    double t_plan0 = -1;    // 初规划时刻(延迟起挥基准)
    BaseRefGen::Schedule mpc_sched;   // MPC 参考:初规划起锚(绝对时间)
    // ocpfly 每 trial 状态:OCP 协调器(热启动/守门态隔离)+ 上一接受解
    WholebodyOcp wbo;
    WboResult ocp_res;
    bool ocp_have = false;      // 已有接受解(底盘可跟踪其参考)
    int ocp_fails_streak = 0;   // 连续拒收/失败拍数
    bool ocp_dead = false;      // 连败回退:本 trial 永久走 pfly(B′)路径
    // wbmpc 每 trial 状态:联合 MPC 实例(热启动/守门态隔离)+ 最后接受计划
    WholebodyMpc wb(ik, m, WmLimits());
    WmPlan wb_plan;
    bool wb_have = false;       // 已有接受计划(臂/底盘直接采样它)
    int wb_fails_streak = 0;    // 从未有解的连续失败拍数
    bool wb_dead = false;       // 连败回退:本 trial 永久走 B′(pfly+挥拍层)
    double wb_tau = 0.0;        // τ̂ 闭环滞后补偿(EMA,采样时钟前移量)
    // nmpc6 每 trial 状态:旧机 NMPC 实例(回合态隔离;首解全零 guess 自触发
    // 新回合)+ 最后接受的 1ms 执行轨迹(Plans 式拷贝,绝对时刻采样)
    LegacyNmpcConfig n6_config;n6_config.enforce_feasibility=corrected;
    n6_config.freeze_window_s=cfg.time_windows.nmpc6.freeze_window_s;
    n6_config.allow_nonuniform_dt=cfg.time_windows.nmpc6.allow_nonuniform_dt;
    n6_config.actuator_prediction=variant=="servo_mpc";n6_config.chassis_prediction=chassis_model;
    n6_config.nlp_so=cfg.nmpc6_nlp_path;
    if (!cfg.legacy_nmpc6_limits) {
      n6_config.sim_joint23_limits = true;
      for (int k=0;k<2;++k) {
        const int jid=mj_name2id(m,mjOBJ_JOINT,k==0?"Joint2":"Joint3");
        if(jid<0 || !m->jnt_limited[jid]) throw std::runtime_error("missing MJCF arm limits");
        n6_config.joint23_lo[k]=m->jnt_range[2*jid];
        n6_config.joint23_hi[k]=m->jnt_range[2*jid+1];
      }
    }
    LegacyNmpc n6(n6_config);
    std::vector<double> n6_pos, n6_vel;
    int n6_ms = 0;
    double n6_t0 = -1;
    bool diagnostic_previous_request=false;double diagnostic_previous_target[3]={0},diagnostic_previous_target_time=0;
    bool n6_have = false, n6_dead = false;
    int n6_fails_streak = 0;int n6_recent_rejects=0;
    double n6_tau = 0.0;        // τ̂ 臂伺服带宽差补偿(EMA,采样时钟前移量)
    double t_min_dist = -1;
    int video_ground_contacts = 0;
    double t_ground = -1;   // 首次落地时刻(判"挥拍击球"还是"落地后撞拍")
    double t_flip = -1;     // 球速方向翻转时刻(跨反弹模式下判翻转是否在挥拍窗内)
    double t_bounce_plan = -1;  // 初规划时预测的反弹时刻(与 t_ground 对照诊断)
    bool touch_printed = false;
    double min_dist_decomp[3] = {-1, -1, -1};  // [法向, 面内1, 面内2]@最近时刻
    double base_speed_hit = -1;                // 触球时刻底盘速度(M3 判据>0.5)
    // 反弹标定数据(拍面支,§3.5):接触首帧的入射量,配方向翻转帧的出射量
    double calib_vin[3] = {0}, calib_win[3] = {0}, calib_n[3] = {0};
    double calib_bp[3] = {0}, calib_sp[3] = {0}, calib_rv[3] = {0};
    char calib_other[32] = "";
    bool calib_saved = false;
    int post_contact = -1;      // 方向翻转后再观察的毫秒数(取稳态出射速度)

    // 发球:fixed = 微扰定点族(M2 回归口径);rand = 随机族(M3,全域均匀,
    // 域按"反弹后甜区有机会落入工作空间"构造:y 半幅 ±1.6 内保证底盘跑道可达)
    ServerParams sp;
    if (gs_serve) {
      // 抽球族(平抽,2026-10-08):平击低 spawn,反弹后上升段进甜区——
      // 无旋(用户定向:旋转球后续迭代)。域两次整定:①z 1.2-2.0/elev 8°
      // → 球反弹前以甜区高度横穿机器人,全选反弹前截击;②speed 8.5-11/
      // spawn 6.8-7.6 → 反弹点 x≈1 贴脸,球升到打击高度前已冲过机器人,
      // 规划器倒退追打(边缘擦击 off_l 0.132,冲量不反 vx)。终版:降速+
      // 拉远,反弹点落在机器人前方 2.4-4.6m,上升段在 reach 内过甜区
      std::uniform_real_distribution<double> u(0.0, 1.0);
      sp.speed = 7.0 + 2.0 * u(rng);
      sp.elevation_deg = 3.0 * u(rng);
      sp.azimuth_deg = 178.0 + 17.0 * u(rng);
      sp.spin_rpm = 0.0;
      sp.spawn[0] = 7.4 + 0.8 * u(rng);
      sp.spawn[1] = -0.9 + 1.4 * u(rng);   // 平抽 y 几乎不漂:域须按打击点可达收窄
      sp.spawn[2] = 1.4 + 0.5 * u(rng);
      std::printf("  [serve·gs] %.1f m/s %.0f° %.0f° from (%.1f,%.1f,%.1f)\n",
                  sp.speed, sp.elevation_deg, sp.azimuth_deg,
                  sp.spawn[0], sp.spawn[1], sp.spawn[2]);
    } else if (rand_serve) {
      std::uniform_real_distribution<double> u(0.0, 1.0);
      sp.speed = 3.8 + 2.0 * u(rng);
      sp.elevation_deg = 28.0 + 22.0 * u(rng);
      sp.azimuth_deg = 178.0 + 17.0 * u(rng);
      sp.spin_rpm = 400.0 + 800.0 * u(rng);
      sp.spawn[0] = 4.4 + 1.0 * u(rng);
      sp.spawn[1] = -1.5 + 2.2 * u(rng);
      sp.spawn[2] = 2.6 + 0.8 * u(rng);
      std::printf("  [serve] %.1f m/s %.0f° %.0f° spin %.0f from (%.1f,%.1f,%.1f)\n",
                  sp.speed, sp.elevation_deg, sp.azimuth_deg, sp.spin_rpm,
                  sp.spawn[0], sp.spawn[1], sp.spawn[2]);
    } else {
      std::normal_distribution<double> jitter(0, 0.3);
      sp.speed = 4.5 + jitter(rng) * 0.15;
      sp.elevation_deg = 40.0 + jitter(rng);
      sp.azimuth_deg = 185.0 + jitter(rng) * 2.0;
      sp.spin_rpm = 800;
      sp.spawn[0] = 4.8; sp.spawn[1] = -0.5; sp.spawn[2] = 3.0;
    }
    // 单独演示族，保留原 fixed/rand/gs 基线。球网 x=11.885 m。
    if (modes.opposite_serve) {
      sp.spawn[0]=16.0; sp.spawn[1]=-0.5; sp.spawn[2]=2.0;
      sp.speed=14.0; sp.elevation_deg=8.0; sp.azimuth_deg=180.0; sp.spin_rpm=0.0;
      std::printf("[serve opposite] spawn=(16,-0.5,2) speed=14 elev=8 net_x=11.885\n");
    }
    BallServer server(sp);
    double q[7], v[3], w[3];
    server.Spawn(q, v, w);
    for (int i = 0; i < 7; ++i) d->qpos[ball_qadr+i] = q[i];
    for (int i = 0; i < 3; ++i) { d->qvel[ball_vadr+i] = v[i]; d->qvel[ball_vadr+3+i] = w[i]; }
    mj_forward(m, d);
    std::function<void(double)> set_measured_reference=[&](double stamp) {
      double p[3],speed[3];measured->Reference(stamp,p,speed);
      for(int j=0;j<3;++j){d->qpos[ball_qadr+j]=p[j];d->qvel[ball_vadr+j]=speed[j];d->qvel[ball_vadr+3+j]=0;}
      mj_forward(m,d);
    };
    if(measured)set_measured_reference(0);
    const double vx0 = v[0];
    VideoTraceContext video_context{
      /* plan */ plan,
      /* measured */ measured,
      /* remote_predictor */ remote_predictor,
      /* m */ m,
      /* d */ d,
      /* video_trace */ video_trace.get(),
      /* base_mode */ base_mode,
      /* n6_mode */ n6_mode,
      /* wb_mode */ wb_mode,
      /* measured_replay */ measured_replay,
      /* wb_dead */ wb_dead,
      /* n6_dead */ n6_dead,
      /* ball_hit */ ball_hit,
      /* video_ground_contacts */ video_ground_contacts,
      /* ball_qadr */ ball_qadr,
      /* trial */ trial,
    };
    auto record_video = [&](double time_s) {
      RecordVideoTrace(video_context, time_s);
    };
    record_video(0.0);
    if (!io_log_prefix.empty()) {
      std::vector<double> initial_act;
      if (m->na > 0) initial_act.assign(d->act, d->act + m->na);
      const nlohmann::json initial_state = {
        {"qpos", std::vector<double>(d->qpos, d->qpos + m->nq)},
        {"qvel", std::vector<double>(d->qvel, d->qvel + m->nv)},
        {"act", initial_act}, {"time", d->time},
        {"purpose", "replay_initialization_only_not_controller_input"}};
      const nlohmann::json metadata = {
        {"backend", "mujoco"}, {"frame_id", "mpc_model"},
        {"arm_feedforward", arm_feedforward}, {"command_clock", "simulation"},
        {"trial", trial}, {"seed", cfg.seed}, {"base_mode", base_mode},
        {"control_variant", variant}, {"steer_rate_mode", steer_rate_mode},
        {"direction_limited", direction_limited}, {"feedforward_limited", feedforward_limited},
        {"initial_motor_hold", {{"kp", 60.0}, {"kd", 1.5}}},
        {"scene_path", scene}, {"motors_path", cfg.motors_path},
        {"robot_variant_path", cfg.robot_variant_path}, {"chassis_path", cfg.chassis_path},
        {"step_s", SynchronousSchedule::kTickSeconds}, {"physics_step_s", m->opt.timestep},
        {"resolved_config", resolved.ToJson()}, {"mujoco_initial_state", initial_state},
        {"observation_phase", "after_command_before_physics"},
        {"joint_torque_source", "simulation_motor_applied_torque"},
        {"joint_temperature_available", false}, {"plan_id_tracking", "unavailable_zero"}};
      robot_io_log = std::make_unique<io::CommandLog>(
          io_log_prefix + "_trial_" + std::to_string(trial) + ".jsonl", metadata);
    }

    HitPlan locked_plan;
    double previous_dq_command[5]={0};bool have_dq_command=false;
    if(corrected)planner.contact_feasible=[&](const HitPlan& hp,const BaseState& bs,double now){
      const double T=hp.t_hit-now-be.motor_config().command_delay_ms*.001;
      if(T<=0){planner.last_reject="time";return false;}
      double dx=hp.base_target[0]-bs.x,dy=hp.base_target[1]-bs.y,D=std::hypot(dx,dy);
      double steer=be.joint(AxisSteer(0)).q;
      double heading=std::remainder(std::atan2(dy,dx)-bs.yaw-steer,M_PI);
      double turn=D>.05&&direction_limited?std::fabs(heading)/be.chassis_config().ramp.dir_rate_limit_rad_s:0;
      double available=std::max(0.0,T-turn);
      double vx=cos(bs.yaw)*bs.vx-sin(bs.yaw)*bs.vy,vy=sin(bs.yaw)*bs.vx+cos(bs.yaw)*bs.vy;
      double v0=D>1e-9?std::clamp((vx*dx+vy*dy)/D,-pp.base_v_max,pp.base_v_max):0;
      double a=be.chassis_config().ramp.accel_limit_mps2;
      double ta=std::clamp((pp.base_v_max-v0)/a,0.0,available);
      double distance=v0*ta+.5*a*ta*ta+pp.base_v_max*(available-ta);
      if(D>distance+.15){planner.last_reject="base_accel_speed_steer";return false;}
      const double* seed=q_cur;auto ikr=ik.Solve(hp.p_hit,hp.base_target[0],hp.base_target[1],0,seed,120);
      if(ikr.err>.03){planner.last_reject="arm_ik";return false;}
      const double alim[5]={30,30,180,400,200},vlim[5]={10,10,20,48,15};
      for(int j=0;j<5;++j){double reach=std::fabs(dq_cur[j])*T+.5*alim[j]*T*T;reach=std::min(reach,vlim[j]*T);if(std::fabs(ikr.q[j]-q_cur[j])>reach+.03){planner.last_reject="arm_travel";return false;}}
      planner.last_reject="accepted";return true;
    };
    HitIntent intent;  // 全程同一意图(旧版重规划误用默认意图,已修)
    intent.v_out_speed = 8.0;
    intent.v_out_elev_deg = 15.0;
    InputSnapshot input_snapshot;
    input_snapshot.ball.identity.track_id="trial-"+std::to_string(trial);

    // 回合装配：绑定已有对象的引用，tick内只传递具名上下文。
    AeroContext aero_context{
      /* measured */ measured,
      /* d */ d,
      /* aero */ aero,
      /* ball_bid */ ball_bid,
      /* ball_vadr */ ball_vadr,
    };

    InputContext input_context{
      /* be */ be,
      /* predictor */ predictor,
      /* measured */ measured,
      /* deduplicate_obs */ deduplicate_obs,
      /* input_snapshot */ input_snapshot,
      /* last_obs_t */ last_obs_t,
      /* q_cur */ q_cur,
      /* dq_cur */ dq_cur,
    };

    PlanningContext planning_context{
      /* be */ be,
      /* servo */ servo,
      /* ik */ ik,
      /* predictor */ predictor,
      /* planner */ planner,
      /* intent */ intent,
      /* plan */ plan,
      /* locked_plan */ locked_plan,
      /* wbo */ wbo,
      /* ocp_res */ ocp_res,
      /* wb */ wb,
      /* wb_plan */ wb_plan,
      /* n6 */ n6,
      /* aswing */ aswing,
      /* swing */ swing,
      /* mpc_sched */ mpc_sched,
      /* mpc_ref */ mpc_ref,
      /* trace */ trace,
      /* cfg */ cfg,
      /* n6_pos */ n6_pos,
      /* n6_vel */ n6_vel,
      /* m */ m,
      /* d */ d,
      /* steer_rate_mode */ steer_rate_mode,
      /* fly */ fly,
      /* n6_mode */ n6_mode,
      /* wb_mode */ wb_mode,
      /* ocp_mode */ ocp_mode,
      /* use_qp */ use_qp,
      /* corrected */ corrected,
      /* coherent */ coherent,
      /* chassis_model */ chassis_model,
      /* chassis_recovery */ chassis_recovery,
      /* chassis_guard */ chassis_guard,
      /* planned */ planned,
      /* ref_built */ ref_built,
      /* ocp_have */ ocp_have,
      /* ocp_dead */ ocp_dead,
      /* wb_have */ wb_have,
      /* wb_dead */ wb_dead,
      /* n6_have */ n6_have,
      /* n6_dead */ n6_dead,
      /* diagnostic_previous_request */ diagnostic_previous_request,
      /* locked_saved */ locked_saved,
      /* replans */ replans,
      /* ocp_fails_streak */ ocp_fails_streak,
      /* wb_fails_streak */ wb_fails_streak,
      /* n6_fails_streak */ n6_fails_streak,
      /* n6_recent_rejects */ n6_recent_rejects,
      /* n6_attempts */ n6_attempts,
      /* n6_fallbacks_t */ n6_fallbacks_t,
      /* wb_fallbacks */ wb_fallbacks,
      /* n6_ms */ n6_ms,
      /* trial */ trial,
      /* t_plan0 */ t_plan0,
      /* t_bounce_plan */ t_bounce_plan,
      /* n6_t0 */ n6_t0,
      /* n6_attempt_ms */ n6_attempt_ms,
      /* n6_attempt_max_ms */ n6_attempt_max_ms,
      /* diagnostic_goal_time */ diagnostic_goal_time,
      /* diagnostic_plan_t */ diagnostic_plan_t,
      /* diagnostic_contact_time */ diagnostic_contact_time,
      /* diagnostic_previous_target_time */ diagnostic_previous_target_time,
      /* lock_t */ lock_t,
      /* kInitDelay */ kInitDelay,
      /* q_cur */ q_cur,
      /* dq_cur */ dq_cur,
      /* diagnostic_goal */ diagnostic_goal,
      /* diagnostic_previous_target */ diagnostic_previous_target,
      /* lock_p */ lock_p,
    };

    ExecutionContext execution_context{
      /* be */ be,
      /* servo */ servo,
      /* plan */ plan,
      /* wb_plan */ wb_plan,
      /* aswing */ aswing,
      /* swing */ swing,
      /* mpc */ mpc,
      /* mpc_lim */ mpc_lim,
      /* mpc_sched */ mpc_sched,
      /* mpc_ref */ mpc_ref,
      /* cfg */ cfg,
      /* identified_gains_config */ identified_gains_config,
      /* n6_pos */ n6_pos,
      /* n6_vel */ n6_vel,
      /* m */ m,
      /* variant */ variant,
      /* fly */ fly,
      /* use_mpc */ use_mpc,
      /* n6_mode */ n6_mode,
      /* wb_mode */ wb_mode,
      /* use_qp */ use_qp,
      /* coherent */ coherent,
      /* inverse_ff */ inverse_ff,
      /* identified_gains */ identified_gains,
      /* planned */ planned,
      /* ref_built */ ref_built,
      /* wb_have */ wb_have,
      /* wb_dead */ wb_dead,
      /* n6_have */ n6_have,
      /* n6_dead */ n6_dead,
      /* have_dq_command */ have_dq_command,
      /* n6_ms */ n6_ms,
      /* mpc_solves */ mpc_solves,
      /* mpc_fails */ mpc_fails,
      /* t_plan0 */ t_plan0,
      /* n6_t0 */ n6_t0,
      /* n6_tau */ n6_tau,
      /* wb_tau */ wb_tau,
      /* mpc_ms_max */ mpc_ms_max,
      /* q_cur */ q_cur,
      /* mpc_cs2 */ mpc_cs2,
      /* previous_dq_command */ previous_dq_command,
      /* issued_body_twist */ issued_body_twist,
      /* zero_arm_feedforward */ zero_arm_feedforward,
    };

    StateTraceContext state_trace_context{
      /* be */ be,
      /* servo */ servo,
      /* planner */ planner,
      /* plan */ plan,
      /* aswing */ aswing,
      /* mpc_ref */ mpc_ref,
      /* trace */ trace,
      /* m */ m,
      /* d */ d,
      /* variant */ variant,
      /* base_mode */ base_mode,
      /* steer_rate_mode */ steer_rate_mode,
      /* n6_mode */ n6_mode,
      /* wb_mode */ wb_mode,
      /* coherent */ coherent,
      /* planned */ planned,
      /* ref_built */ ref_built,
      /* wb_dead */ wb_dead,
      /* n6_dead */ n6_dead,
      /* locked_saved */ locked_saved,
      /* site_id */ site_id,
      /* n6_tau */ n6_tau,
      /* wb_tau */ wb_tau,
      /* diagnostic_goal_time */ diagnostic_goal_time,
      /* diagnostic_plan_t */ diagnostic_plan_t,
      /* diagnostic_contact_time */ diagnostic_contact_time,
      /* lock_t */ lock_t,
      /* diagnostic_goal */ diagnostic_goal,
      /* lock_p */ lock_p,
    };

    PhysicsContext physics_context{
      /* be */ be,
      /* measured */ measured,
      /* set_measured_reference */ set_measured_reference,
      /* m */ m,
      /* d */ d,
      /* ric_pp */ ric_pp,
      /* gnd_pp */ gnd_pp,
      /* calib_saved */ calib_saved,
      /* video_ground_contacts */ video_ground_contacts,
      /* ball_qadr */ ball_qadr,
      /* ball_vadr */ ball_vadr,
      /* face_gid */ face_gid,
      /* site_id */ site_id,
      /* t_ground */ t_ground,
      /* calib_vin */ calib_vin,
      /* calib_win */ calib_win,
      /* calib_n */ calib_n,
      /* calib_bp */ calib_bp,
      /* calib_sp */ calib_sp,
      /* calib_rv */ calib_rv,
      /* calib_other */ calib_other,
    };

    for (int t = 1; t <= SynchronousSchedule::kMaxTicks; ++t) {  // 6s max
      StageProfile::Scope loop_profile("complete_tick");
      const double ts = SynchronousSchedule::Time(t);
      ApplyFlightForces(aero_context);
      UpdateInputs(input_context, ts);
      trace.SetInput(input_snapshot.ball,input_snapshot.robot,input_snapshot.have_ball);
      UpdatePlanning(planning_context, t, ts);
      // 诊断②:锁止时刻快照;到 t_hit 与真值对比(wbmpc 锁止=末段冻结窗
      // 0.15 且已有接受计划,与 aswing.locked() 同窗)
      const bool wb_lock =
          (wb_mode && !wb_dead && wb_have && plan.t_hit - ts < 0.15) ||
          (n6_mode && !n6_dead && n6_have && plan.t_hit - ts < cfg.time_windows.target_lock_s);
      if (planned && !locked_saved &&
          (wb_lock || (use_qp ? (aswing.active() && aswing.locked())
                              : (swing.active() && swing.locked())))) {
        locked_saved = true;locked_plan=plan;
        for (int i = 0; i < 3; ++i) lock_p[i] = plan.p_hit[i];
        lock_t = plan.t_hit;
        lock_base[0] = (wb_mode && wb_lock) ? wb_plan.term_base[0]
                                           : plan.base_target[0];
        lock_base[1] = (wb_mode && wb_lock) ? wb_plan.term_base[1]
                                           : plan.base_target[1];
        std::printf("  [lock] t=%.3f t_hit=%.3f p=(%.2f,%.2f,%.2f) replans=%d base_err=%.2f\n",
                    ts, lock_t, lock_p[0], lock_p[1], lock_p[2], replans,
                    std::hypot(be.base_state().x-plan.base_target[0],
                               be.base_state().y-plan.base_target[1]));
        {
          const double* xs = predictor.state();
          std::printf("  [lock] flt t=%.3f p=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n"
                      "  [lock] tru      p=(%.2f,%.2f,%.2f) v=(%.2f,%.2f,%.2f)\n",
                      predictor.t_last(), xs[0], xs[1], xs[2], xs[3], xs[4], xs[5],
                      d->qpos[ball_qadr], d->qpos[ball_qadr+1], d->qpos[ball_qadr+2],
                      d->qvel[ball_vadr], d->qvel[ball_vadr+1], d->qvel[ball_vadr+2]);
        }
      }
      if (locked_saved && !hit_eval_done && ts >= lock_t) {
        hit_eval_done = true;
        plan_err = std::hypot(d->qpos[ball_qadr]-lock_p[0],
                              d->qpos[ball_qadr+1]-lock_p[1],
                              d->qpos[ball_qadr+2]-lock_p[2]);
        plan_err_sum += plan_err; ++plan_err_n;
        double rx[3] = {0,0,0};
        mj_kinematics(m, d);
        mju_rotVecQuat(rx, m->site_pos+site_id*3, d->xquat+m->site_bodyid[site_id]*4);
        const double* sb = d->xpos + m->site_bodyid[site_id]*3;
        std::printf("  [eval] ball@(t_hit)=(%.2f,%.2f,%.2f) plan_err=%.3f racket=(%.2f,%.2f,%.2f) rk_err=%.3f\n",
                    d->qpos[ball_qadr], d->qpos[ball_qadr+1], d->qpos[ball_qadr+2], plan_err,
                    sb[0]+rx[0], sb[1]+rx[1], sb[2]+rx[2],
                    std::hypot(sb[0]+rx[0]-lock_p[0], sb[1]+rx[1]-lock_p[1], sb[2]+rx[2]-lock_p[2]));
        IkResult chk = ik.Solve(lock_p, lock_base[0], lock_base[1], 0.0, q_cur);
        double qm[5], dq_max = 0;
        for (int j = 0; j < 5; ++j) {
          qm[j] = be.joint_measured(AxisArm(j)).q;
          dq_max = std::max(dq_max, std::fabs(chk.q[j] - qm[j]));
        }
        std::printf("  [eval] ik conv=%d err=%.4f  max|q_ik-q_meas|=%.3f rad  base=(%.2f,%.2f)\n",
                    (int)chk.converged, chk.err, dq_max,
                    be.base_state().x, be.base_state().y);
        // M3 指标:t_hit 时刻底盘速度(fly-through 达标判据 >0.5 m/s)
        {
          const BaseState bse = be.base_state();
          base_speed_hit = std::hypot(bse.vx, bse.vy);
          base_spd_sum += base_speed_hit; ++base_spd_n;
          std::printf("  [eval] 触球底盘速度=%.2f m/s%s\n", base_speed_hit,
                      base_speed_hit > 0.5 ? "(>0.5 达标)" : "");
        }
        // nmpc6 误差分解:t_hit 时最后计划的底盘/臂 vs 实测(定位哪层吃掉时序)
        if (n6_mode && n6_have && !n6_dead && n6_ms > 0) {
          const int W = n6_ms + 1;
          const int idx = std::clamp(
              (int)std::lround((lock_t - n6_t0) * 1000.0), 0, n6_ms);
          const BaseState bse = be.base_state();
          double dq_lag = 0, pdq_max = 0, mdq_max = 0;
          for (int j = 0; j < 5; ++j) {
            dq_lag = std::max(dq_lag, std::fabs(
                n6_pos[(2 + j) * W + idx] - q_cur[j]));
            pdq_max = std::max(pdq_max, std::fabs(n6_vel[(2 + j) * W + idx]));
            mdq_max = std::max(mdq_max, std::fabs(dq_cur[j]));
          }
          std::printf("  [eval·n6] 底盘误差=(%.3f,%.3f) 臂maxΔq=%.3f rad "
                      "dq: plan=%.1f meas=%.1f rad/s\n",
                      n6_pos[idx] - bse.x, n6_pos[W + idx] - bse.y,
                      dq_lag, pdq_max, mdq_max);
        }
      }
      const auto execution_frame=ExecuteTrajectory(execution_context, t, ts);
      if (robot_io_log)
        robot_io_log->Write(execution_frame.command,
                            recording_backend.Read({ts, io::Clock::kSimulation}),
                            {true, "submitted_to_mujoco"});
      RecordControlState(state_trace_context, execution_frame, t, ts);
      AdvanceSimulation(physics_context, ts);
      // 接触检测:球速方向改变(-x → +x)
      const double vx = d->qvel[ball_vadr];
      // 诊断:击球窗内首次球接触(geom 对/接触法向/球速)
      if (locked_saved && !touch_printed && ts > lock_t - 0.30 && ts < lock_t + 0.30 &&
          d->ncon > 0) {
        for (int ci = 0; ci < d->ncon; ++ci) {
          const int g1 = d->contact[ci].geom1, g2 = d->contact[ci].geom2;
          const bool has_ball = (g1 == ball_gid || g2 == ball_gid);
          if (!has_ball) continue;
          const int other = (g1 == ball_gid) ? g2 : g1;
          const double* cn = d->contact[ci].frame;  // 法向 = frame[0..2](g1→g2)
          double nball[3] = {cn[0], cn[1], cn[2]};
          if (g2 == ball_gid) for (int k = 0; k < 3; ++k) nball[k] = -nball[k];
          std::printf("  [touch] t=%.3f vs %s n=(%.2f,%.2f,%.2f) bv=(%.1f,%.1f,%.1f)\n",
                      ts, mj_id2name(m, mjOBJ_GEOM, other),
                      nball[0], nball[1], nball[2],
                      d->qvel[ball_vadr], d->qvel[ball_vadr+1], d->qvel[ball_vadr+2]);
          if (!calib_saved) {   // 首个球接触帧:入射量快照(标定数据,§3.5)
            mj_kinematics(m, d);
            double rp[3] = {0, 0, 0};
            mju_rotVecQuat(rp, m->site_pos+site_id*3, d->xquat+m->site_bodyid[site_id]*4);
            const double* sbp = d->xpos + m->site_bodyid[site_id]*3;
            double rv6[6];
            mj_objectVelocity(m, d, mjOBJ_SITE, site_id, rv6, /*flg_local=*/0);
            for (int k = 0; k < 3; ++k) {
              calib_vin[k] = d->qvel[ball_vadr+k]; calib_win[k] = d->qvel[ball_vadr+3+k];
              calib_n[k] = nball[k]; calib_bp[k] = d->qpos[ball_qadr+k];
              calib_sp[k] = sbp[k] + rp[k]; calib_rv[k] = rv6[3+k];
            }
            std::snprintf(calib_other, sizeof(calib_other), "%s",
                          mj_id2name(m, mjOBJ_GEOM, other));
            calib_saved = true;
          }
          touch_printed = true;
        }
      }
      if (t_ground < 0 && d->qpos[ball_qadr+2] < 0.033 && d->qvel[ball_vadr+2] < 0)
        t_ground = ts;
      if (!measured && !ball_hit && vx > 0 && vx0 < 0) {   // 一次性(观察窗内不再重复计)
        ball_hit = true;
        t_flip = ts;
        // 翻转拍≠冲量完毕,再走 30ms 取稳态出射;GUI 下拉长到 0.8s 让球飞
        // 一段(用户实测"没看到击球就跳下一个"—— 批处理口径不受影响,
        // 命中判定在 t_flip 已定,观察窗只影响 [calib] 取数与结束时刻)
        if (post_contact < 0) post_contact = sh ? 830 : 30;
        // M3 拍速分解:拍速 = 底盘贡献 + 臂贡献;出球速度 vs 意图(8 m/s)
        {
          double rv[6];
          mj_objectVelocity(m, d, mjOBJ_SITE, site_id, rv, /*flg_local=*/0);
          const double rk = std::hypot(rv[3], rv[4], rv[5]);        // 拍速(世界系)
          const BaseState bsc = be.base_state();
          const double vbx = std::cos(bsc.yaw)*bsc.vx - std::sin(bsc.yaw)*bsc.vy;
          const double vby = std::sin(bsc.yaw)*bsc.vx + std::cos(bsc.yaw)*bsc.vy;
          const double vbp = (rk > 0.1) ? (vbx*rv[3] + vby*rv[4]) / rk : 0.0;
          const double vout = std::hypot(d->qvel[ball_vadr], d->qvel[ball_vadr+1],
                                         d->qvel[ball_vadr+2]);
          share_sum += vbp / rk; ++share_n;
          out_spd_sum += vout; ++out_spd_n;
          if (std::hypot(bsc.vx, bsc.vy) > 0.5) ++fly_ok;   // 触球时刻底盘速度达标
          std::printf("  [contact] t=%.3f ball_z=%.3f %s 拍速=%.1f 底盘贡献=%.1f(%.0f%%) 出球=%.1f\n",
                      ts, d->qpos[ball_qadr+2],
                      (t_ground > 0) ? "(落地后!非挥拍击球)" : "(落地前·有效)",
                      rk, vbp, 100.0*vbp/rk, vout);
        }
        // 不立即 break:翻转拍≠冲量完毕,再观察 30ms 取稳态出射(标定数据质量)
      }
      // 拍-球距离
      {
        double x[3]={0,0,0};
        mj_kinematics(m, d);
        mju_rotVecQuat(x, m->site_pos+site_id*3, d->xquat+m->site_bodyid[site_id]*4);
        const double* sp2 = d->xpos + m->site_bodyid[site_id]*3;
        const double dd = std::hypot(d->qpos[ball_qadr]-sp2[0]-x[0],
                                     d->qpos[ball_qadr+1]-sp2[1]-x[1],
                                     d->qpos[ball_qadr+2]-sp2[2]-x[2]);
        if (dd < min_ball_dist) {
          min_ball_dist = dd; t_min_dist = ts;
          // 分解:球-拍心偏差在拍面系(法向=盒薄轴 y_local)
          const int rb = m->site_bodyid[site_id];
          double nrm[3] = {0, 1, 0}, nrm_w[3];
          mju_rotVecQuat(nrm_w, nrm, d->xquat + rb * 4);
          const double dvec[3] = {d->qpos[ball_qadr]-(sp2[0]+x[0]),
                                  d->qpos[ball_qadr+1]-(sp2[1]+x[1]),
                                  d->qpos[ball_qadr+2]-(sp2[2]+x[2])};
          min_dist_decomp[0] = dvec[0]*nrm_w[0] + dvec[1]*nrm_w[1] + dvec[2]*nrm_w[2];
          const double ip = std::hypot(dvec[0], dvec[1], dvec[2]);
          min_dist_decomp[1] = std::sqrt(std::max(0.0, ip*ip - min_dist_decomp[0]*min_dist_decomp[0]));
          min_dist_decomp[2] = ip;
        }
      }
      if (t % 10 == 0 || (measured && ts>=measured->end()) || video_ground_contacts >= 2 || (post_contact == 1) ||
          (d->qpos[ball_qadr+2] < 0.05 && (locked_saved ? ts > lock_t + 0.3 : ts > 1.0)))
        record_video(ts);
      // GUI 同步(60Hz,M2_VIEW)+ 实时节拍(m2_demo 同协议:seq/nq/nv/qpos/qvel)
      if (sh && SynchronousSchedule::Viewer(t)) {
        uint64_t vseq = static_cast<uint64_t>(trial) * 100000 + t;
        int32_t nq = m->nq, nv = m->nv;
        fseek(sh, 0, SEEK_SET);
        fwrite(&vseq, 8, 1, sh); fwrite(&nq, 4, 1, sh); fwrite(&nv, 4, 1, sh);
        fwrite(d->qpos, 8, m->nq, sh); fwrite(d->qvel, 8, m->nv, sh);
        fflush(sh);
      }
      if (sh) {
        while (true) {
          const double wall = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - wall0).count();
          if (wall >= ts - 0.0015) break;
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
      }
      // 终止:落地(越过规划击球时刻后)/超时
      if (post_contact > 0 && --post_contact == 0) {
        // 稳态出射:反弹标定 CSV(拍面支,python 拟合 a/b 用;§3.5)
        if (calib_saved)
          std::printf("[calib] other=%s vin=%.3f,%.3f,%.3f rv=%.3f,%.3f,%.3f "
                      "n=%.3f,%.3f,%.3f bp=%.3f,%.3f,%.3f sp=%.3f,%.3f,%.3f "
                      "vout=%.3f,%.3f,%.3f win=%.2f,%.2f,%.2f wout=%.2f,%.2f,%.2f\n",
                      calib_other,
                      calib_vin[0], calib_vin[1], calib_vin[2],
                      calib_rv[0], calib_rv[1], calib_rv[2],
                      calib_n[0], calib_n[1], calib_n[2],
                      calib_bp[0], calib_bp[1], calib_bp[2],
                      calib_sp[0], calib_sp[1], calib_sp[2],
                      d->qvel[ball_vadr], d->qvel[ball_vadr+1], d->qvel[ball_vadr+2],
                      calib_win[0], calib_win[1], calib_win[2],
                      d->qvel[ball_vadr+3], d->qvel[ball_vadr+4], d->qvel[ball_vadr+5]);
        break;
      }
      if(measured) {if(ts>=measured->end())break;}
      else if (d->qpos[ball_qadr+2] < 0.05 && (locked_saved ? ts > lock_t + 0.3 : ts > 1.0)) break;
      if (ts > 5.0) break;
    }
    // ocpfly 诊断累计(跨 trial)
    ocp_solves += wbo.solves;
    ocp_rejects += wbo.rejects;
    ocp_fails += wbo.fails;
    ocp_ms_max = std::max(ocp_ms_max, wbo.ms_max);
    // wbmpc 诊断累计(跨 trial)
    wb_solves += wb.solves;
    wb_rejects += wb.rejects;
    wb_fails += wb.fails;
    wb_ms_max = std::max(wb_ms_max, wb.ms_max);
    // nmpc6 诊断累计(跨 trial)
    n6_solves_t += n6.solves;
    n6_rejects_t += n6.rejects;
    n6_ms_max_t = std::max(n6_ms_max_t, n6.ms_max);
    n6_ms_sum_t += n6.ms_sum;
    // 有效命中:拍接近 + 方向翻转。基线口径不变(落地前翻转,与 M2/M3 历史
    // 逐位可比);跨反弹模式另加挥拍窗因果(翻转须在锁止计划 ±0.35s 内,且
    // 抽球为反弹后翻转/截击为反弹前——排除"落地后滚进静止拍"的假命中)
    if (cross_bounce) {
      const bool swing_cause = locked_saved && t_flip > 0 && std::fabs(t_flip - lock_t) < 0.35;
      if (ball_hit && min_ball_dist < 0.30 && swing_cause &&
          (t_ground < 0 || t_flip > t_ground)) ++hits;
    } else if (ball_hit && min_ball_dist < 0.30 &&
               (t_ground < 0 || t_flip < t_ground)) {
      // 有效=方向翻转先于首次落地。原判 t_ground<0 把"观察窗内没落地"错当
      // "击球在落地前":M2_VIEW 观察窗 830ms 内命中球飞完落地会写 t_ground
      // → 真有效击球被全数误拒(用户实测 3/3 显示 FAIL,2026-10-10 定位)
      ++hits;
    }
    if (min_ball_dist < 0.15) ++contacts;
    // GUI trial 间驻留:末帧停 0.7s 再进下一发(批处理无感)
    if (sh) std::this_thread::sleep_for(std::chrono::milliseconds(700));
    char pe_str[16];
    if (plan_err >= 0) std::snprintf(pe_str, sizeof(pe_str), "%.3f", plan_err);
    else std::snprintf(pe_str, sizeof(pe_str), "n/a");
    std::printf("trial %2d: hit=%d min_dist=%.3fm@t=%.3f(plan t_hit=%.3f) 法向=%.3f 面内=%.3f plan_err=%s"
                " 落地=%.2f(预测%.2f)\n",
                trial+1, ball_hit, min_ball_dist, t_min_dist, lock_t,
                min_dist_decomp[0], min_dist_decomp[1], pe_str, t_ground, t_bounce_plan);
    if (planned) ++feasible_serves;
  }
  const double rate = 100.0 * hits / kTrials;
  const double crate = 100.0 * contacts / kTrials;
  const double mean_pe = plan_err_n ? plan_err_sum / plan_err_n : -1;
  char mean_pe_str[16];
  if (mean_pe >= 0) std::snprintf(mean_pe_str, sizeof(mean_pe_str), "%.3f", mean_pe);
  else std::snprintf(mean_pe_str, sizeof(mean_pe_str), "n/a");
  std::printf("\n=== M2 拦截验收 ===\n");
  std::printf("命中 %d/%d(%.0f%%) 接近率 %.0f%% 平均击球点偏差 %sm\n",
              hits, kTrials, rate, crate, mean_pe_str);
  if (use_mpc)
    std::printf("底盘 MPC:%d 解 %d 失败(%.1f%%) 最大单解 %.2fms\n",
                mpc_solves, mpc_fails, 100.0 * mpc_fails / std::max(mpc_solves, 1),
                mpc_ms_max);
  if (ocp_mode)
    std::printf("在线OCP: %d 解 %d 拒收 %d 失败,最大单解 %.2fms(20Hz 拍,守门口径)\n",
                ocp_solves, ocp_rejects, ocp_fails, ocp_ms_max);
  if (wb_mode)
    std::printf("全身MPC: %d 解 %d 拒收 %d 失败,最大单解 %.2fms(20Hz 拍,研究口径) 回退 %d trial\n",
                wb_solves, wb_rejects, wb_fails, wb_ms_max, wb_fallbacks);
  if (n6_mode)
    std::printf("旧机NMPC: %d 解 %d 拒收 平均 %.2fms 最大单解 %.2fms(50Hz 重解,守门口径) 回退 %d trial\n",
                n6_solves_t, n6_rejects_t,
                n6_solves_t ? n6_ms_sum_t / n6_solves_t : 0.0,
                n6_ms_max_t, n6_fallbacks_t);
  if (n6_mode) std::printf("NMPC attempts=%d mean_ms=%.3f max_ms=%.3f\n", n6_attempts, n6_attempt_ms/std::max(1,n6_attempts), n6_attempt_max_ms);
  if (base_spd_n > 0)
    std::printf("M3 指标: t_hit 底盘速度均值 %.2f m%s | 命中且触球底盘速度>0.5: %d/%d"
                " | 拍速底盘贡献 %.0f%% | 出球速度 %.1f m/s\n",
                base_spd_sum / base_spd_n, (fly ? "(fly-through)" : "(定点)"),
                fly_ok, hits,
                share_n ? 100.0 * share_sum / share_n : 0.0,
                out_spd_n ? out_spd_sum / out_spd_n : 0.0);
  std::printf("诊断: ②击球点偏差<0.2m ③拍球距离<0.15m ④命中率≥80%%\n");
  if (rand_serve || gs_serve) {
    // M3 随机/抽球族口径:击球率分母 = 可达发球(规划可行);不可达单独报告
    const double fr = feasible_serves ? 100.0 * hits / feasible_serves : 0.0;
    const double in_dom = 100.0 * feasible_serves / kTrials;
    std::printf("随机发球族: 可达 %d/%d(%.0f%%) | 命中(可达域内) %d/%d = %.0f%%\n",
                feasible_serves, kTrials, in_dom, hits, feasible_serves, fr);
    std::printf("%s (M3 门槛: 可达域内 ≥60%%)\n", fr >= 60.0 ? "[PASS]" : "[FAIL]");
    if (sh) { fclose(sh); remove(cfg.viewer_state_path.c_str()); remove(cfg.viewer_lock_path.c_str()); }
    mj_deleteData(d_ik);
    return fr >= 60.0 ? 0 : 1;
  }
  std::printf("%s (门槛 ≥80%%)\n", rate >= 80.0 ? "[PASS]" : "[FAIL]");
  if(measured_replay)std::printf("[MEASURED REPLAY] 外源实测参考；无真实击球/出球判定，不能使用上述命中率作为验收。\n");
  if (sh) { fclose(sh); remove(cfg.viewer_state_path.c_str()); remove(cfg.viewer_lock_path.c_str()); }
  mj_deleteData(d_ik);
  return rate >= 80.0 ? 0 : 1;
}
