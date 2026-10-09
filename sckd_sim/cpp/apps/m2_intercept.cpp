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
#include <cstring>
#include <random>

#include <mujoco/mujoco.h>

#include "control/arm_ik.h"
#include "control/arm_swing.h"
#include "control/base_ref_gen.h"
#include "control/chassis_mpc.h"
#include "control/chassis_servo.h"
#include "control/swing_generator.h"
#include "control/wholebody_ocp.h"
#include "estimation/ball_predictor.h"
#include "hal/sim_backend.h"
#include "planning/hit_planner.h"
#include "sim/ball_aero.h"
#include "sim/ball_server.h"
#include "sim/analytic_contact.h"

using namespace sckd;

// v2 triage(2026-10-08):预测实际底盘位姿@t_hit,ArmSwing 终值 IK/J/J5 对准
// 按它解——fly-through 底盘滞后 0.1-0.2m,按规划 base_target 解则拍随基座
// 平移、无人补偿(面内 miss 0.16-0.47,抽球族定位)。
// 模型 = P 控误差动力学:pred = base_target − lag·exp(−dtg/τ),τ=1/kP=0.4s
// (t_hit 时参考恰为 base_target,当前位置误差按闭环时间常数衰减)。比线性
// 外推稳:加速段外推会把锚拖离目标 0.4-0.7m(fixed 族实测 100%→30-40%);
// 误差衰减模型对 lag≈0 的族自然回目标(无阈值跳变)。仅 fly 模式调用
static void FillBasePred(HitPlan& plan, SimBackend& be, double t_now,
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
static bool RunOcp(WholebodyOcp& wbo, HitPlan& plan, SimBackend& be, double ts,
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

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_demo.mjcf";
  const int kTrials = argc > 2 ? atoi(argv[2]) : 10;
  // 挥拍层选择:quintic(五次多项式基线)/ qp(轨迹 QP,§6.4 臂优化)
  const bool use_qp = (argc > 3 && std::strcmp(argv[3], "qp") == 0);
  // QP 起挥延迟(秒,argv[4],默认 0.25):初规划后等待该时长再 Init。
  // 延迟起挥缩短轨迹时窗 T,伺服滞后显著减小 —— 三种子扫 0/0.20/0.25/0.30/0.35
  // 得 40%/86.7%/93.3%/70%/60%,0.25 为平台中心(2026-10-06,验收达标配置)。
  const double kInitDelay = (argc > 4 ? atof(argv[4]) : 0.25);
  // 底盘走位层(argv[6]):默认 P 控到位即停 / "mpc" MPC 到位即停 / "mpcfly"
  // MPC fly-through / "pfly" P+前馈 fly-through(方案 B′ 移动击球,2026-10-07:
  // 触球时刻底盘以 v_base* 通过目标,拍速获底盘贡献,臂终值改差项——联合优化的
  // 协调分解。mpcfly 首测:速度到位(1.26 m/s)但 MPC 位置滞后 0.2-0.3m → 有效
  // 命中 2/10;pfly 用速度前馈消相位滞后)
  // "ocpfly" = 方案 B v2 在线底盘 OCP(2026-10-09):WholebodyOcp 从实测状态
  // 20Hz 重解,终端底盘位置(可达集内 anchor*)/速度(v*_xy 方向)替换 B′ 的
  // back=0.8+投影启发式;底盘 P+ff 直接跟踪 OCP 轨迹(不经 BaseRefGen);
  // ArmSwing 终值锚 base_pred=OCP 终端(替代 FillBasePred 外推)。三轮蒸馏
  // (v_base 规则/信念对齐/back(z))闭环全败的共同根因 = 目标超出可达集,
  // 本模式把可达性放进约束内(PROGRESS 待办 #1)
  const std::string base_mode = (argc > 6) ? argv[6] : "";
  const bool fly = (base_mode == "mpcfly" || base_mode == "pfly" || base_mode == "ocpfly");
  const bool ocp_mode = (base_mode == "ocpfly");
  const bool use_mpc = (base_mode == "mpc" || base_mode == "mpcfly");
  // 发球族(argv[7],默认 "fixed"):fixed = M2 定点族(回归口径不变);
  // rand = 随机发球族(M3 验收):速度/仰角/方位/转速/落点全域随机;
  // gs = 抽球族(平抽,2026-10-08):平击低 spawn,甜区只在反弹后上升段
  // (跨反弹规划端到端演示;配 argv[8]="xb")。
  // "可达域内"的判定 = 规划器找到可行计划(IK 可达+时间裕度+甜区),不可达的
  // 发球单独计数、不入击球率分母(参数域按反弹后甜区落在工作空间构造,
  // 不可达占比应 <20%,否则说明域取得过宽)
  const bool rand_serve = (argc > 7 && std::strcmp(argv[7], "rand") == 0);
  const bool gs_serve = (argc > 7 && std::strcmp(argv[7], "gs") == 0);
  // v_base* 规则(argv[8],默认 0=B′ 原规则):"v2" = 方案 B v0 OCP 蒸馏规则
  // (2.2x/(1+0.55x) 饱和上限,wholebody_ocp sweep 蒸馏;闭环 A/B 用);
  // "xb" = 跨反弹规划(反弹后候选入窗,§3.5 地面支落地后开启)
  const bool cross_bounce = (argc > 8 && std::strcmp(argv[8], "xb") == 0);
  const int fly_rule = (argc > 8 && std::strcmp(argv[8], "v2") == 0) ? 1 : 0;
  SimBackend be(scene, "models/config/motors.json", "models/config/x4_tennis.json",
                "models/config/chassis_m1.json");
  if (!be.ok()) { std::printf("[FAIL] init\n"); return 1; }
  const mjModel* m = be.model();
  mjData* d = be.data_mut();
  const int ball_qadr = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_vadr = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_bid = mj_name2id(m, mjOBJ_BODY, "ball");
  const int ball_gid = mj_name2id(m, mjOBJ_GEOM, "ball_geom");
  const int face_gid = mj_name2id(m, mjOBJ_GEOM, "racket_face");
  const RacketImpulseParams ric_pp;   // 拍-球解析冲量(§3.5;k=0.5 待实机替换)
  const GroundImpulseParams gnd_pp;   // 地面解析冲量(§3.5 地面支;模型已排除对)
  const int site_id = mj_name2id(m, mjOBJ_SITE, "racket_center");

  BallAeroParams aero;
  BounceParams bp;
  PlannerParams pp;
  pp.fly_rule = fly_rule;
  pp.cross_bounce = cross_bounce ? 1 : 0;
  ChassisServo servo(be.chassis_config(), &be);
  // IK 必须用独立 scratch mjData:ArmIk::Solve 直接写 qpos(base+臂),
  // 传活体 d 会每次重规划把整车瞬移到 IK 姿态(假接触/物理污染,2026-10-06 定位)
  mjData* d_ik = mj_makeData(const_cast<mjModel*>(m));
  mj_resetDataKeyframe(m, d_ik, 0);
  ArmIk ik(const_cast<mjModel*>(m), d_ik);
  std::mt19937 rng(argc > 5 ? std::atoi(argv[5]) : 2026);   // 种子( argv[5],鲁棒性验证 )
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

  int hits = 0, contacts = 0;
  int feasible_serves = 0;   // 随机族:规划器找到可行计划的发球数(可达域分母)
  double plan_err_sum = 0; int plan_err_n = 0;
  // M3 移动击球指标:触球底盘速度 / 拍速底盘贡献占比 / 出球速度
  double base_spd_sum = 0; int base_spd_n = 0;
  double share_sum = 0; int share_n = 0;
  double out_spd_sum = 0; int out_spd_n = 0;
  int fly_ok = 0;   // 命中且触球底盘速度>0.5 m/s
  for (int trial = 0; trial < kTrials; ++trial) {
    be.reset(0);
    be.enable(-1, true);
    be.hold_arm(60.0, 1.5);
    servo = ChassisServo(be.chassis_config(), &be);
    BallPredictor predictor(aero, 0.030, bp);
    HitPlanner planner(pp, aero);
    SwingGenerator swing;
    ArmSwing aswing;
    double dq_cur[5] = {0};
    HitPlan plan; plan.feasible = false;
    bool planned = false;
    double q_cur[5] = {0, -1.0, 0.5, 0, 0.8};
    double min_ball_dist = 1e9;
    bool ball_hit = false;
    // 诊断②:锁止时刻的击球点/时刻,与真值到时对比
    bool locked_saved = false, hit_eval_done = false;
    double lock_p[3] = {0}, lock_t = 0, plan_err = -1;
    int replans = 0;
    double t_plan0 = -1;    // 初规划时刻(延迟起挥基准)
    BaseRefGen::Schedule mpc_sched;   // MPC 参考:初规划起锚(绝对时间)
    // ocpfly 每 trial 状态:OCP 协调器(热启动/守门态隔离)+ 上一接受解
    WholebodyOcp wbo;
    WboResult ocp_res;
    bool ocp_have = false;      // 已有接受解(底盘可跟踪其参考)
    int ocp_fails_streak = 0;   // 连续拒收/失败拍数
    bool ocp_dead = false;      // 连败回退:本 trial 永久走 pfly(B′)路径
    double t_min_dist = -1;
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
    BallServer server(sp);
    double q[7], v[3], w[3];
    server.Spawn(q, v, w);
    for (int i = 0; i < 7; ++i) d->qpos[ball_qadr+i] = q[i];
    for (int i = 0; i < 3; ++i) { d->qvel[ball_vadr+i] = v[i]; d->qvel[ball_vadr+3+i] = w[i]; }
    mj_forward(m, d);
    const double vx0 = v[0];

    HitIntent intent;  // 全程同一意图(旧版重规划误用默认意图,已修)
    intent.v_out_speed = 8.0;
    intent.v_out_elev_deg = 15.0;

    for (int t = 1; t <= 6000; ++t) {  // 6s max
      const double ts = t * 0.001;
      // 球气动(真值侧;无旋简化,与预测器 T1 模型一致)
      // 单位:xfrc_applied 为力(N)→ 系数 0.5ρCdA(勿除质量,见 m2_predictor 注)
      {
        const double* bv = d->qvel + ball_vadr;
        const double vm = std::sqrt(bv[0]*bv[0]+bv[1]*bv[1]+bv[2]*bv[2]);
        const double area = M_PI * aero.radius * aero.radius;
        const double kk = 0.5 * aero.rho * area * aero.Cd;   // F = -kk·|v|·v [N]
        for (int i = 0; i < 3; ++i)
          d->xfrc_applied[ball_bid*6+i] = -kk*vm*bv[i];
        d->xfrc_applied[ball_bid*6+3] = d->xfrc_applied[ball_bid*6+4]
            = d->xfrc_applied[ball_bid*6+5] = 0;
      }
      // 观测 → 预测;当前臂构型取实测
      BallObs obs;
      if (be.get_ball_obs(&obs)) predictor.Update(obs);
      for (int j = 0; j < 5; ++j) {
        const JointReport jr = be.joint_measured(AxisArm(j));
        q_cur[j] = jr.q; dq_cur[j] = jr.dq;
      }

      // 初次规划:预测器就绪即规划(越早裕度越大)
      if (!planned && predictor.initialized()) {
        const BaseState bs = be.base_state();
        plan = planner.Plan(predictor, bs, ts, intent);
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
          if (use_qp && kInitDelay > 0) {
            // 延迟起挥:此处不 Init,由下方触发块在 t_plan0+kInitDelay 起挥
            // (等待期臂保持就位构型,RefineAt 持续精化计划)
          } else if (use_qp) {
            if (fly && !ocp_mode && ref_built)
              FillBasePred(plan, be, ts, mpc_ref[0], mpc_ref[1]);
            aswing.Init(plan, q_cur, dq_cur, ik, ts);
          }
          else swing.Init(plan, q_cur, ik);
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
      // QP 延迟起挥触发
      if (use_qp && planned && kInitDelay > 0 && !aswing.active() && ts >= t_plan0 + kInitDelay) {
        if (fly && !ocp_mode && ref_built)
          FillBasePred(plan, be, ts, mpc_ref[0], mpc_ref[1]);
        aswing.Init(plan, q_cur, dq_cur, ik, ts);
        if (!aswing.active() && ts - t_plan0 < kInitDelay + 0.60)   // 初解失败诊断(限窗打印)
          std::printf("  [init-fail] t=%.2f p=(%.2f,%.2f,%.2f) base_t=(%.2f,%.2f) t_hit=%.2f\n",
                      ts, plan.p_hit[0], plan.p_hit[1], plan.p_hit[2],
                      plan.base_target[0], plan.base_target[1], plan.t_hit);
      }
      // 重规划(50Hz):t_hit 锚定,RefineAt 仅精化 p_hit/base_target
      // (评分比较会拒绝精化:高度偏好偏爱陈旧高 z 计划 —— 见 hit_planner 注)
      // QP 路径锁止后仍调用(UpdatePlan 内部按 §6.5 限微调幅度,大改自拒);
      // 五次基线保持锁止即冻结。aswing 未激活时 locked() 恒真,须判活
      if (planned && (use_qp ? aswing.active() : !swing.locked()) && t % 20 == 0) {
        const BaseState bs2 = be.base_state();
        planner.RefineAt(predictor, bs2, &plan, ts);
        ++replans;
        // 底盘参考终点:锁止后冻结(fly-through 末段 chasing 收敛中的目标会
        // 拉偏触球几何,随机族 plan_err 0.94/1.05 反例,2026-10-07);臂仍可微调
        // ocpfly:终点=OCP 终端(下方 RunOcp 后更新),非 ocp 模式随精化走 base_target
        if (!(use_qp && aswing.locked()) && !ocp_mode) {
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
          if (aswing.active()) aswing.UpdatePlan(plan, ik, ts);
        }
        else if (swing.active()) swing.UpdatePlan(plan, ik);
        else swing.Init(plan, q_cur, ik);
      }
      // 诊断②:锁止时刻快照;到 t_hit 与真值对比
      if (planned && !locked_saved && (use_qp ? (aswing.active() && aswing.locked())
                                              : (swing.active() && swing.locked()))) {
        locked_saved = true;
        for (int i = 0; i < 3; ++i) lock_p[i] = plan.p_hit[i];
        lock_t = plan.t_hit;
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
        IkResult chk = ik.Solve(lock_p, plan.base_target[0], plan.base_target[1], 0.0, q_cur);
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
      }
      // 底盘就位:MPC(50Hz 重解,到位目标 = 锁止前 0.15s 到达;非解算拍保持上一拍
      // twist)或 P 控(argv[6]="mpc" 切换;P 控解算失败时兜底)
      if (planned && plan.feasible) {
        const BaseState bs = be.base_state();
        bool mpc_applied = false;
        if (fly && !use_mpc) {   // pfly 与 ocpfly 同路(BaseRefGen 绝对时刻
                                 // 参考;ocpfly 的调度终点=OCP 终端,见上)
          // pfly:P + 速度前馈跟踪 BaseRefGen 参考(每拍)。前馈承担巡航速度,
          // P 项只纠残差 —— 无 MPC 的位置相位滞后;参考 k=0 即当前时刻值
          double x0p[6] = {bs.x, bs.y, bs.yaw, bs.vx, bs.vy, 0};
          BaseRefGen::Build(mpc_sched, ts, x0p, mpc_ref, mpc_cs2);
          ref_built = true;
          const double* r0 = &mpc_ref[0];
          const double c = std::cos(bs.yaw), sn = std::sin(bs.yaw);
          const double vwx = c*r0[3] - sn*r0[4], vwy = sn*r0[3] + c*r0[4];
          const double cx = vwx + 2.5*(r0[0]-bs.x);   // 世界系命令
          const double cy = vwy + 2.5*(r0[1]-bs.y);
          servo.set_twist(Twist{std::clamp(c*cx + sn*cy, -1.5, 1.5),
                                std::clamp(-sn*cx + c*cy, -1.5, 1.5), 0});
        } else if (use_mpc && t % 20 == 0) {
          double x0[6] = {bs.x, bs.y, bs.yaw, bs.vx, bs.vy,
                          std::clamp(bs.wz, -mpc_lim.wz_max + 0.01,
                                     mpc_lim.wz_max - 0.01)};
          const double vn = std::hypot(x0[3], x0[4]);
          const double vcap = mpc_lim.v_max - 0.05;
          if (vn > vcap) { x0[3] *= vcap / vn; x0[4] *= vcap / vn; }
          // 到位时窗在起锚时已定(Build 按 Schedule 绝对时刻取值)
          BaseRefGen::Build(mpc_sched, ts, x0, mpc_ref, mpc_cs2);
          double pp[ChassisMpc::kNp];
          ChassisMpc::PackParams(pp, x0, mpc_ref, mpc_cs2, mpc_lim);
          const auto tc0 = std::chrono::steady_clock::now();
          const bool ok = mpc.Solve(pp);
          const double ms = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - tc0).count() * 1e3;
          mpc_ms_max = std::max(mpc_ms_max, ms);
          ++mpc_solves;
          if (ok) {
            double vx, vy, wz;
            mpc.NextTwist(&vx, &vy, &wz);
            servo.set_twist(Twist{vx, vy, wz});
            mpc_applied = true;
          } else {
            ++mpc_fails;
          }
        }
        // P 控兜底:仅定点模式 MPC 求解失败时(非解算拍保持 MPC twist,勿覆盖);
        // fly 模式禁用——P 控奔向目标收敛会把飞行中的底盘中途刹停,触球时刻
        // v_base 归零;解算失败时保持上一拍 twist 更安全(MPC 全程 0 失败)
        if (!use_mpc && !fly) {
          const double ex = plan.base_target[0] - bs.x;
          const double ey = plan.base_target[1] - bs.y;
          servo.set_twist(Twist{std::clamp(2.5*ex, -1.5, 1.5),
                                std::clamp(2.5*ey, -1.5, 1.5), 0});
        }
      }
      servo.update(ts);
      // 臂:挥拍(绝对时序)或保持实测构型
      ArmHybridCmd ac;
      for (int j = 0; j < 5; ++j) { ac.kp[j]=60; ac.kd[j]=(j==4?1.5:6.0); }
      SwingCmd sc;
      if (use_qp ? aswing.Update(ts, q_cur, sc) : (swing.active() && swing.Update(ts, sc))) {
        for (int j = 0; j < 5; ++j) {
          ac.q[j]=sc.q[j]; ac.dq[j]=sc.dq[j]; ac.kp[j]=sc.kp[j]; ac.kd[j]=sc.kd[j];
        }
      } else {
        for (int j = 0; j < 5; ++j) ac.q[j] = q_cur[j];
      }
      double ff[5];
      if (be.arm_gravity(ac.q, ff)) for (int j=0;j<5;++j) ac.tau_ff[j]=ff[j];
      // 惯性前馈:τ_ff += Î·q̈(对角估计;挥拍段跟踪误差 I·a/kp 的直接抵消)
      static constexpr double kArmInertia[5] = {1.2, 2.0, 0.8, 0.3, 0.05};
      for (int j = 0; j < 5; ++j) ac.tau_ff[j] += kArmInertia[j] * sc.ddp[j];
      be.set_arm_hybrid(ac);
      // 拍-球解析冲量(§3.5 已知碰撞模型;求解器接触对已排除,真值侧在此施
      // 加)。命中事件接管 [calib] 标定捕获
      {
        const RacketImpulseEvent ev =
            ApplyRacketImpulse(m, d, face_gid, ball_qadr, ball_vadr, ric_pp);
        if (ev.fired) {
          mj_kinematics(m, d);
          double rp[3] = {0, 0, 0};
          mju_rotVecQuat(rp, m->site_pos+site_id*3, d->xquat+m->site_bodyid[site_id]*4);
          const double* sbp = d->xpos + m->site_bodyid[site_id]*3;
          for (int k = 0; k < 3; ++k) {
            calib_vin[k] = ev.v_in[k]; calib_win[k] = d->qvel[ball_vadr+3+k];
            calib_n[k] = -ev.n[k];      // [calib] 口径:n=球→拍(内向)
            calib_bp[k] = d->qpos[ball_qadr+k];
            calib_sp[k] = sbp[k] + rp[k]; calib_rv[k] = ev.v_face[k];
          }
          std::snprintf(calib_other, sizeof(calib_other), "racket_face");
          calib_saved = true;
          std::printf("  [touch-a] v_close=%.2f → vn_out=%.2f off=(%.3f,%.3f) n=(%.2f,%.2f,%.2f) vout=(%.1f,%.1f,%.1f)\n",
                      ev.v_close, ev.vn_out, ev.off_l, ev.off_h,
                      ev.n[0], ev.n[1], ev.n[2],
                      d->qvel[ball_vadr], d->qvel[ball_vadr+1], d->qvel[ball_vadr+2]);
        }
      }
      // 地面解析冲量(§3.5 地面支;模型已排除 ball↔floor 求解器对,不调则
      // 球穿地)。事件置 t_ground:保持"落地后撞拍无效"判据语义(解析冲量
      // 下球不会持续处于 z<R 且 vz<0,旧观测式永不触发)
      {
        const GroundImpulseEvent gev =
            ApplyGroundImpulse(d, ball_qadr, ball_vadr, gnd_pp);
        if (gev.fired && !gev.resting && t_ground < 0) t_ground = ts;
      }
      be.run(1);

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
      if (!ball_hit && vx > 0 && vx0 < 0) {   // 一次性(观察窗内不再重复计)
        ball_hit = true;
        t_flip = ts;
        if (post_contact < 0) post_contact = 30;   // 翻转拍≠冲量完毕,再走 30ms
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
      if (d->qpos[ball_qadr+2] < 0.05 && (locked_saved ? ts > lock_t + 0.3 : ts > 1.0)) break;
      if (ts > 5.0) break;
    }
    // ocpfly 诊断累计(跨 trial)
    ocp_solves += wbo.solves;
    ocp_rejects += wbo.rejects;
    ocp_fails += wbo.fails;
    ocp_ms_max = std::max(ocp_ms_max, wbo.ms_max);
    // 有效命中:拍接近 + 方向翻转。基线口径不变(落地前翻转,与 M2/M3 历史
    // 逐位可比);跨反弹模式另加挥拍窗因果(翻转须在锁止计划 ±0.35s 内,且
    // 抽球为反弹后翻转/截击为反弹前——排除"落地后滚进静止拍"的假命中)
    if (cross_bounce) {
      const bool swing_cause = locked_saved && t_flip > 0 && std::fabs(t_flip - lock_t) < 0.35;
      if (ball_hit && min_ball_dist < 0.30 && swing_cause &&
          (t_ground < 0 || t_flip > t_ground)) ++hits;
    } else if (ball_hit && min_ball_dist < 0.30 && t_ground < 0) {
      ++hits;
    }
    if (min_ball_dist < 0.15) ++contacts;
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
    mj_deleteData(d_ik);
    return fr >= 60.0 ? 0 : 1;
  }
  std::printf("%s (门槛 ≥80%%)\n", rate >= 80.0 ? "[PASS]" : "[FAIL]");
  mj_deleteData(d_ik);
  return rate >= 80.0 ? 0 : 1;
}
