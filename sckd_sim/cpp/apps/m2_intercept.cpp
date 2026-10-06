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
#include "estimation/ball_predictor.h"
#include "hal/sim_backend.h"
#include "planning/hit_planner.h"
#include "sim/ball_aero.h"
#include "sim/ball_server.h"

using namespace sckd;

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
  const std::string base_mode = (argc > 6) ? argv[6] : "";
  const bool fly = (base_mode == "mpcfly" || base_mode == "pfly");
  const bool use_mpc = (base_mode == "mpc" || base_mode == "mpcfly");
  SimBackend be(scene, "models/config/motors.json", "models/config/x4_tennis.json",
                "models/config/chassis_m1.json");
  if (!be.ok()) { std::printf("[FAIL] init\n"); return 1; }
  const mjModel* m = be.model();
  mjData* d = be.data_mut();
  const int ball_qadr = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_vadr = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_bid = mj_name2id(m, mjOBJ_BODY, "ball");
  const int ball_gid = mj_name2id(m, mjOBJ_GEOM, "ball_geom");
  const int site_id = mj_name2id(m, mjOBJ_SITE, "racket_center");

  BallAeroParams aero;
  BounceParams bp;
  PlannerParams pp;
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
  double mpc_cs2[2 * ChassisMpc::kN] = {0};
  int mpc_solves = 0, mpc_fails = 0;
  double mpc_ms_max = 0;

  int hits = 0, contacts = 0;
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
    double t_min_dist = -1;
    double t_ground = -1;   // 首次落地时刻(判"挥拍击球"还是"落地后撞拍")
    bool touch_printed = false;
    double min_dist_decomp[3] = {-1, -1, -1};  // [法向, 面内1, 面内2]@最近时刻
    double base_speed_hit = -1;                // 触球时刻底盘速度(M3 判据>0.5)

    // 发球(微扰):慢弧线喂球,保证甜区落在管线可达时间窗内
    ServerParams sp;
    std::normal_distribution<double> jitter(0, 0.3);
    sp.speed = 4.5 + jitter(rng) * 0.15;
    sp.elevation_deg = 40.0 + jitter(rng);
    sp.azimuth_deg = 185.0 + jitter(rng) * 2.0;
    sp.spin_rpm = 800;
    sp.spawn[0] = 4.8; sp.spawn[1] = -0.5; sp.spawn[2] = 3.0;
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
          if (use_qp && kInitDelay > 0) {
            // 延迟起挥:此处不 Init,由下方触发块在 t_plan0+kInitDelay 起挥
            // (等待期臂保持就位构型,RefineAt 持续精化计划)
          } else if (use_qp) aswing.Init(plan, q_cur, dq_cur, ik, ts);
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
      if (use_qp && planned && kInitDelay > 0 && !aswing.active() && ts >= t_plan0 + kInitDelay)
        aswing.Init(plan, q_cur, dq_cur, ik, ts);
      // 重规划(50Hz,后引锁止前):t_hit 锚定,RefineAt 仅精化 p_hit/base_target
      // (评分比较会拒绝精化:高度偏好偏爱陈旧高 z 计划 —— 见 hit_planner 注)
      // (aswing 未激活时 locked() 恒真,须判活,否则延迟起挥期会跳过精化)
      if (planned && !(use_qp ? (aswing.active() && aswing.locked())
                              : swing.locked()) && t % 20 == 0) {
        const BaseState bs2 = be.base_state();
        planner.RefineAt(predictor, bs2, &plan, ts);
        ++replans;
        mpc_sched.tx = plan.base_target[0];   // 终点随精化更新(起锚不变)
        mpc_sched.ty = plan.base_target[1];
        if (use_qp) { if (aswing.active()) aswing.UpdatePlan(plan, ik, ts); }
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
        if (fly && !use_mpc) {
          // pfly:P + 速度前馈跟踪 BaseRefGen 参考(每拍)。前馈承担巡航速度,
          // P 项只纠残差 —— 无 MPC 的位置相位滞后;参考 k=0 即当前时刻值
          double x0p[6] = {bs.x, bs.y, bs.yaw, bs.vx, bs.vy, 0};
          BaseRefGen::Build(mpc_sched, ts, x0p, mpc_ref, mpc_cs2);
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
          touch_printed = true;
        }
      }
      if (t_ground < 0 && d->qpos[ball_qadr+2] < 0.033 && d->qvel[ball_vadr+2] < 0)
        t_ground = ts;
      if (vx > 0 && vx0 < 0) {
        ball_hit = true;
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
        break;
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
      if (d->qpos[ball_qadr+2] < 0.05 && (locked_saved ? ts > lock_t + 0.3 : ts > 1.0)) break;
      if (ts > 5.0) break;
    }
    if (ball_hit && min_ball_dist < 0.30 && t_ground < 0) ++hits;  // 有效:落地前方向改变且拍接近
    if (min_ball_dist < 0.15) ++contacts;
    char pe_str[16];
    if (plan_err >= 0) std::snprintf(pe_str, sizeof(pe_str), "%.3f", plan_err);
    else std::snprintf(pe_str, sizeof(pe_str), "n/a");
    std::printf("trial %2d: hit=%d min_dist=%.3fm@t=%.3f(plan t_hit=%.3f) 法向=%.3f 面内=%.3f plan_err=%s\n",
                trial+1, ball_hit, min_ball_dist, t_min_dist, lock_t,
                min_dist_decomp[0], min_dist_decomp[1], pe_str);
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
  if (base_spd_n > 0)
    std::printf("M3 指标: t_hit 底盘速度均值 %.2f m%s | 命中且触球底盘速度>0.5: %d/%d"
                " | 拍速底盘贡献 %.0f%% | 出球速度 %.1f m/s\n",
                base_spd_sum / base_spd_n, (fly ? "(fly-through)" : "(定点)"),
                fly_ok, hits,
                share_n ? 100.0 * share_sum / share_n : 0.0,
                out_spd_n ? out_spd_sum / out_spd_n : 0.0);
  std::printf("诊断: ②击球点偏差<0.2m ③拍球距离<0.15m ④命中率≥80%%\n");
  std::printf("%s (门槛 ≥80%%)\n", rate >= 80.0 ? "[PASS]" : "[FAIL]");
  mj_deleteData(d_ik);
  return rate >= 80.0 ? 0 : 1;
}
