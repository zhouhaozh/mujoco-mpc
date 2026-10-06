// M2 闭环拦截 GUI 演示:与 m2_intercept 相同管线(预测器差分预热→规划→RefineAt
// 精化→IK(scratch)→绝对时序挥拍→底盘 P 控),增加:
//   1) 实时节拍(墙钟对齐,观感真实速度)
//   2) 60Hz 状态写 /tmp/sckd_demo_state.bin 供 python/tools/demo_viewer.py 同步
// 用法:先启动 viewer(DISPLAY=:1 python python/tools/demo_viewer.py),再运行本程序。
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>

#include <mujoco/mujoco.h>

#include "control/arm_ik.h"
#include "control/chassis_servo.h"
#include "control/swing_generator.h"
#include "estimation/ball_predictor.h"
#include "hal/sim_backend.h"
#include "planning/hit_planner.h"
#include "sim/ball_aero.h"
#include "sim/ball_server.h"

using namespace sckd;

int main(int argc, char** argv) {
  const int kTrials = argc > 1 ? atoi(argv[1]) : 3;
  SimBackend be("models/generated/x4_tennis_demo.mjcf", "models/config/motors.json",
                "models/config/x4_tennis.json", "models/config/chassis_m1.json");
  if (!be.ok()) { std::printf("[FAIL] init\n"); return 1; }
  const mjModel* m = be.model();
  mjData* d = be.data_mut();
  const int ball_qadr = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_vadr = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_bid = mj_name2id(m, mjOBJ_BODY, "ball");
  const int site_id = mj_name2id(m, mjOBJ_SITE, "racket_center");

  BallAeroParams aero;
  BounceParams bp;
  PlannerParams pp;
  ChassisServo servo(be.chassis_config(), &be);
  mjData* d_ik = mj_makeData(const_cast<mjModel*>(m));  // IK 专用 scratch(勿写活体 d)
  mj_resetDataKeyframe(m, d_ik, 0);
  ArmIk ik(const_cast<mjModel*>(m), d_ik);
  std::mt19937 rng(2026);

  FILE* sh = fopen("/tmp/sckd_demo_state.bin", "wb");
  std::printf("=== M2 闭环拦截 GUI 演示(%d 次喂球,实时节拍)===\n", kTrials);
  int hits = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    be.reset(0);
    be.enable(-1, true);
    be.hold_arm(60.0, 1.5);
    servo = ChassisServo(be.chassis_config(), &be);
    BallPredictor predictor(aero, 0.030, bp);
    HitPlanner planner(pp, aero);
    SwingGenerator swing;
    HitPlan plan; plan.feasible = false;
    bool planned = false;
    // 待机位姿 = 实机 sckd-robot home(J2 抬臂 35°);挥拍就位 = 验收管线就位构型。
    // 首次规划后 0.25s 五次平滑过渡 待机→就位(后引开始前完成),
    // 既保待机视觉正确,又保已验证的挥拍几何(直接以抬臂待机作 IK 种子会使
    // 解支漂移、命中降到 1-2/10 —— 2026-10-06 实测)
    static constexpr double kStandby[5] = {-0.0244, 0.6109, -0.3927, -1.5359, 0.0};
    static constexpr double kReady[5] = {0, -1.0, 0.5, 0, 0.8};
    double q_cur[5] = {kStandby[0], kStandby[1], kStandby[2], kStandby[3], kStandby[4]};
    double t_ready0 = -1;
    double min_ball_dist = 1e9;
    bool ball_hit = false;
    double t_ground = -1;

    // 发球(与验收同分布微扰)
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

    HitIntent intent;
    intent.v_out_speed = 8.0;
    intent.v_out_elev_deg = 15.0;
    std::printf("\n--- trial %d: 发球 %.1f m/s / %.0f° ---\n",
                trial + 1, sp.speed, sp.elevation_deg);

    const auto wall0 = std::chrono::steady_clock::now();
    for (int t = 1; t <= 6000; ++t) {
      const double ts = t * 0.001;
      // 球气动(真值侧;F = -0.5ρCdA·|v|·v,力单位)
      {
        const double* bv = d->qvel + ball_vadr;
        const double vm = std::sqrt(bv[0]*bv[0]+bv[1]*bv[1]+bv[2]*bv[2]);
        const double kk = 0.5 * aero.rho * M_PI * aero.radius * aero.radius * aero.Cd;
        for (int i = 0; i < 3; ++i)
          d->xfrc_applied[ball_bid*6+i] = -kk*vm*bv[i];
        d->xfrc_applied[ball_bid*6+3] = d->xfrc_applied[ball_bid*6+4]
            = d->xfrc_applied[ball_bid*6+5] = 0;
      }
      // 观测 → 预测;臂构型取实测
      BallObs obs;
      if (be.get_ball_obs(&obs)) predictor.Update(obs);
      for (int j = 0; j < 5; ++j) q_cur[j] = be.joint_measured(AxisArm(j)).q;

      // 初次规划(启动待机→就位过渡;IK 种子=就位构型)
      if (!planned && predictor.initialized()) {
        const BaseState bs = be.base_state();
        plan = planner.Plan(predictor, bs, ts, intent);
        if (plan.feasible) {
          planned = true;
          t_ready0 = ts;
          swing.Init(plan, kReady, ik);
          std::printf("[规划] t=%.2f 击球点=(%.2f,%.2f,%.2f) t_hit=%.2f 底盘目标=(%.2f,%.2f)\n",
                      ts, plan.p_hit[0], plan.p_hit[1], plan.p_hit[2],
                      plan.t_hit, plan.base_target[0], plan.base_target[1]);
        }
      }
      // 重规划精化(50Hz,穿击锁止前):t_hit 锚定,仅精化 p_hit
      if (planned && !swing.locked() && t % 20 == 0) {
        const BaseState bs2 = be.base_state();
        planner.RefineAt(predictor, bs2, &plan, ts);
        if (swing.active()) swing.UpdatePlan(plan, ik);
        else swing.Init(plan, kReady, ik);
      }
      // 底盘 P 控就位
      if (planned && plan.feasible) {
        const BaseState bs = be.base_state();
        const double ex = plan.base_target[0] - bs.x;
        const double ey = plan.base_target[1] - bs.y;
        servo.set_twist(Twist{std::clamp(2.5*ex, -1.5, 1.5),
                              std::clamp(2.5*ey, -1.5, 1.5), 0});
      }
      servo.update(ts);
      // 臂:挥拍(绝对时序)或保持
      ArmHybridCmd ac;
      for (int j = 0; j < 5; ++j) { ac.kp[j]=60; ac.kd[j]=(j==4?1.5:6.0); }
      SwingCmd sc;
      if (swing.active() && swing.Update(ts, sc)) {
        for (int j = 0; j < 5; ++j) {
          ac.q[j]=sc.q[j]; ac.dq[j]=sc.dq[j]; ac.kp[j]=sc.kp[j]; ac.kd[j]=sc.kd[j];
        }
      } else {
        // 待机,或待机→就位过渡(五次平滑,0.25s)
        double s = 1.0;
        if (t_ready0 >= 0.0) {
          s = std::clamp((ts - t_ready0) / 0.25, 0.0, 1.0);
          const double p = 10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s;
          for (int j = 0; j < 5; ++j)
            ac.q[j] = kStandby[j] + (kReady[j] - kStandby[j]) * p;
        } else {
          for (int j = 0; j < 5; ++j) ac.q[j] = kStandby[j];
        }
      }
      double ff[5];
      if (be.arm_gravity(ac.q, ff)) for (int j=0;j<5;++j) ac.tau_ff[j]=ff[j];
      static constexpr double kArmInertia[5] = {1.2, 2.0, 0.8, 0.3, 0.05};
      for (int j = 0; j < 5; ++j) ac.tau_ff[j] += kArmInertia[j] * sc.ddp[j];
      be.set_arm_hybrid(ac);
      be.run(1);

      // 接触:球速方向改变(须在首次落地前,判"挥拍击球"而非"落地后撞拍")
      const double vx = d->qvel[ball_vadr];
      if (t_ground < 0 && d->qpos[ball_qadr+2] < 0.033 && d->qvel[ball_vadr+2] < 0)
        t_ground = ts;
      if (vx > 0 && vx0 < 0 && t_ground < 0) { ball_hit = true; }
      // 拍-球距离
      {
        double x[3]={0,0,0};
        mj_kinematics(m, d);
        mju_rotVecQuat(x, m->site_pos+site_id*3, d->xquat+m->site_bodyid[site_id]*4);
        const double* sp2 = d->xpos + m->site_bodyid[site_id]*3;
        const double dd = std::hypot(d->qpos[ball_qadr]-sp2[0]-x[0],
                                     d->qpos[ball_qadr+1]-sp2[1]-x[1],
                                     d->qpos[ball_qadr+2]-sp2[2]-x[2]);
        if (dd < min_ball_dist) min_ball_dist = dd;
      }
      // 状态同步(60Hz) + 实时节拍
      if (t % 17 == 0 && sh) {
        uint64_t seq = static_cast<uint64_t>(trial) * 100000 + t;
        int32_t nq = m->nq, nv = m->nv;
        fseek(sh, 0, SEEK_SET);
        fwrite(&seq, 8, 1, sh); fwrite(&nq, 4, 1, sh); fwrite(&nv, 4, 1, sh);
        fwrite(d->qpos, 8, m->nq, sh); fwrite(d->qvel, 8, m->nv, sh);
        fflush(sh);
      }
      while (true) {
        const double wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall0).count();
        if (wall >= ts - 0.0015) break;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
      // 终止(演示:落地后稍停;命中后多看球飞走;超时结束)
      if (d->qpos[ball_qadr+2] < 0.05 && ts > 1.0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        break;
      }
      if (ball_hit && ts > 1.8) break;
      if (ts > 4.5) break;
    }
    if (ball_hit) ++hits;
    std::printf("[结果] trial %d: %s(拍-球最小距离 %.3fm)\n",
                trial + 1, ball_hit ? "命中!" : "未中", min_ball_dist);
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  }
  if (sh) fclose(sh);
  std::printf("\n=== 演示结束: %d/%d 命中 ===\n", hits, kTrials);
  mj_deleteData(d_ik);
  return 0;
}
