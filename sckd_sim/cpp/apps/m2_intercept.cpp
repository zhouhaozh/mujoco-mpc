// M2 验收:定点喂球拦截闭环(集成验收,≥80% 命中)。
// 链路:发球 → 观测(300Hz/σ=30mm) → 预测器(EKF) → 命中点规划 → IK → 挥拍 → 触球 → 统计
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

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
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_demo.mjcf";
  const int kTrials = argc > 2 ? atoi(argv[2]) : 10;
  SimBackend be(scene, "models/config/motors.json", "models/config/x4_tennis.json",
                "models/config/chassis_m1.json");
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
  ArmIk ik(const_cast<mjModel*>(m), d);
  std::mt19937 rng(2026);

  int hits = 0, contacts = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    be.reset(0);
    be.enable(-1, true);
    be.hold_arm(60.0, 1.5);
    servo = ChassisServo(be.chassis_config(), &be);
    BallPredictor predictor(aero, 0.030, bp);
    HitPlanner planner(pp, aero);
    SwingGenerator swing;
    HitPlan plan; plan.feasible = false;
    bool planned = false, swinging = false;
    double q_cur[5] = {0, -1.0, 0.5, 0, 0.8};
    double t_swing_start = -1;
    double min_ball_dist = 1e9;
    bool ball_hit = false, direction_changed = false;

    // 发球(微扰)
    ServerParams sp;
    std::normal_distribution<double> jitter(0, 0.3);
    sp.speed = 8.0 + jitter(rng) * 0.5;
    sp.elevation_deg = 18.0 + jitter(rng);
    sp.azimuth_deg = 188.0 + jitter(rng) * 2;
    sp.spin_rpm = 1500;
    sp.spawn[0] = 6.0; sp.spawn[1] = -0.8; sp.spawn[2] = 1.8;
    BallServer server(sp);
    double q[7], v[3], w[3];
    server.Spawn(q, v, w);
    for (int i = 0; i < 7; ++i) d->qpos[ball_qadr+i] = q[i];
    for (int i = 0; i < 3; ++i) { d->qvel[ball_vadr+i] = v[i]; d->qvel[ball_vadr+3+i] = w[i]; }
    mj_forward(m, d);
    const double vx0 = v[0];

    for (int t = 1; t <= 5000; ++t) {  // 5s max
      const double ts = t * 0.001;
      // 球气动(真值侧)
      {
        const double* bv = d->qvel + ball_vadr;
        const double vm = std::sqrt(bv[0]*bv[0]+bv[1]*bv[1]+bv[2]*bv[2]);
        const double area = M_PI * aero.radius * aero.radius;
        const double kk = 0.5 * aero.rho * area / aero.mass;
        for (int i = 0; i < 3; ++i)
          d->xfrc_applied[ball_bid*6+i] = -kk*aero.Cd*vm*bv[i];
        d->xfrc_applied[ball_bid*6+3]=d->xfrc_applied[ball_bid*6+4]=d->xfrc_applied[ball_bid*6+5]=0;
      }
      // 观测 → 预测
      BallObs obs;
      if (be.get_ball_obs(&obs)) predictor.Update(obs);
      // 规划(预测器就绪且未规划)
      if (!swinging && predictor.initialized() && t > 200) {
        const BaseState bs = be.base_state();
        HitIntent intent;
        intent.v_out_speed = 8.0;
        intent.v_out_elev_deg = 15.0;
        plan = planner.Plan(predictor, bs, ts, intent);
        if (plan.feasible) {
          planned = true;
          swing.Init(plan, q_cur, ik);
          if (t % 500 == 0)
            std::printf("  t=%.2f plan: hit=(%.2f,%.2f,%.2f) t_hit=%.2f margin=%.2f base=(%.2f,%.2f)\n",
                        ts, plan.p_hit[0],plan.p_hit[1],plan.p_hit[2],
                        plan.t_hit, plan.time_margin, plan.base_target[0], plan.base_target[1]);
        }
      }
      // 底盘:就位
      if (planned && plan.feasible) {
        const BaseState bs = be.base_state();
        const double ex = plan.base_target[0] - bs.x;
        const double ey = plan.base_target[1] - bs.y;
        servo.set_twist(Twist{std::clamp(2.5*ex, -1.5, 1.5),
                              std::clamp(2.5*ey, -1.5, 1.5), 0});
      }
      servo.update(ts);
      // 臂:挥拍或保持
      ArmHybridCmd ac;
      for (int j = 0; j < 5; ++j) { ac.kp[j]=60; ac.kd[j]=(j==4?1.5:6.0); }
      // 重规划(50Hz,直到挥拍锁止)
      if (planned && !swinging && predictor.initialized() && t % 20 == 0) {
        const BaseState bs2 = be.base_state();
        HitPlan np = planner.Plan(predictor, bs2, ts, HitIntent{});
        if (np.feasible && np.score > plan.score) {
          plan = np;
          swing.Init(plan, q_cur, ik);
        }
      }
      if (planned && plan.feasible && swing.active()) {
        if (!swinging && ts >= plan.t_hit - pp.swing_time - pp.prep_time) {
          swinging = true; t_swing_start = ts;
        }
        if (swinging) {
          SwingCmd sc;
          if (swing.Update(ts - t_swing_start, sc)) {
            for (int j = 0; j < 5; ++j) {
              ac.q[j]=sc.q[j]; ac.dq[j]=sc.dq[j]; ac.kp[j]=sc.kp[j]; ac.kd[j]=sc.kd[j];
            }
          }
        }
      } else {
        for (int j = 0; j < 5; ++j) ac.q[j] = q_cur[j];
      }
      double ff[5];
      if (be.arm_gravity(ac.q, ff)) for (int j=0;j<5;++j) ac.tau_ff[j]=ff[j];
      be.set_arm_hybrid(ac);
      be.run(1);

      // 接触检测:球速方向变化(从 -x 变 +x)
      const double vx = d->qvel[ball_vadr];
      if (vx > 0 && vx0 < 0) { direction_changed = true; }
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
      // 终止条件
      if (direction_changed) { ball_hit = true; break; }
      if (d->qpos[ball_qadr+2] < 0.05 && ts > 1.0) break;  // 球落地
      if (ts > 4.0) break;
    }
    if (ball_hit) ++hits;
    if (min_ball_dist < 0.15) ++contacts;  // 拍球接近(近似接触)
    std::printf("trial %2d: hit=%d contact=%d min_dist=%.3fm v_final_x=%.1f\n",
                trial+1, ball_hit, min_ball_dist<0.15, min_ball_dist,
                d->qvel[ball_vadr]);
  }
  const double rate = 100.0 * hits / kTrials;
  const double crate = 100.0 * contacts / kTrials;
  std::printf("\n=== M2 拦截验收: %d/%d 命中(%.0f%%), 接近率 %.0f%% ===\n",
              hits, kTrials, rate, crate);
  std::printf("%s (门槛 ≥80%%)\n", rate >= 80.0 ? "[PASS]" : "[FAIL]");
  return rate >= 80.0 ? 0 : 1;
}
