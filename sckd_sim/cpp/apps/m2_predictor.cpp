// M2 验证①:预测器前视 RMSE(0.3s 真前视,闭环关键窗口)+ A/B 对照(逐轴 CA 基线)。
// 场景:参数化发球(12 m/s / 仰角 25° / 2000 rpm)落入机器人前方,含 1~2 次反弹。
// 2026-10-06 改版:真值对照由"零前视近似"改为延迟队列到期对拍(每 5ms 采样一次
// 前视预测,仿真推进到 t_target 时与真值比,时间偏差 ≤1ms);
// 增加初始化后 0.3s"早期收敛"分段(闭环规划恰恰发生在这个窗口)。
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "estimation/ball_predictor.h"
#include "hal/sim_backend.h"
#include "sim/analytic_contact.h"
#include "sim/ball_aero.h"
#include "sim/ball_server.h"

using namespace sckd;

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_demo.mjcf";
  SimBackend be(scene, "models/config/motors.json", "models/config/x4_tennis.json",
                "models/config/chassis_m1.json");
  if (!be.ok() || !be.reset(0)) {
    std::printf("[FAIL] backend init\n");
    return 1;
  }
  const mjModel* m = be.model();
  mjData* d = be.data_mut();
  const int ball_qadr = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_vadr = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_bid = mj_name2id(m, mjOBJ_BODY, "ball");

  // 发球
  BallAeroParams aero;
  BounceParams bp;
  ServerParams sp;
  sp.speed = 12.0; sp.elevation_deg = 25.0; sp.azimuth_deg = 200.0;
  sp.spin_rpm = 2000; sp.spawn[0] = 7.0; sp.spawn[1] = -1.5; sp.spawn[2] = 2.5;
  BallServer server(sp);
  double q[7], v[3], w[3];
  server.Spawn(q, v, w);
  for (int i = 0; i < 7; ++i) d->qpos[ball_qadr + i] = q[i];
  d->qvel[ball_vadr] = v[0]; d->qvel[ball_vadr + 1] = v[1]; d->qvel[ball_vadr + 2] = v[2];
  d->qvel[ball_vadr + 3] = w[0]; d->qvel[ball_vadr + 4] = w[1]; d->qvel[ball_vadr + 5] = w[2];
  mj_forward(m, d);

  BallPredictor ekf(aero, 0.030, bp);   // 内置 40ms 差分预热初始化
  CaAxisFilter ca(0.030, 4.0e6);

  struct Pending { double t_pred, t_target; double pe[3], pc[3]; };
  std::vector<Pending> pend;
  // 分段:[0]全程 [1]干净反弹前(预测与目标均在反弹前) [2]跨反弹(t_pred前、t_target后)
  //       [3]反弹后0.5s [4]早期(初始化后0.3s,闭环规划窗口)
  double e2[5] = {0,0,0,0,0}, c2[5] = {0,0,0,0,0};
  int n[5] = {0,0,0,0,0};
  double t_ekf_init = -1;
  double last_z = d->qpos[ball_qadr + 2];
  bool bounced = false;
  double t_bounce = -1;
  const int kSteps = 4000;

  for (int t = 1; t <= kSteps; ++t) {
    // 气动外力(真值侧;无旋简化,与 T1 预测器模型一致)。
    // 注意单位:xfrc_applied 是力(N)→ 系数 0.5ρCdA,勿再除质量(旧版写成加速度,
    // MuJoCo 再 /m 使阻力大 17.5×,滤波速度稳态偏差 ~1 m/s —— 2026-10-06 定位)
    const double* bv = d->qvel + ball_vadr;
    const double vm = std::sqrt(bv[0]*bv[0] + bv[1]*bv[1] + bv[2]*bv[2]);
    const double area = M_PI * aero.radius * aero.radius;
    const double kk = 0.5 * aero.rho * area * aero.Cd;   // F = -kk·|v|·v [N]
    for (int i = 0; i < 3; ++i)
      d->xfrc_applied[ball_bid*6+i] = -kk*vm*bv[i];
    d->xfrc_applied[ball_bid*6+3] = d->xfrc_applied[ball_bid*6+4] = d->xfrc_applied[ball_bid*6+5] = 0;

    // 地面解析冲量(§3.5 地面支,2026-10-08):真值侧反弹 = 已知模型
    // (e_z=0.73/μ=0.20),与预测器 BounceParams 同参数——跨反弹段残差从此
    // 只含估计误差,不再有模型失配(旧:预测器 0.73 vs 求解器实测 0.09)
    static const GroundImpulseParams gnd_pp;
    ApplyGroundImpulse(d, ball_qadr, ball_vadr, gnd_pp);

    be.run(1);
    const double ts = t * 0.001;
    // 反弹检测(真值 z)
    const double z = d->qpos[ball_qadr + 2];
    if (!bounced && z < 0.1 && last_z >= 0.1 && d->qvel[ball_vadr+2] < 0) {
      bounced = true; t_bounce = ts;
    }
    last_z = z;

    // 观测 -> 预测器
    BallObs obs;
    if (be.get_ball_obs(&obs)) {
      ekf.Update(obs);   // 预热差分自动初始化
      ca.Update(obs);
      if (t_ekf_init < 0 && ekf.initialized()) t_ekf_init = obs.t;
    }
    // 前视预测入队(200Hz 采样)
    if (t % 5 == 0 && ekf.initialized() && ca.initialized()) {
      double pe[3], pc[3];
      if (ekf.Predict(ts + 0.3, pe) && ca.Predict(ts + 0.3, pc)) {
        Pending p; p.t_pred = ts; p.t_target = ts + 0.3;
        for (int i = 0; i < 3; ++i) { p.pe[i] = pe[i]; p.pc[i] = pc[i]; }
        pend.push_back(p);
      }
    }
    // 到期对拍(时间偏差 ≤1ms)
    while (!pend.empty() && pend.front().t_target <= ts) {
      const Pending& p = pend.front();
      const double te[3] = {d->qpos[ball_qadr], d->qpos[ball_qadr+1], z};
      const double de = std::hypot(p.pe[0]-te[0], p.pe[1]-te[1], p.pe[2]-te[2]);
      const double dc = std::hypot(p.pc[0]-te[0], p.pc[1]-te[1], p.pc[2]-te[2]);
      e2[0] += de*de; c2[0] += dc*dc; ++n[0];
      if (t_bounce < 0 || p.t_target < t_bounce) { e2[1] += de*de; c2[1] += dc*dc; ++n[1]; }
      else if (p.t_pred < t_bounce)              { e2[2] += de*de; c2[2] += dc*dc; ++n[2]; }
      else if (ts - t_bounce < 0.5)              { e2[3] += de*de; c2[3] += dc*dc; ++n[3]; }
      if (t_ekf_init >= 0 && p.t_pred - t_ekf_init < 0.3) { e2[4] += de*de; c2[4] += dc*dc; ++n[4]; }
      pend.erase(pend.begin());
    }
    if (z < 0.03 && ts > 3.0) break;  // 球停滚,提前结束
  }

  std::printf("=== M2 验证①:预测器 0.3s 前视 RMSE(m;门槛:干净反弹前<0.10 且 早期<0.15)==\n");
  const char* seg[5] = {"全程", "干净反弹前", "跨反弹", "反弹后0.5s", "早期0.3s(闭环窗口)"};
  for (int i = 0; i < 5; ++i) {
    if (n[i] == 0) { std::printf("  %-18s: 无数据\n", seg[i]); continue; }
    const double re = std::sqrt(e2[i]/n[i]);
    const double rc = std::sqrt(c2[i]/n[i]);
    const double imp = (rc > 1e-9) ? (1.0 - re/rc) * 100.0 : 0.0;
    std::printf("  %-18s: EKF=%.4f  CA=%.4f  提升 %+.1f%%  (n=%d)\n", seg[i], re, rc, imp, n[i]);
  }
  const double pre = (n[1] ? std::sqrt(e2[1]/n[1]) : 1e9);
  const double early = (n[4] ? std::sqrt(e2[4]/n[4]) : 1e9);
  std::printf("[INFO] EKF 初始化 t=%.3fs,反弹 t=%.2fs,未对拍 %zu 条;"
              "反弹 = 已知模型(真值侧 ApplyGroundImpulse 与预测器同参数,"
              "§3.5 地面支 2026-10-08;跨反弹段残差 = 估计误差)\n",
              t_ekf_init, t_bounce, pend.size());
  std::printf("%s (干净反弹前 RMSE %.4f 门槛 0.10;早期 RMSE %.4f 门槛 0.15)\n",
              (pre < 0.10 && early < 0.15) ? "[PASS]" : "[FAIL]", pre, early);
  return (pre < 0.10 && early < 0.15) ? 0 : 1;
}
