// 击球演示:击球意图参数 -> 底盘就位 + 引拍 + 挥拍触球(物理接触),
// 状态 60Hz 写 /tmp/sckd_demo_state.bin 供 GUI(python passive viewer)同步。
// 用法: ./build/demo_hit [hit_x hit_y hit_z dt_ms]  (默认 0.90 0.35 1.35 3200)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <mujoco/mujoco.h>

#include "control/chassis_servo.h"
#include "hal/sim_backend.h"

using namespace sckd;

namespace {
constexpr double kRho = 1.225, kCd = 0.55, kBallR = 0.033, kBallM = 0.057;
constexpr double kDragK = 0.5 * kRho * kCd * 3.14159265 * kBallR * kBallR;  // F=kDragK|v|v
}  // namespace

int main(int argc, char** argv) {
  const double hx = argc > 1 ? atof(argv[1]) : 0.90;    // 击球点(世界系,机器人面 +x)
  const double hy = argc > 2 ? atof(argv[2]) : 0.35;
  const double hz = argc > 3 ? atof(argv[3]) : 1.35;
  const double t_hit = argc > 4 ? atof(argv[4]) / 1000.0 : 3.2;

  SimBackend be("models/generated/x4_tennis_demo.mjcf", "models/config/motors.json",
                "models/config/x4_tennis.json", "models/config/chassis_m1.json");
  if (!be.ok() || !be.reset(0)) {
    std::printf("[FAIL] init\n");
    return 1;
  }
  mjModel* m = const_cast<mjModel*>(be.model());
  mjData* d = be.data_mut();
  const int site_id = mj_name2id(m, mjOBJ_SITE, "racket_center");
  const int ball_bid = mj_name2id(m, mjOBJ_BODY, "ball");
  const int ball_qadr = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int ball_vadr = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "ball_free")];
  const int base_qadr = 0;  // root freejoint 在最前
  auto qadr = [&](int axis) {
    return m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, kAxisJointName[axis])];
  };
  auto vadr = [&](int axis) {
    return m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, kAxisJointName[axis])];
  };
  ChassisServo servo(be.chassis_config(), &be);
  be.enable(-1, true);

  // ---- 球:自击球点反向积分生成来球(t* 时刻:位置=P_hit,来球速度) ----
  double vin[3] = {-7.5, 0.0, -1.6};  // 来球速度(自 +x 方向来)
  double bp[3] = {hx, hy, hz}, bv[3] = {vin[0], vin[1], vin[2]};
  for (double t = t_hit; t > 0.0; t -= 0.001) {  // 显式欧拉反向
    const double v = std::sqrt(bv[0] * bv[0] + bv[1] * bv[1] + bv[2] * bv[2]);
    for (int i = 0; i < 3; ++i)
      bp[i] -= bv[i] * 0.001,
      bv[i] -= 0.001 * (-9.81 * (i == 2) - kDragK * v * bv[i] / kBallM);
  }
  d->qpos[ball_qadr + 0] = bp[0]; d->qpos[ball_qadr + 1] = bp[1]; d->qpos[ball_qadr + 2] = bp[2];
  d->qpos[ball_qadr + 3] = 1; d->qpos[ball_qadr + 4] = d->qpos[ball_qadr + 5] = d->qpos[ball_qadr + 6] = 0;
  d->qvel[ball_vadr + 0] = bv[0]; d->qvel[ball_vadr + 1] = bv[1]; d->qvel[ball_vadr + 2] = bv[2];
  mj_forward(m, d);

  // ---- IK:拍头到达 P_hit(基座置于预击球位姿后求解) ----
  const double base_y = hy;  // 预击球基座位姿(y 对齐击球点,使击球点在车体正前)
  double q_hit[5] = {0, -1.05, 0.55, 0, 0.9};
  {
    // 临时把基座放到目标位姿
    const double save_q = d->qpos[1];
    d->qpos[1] = base_y;
    for (int j = 0; j < 5; ++j) d->qpos[qadr(AxisArm(j))] = q_hit[j];
    mj_forward(m, d);
    double best_err = 1e9, best_q[5];
    for (int it = 0; it < 600; ++it) {
      double err[3];
      std::vector<double> jacp(3 * m->nv), jacr(3 * m->nv);  // 3 x nv
      mjtNum x[3] = {0, 0, 0};
      mj_kinematics(m, d);
      mju_rotVecQuat(x, m->site_pos + site_id * 3, d->xquat + (m->site_bodyid[site_id] * 4));
      const double tgt[3] = {hx, hy, hz};
      for (int i = 0; i < 3; ++i)
        err[i] = tgt[i] - ((d->xpos + m->site_bodyid[site_id] * 3)[i] + x[i]);
      mj_jacSite(m, d, jacp.data(), jacr.data(), site_id);
      // DLS: dq = J^T (J J^T + λI)^-1 err,只取臂 5 列(车体系:基座固定)
      // 简化:全 6 DOF jacobian 的臂列
      double J[3][5] = {{0}};
      for (int c = 0; c < 5; ++c) {
        const int dof = vadr(AxisArm(c));   // 臂关节 dof 地址(按名查,不假设顺序)
        J[0][c] = jacp[0 * m->nv + dof];    // jacp: 3 x nv 行优先
        J[1][c] = jacp[1 * m->nv + dof];
        J[2][c] = jacp[2 * m->nv + dof];
      }
      double A[3][3] = {{0}}, lam = 0.05;
      for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k)
          for (int c = 0; c < 5; ++c) A[i][k] += J[i][c] * J[k][c];
        A[i][i] += lam * lam;
      }
      // 3x3 求逆(Cramer)
      double det = A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                 - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
                 + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
      if (std::fabs(det) < 1e-12) break;
      double Ai[3][3] = {{
          (A[1][1]*A[2][2]-A[1][2]*A[2][1])/det,
          -(A[0][1]*A[2][2]-A[0][2]*A[2][1])/det,
          (A[0][1]*A[1][2]-A[0][2]*A[1][1])/det},
          {-(A[1][0]*A[2][2]-A[1][2]*A[2][0])/det,
           (A[0][0]*A[2][2]-A[0][2]*A[2][0])/det,
           -(A[0][0]*A[1][2]-A[0][2]*A[1][0])/det},
          {(A[1][0]*A[2][1]-A[1][1]*A[2][0])/det,
           -(A[0][0]*A[2][1]-A[0][1]*A[2][0])/det,
           (A[0][0]*A[1][1]-A[0][1]*A[1][0])/det}};
      double dq[5] = {0};
      for (int c = 0; c < 5; ++c)
        for (int i = 0; i < 3; ++i)
          for (int k = 0; k < 3; ++k) dq[c] += J[k][c] * Ai[k][i] * err[i];
      for (int j = 0; j < 5; ++j) dq[j] = std::clamp(dq[j], -0.05, 0.05);  // 步长钳制
      for (int j = 0; j < 5; ++j) q_hit[j] += dq[j];
      q_hit[1] = std::clamp(q_hit[1], -3.14, 0.5);   // J2/J3 限位
      q_hit[2] = std::clamp(q_hit[2], -3.14, 0.5);
      const double e2 = std::sqrt(err[0]*err[0] + err[1]*err[1] + err[2]*err[2]);
      if (e2 < best_err) { best_err = e2; for (int j=0;j<5;++j) best_q[j]=q_hit[j]; }
      if (best_err < 0.005) break;
      for (int j = 0; j < 5; ++j) d->qpos[qadr(AxisArm(j))] = q_hit[j];
      mj_forward(m, d);
      if (it % 100 == 0)
        std::printf("ik it%d err=%.4f q=[%.2f %.2f %.2f %.2f %.2f]\n", it,
                    std::sqrt(err[0]*err[0]+err[1]*err[1]+err[2]*err[2]),
                    q_hit[0], q_hit[1], q_hit[2], q_hit[3], q_hit[4]);
    }
    for (int j = 0; j < 5; ++j) q_hit[j] = best_q[j];
    d->qpos[1] = save_q;
    mj_forward(m, d);
    mjtNum xs[3] = {0,0,0};
    mj_kinematics(m, d);
    mju_rotVecQuat(xs, m->site_pos + site_id*3, d->xquat + m->site_bodyid[site_id]*4);
    const double* sp = d->xpos + m->site_bodyid[site_id]*3;
    std::printf("ik done: err=%.4f site=(%.3f,%.3f,%.3f) target=(%.3f,%.3f,%.3f)\n",
                best_err, sp[0]+xs[0], sp[1]+xs[1], sp[2]+xs[2], hx, hy, hz);
  }
  double q_back[5];
  for (int j = 0; j < 5; ++j) q_back[j] = q_hit[j];
  q_back[0] -= 0.55; q_back[1] += 0.30; q_back[2] += 0.12;  // 后引拍

  FILE* sh = fopen("/tmp/sckd_demo_state.bin", "wb");
  const auto wall0 = std::chrono::steady_clock::now();
  double arm_q[5] = {0, 0, 0, 0, 0}, arm_dq[5] = {0, 0, 0, 0, 0};
  const int kSteps = 12000;  // 12 s
  int last_phase = -1;
  for (int t = 1; t <= kSteps; ++t) {
    const double ts = t * 0.001;
    // ---- 底盘:位置伺服(预击球位姿) ----
    const BaseState bs = be.base_state();
    double ex = 0.0 - bs.x, ey = base_y - bs.y;
    double vx = std::clamp(2.5 * ex, -1.2, 1.2), vy = std::clamp(2.5 * ey, -1.2, 1.2);
    if (ts > t_hit) { vx = 0; vy = 0; }
    servo.set_twist(Twist{vx, vy, 0});
    servo.update(ts);
    // ---- 臂相位 ----
    int phase;
    double kp[5] = {120, 120, 120, 120, 60}, kd[5] = {8, 8, 8, 8, 1.5};
    if (ts < t_hit - 0.35) {           // 就位+引拍
      phase = 0;
      for (int j = 0; j < 5; ++j) arm_q[j] = q_back[j], arm_dq[j] = 0;
    } else if (ts < t_hit) {           // 挥拍(五次多项式,末端速度>0)
      phase = 1;
      const double s = (ts - (t_hit - 0.35)) / 0.35;
      const double p = 10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s;
      const double dp = (30*s*s - 60*s*s*s + 30*s*s*s*s) / 0.35;
      for (int j = 0; j < 5; ++j)
        arm_q[j] = q_back[j] + (q_hit[j] - q_back[j]) * p,
        arm_dq[j] = (q_hit[j] - q_back[j]) * dp;
      for (int j = 0; j < 5; ++j) kp[j] = 200, kd[j] = (j == 4 ? 1.5 : 10);
    } else if (ts < t_hit + 0.25) {    // 随挥
      phase = 2;
      for (int j = 0; j < 5; ++j)
        arm_q[j] = q_hit[j] + 0.10 * (q_hit[j] - q_back[j]), arm_dq[j] = 0;
    } else {                            // 回位
      phase = 3;
      for (int j = 0; j < 5; ++j) arm_q[j] *= 0.9995, arm_dq[j] = 0;
    }
    if (phase != last_phase) {
      std::printf("t=%.2f phase=%d\n", ts, phase);
      last_phase = phase;
    }
    ArmHybridCmd ac;
    double ff[5];
    if (be.arm_gravity(arm_q, ff))
      for (int j = 0; j < 5; ++j) ac.tau_ff[j] = ff[j];
    for (int j = 0; j < 5; ++j) ac.q[j] = arm_q[j], ac.dq[j] = arm_dq[j], ac.kp[j] = kp[j], ac.kd[j] = kd[j];
    be.set_arm_hybrid(ac);
    be.run(1);
    // ---- 球气动(T1 阻力) ----
    {
      const double* bv2 = d->qvel + ball_vadr;
      const double v = std::sqrt(bv2[0]*bv2[0] + bv2[1]*bv2[1] + bv2[2]*bv2[2]);
      const double* bp2 = d->qpos + ball_qadr;
      (void)bp2;
      for (int i = 0; i < 3; ++i)
        d->xfrc_applied[ball_bid * 6 + i] = -kDragK * v * bv2[i];
      d->xfrc_applied[ball_bid * 6 + 3] = d->xfrc_applied[ball_bid * 6 + 4] = d->xfrc_applied[ball_bid * 6 + 5] = 0;
    }
    // ---- 状态同步(60Hz) + 墙钟节拍 ----
    if (t % 17 == 0 && sh) {
      uint64_t seq = t; int32_t nq = m->nq, nv = m->nv;
      fseek(sh, 0, SEEK_SET);
      fwrite(&seq, 8, 1, sh); fwrite(&nq, 4, 1, sh); fwrite(&nv, 4, 1, sh);
      fwrite(d->qpos, 8, m->nq, sh); fwrite(d->qvel, 8, m->nv, sh);
      fflush(sh);
    }
    while (true) {  // 实时节拍(GUI 观感)
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
      if (wall >= ts - 0.0015) break;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
  }
  if (sh) fclose(sh);
  const BaseState bf = be.base_state();
  const double* bpos = d->qpos + ball_qadr;
  std::printf("done: ball=(%.2f,%.2f,%.2f) base=(%.2f,%.2f)\n",
              bpos[0], bpos[1], bpos[2], bf.x, bf.y);
  return 0;
}
