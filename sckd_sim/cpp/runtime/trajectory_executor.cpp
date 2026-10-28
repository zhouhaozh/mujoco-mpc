#include "runtime/intercept_modules.h"
#include "runtime/controller_adapters.h"
#include "runtime/plan_supervisor.h"
#include "runtime/synchronous_schedule.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
namespace sckd::runtime {
// WmPlan 绝对时刻采样(节点间线性插值)。t 超出终端节点后按终端速度恒速
// 随挥外推(≤0.2s,之后停住)——峰值拍速在触球带内,不在终端节点后急刹
void SampleWm(const WmPlan& p, double t, double q[5], double dq[5],
                     double ddq[5], double base[2], double vbase[2]) {
  constexpr int kN = WholebodyMpc::kN;
  const double tau = (t - p.t0) / p.h;
  const double kc = std::clamp(tau, 0.0, (double)kN);
  const int k = std::min((int)kc, kN - 1);
  const double a = kc - k;
  base[0] = p.rx[k] + a * (p.rx[k + 1] - p.rx[k]);
  base[1] = p.ry[k] + a * (p.ry[k + 1] - p.ry[k]);
  vbase[0] = p.rvx[k] + a * (p.rvx[k + 1] - p.rvx[k]);
  vbase[1] = p.rvy[k] + a * (p.rvy[k + 1] - p.rvy[k]);
  for (int j = 0; j < 5; ++j) {
    q[j] = p.rq[j * (kN + 1) + k] +
           a * (p.rq[j * (kN + 1) + k + 1] - p.rq[j * (kN + 1) + k]);
    dq[j] = p.rdq[j * (kN + 1) + k] +
            a * (p.rdq[j * (kN + 1) + k + 1] - p.rdq[j * (kN + 1) + k]);
    ddq[j] = p.rddq[j * kN + k];
  }
  if (tau > kN) {   // 随挥外推:终端速度恒速(接触窗内拍速不塌;0.15s 上限)
    const bool capped = (tau - kN) * p.h > 0.15;
    const double ext = std::min((tau - kN) * p.h, 0.15);
    base[0] += vbase[0] * ext;
    base[1] += vbase[1] * ext;
    for (int j = 0; j < 5; ++j) {
      q[j] += dq[j] * ext;
      ddq[j] = 0.0;
      if (capped) dq[j] = 0.0;
    }
    if (capped) { vbase[0] = 0.0; vbase[1] = 0.0; }
  }
}


ExecutionFrame ExecuteTrajectory(ExecutionContext c, int t, double ts) {
  ExecutionFrame out;
  auto& be = c.be;
  const auto& arm_servo = be.motor_config().arm_servo;
  auto& servo = c.servo;
  auto& plan = c.plan;
  auto& wb_plan = c.wb_plan;
  auto& aswing = c.aswing;
  auto& swing = c.swing;
  auto& mpc = c.mpc;
  auto& mpc_lim = c.mpc_lim;
  auto& mpc_sched = c.mpc_sched;
  auto& mpc_ref = c.mpc_ref;
  auto& cfg = c.cfg;
  auto& identified_gains_config = c.identified_gains_config;
  auto& n6_pos = c.n6_pos;
  auto& n6_vel = c.n6_vel;
  auto& m = c.m;
  auto& variant = c.variant;
  auto& fly = c.fly;
  auto& use_mpc = c.use_mpc;
  auto& n6_mode = c.n6_mode;
  auto& wb_mode = c.wb_mode;
  auto& use_qp = c.use_qp;
  auto& coherent = c.coherent;
  auto& inverse_ff = c.inverse_ff;
  auto& identified_gains = c.identified_gains;
  auto& planned = c.planned;
  auto& ref_built = c.ref_built;
  auto& wb_have = c.wb_have;
  auto& wb_dead = c.wb_dead;
  auto& n6_have = c.n6_have;
  auto& n6_dead = c.n6_dead;
  auto& have_dq_command = c.have_dq_command;
  auto& n6_ms = c.n6_ms;
  auto& mpc_solves = c.mpc_solves;
  auto& mpc_fails = c.mpc_fails;
  auto& t_plan0 = c.t_plan0;
  auto& n6_t0 = c.n6_t0;
  auto& n6_tau = c.n6_tau;
  auto& wb_tau = c.wb_tau;
  auto& mpc_ms_max = c.mpc_ms_max;
  auto& q_cur = c.q_cur;
  auto& mpc_cs2 = c.mpc_cs2;
  auto& previous_dq_command = c.previous_dq_command;
  // wbmpc 执行采样(绝对时刻;一次采样底盘/臂两用)。τ̂ 闭环滞后补偿
  // (ArmSwing 同法):实测对指令的最小二乘时间移位 + 固定前移 20ms,
  // 播放时钟前移 —— 拍在 t_hit 实际抵达(伺服滞后下高速挥拍不补会
  // 面内 miss 10-20cm)
  auto& wb_q=out.wb_q; auto& wb_dq=out.wb_dq; auto& wb_ddq=out.wb_ddq;
  auto& wb_base=out.wb_base; auto& wb_vbase=out.wb_vbase;
  const bool wb_exec = out.wb_exec = (wb_mode && wb_have && !wb_dead);
  if (wb_exec) {
    double qn[5], dqn[5], ddn[5], bn[2], vn[2];
    SampleWm(wb_plan, ts, qn, dqn, ddn, bn, vn);
    double num = 0, den = 0;
    for (int j = 0; j < 5; ++j) {
      num += (q_cur[j] - qn[j]) * dqn[j];
      den += dqn[j] * dqn[j];
    }
    // τ̂ 符号约定与 arm_swing.cpp 同源:纯滞后 τ 下 e≈−dq·τ → τ̂=−num/den
    // ≥0(只前移不延迟;首版抄成正号,负 lead+外推=拍提前冲过截面球追拍,
    // min_dist 集中在 t_hit 后 20-30ms,2/10 回归定位,2026-10-09)
    const double tau_hat = (den > 1e-6) ? std::clamp(-num / den, 0.0, 0.10) : wb_tau;
    wb_tau = 0.7 * wb_tau + 0.3 * tau_hat;
    SampleWm(wb_plan, ts + wb_tau, wb_q, wb_dq, wb_ddq, wb_base, wb_vbase);
  }
  // nmpc6 执行采样:底盘=Plans 速度直下(真机语义;ChassisServo 内部
  // 8 m/s² 斜坡=旧机底盘同款,且每拍新 Plans 从实测状态起锚 → 速度
  // 跟踪近无差,无需 P 位置环)。臂=位置+速度参考 hybrid(Plans 同时
  // 携带两者,协议同构;纯速度模式 kp=0 在 sim 电机模型通路上未标定,
  // 实测失稳 dq 冲 24 rad/s,负结果留档)。τ̂ 补偿 sim 伺服与真机驱动
  // 器的带宽差(wbmpc 同法,机器本身驱动器快不需要)
  auto& n6_q=out.n6_q; auto& n6_dq=out.n6_dq;
  auto& n6_vbase=out.n6_vbase; auto& n6_base=out.n6_base;
  const bool n6_exec = out.n6_exec = (n6_mode && n6_have && !n6_dead);
  if (n6_exec) {
    const int W = n6_ms + 1;
    auto sample6 = [&](double t_abs, double q_out[5], double dq_out[5]) {
      const double tau = t_abs - n6_t0;
      const double over = tau - n6_ms * 0.001;
      const int idx = std::clamp((int)std::lround(tau * 1000.0), 0, n6_ms);
      for (int j = 0; j < 5; ++j) {
        q_out[j] = n6_pos[(2 + j) * W + idx];
        dq_out[j] = n6_vel[(2 + j) * W + idx];
      }
      if (over > 0) {   // 终端后恒速随挥 ≤0.15s 再停(SampleWm 同法)
        const double ext = std::min(over, 0.15);
        for (int j = 0; j < 5; ++j) q_out[j] += dq_out[j] * ext;
        if (over > 0.15)
          for (int j = 0; j < 5; ++j) dq_out[j] = 0.0;
      }
    };
    double qn6[5], dqn6[5];
    sample6(ts, qn6, dqn6);
    double num = 0, den = 0;
    for (int j = 0; j < 5; ++j) {
      num += (q_cur[j] - qn6[j]) * dqn6[j];
      den += dqn6[j] * dqn6[j];
    }
    const double tau_hat6 = (den > 1e-6)
        ? std::clamp(-num / den, 0.0, 0.10) : n6_tau;
    n6_tau = 0.7 * n6_tau + 0.3 * tau_hat6;
    if(coherent || arm_servo.enabled)n6_tau=variant=="servo_mpc"?0:be.motor_config().command_delay_ms*.001;
    sample6(ts + n6_tau, n6_q, n6_dq);
    {   // 底盘速度:ts 时刻(无 τ̂ —— 伺服斜坡与 Plans 斜坡同 8 m/s²)
      const double tau6 = ts - n6_t0;
      const int idx = std::clamp((int)std::lround(tau6 * 1000.0), 0, n6_ms);
      n6_base[0]=n6_pos[idx];n6_base[1]=n6_pos[W+idx];
      n6_vbase[0] = n6_vel[idx];
      n6_vbase[1] = n6_vel[W + idx];
      if (tau6 - n6_ms * 0.001 > 0.15)
        n6_vbase[0] = n6_vbase[1] = 0.0;
    }
  }
  // 底盘就位:MPC(50Hz 重解,到位目标 = 锁止前 0.15s 到达;非解算拍保持上一拍
  // twist)或 P 控(argv[6]="mpc" 切换;P 控解算失败时兜底)
  if (planned && plan.feasible) {
    const BaseState bs = be.base_state();
    bool mpc_applied = false;
    if (n6_exec) {
      // nmpc6:Plans 速度直下(真机语义,无 P 位置环 —— 见上执行采样注)
      const double c = std::cos(bs.yaw), sn = std::sin(bs.yaw);
      double vx=n6_vbase[0],vy=n6_vbase[1];
      if(variant=="base_feedback") {vx+=2.5*(n6_base[0]-bs.x);vy+=2.5*(n6_base[1]-bs.y);double speed=std::hypot(vx,vy);if(speed>2.1){vx*=2.1/speed;vy*=2.1/speed;}}
      servo.set_twist(Twist{c*vx+sn*vy,-sn*vx+c*vy,0});
    } else if (wb_exec) {
      // wbmpc:P+ff 跟踪优化底盘段(pfly 同增益同口径,参考=WmPlan 轨迹)
      const double c = std::cos(bs.yaw), sn = std::sin(bs.yaw);
      const double cx = wb_vbase[0] + 2.5 * (wb_base[0] - bs.x);
      const double cy = wb_vbase[1] + 2.5 * (wb_base[1] - bs.y);
      servo.set_twist(Twist{std::clamp(c * cx + sn * cy, -1.5, 1.5),
                            std::clamp(-sn * cx + c * cy, -1.5, 1.5), 0});
      if (cfg.wb_chassis_trace && (t % 100 == 0))
        std::printf("  [wbC] t=%.2f base=(%.2f,%.2f) ref=(%.2f,%.2f) vref=(%.2f,%.2f) tau=%.3f planT0=%.2f h=%.3f term=(%.2f,%.2f)\n",
                    ts, bs.x, bs.y, wb_base[0], wb_base[1], wb_vbase[0],
                    wb_vbase[1], wb_tau, wb_plan.t0, wb_plan.h,
                    wb_plan.term_base[0], wb_plan.term_base[1]);
    } else if (fly && !use_mpc) {   // pfly 与 ocpfly 同路(BaseRefGen 绝对时刻
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
    } else if (use_mpc && SynchronousSchedule::Replan(t)) {
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
  StageProfile::Measure("chassis_servo", [&]{servo.update(ts);});
  // 臂:wbmpc 直接跟踪 OCP 解(挥拍层旁路)/ 挥拍(绝对时序)/ 保持实测构型
  auto& ac=out.arm;
  for (int j = 0; j < 5; ++j) { ac.kp[j]=60; ac.kd[j]=(j==4?1.5:6.0); }
  SwingCmd sc;
  if (wb_exec) {
    for (int j = 0; j < 5; ++j) {
      ac.q[j] = wb_q[j]; ac.dq[j] = wb_dq[j];
      ac.kp[j] = (j == 0) ? 250 : 120;   // ArmSwing 播放段增益同源
      ac.kd[j] = (j == 0) ? 3.0 : (j == 4 ? 1.5 : 8.0);   // (arm_swing.cpp)
    }
  } else if (n6_exec) {
    for (int j = 0; j < 5; ++j) {
      ac.q[j] = n6_q[j]; ac.dq[j] = n6_dq[j];
      ac.kp[j] = (j == 0) ? 250 : 120;   // wbmpc 执行段增益同源(已标定)
      ac.kd[j] = (j == 0) ? 3.0 : (j == 4 ? 1.5 : 8.0);
    }
  } else if (use_qp ? StageProfile::Measure("arm_sample", [&]{return aswing.Update(ts, q_cur, sc);}) : (swing.active() && swing.Update(ts, sc))) {
    for (int j = 0; j < 5; ++j) {
      ac.q[j]=sc.q[j]; ac.dq[j]=sc.dq[j]; ac.kp[j]=sc.kp[j]; ac.kd[j]=sc.kd[j];
    }
  } else {
    // 待机位姿(实机 sckd-robot home,J2 抬臂 35°;用户定向 2026-10-09:
    // 待机大臂应稍微举起,hold_arm 零目标=水平是错的)。挥拍层模式
    // (pfly/B′)首规划后 0.25s 五次过渡到就位构型(m2_demo 同款,直接以
    // 抬臂待机作 IK 种子会解支漂移掉到 1-2/10,2026-10-06 实测);优化器
    // 接管型(nmpc6/wbmpc)**不做人工过渡**——过渡期 5-6 rad/s 移动起点
    // 会让 NMPC 重解连续拒收(实测 65%→10%),静持待机位由 OCP 自己规划
    // 待机→击球全程(与实机从 home 起解一致)
    // kStandby(2026-10-10 定稿,与 viewer/m2_demo 同源):sim 零位构型=
    // 大臂水平/小臂竖直/拍水平后折,实机待机不可代数映射(电机零位≠模型
    // 零位,ini joint_offset;两模型连杆构型不同),按用户三几何规约直解:
    // J2=−35° 抬大臂,J3=−35°(净转角 0 小臂近竖直),J4=0 拍水平回折;
    // FK 验证 大臂仰43.6°/小臂78.8°/拍仰0°,拍心(−0.06,0,1.26)
    static constexpr double kStandby[5] = {0.0, -0.6109, -0.6109, 0.0, 0.0};
    static constexpr double kReady[5] = {0, -1.0, 0.5, 0, 0.8};
    const bool swing_mode = !n6_mode && !wb_mode;
    if (arm_servo.enabled) {
      for (int j = 0; j < 5; ++j) ac.q[j] = arm_servo.standby_q[j];
    } else if (t_plan0 >= 0.0 && swing_mode) {
      const double transition = cfg.time_windows.pfly_ready_transition_s;
      const double s = transition > 0 ? std::clamp((ts - t_plan0) / transition, 0.0, 1.0) : 1.0;
      const double p = 10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s;
      for (int j = 0; j < 5; ++j)
        ac.q[j] = kStandby[j] + (kReady[j] - kStandby[j]) * p;
    } else {
      for (int j = 0; j < 5; ++j) ac.q[j] = kStandby[j];
    }
  }
  double ff[5];
  if (StageProfile::Measure("arm_gravity", [&]{return be.arm_gravity(ac.q, ff);})) for (int j=0;j<5;++j) ac.tau_ff[j]=ff[j];
  // 惯性前馈:τ_ff += Î·q̈(对角估计;挥拍段跟踪误差 I·a/kp 的直接抵消)
  static constexpr double kArmInertia[5] = {1.2, 2.0, 0.8, 0.3, 0.05};
  for (int j = 0; !arm_servo.enabled && j < 5; ++j)
    ac.tau_ff[j] += kArmInertia[j] *
                    (wb_exec ? wb_ddq[j] : (n6_exec ? 0.0 : sc.ddp[j]));
  if(inverse_ff&&!wb_mode&&!arm_servo.enabled) {
    double accel[5],inv[5],qa[5],dqa[5];
    for(int j=0;j<5;++j){qa[j]=ac.q[j];dqa[j]=ac.dq[j];accel[j]=n6_exec?(have_dq_command?std::clamp((ac.dq[j]-previous_dq_command[j])/.001,-(j<2?30.0:j==2?180.0:j==3?400.0:200.0),(j<2?30.0:j==2?180.0:j==3?400.0:200.0)):0):sc.ddp[j];}
    if(be.arm_inverse(qa,dqa,accel,inv))for(int j=0;j<5;++j)ac.tau_ff[j]=inv[j];
  }
  if(identified_gains) {
    // 候选受原协议/限矩约束；具体辨识参数由独立文件加载，未设置则拒绝运行。
    const auto& gains=identified_gains_config;
    for(int j=0;j<5;++j){ac.kp[j]=gains["kp"][j];ac.kd[j]=gains["kd"][j];}
  }
  if (arm_servo.enabled) {
    for (int j = 0; j < 5; ++j) {
      ac.kp[j] = arm_servo.kp[j];
      ac.kd[j] = arm_servo.kd[j];
    }
  }
  for(int j=0;j<5;++j)previous_dq_command[j]=ac.dq[j];have_dq_command=true;
  StageProfile::Measure("actuator_write", [&]{be.set_arm_hybrid(ac);});
  if(wb_exec) StageProfile::Get().Count("wbmpc_execution_ticks");
  if(n6_exec) StageProfile::Get().Count("nmpc6_execution_ticks");

  return out;

}
}
