// M1b 验收:底盘 MPC(OSQP QP,50Hz)圆弧跟踪。
// 反馈用里程计估计(陀螺融合)——真值仅评估(D4);指标:横向误差/速度误差/
// 求解耗时(mean/p99/max)与失败次数(零失败硬门槛)。
#include <algorithm>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "control/chassis_mpc.h"
#include "control/chassis_servo.h"
#include "estimation/wheel_odom.h"
#include "hal/sim_backend.h"

using namespace sckd;

int main(int argc, char** argv) {
  const std::string scene = argc > 1 ? argv[1] : "models/generated/x4_tennis_scene.mjcf";
  const std::string cj = argc > 4 ? argv[4] : "models/config/chassis_m1.json";
  SimBackend be(scene, argc > 2 ? argv[2] : "models/config/motors.json",
                argc > 3 ? argv[3] : "models/config/x4_tennis.json", cj);
  if (!be.ok() || !be.reset(0)) {
    std::printf("[FAIL] backend init\n");
    return 1;
  }
  ChassisServo servo(be.chassis_config(), &be);
  WheelOdom odom(be.chassis_config());
  ChassisMpc mpc;
  ChassisMpc::Limits lim;
  be.enable(-1, true);
  be.hold_arm(60.0, 1.5);

  const double R = 1.5, V = 1.0, WZ = V / R;      // 圆弧:圆心(0,R),自(0,0)起
  const double kMpcDt = 0.02;
  const int kTotal = 30000;                        // 30 s
  double theta_ref = 0.0;                          // 参考弧参数(连续化)
  double ct_sum2 = 0, v_sum2 = 0;
  int n_meas = 0, solve_fail = 0, n_solve = 0;
  std::vector<double> solve_ms;

  double x0[6] = {0, 0, 0, 0, 0, 0};
  double wheels[kDriveWheels], steer[kSwerveCorners];
  double yaw_cont = 0.0, yaw_prev_raw = 0.0;   // 连续航向(避免 WrapPi 与连续参考失配>π)
  for (int t = 1; t <= kTotal; ++t) {
    // 里程计(250Hz 更新,反馈源)
    if (t % 4 == 0) {
      int wi = 0;
      for (int c = 0; c < kSwerveCorners; ++c) {
        steer[c] = be.joint_measured(AxisSteer(c)).q;
        wheels[wi++] = be.joint_measured(AxisWheelL(c)).dq;
        wheels[wi++] = be.joint_measured(AxisWheelR(c)).dq;
      }
      ImuSample imu;
      const bool have = be.get_imu(&imu);
      odom.update(0.004, wheels, steer, have ? &imu : nullptr, be.tick() * 1000);
      yaw_cont += WrapPi(odom.yaw() - yaw_prev_raw);  // 连续化
      yaw_prev_raw = odom.yaw();
    }
    // MPC 重解(50Hz)
    if (t % 20 == 0) {
      x0[0] = odom.x();
      x0[1] = odom.y();
      x0[2] = yaw_cont;
      x0[3] = odom.twist().vx;
      x0[4] = odom.twist().vy;
      x0[5] = odom.twist().wz;
      // 状态投影到可行集(须在 odom 赋值之后):里程计噪声/滑移可致 |wz|、|v| 估计
      // 越过状态上界,k=0 行随之不可行(OSQP 原始不可行 -> 失败/长尾求解)
      x0[5] = std::clamp(x0[5], -lim.wz_max + 0.01, lim.wz_max - 0.01);
      {
        const double vn = std::hypot(x0[3], x0[4]);
        const double vcap = lim.v_max * 0.92 - 0.01;   // 0.92=内接多边形系数
        if (vn > vcap) {
          x0[3] *= vcap / vn;
          x0[4] *= vcap / vn;
        }
      }
      // 绝对时间参考:θ(t)=lead+WZ·t(自 t=0 起算;机器人自然追及,
      // 杜绝"投影锚定->参考每拍回退->前冲超调"循环)
      (void)theta_ref;
      const double lead = 0.10 / R;
      std::array<double, ChassisMpc::kNx * (ChassisMpc::kN + 1)> ref{};
      // 参考航向基准取"与连续航向最近的分支",避免 2π 跳变
      double a0 = lead + WZ * t * 0.001;
      while (a0 - yaw_cont > M_PI) a0 -= 2 * M_PI;
      while (a0 - yaw_cont < -M_PI) a0 += 2 * M_PI;
      for (int k = 0; k <= ChassisMpc::kN; ++k) {
        const double a = a0 + WZ * k * kMpcDt;
        double* r = &ref[k * ChassisMpc::kNx];
        r[0] = R * std::sin(a);
        r[1] = R * (1 - std::cos(a));
        // k=0 航向参考钳入当前状态 ±0.4·θ_max,防初始违反航向窗导致不可行
        double psi_ref = a;
        if (k == 0) psi_ref = std::clamp(a, x0[2] - 0.2, x0[2] + 0.2);
        r[2] = psi_ref;
        r[3] = V;                                  // 车体系速度:沿切向前进(V,0)
        r[4] = 0.0;                                //(模型 vx,vy 为车体系)
        r[5] = WZ;
      }
      double cs2[2 * ChassisMpc::kN];
      for (int k = 0; k < ChassisMpc::kN; ++k)
        cs2[2 * k] = std::cos(ref[(k + 1) * ChassisMpc::kNx + 2]),
        cs2[2 * k + 1] = std::sin(ref[(k + 1) * ChassisMpc::kNx + 2]);
      double p[ChassisMpc::kNp];
      ChassisMpc::PackParams(p, x0, ref, cs2, lim);
      const auto t0 = std::chrono::steady_clock::now();
      const bool ok = mpc.Solve(p);
      const auto t1 = std::chrono::steady_clock::now();
      solve_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      ++n_solve;
      if (!ok) {
        ++solve_fail;
        std::printf("FAIL@t=%.3f x0=[%.3f %.3f %.3f %.3f %.3f %.3f] a0=%.3f psi_ref0=%.3f\n",
                    t*0.001, x0[0],x0[1],x0[2],x0[3],x0[4],x0[5],
                    lead + WZ*t*0.001, ref[2]);
      }
      double vx, vy, wz;
      mpc.NextTwist(&vx, &vy, &wz);
      servo.set_twist(Twist{vx, vy, wz});
      if (t % 4000 == 0) {
        const BaseState b = be.base_state();
        std::printf("t=%4.1f pos=(%.2f,%.2f) yaw=%.2f v=(%.2f,%.2f) wz=%.2f | cmd=(%.2f,%.2f,%.2f) | odom=(%.2f,%.2f,%.2f)\n",
                    t*0.001,b.x,b.y,b.yaw,b.vx,b.vy,b.wz,vx,vy,wz,
                    odom.twist().vx,odom.twist().vy,odom.twist().wz);
      }
    }
    servo.update(t * 0.001);
    be.run(1);
    // 指标(3 s 后,真值评估)
    if (t > 3000) {
      const BaseState b = be.base_state();
      const double ct = std::fabs(std::hypot(b.x, b.y - R) - R);
      const double ve = std::hypot(b.vx - V, b.vy);
      ct_sum2 += ct * ct;
      v_sum2 += ve * ve;
      ++n_meas;
    }
  }

  std::sort(solve_ms.begin(), solve_ms.end());
  const double p99 = solve_ms[static_cast<size_t>(0.99 * solve_ms.size()) - 1];
  const double ct_rms = std::sqrt(ct_sum2 / n_meas);
  const double v_rms = std::sqrt(v_sum2 / n_meas);
  std::printf("mpc: solves=%d fail=%d | solve_ms mean=%.3f p99=%.3f max=%.3f\n",
              n_solve, solve_fail, [&]{ double s=0; for(double v:solve_ms)s+=v; return s/solve_ms.size(); }(),
              p99, solve_ms.back());
  std::printf("track: cross_track_rms=%.3f m  v_rms=%.3f m/s\n", ct_rms, v_rms);
  const bool pass = ct_rms < 0.15 && v_rms < 0.15 && solve_fail == 0 && p99 < 5.0;
  if (pass) {
    std::printf("[PASS] m1_mpc_tracking\n");
    return 0;
  }
  std::printf("[FAIL] m1_mpc_tracking\n");
  return 1;
}
