#pragma once
// 球轨迹预测器(§6.1/§3.6):物理 EKF(T1:重力+阻力,可选残余加速度)+ 反弹事件分段;
// 附逐轴 CA 滤波(trobot 现役路线)作 A/B 基线——用数据说话,不被其绑架。
#include "core/types.h"
#include "sim/ball_aero.h"

namespace sckd {

struct BounceParams {
  double restitution_z = 0.73;   // 参数化映射的常数起点(F3/仿真真值拟合替换)
  double friction_xy = 0.80;
  double min_interval_s = 0.10;  // 去抖
};

// ---------- 物理 EKF(主线):状态 [p(3), v(3)] ----------
class BallPredictor {
 public:
  BallPredictor(const BallAeroParams& a, double obs_noise_std,
                const BounceParams& bp = {});

  void Reset(const BallObs& o);
  void Update(const BallObs& o);                       // 预测至 o.t + 观测更新
  bool Predict(double t_query, double p_out[3]) const;
  bool PredictFull(double t_query, double p_out[3], double v_out[3]) const;
  // 拦截采样:t 自 t_now+0.15 起 20ms 步进,首个 z∈[zlo,zhi] 且水平距 base ≤ reach
  bool Intercept(const double base[3], double t_now, double zlo, double zhi,
                 double reach, double p_hit[3], double* t_hit) const;

  bool initialized() const { return init_; }
  double t_last() const { return t_last_; }
  const double* state() const { return x_; }

 private:
  void Propagate(double dt);   // 状态+协方差(Euler+阻力雅可比);含反弹事件
  BallAeroParams a_;
  BounceParams bp_;
  double r_std_ = 0.030;
  bool init_ = false;
  double x_[6] = {0};
  double P_[6][6] = {{0}};
  double t_last_ = 0;
  double t_last_bounce_ = -10.0;
};

// ---------- 逐轴 CA 滤波(A/B 基线:trobot 现役路线)----------
// 每轴 [p,v,a] 3x3 协方差,jerk 白噪声过程;无物理模型(反弹/气动全靠 a 吸收)。
class CaAxisFilter {
 public:
  CaAxisFilter(double meas_sigma, double jerk_q);
  void Reset(const BallObs& o);
  void Update(const BallObs& o);
  bool Predict(double t_query, double p_out[3]) const;  // 三阶外推
  bool initialized() const { return init_; }

 private:
  struct Axis {
    double p = 0, v = 0, acc = 0;
    double P[3][3] = {{0}};
  };
  void PredictAxis(Axis& ax, double dt) const;
  Axis ax_[3];
  double meas_sigma_ = 0.030, jerk_q_ = 4.0e6;  // (mm/s³)² 语义换算为 m 制
  bool init_ = false;
  double t_last_ = 0;
};

}  // namespace sckd
