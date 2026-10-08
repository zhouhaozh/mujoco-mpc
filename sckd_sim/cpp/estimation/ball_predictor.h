#pragma once
// 球轨迹预测器(§6.1/§3.6):物理 EKF(T1:重力+阻力,可选残余加速度)+ 反弹事件分段;
// 附逐轴 CA 滤波(trobot 现役路线)作 A/B 基线——用数据说话,不被其绑架。
#include "core/types.h"
#include "sim/ball_aero.h"

namespace sckd {

struct BounceParams {
  // 已知碰撞模型(§3.5 地面支,2026-10-08):与真值侧 analytic_contact.h
  // ApplyGroundImpulse 同参数(自洽);e_z=0.73 ITF 文献初值、μ=0.20 为
  // ground_calib.py 无旋台架拟合初值——实机实测后同形式直接替换。
  // 平抽 v1 无旋(用户定向 2026-10-08:旋转球后续迭代)。
  double restitution_z = 0.73;   // 法向恢复 vz' = −e_z·vz
  double mu = 0.20;              // Coulomb 滑动摩擦(切向粘滞/滑动取小)
  double min_interval_s = 0.10;  // 去抖
};

// ---------- 物理 EKF(主线):状态 [p(3), v(3)] ----------
class BallPredictor {
 public:
  // warmup_span_s:首次观测后等待该时距再差分定初速(300Hz 下约 18 帧,
  // σ_v≈√2·σ/span≈0.7 m/s),避免 v=0/P_v=100 的慢收敛拖垮 200ms 级前视
  BallPredictor(const BallAeroParams& a, double obs_noise_std,
                const BounceParams& bp = {}, double warmup_span_s = 0.06);

  void Reset(const BallObs& o);                        // 显式重置(v=0,供单测/对拍)
  void Update(const BallObs& o);                       // 未初始化时内部走预热路径
  bool Predict(double t_query, double p_out[3]) const;
  bool PredictFull(double t_query, double p_out[3], double v_out[3]) const;
  // 拦截采样:t 自 t_now+0.15 起 20ms 步进,首个 z∈[zlo,zhi] 且水平距 base ≤ reach
  bool Intercept(const double base[3], double t_now, double zlo, double zhi,
                 double reach, double p_hit[3], double* t_hit) const;

  bool initialized() const { return init_; }
  double t_last() const { return t_last_; }
  const double* state() const { return x_; }
  // 当前估计外推:首次反弹的绝对时刻(2.5s 内无反弹返回 -1)。
  // 规划器用它把候选限制在反弹前(基线策略,对齐 trobot MLP 域边界;
  // 跨反弹规划待反弹模型标定后开启)
  double NextBounceTime() const;

 private:
  void Propagate(double dt);   // 状态+协方差(Euler+阻力雅可比);含反弹事件
  BallAeroParams a_;
  BounceParams bp_;
  double r_std_ = 0.030;
  double warm_span_ = 0.04;
  bool warming_ = false;
  double t_w0_ = 0;
  double p_w0_[3] = {0};
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
