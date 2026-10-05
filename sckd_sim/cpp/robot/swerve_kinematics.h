#pragma once
// 舵轮运动学(紧凑矩阵式):分配(含 ADR-0009 协同策略)与正解(最小二乘)。
// 单位 SI(m, m/s, rad);角序为契约序 FL,FR,RL,RR;双轮 j=0(wheel_L)在
// +n=(−sinθ,cosθ) 侧(与 URDF 轮心 ±0.0275 m 一致,见 §3.2.1)。
// 参考:实机验证过的 sckd-robot chassis/kinematics(其 mm 单位/FR 起始角序已转换)。
#include <array>

#include <Eigen/Dense>

namespace sckd {

constexpr int kSwerveCorners = 4;
constexpr int kDrivesPerCorner = 2;   // 8W4S
constexpr int kDriveWheels = kSwerveCorners * kDrivesPerCorner;

struct SwerveGeometry {
  std::array<double, kSwerveCorners> corner_x{}, corner_y{};  // 车体系,契约角序
  double wheel_radius = 0.0828;      // m
  double dual_offset = 0.055;        // 双轮心全间距 m;d_j = ±dual_offset/2
};

struct CoordinationParams {
  bool enable_cos_projection = true;
  double cos_deadband = 0.10;        // rad,死区内不降速
  double cos_floor = 0.0;            // 投影下限
  double reverse_threshold = 1.5708; // rad,>90° 允许反转
  double reverse_hysteresis = 0.20;  // rad,反转迟滞(防临界抖动)
  double zero_speed = 1e-3;          // m/s,速度死区:保持当前舵角、驱动零
};

struct Twist {
  double vx = 0, vy = 0, wz = 0;     // 车体系 m/s, rad/s
};

struct CornerCmd {
  double steer_target = 0;           // rad(最近角语义:实测角 + 最短误差)
  double drive_mps[kDrivesPerCorner] = {0, 0};
};

double WrapPi(double a);
double AngleDiff(double to, double from);

class SwerveKinematics {
 public:
  SwerveKinematics(const SwerveGeometry& g, const CoordinationParams& c);

  // 逆解:twist -> 每角(目标舵角, 模块速度)。reversed 为有状态迟滞,调用方持有并
  // 跨周期传入;舵角误差大时 cos 投影自动收缩驱动速度(消除侧向刮擦分量)。
  void Allocate(const Twist& t, const double steer_meas[kSwerveCorners],
                bool reversed[kSwerveCorners], CornerCmd out[kSwerveCorners]) const;

  // 双轮差速:v_j = v_module − d_j·(wz + θ̇);wz 与 θ̇ 统一处理(实测验证的公式)
  static void DualWheel(double module_mps, double wz, double steer_rate,
                        double dual_offset, double out_mps[kDrivesPerCorner]);

  // 正解:8 轮线速度 + 舵角 -> 车体 twist(最小二乘;轮位置含双轮精确偏置)
  Twist EstimateTwist(const double drive_mps[kDriveWheels],
                      const double steer_meas[kSwerveCorners]) const;

  const SwerveGeometry& geo() const { return g_; }

 private:
  SwerveGeometry g_;
  CoordinationParams c_;
};

}  // namespace sckd
