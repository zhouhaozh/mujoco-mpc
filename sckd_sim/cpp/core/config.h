#pragma once
// 配置加载:motors.json(电机手册参数)+ <variant>.json(机器人变体)
// 运行时只读;来源与假设标注在 json 内(calibrated/sources 字段)。
#include <string>
#include <vector>
#include <array>
#include "core/types.h"

namespace sckd {

struct EnvelopePoint {
  double speed_rad_s = 0;
  double torque_nm = 0;
};

struct MotorTypeConfig {
  std::string name;
  double peak_torque_nm = 0;
  double continuous_torque_nm = 0;
  double braking_torque_nm = 0;
  double no_load_speed_rad_s = 0;
  double hybrid_speed_limit_rad_s = 0;   // 运控模式速度量程(协议),非机构安全范围
  double hybrid_position_limit_rad = 0;
  double kp_min = 0, kp_max = 0;
  double kd_min = 0, kd_max = 0;
  std::vector<EnvelopePoint> motoring;   // 升序;|dq| 插值,仅推动方向
};

struct JointConfig {
  std::string joint;                     // 模型关节名(= 契约轴序表)
  std::string motor;                     // RS04 / RS06 / RS05
  double mechanism_limit_nm = 0;
};

struct ArmServoConfig {
  bool enabled = false;  // 未配置时保持网球执行器原有增益和准备姿态。
  std::array<double, kNumArm> kp{}, kd{}, standby_q{};
};

struct MotorSystemConfig {
  std::string robot_variant;
  bool truth_feedback = false;
  ArmServoConfig arm_servo;
  double command_delay_ms = 9.0;
  double control_dt_s = 0.001;
  int encoder_bits = 14;                 // 占位,待实机确认(设计 §3.3)
  bool calibrated = false;               // T-N 读图估值;报告必须体现
  bool profile_peak = true;              // true=峰值档,false=连续档
  std::vector<MotorTypeConfig> motors;
  std::vector<JointConfig> joints;       // 按契约轴序 0..16
};

// 解析失败抛 std::runtime_error(带字段名)。
MotorSystemConfig LoadMotorConfig(const std::string& motors_json_path,
                                  const std::string& variant_json_path);

// --- 底盘配置：网球与 X4_950 分别加载各自 JSON ---

struct ChassisCoordination {
  bool enable_cos_projection = true;
  double cos_deadband = 0.10, cos_floor = 0.0;
  double reverse_threshold = 1.5708, reverse_hysteresis = 0.20;
  double zero_speed = 1e-3;
};

struct ChassisRamp {
  double accel_limit_mps2 = 8.0;
  double ang_accel_limit_rad_s2 = 20.0;
  double dir_rate_limit_rad_s = 3.0;   // 平移方向角速率限幅(防甩头)
};

struct ChassisServoParams {
  double period_ms = 4;                // 250 Hz 分配(对齐实机 4ms)
  double steer_kp = 20.0, steer_kd = 1.0;   // 舵位置 PD(混合律;等效 CSP loc_kp)
  double wheel_kp = 2.0, wheel_ki = 0.5;    // 轮速 PI -> tau_ff(显式环路)
  double wheel_int_max_rad = 8.0;           // 积分限幅(防缠绕;τ_int 上限 = ki*此值)
  double wheel_target_accel_mps2 = 8.0;     // 轮目标加速度限幅(≈实机 97 rad/s²·r)
};

struct ChassisOdomParams {
  std::array<double, 4> steer_offset_rad{};  // 舵角零位偏差(实机 x4-04 约 ±45°)
  double wheel_speed_noise_rad_s = 0.01;
  double imu_rate_hz = 200.0, imu_latency_ms = 2.0;
  double gyro_bias_rad_s = 0.002, gyro_noise_rad_s = 0.002;
  double acc_noise_mps2 = 0.05;
  double gyro_fallback_ms = 300.0;           // IMU 龄期超时回退轮式航向
};

struct ChassisConfig {
  std::string robot_variant;
  std::array<double, 4> corner_x{}, corner_y{};   // 契约角序 FL,FR,RL,RR
  double wheel_radius = 0.0828;
  double dual_offset = 0.055;
  std::array<double, 8> wheel_sign{};             // 线速度->关节速度符号(L=+1/R=−1,镜像轴)
  double robot_mass_kg = 51.26;                   // 加速度前馈用(CAD 求和,§3.2.1)
  ChassisCoordination coord;
  ChassisRamp ramp;
  ChassisServoParams servo;
  ChassisOdomParams odom;
};

ChassisConfig LoadChassisConfig(const std::string& chassis_json_path);

}  // namespace sckd
