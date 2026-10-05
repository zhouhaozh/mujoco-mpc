#pragma once
// =============================================================================
// sckd_sim / cpp/core/types.h — 单一事实来源:单位、坐标系、轴序契约
// 依据: docs/SIM_SYSTEM_DESIGN.md §4.2(C++17 为准,Python 仅表意)
//   - 单位: m, s, rad, rad/s, N·m(HMI 显示层才允许 degree)
//   - 车体系: +x 前, +y 左, +z 上;俯视逆时针为 +wz
//   - 契约轴序 0..16:
//       底盘 FL,FR,RL,RR 每组 (steer, wheel_L, wheel_R) -> 0..11
//       臂 Joint1..Joint5 -> 12..16
//     模型内关节顺序与此无关,一律按名映射(mj_name2id)。
// =============================================================================
#include <array>
#include <cstdint>

namespace sckd {

constexpr int kNumAxes    = 17;
constexpr int kNumChassis = 12;
constexpr int kNumArm     = 5;
constexpr int kNumCorners = 4;

// 契约轴序对应的模型关节名(x4_tennis / x4_badminton 相同)
inline constexpr std::array<const char*, kNumAxes> kAxisJointName = {
    "FL_steer_joint", "FL_wheel_left_joint", "FL_wheel_right_joint",
    "FR_steer_joint", "FR_wheel_left_joint", "FR_wheel_right_joint",
    "RL_steer_joint", "RL_wheel_left_joint", "RL_wheel_right_joint",
    "RR_steer_joint", "RR_wheel_left_joint", "RR_wheel_right_joint",
    "Joint1", "Joint2", "Joint3", "Joint4", "Joint5"};

inline constexpr std::array<const char*, kNumCorners> kCornerName = {"FL", "FR", "RL", "RR"};

constexpr int AxisSteer(int corner)   { return 3 * corner + 0; }
constexpr int AxisWheelL(int corner)  { return 3 * corner + 1; }
constexpr int AxisWheelR(int corner)  { return 3 * corner + 2; }
constexpr int AxisArm(int j)          { return kNumChassis + j; }  // j=0..4 -> J1..J5

// --- 模块间消息(仅 POD;单位见契约) ---

struct ArmHybridCmd {          // 电机混合模式命令,与实机协议同构
  double q[kNumArm]     = {0};  // rad
  double dq[kNumArm]    = {0};  // rad/s
  double kp[kNumArm]    = {0};  // N·m/rad
  double kd[kNumArm]    = {0};  // N·m·s/rad
  double tau_ff[kNumArm] = {0}; // N·m(实机当前契约要求 0,接口预留)
};

struct BaseState {
  double x = 0, y = 0, z = 0;    // world 系
  double yaw = 0, roll = 0, pitch = 0;
  double vx = 0, vy = 0;         // 车体系
  double wz = 0;                 // world z
};

struct JointReport {
  double q = 0, dq = 0;
  double tau_cmd = 0;             // 电机模型输出力矩(限幅后)
  bool   enabled = false;
};

// IMU 样本(对齐实机 HiPNUC 通路:SI 单位,车体系,比力含重力反作用)
struct ImuSample {
  uint64_t t_us = 0;              // 采样时刻(仿真钟)
  double acc[3] = {0, 0, 0};      // m/s²(车体系)
  double gyr[3] = {0, 0, 0};      // rad/s(车体系)
};

// 球观测(§3.6;时间戳=采样时刻-感知延迟)
struct BallObs {
  double t = 0;
  double p[3] = {0, 0, 0};
};

// 单轴电机混合模式命令(核心命令契约;实机协议同构)
struct MotorCmd {
  double q = 0, dq = 0, kp = 0, kd = 0, tau_ff = 0;
};

}  // namespace sckd
