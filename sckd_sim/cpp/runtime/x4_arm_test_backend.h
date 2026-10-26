#pragma once
// 逐轴测试的 SDK 机械臂后端。构造/读取配置不会启动设备；StartHardware
// 会使能并发送配置 home，调用层必须在启动前让现场操作者确认该完整动作。
#include <array>
#include <memory>
#include <string>

#include "runtime/arm_axis_check.h"

namespace sckd::io::axischeck {

struct Setup {
  Joints home{}, kp{}, kd{}, kp_exit{}, kd_exit{}, sign{}, offset{};
  std::array<double, 4> dims{};
  std::array<std::string, 3> can_if{};
  double period_s = 0;
  // 明确是软件流程验证参数，不是已辨识的真机动力学。
  Joints sim_gravity_amplitude{};
  Joints sim_inertia{1., 2., .3, .1, .05};
  Joints sim_damping{.5, 1., .1, .05, .03};
  double sim_noise_deg = 0;
  int sim_command_delay_ms = 9;
  bool sim_floor_check = false;
  // 仿真线程启动前显式预置home，避免SDK未seed期间把零命令写入延迟队列。
  bool sim_preseed_home = true;
};

struct Feedback {
  Sample sample;
  std::array<bool, 3> can_open{};
  std::array<int, 5> raw_mode_state{};
  Joints torque{}, temperature{};
  bool running = false;
  // 本 SDK 的 ArmServerSim 未填 Motor mode（实测固定0）。仅在该模拟后端
  // 运行时给状态机合成mode=2；原值保留，时间戳/故障位始终不修改。
  bool mode_state_synthetic = false;
  unsigned snapshot_attempts = 0;
};

class SdkArmBackend {
 public:
  // 只解析配置。缺失、类型错误、符号/编码/增益不合法会抛 invalid_argument。
  explicit SdkArmBackend(const std::string& config_path);
  ~SdkArmBackend();
  SdkArmBackend(const SdkArmBackend&) = delete;
  SdkArmBackend& operator=(const SdkArmBackend&) = delete;

  const Setup& setup() const;
  // 每个实例只允许一次启动尝试；失败也不能复用。模拟端仅创建 ArmServerSim。
  bool StartSim(const std::string& log_dir, std::string* error);
  // 创建 ArmServer 并 Start；不创建 Robot/底盘/手柄/IMU，不配置或拉起 CAN。
  // 此方法会实际使能并发送 home；硬件调用权限由交互 CLI 的确认门管理。
  bool StartHardware(std::string* error);
  Feedback Read() const;
  // 首次 Write 前返回配置 home：它是实际启动目标，不是测得姿态。
  Joints LastTarget() const;
  // 整组设置目标，使用显式 hold kp/kd；首次成功提交后开启SDK目标接管。
  // true 仅表示交给 SDK，不是电机确认。Start及启动稳定窗仍沿用SDK home。
  bool Write(const Output& output, std::string* error);
  // 幂等。SDK 终末帧=最近目标/home+exit 增益；不是失能/急停，不调用 Hold。
  void Stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sckd::io::axischeck
