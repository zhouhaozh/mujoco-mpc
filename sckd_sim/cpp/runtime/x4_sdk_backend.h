#pragma once
// X4 SDK 边界。SDK 头与对象布局仅在 .cpp 中可见；本头可供纯逻辑测试使用。
#include <memory>
#include <string>

#include "runtime/robot_io.h"

namespace sckd::sdk { class Robot; }

namespace sckd::io {

// SDK 使用 CLOCK_MONOTONIC；设备仿真也按该时钟运行，不能传 MuJoCo 仿真时间。
Time X4SdkMonotonicNow();

// SDK模型角到电机编码角的边界：motor=sign*(model-offset)。
// 必须来自初始化Robot的同一配置；默认sign=0刻意表示未提供上下文。
// 这是协议编码范围检查所需的转换，不代表已经标定机械运动限位。
struct X4SdkJointEncoding {
  std::array<double, 5> sign{}, offset_rad{};
};
bool LoadX4SdkJointEncoding(const std::string& config_ini,
                           X4SdkJointEncoding* out, std::string* err);

// 与真实派发完全相同的静态预检；回放可在启动模拟线程前检查整个文件。
// 不读取设备状态、不消耗序号、不产生写入。now 必须为 SDK 单调钟时间。
WriteResult ValidateX4SdkCommand(const Command&, Time now, const Capabilities&,
                                 const X4SdkJointEncoding&,
                                 std::uint64_t previous_sequence = 0);

// 仅构造 ChassisServerSim / ArmServerSim，不打开 CAN、IMU 或手柄。
// 要求 [Sim] enabled=true，log_dir 不存在或为空；拒绝覆盖既有运行证据。
// 日志和实际加载参数清单写入显式 log_dir，清单以独占方式创建。
// 底盘为理想运动学；臂为 SDK 简化动力学反馈，均不是 MuJoCo plant。
std::unique_ptr<RobotBackend> CreateX4SdkSimulation(
    const std::string& config_ini, const std::string& log_dir, std::string* err);

// 借用已由调用者初始化的 Robot；不调用 Init、Shutdown、ArmHome 或 Enable。
// 调用者须保证 Robot 生存期长于返回对象，并负责整个硬件生命周期。
// frame_id 只是已核实映射的声明；适配器不执行 sdk_model / mpc_model 变换。
// 两次 SDK 下发非整机原子；submitted 不代表驱动确认，也不消除库内仲裁竞态。
std::unique_ptr<RobotBackend> BorrowX4Robot(
    sckd::sdk::Robot& robot, const X4SdkJointEncoding& encoding,
    std::string frame_id = "sdk_model");

namespace testing {
// 无设备测试接缝。生产工厂与 fake 共用相同预检及派发实现。
struct X4SdkEndpointStatus {
  bool ready = false, arm_degraded = false, manual_takeover = false;
};
class X4SdkEndpoint {
 public:
  virtual ~X4SdkEndpoint() = default;
  virtual Capabilities capabilities() const = 0;
  virtual X4SdkJointEncoding joint_encoding() const = 0;
  virtual X4SdkEndpointStatus status() const = 0;
  virtual void Submit(const Command&) = 0;
  virtual Observation Snapshot(Time now) = 0;
};
std::unique_ptr<RobotBackend> WrapX4SdkEndpoint(std::unique_ptr<X4SdkEndpoint>);
}  // namespace testing
}  // namespace sckd::io
