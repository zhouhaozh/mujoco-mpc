#pragma once
// 机器人抽象层(M1 扩展:轴级命令 / 测量视角 / IMU / 重力前馈服务)
// 算法模块只依赖此接口与 core/types.h —— 移植规则 §4.3:算法内禁止仿真器头文件。
// 语义:joint()=真值(评估器用);joint_measured()=控制器可见(量化/偏置/噪声注入)。
#include "core/types.h"

namespace sckd {

class RobotInterface {
 public:
  virtual ~RobotInterface() = default;
  virtual bool enable(int axis, bool on) = 0;                 // axis=-1 全部
  virtual void set_arm_hybrid(const ArmHybridCmd& cmd) = 0;   // 契约轴 12..16
  virtual void set_axis_cmd(int axis, const MotorCmd& cmd) = 0;  // 轴级原始命令(底盘伺服用)
  virtual BaseState base_state() const = 0;                   // 真值(仅评估)
  virtual JointReport joint(int axis) const = 0;              // 真值(仅评估)
  virtual JointReport joint_measured(int axis) = 0;           // 测量视角(控制器)
  virtual bool get_imu(ImuSample* out) = 0;                   // 最新 IMU 样本(带龄期)
  // 臂重力前馈(由模型计算;实机后端用同源模型文件实现——接口不变)
  virtual bool arm_gravity(const double q[kNumArm], double tau_ff[kNumArm]) = 0;
};

}  // namespace sckd
