#pragma once
// 机器人抽象层最小接口(M0 子集;M1 扩展底盘速度/舵轮分配,完整契约见设计 §4.1)
// 算法模块只依赖此接口与 core/types.h —— 移植规则 §4.3:算法内禁止仿真器头文件。
#include "core/types.h"

namespace sckd {

class RobotInterface {
 public:
  virtual ~RobotInterface() = default;
  virtual bool enable(int axis, bool on) = 0;                // axis=-1 全部
  virtual void set_arm_hybrid(const ArmHybridCmd& cmd) = 0;  // 契约轴 12..16
  virtual BaseState base_state() const = 0;
  virtual JointReport joint(int axis) const = 0;
};

}  // namespace sckd
