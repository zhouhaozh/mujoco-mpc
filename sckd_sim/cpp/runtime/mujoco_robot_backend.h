#pragma once
// 共用语义接口的MuJoCo适配。调用者拥有后端和伺服，并串行推进物理。
// Read只读取仿真真值，不消费带噪传感器，不推进物理或共享RNG。
#include "runtime/robot_io.h"

namespace sckd {
class SimBackend;
class ChassisServo;
namespace io {

class MujocoRobotBackend final : public RobotBackend {
 public:
  MujocoRobotBackend(SimBackend& simulation, ChassisServo& servo);
  Capabilities capabilities() const override;
  WriteResult Write(const Command& command, Time now) override;
  Observation Read(Time now) override;

  // 调用者每1ms显式调用；只更新底盘伺服，物理仍由SimBackend::run推进。
  void Tick(Time now);

 private:
  SimBackend& simulation_;
  ChassisServo& servo_;
  std::uint64_t last_written_sequence_ = 0;
};

}  // namespace io
}  // namespace sckd
