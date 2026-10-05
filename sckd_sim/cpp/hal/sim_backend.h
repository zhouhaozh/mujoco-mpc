#pragma once
// in-process 仿真后端:直连 MuJoCo C API(设计 §2.3 形态一、§4.3)
// 电机模型作为执行器外壳:先算 τ_req 再写 ctrl,物理只受力矩。
#include <memory>
#include <string>

#include <mujoco/mujoco.h>

#include "hal/robot_interface.h"
#include "robot/motor_model.h"

namespace sckd {

class SimBackend : public RobotInterface {
 public:
  SimBackend(const std::string& scene_xml, const std::string& motors_json,
             const std::string& variant_json);
  ~SimBackend() override;

  bool ok() const { return m_ != nullptr && d_ != nullptr; }

  bool reset(int keyframe = 0);          // 重置物理与电机模型状态(含延迟队列)
  void hold_chassis(double kd);          // M0:底盘 12 轴 kp=0 阻尼保持(分配器 M1 提供)
  void hold_arm(double kp, double kd);   // 臂保持零目标

  // RobotInterface
  bool enable(int axis, bool on) override;
  void set_arm_hybrid(const ArmHybridCmd& cmd) override;
  BaseState base_state() const override;
  JointReport joint(int axis) const override;

  bool run(int steps);                   // 每步:tick -> compute -> ctrl -> mj_step
  uint64_t tick() const { return tick_; }
  double quantize(double q) const;       // 编码器量化(固件视角;评估侧用真值)
  const MotorSystemConfig& motor_config() const { return cfg_; }
  const mjModel* model() const { return m_; }
  const mjData* data() const { return d_; }

 private:
  void write_ctrl();
  bool build_maps();

  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;
  MotorSystemConfig cfg_;
  std::unique_ptr<MotorModel> motor_;
  int jid_[kNumAxes] = {}, qadr_[kNumAxes] = {}, vadr_[kNumAxes] = {}, aid_[kNumAxes] = {};
  int base_bid_ = -1, base_qadr_ = 0;
  uint64_t tick_ = 0;
  double dt_ = 0.001;
};

}  // namespace sckd
