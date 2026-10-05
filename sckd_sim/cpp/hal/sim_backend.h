#pragma once
// in-process 仿真后端:直连 MuJoCo C API(设计 §2.3 形态一、§4.3)
// 电机模型作为执行器外壳:先算 τ_req 再写 ctrl,物理只受力矩。
// M1:IMU 仿真(200Hz,偏置/噪声/龄期)、测量视角(量化+舵角偏置+轮速噪声)、
//     臂重力前馈(mj_inverse,悬空 scratch)。
#include <memory>
#include <random>
#include <string>

#include <mujoco/mujoco.h>

#include "core/config.h"
#include "hal/robot_interface.h"
#include "robot/motor_model.h"

namespace sckd {

class SimBackend : public RobotInterface {
 public:
  SimBackend(const std::string& scene_xml, const std::string& motors_json,
             const std::string& variant_json,
             const std::string& chassis_json = "models/config/chassis_m1.json");
  ~SimBackend() override;

  bool ok() const { return m_ != nullptr && d_ != nullptr; }

  bool reset(int keyframe = 0);          // 重置物理与电机模型状态(含延迟队列/噪声 RNG)
  void hold_chassis(double kd);          // M0:底盘 12 轴 kp=0 阻尼保持(分配器 M1 提供)
  void hold_arm(double kp, double kd);   // 臂保持零目标

  // RobotInterface
  bool enable(int axis, bool on) override;
  void set_arm_hybrid(const ArmHybridCmd& cmd) override;
  void set_axis_cmd(int axis, const MotorCmd& cmd) override;
  BaseState base_state() const override;
  JointReport joint(int axis) const override;
  JointReport joint_measured(int axis) override;
  bool get_imu(ImuSample* out) override;
  bool arm_gravity(const double q[kNumArm], double tau_ff[kNumArm]) override;

  bool run(int steps);                   // 每步:tick -> compute -> ctrl -> mj_step(+IMU 采样)
  uint64_t tick() const { return tick_; }
  double quantize(double q) const;       // 编码器量化(固件视角;评估侧用真值)
  const MotorSystemConfig& motor_config() const { return cfg_; }
  const ChassisConfig& chassis_config() const { return chcfg_; }
  const mjModel* model() const { return m_; }
  const mjData* data() const { return d_; }

 private:
  void write_ctrl();
  bool build_maps();
  void sample_imu();                     // 200Hz:真值 + 偏置/噪声 -> imu_(带龄期戳)

  mjModel* m_ = nullptr;
  mjData* d_ = nullptr;
  mjData* scratch_ = nullptr;            // 重力前馈专用(悬空,无接触)
  MotorSystemConfig cfg_;
  ChassisConfig chcfg_;
  std::unique_ptr<MotorModel> motor_;
  int jid_[kNumAxes] = {}, qadr_[kNumAxes] = {}, vadr_[kNumAxes] = {}, aid_[kNumAxes] = {};
  int base_bid_ = -1, base_qadr_ = 0;
  uint64_t tick_ = 0;
  double dt_ = 0.001;
  // IMU/测量注入(确定性:固定种子,reset 复位)
  std::mt19937 rng_{12345};
  std::normal_distribution<double> gauss_{0.0, 1.0};
  ImuSample imu_{};
  bool imu_valid_ = false;
  uint64_t next_imu_tick_ = 0;
  double last_bvx_ = 0, last_bvy_ = 0, last_bvz_ = 0;
};

}  // namespace sckd
