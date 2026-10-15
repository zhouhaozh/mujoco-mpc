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
  bool get_ball_obs(BallObs* out);
  bool arm_gravity(const double q[kNumArm], double tau_ff[kNumArm]) override;

  bool run(int steps);                   // 每步:tick -> compute -> ctrl -> mj_step(+IMU 采样)
  uint64_t tick() const { return tick_; }
  double quantize(double q) const;       // 编码器量化(固件视角;评估侧用真值)
  void arm_effort(double raw[5], double applied[5], double cap[5]) const {
    for(int j=0;j<5;++j){raw[j]=motor_->last_raw_tau(AxisArm(j));applied[j]=motor_->last_tau_cmd(AxisArm(j));cap[j]=motor_->last_tau_cap(AxisArm(j));}
  }
  const MotorSystemConfig& motor_config() const { return cfg_; }
  const ChassisConfig& chassis_config() const { return chcfg_; }
  const mjModel* model() const { return m_; }
  const mjData* data() const { return d_; }
  // 可写访问(演示/气动外力等进程内扩展;正式算法层禁用,§4.3)
  mjData* data_mut() { return d_; }

 private:
  void write_ctrl();
  bool build_maps();
  void sample_imu();                     // 200Hz:真值 + 偏置/噪声 -> imu_(带龄期戳)
  void sample_ball_obs();                // 300Hz:球位置 + 噪声 -> ball_obs_(带龄期戳)

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
  // 球观测(§3.6 源①)
  BallObs ball_obs_{};       // 带噪声+延迟戳的观测
  bool ball_obs_valid_ = false;
  uint64_t next_ball_obs_tick_ = 0;
  // 延迟回放缓冲:每 ms 记球真值,emit 时回取 latency 前的样本,保证 obs.p 与
  // obs.t 时刻一致(旧实现"当前位置+过去时间戳"使 EKF 系统性滞后一个延迟,
  // 12 m/s 下 0.18m 常值偏差 —— 2026-10-06 验证①定位)
  static constexpr int kBallHist = 64;   // 64ms @1kHz > 15ms 延迟
  double ball_hist_p_[kBallHist][3] = {};
  uint64_t ball_hist_tick_[kBallHist] = {};
  int ball_bid_ = -1, ball_qadr_ = -1;
  double ball_obs_noise_ = 0.030;   // σ(m,对齐真机 trobot)
  double ball_obs_rate_hz_ = 300.0;
  double ball_obs_latency_ms_ = 15.0;
  double last_bvx_ = 0, last_bvy_ = 0, last_bvz_ = 0;
};

}  // namespace sckd
