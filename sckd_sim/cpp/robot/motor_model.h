#pragma once
// 电机模型:手册混合控制律 + 限幅链 + T-N 包络 + 命令延迟 + 使能语义。
// 依据: docs/motors 手册(RS04 p43 / RS06 p41 混合公式;T-N 读图包络)与设计 §3.3。
// 口径:减速器输出端(关节侧);力矩上限 = min(机构限矩, 峰值/连续档, T-N 包络)。
#include <array>
#include <deque>

#include "core/config.h"

namespace sckd {

// 单轴混合模式命令
struct MotorCmd {
  double q = 0, dq = 0, kp = 0, kd = 0, tau_ff = 0;
};

class MotorModel {
 public:
  explicit MotorModel(const MotorSystemConfig& cfg);

  // 命令下发(kp/kd/目标位置按协议范围截断);入延迟队列,不立即生效
  void set_cmd(int axis, const MotorCmd& cmd);
  // 使能/失能:失能输出零,但物理(重力/接触)继续;返回之前状态。axis=-1 全部
  bool enable(int axis, bool on);
  bool enabled(int axis) const;

  void tick();  // 每控制周期推进一次延迟队列;断流时目标锁存(看门狗语义)

  // 由测量值算输出力矩(限幅后);不推进时间
  double compute(int axis, double q_meas, double dq_meas);

  double last_tau_cmd(int axis) const;
  const MotorSystemConfig& config() const { return cfg_; }
  int delay_slots() const { return delay_slots_; }

 private:
  struct AxisState {
    bool enabled = false;
    MotorCmd issued;                 // 最近下发命令(断流时锁存 = 看门狗语义)
    std::deque<MotorCmd> pipe;       // 延迟管道:front = 当前生效命令
    double last_tau = 0.0;
  };
  const MotorTypeConfig& motor_of(int axis) const;
  const JointConfig& joint_of(int axis) const;
  static double envelope(const MotorTypeConfig& m, double abs_speed);  // T-N 推动包络

  MotorSystemConfig cfg_;
  int delay_slots_ = 9;
  std::array<AxisState, kNumAxes> st_;
};

}  // namespace sckd
