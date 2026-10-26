#include "runtime/mujoco_robot_backend.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "control/chassis_servo.h"
#include "hal/sim_backend.h"

namespace sckd::io {

MujocoRobotBackend::MujocoRobotBackend(SimBackend& simulation, ChassisServo& servo)
    : simulation_(simulation), servo_(servo) {}

Capabilities MujocoRobotBackend::capabilities() const {
  return {true, false, "mpc_model"};
}

WriteResult MujocoRobotBackend::Write(const Command& command, Time now) {
  if (now.clock != Clock::kSimulation) return {false, "clock_mismatch"};
  const auto checked = Validate(command, now, capabilities());
  if (!checked.submitted) return checked;
  if (!simulation_.ok()) return {false, "simulation_unavailable"};
  if (command.sequence <= last_written_sequence_) return {false, "out_of_order_sequence"};

  ArmHybridCmd arm;
  for (int j = 0; j < kNumArm; ++j) {
    arm.q[j] = command.q[j];
    arm.dq[j] = command.dq[j];
    arm.kp[j] = command.kp[j];
    arm.kd[j] = command.kd[j];
    arm.tau_ff[j] = command.tau_ff[j];
  }
  servo_.set_twist(Twist{command.body_twist[0], command.body_twist[1],
                         command.body_twist[2]});
  simulation_.set_arm_hybrid(arm);
  last_written_sequence_ = command.sequence;
  return {true, "submitted_to_mujoco"};
}

void MujocoRobotBackend::Tick(Time now) {
  if (now.clock != Clock::kSimulation || !std::isfinite(now.seconds) || now.seconds < 0)
    throw std::invalid_argument("MuJoCo servo tick requires nonnegative simulation time");
  if (!simulation_.ok()) throw std::runtime_error("MuJoCo simulation unavailable");
  servo_.update(now.seconds);
}

Observation MujocoRobotBackend::Read(Time now) {
  Observation out;
  out.received = now;
  out.frame_id = "mpc_model";
  if (now.clock != Clock::kSimulation || !std::isfinite(now.seconds) ||
      !simulation_.ok()) return out;

  // 与调用时间分开：原m2循环在执行后、物理推进前记录，状态仍是上一物理拍。
  const Time sampled{simulation_.data()->time, Clock::kSimulation};
  out.sequence = simulation_.tick() + 1;
  const auto base = simulation_.base_state();
  out.base_pose = {base.x, base.y, base.yaw};
  out.body_twist = {base.vx, base.vy, base.wz};
  out.base_stamp = sampled;
  out.base_available = true;
  out.base_stamp_known = true;
  out.base_source = Source::kSimulationTruth;
  out.coherent_snapshot = true;  // 约定：与物理推进串行调用，不宣称线程安全。
  out.ready = true;
  for (int j = 0; j < kNumArm; ++j) {
    const auto truth = simulation_.joint(AxisArm(j));
    auto& feedback = out.joints[j];
    feedback.q = truth.q;
    feedback.dq = truth.dq;
    feedback.torque = truth.tau_cmd;  // 仿真电机模型施加力矩，非硬件传感器测量。
    feedback.temperature = std::numeric_limits<double>::quiet_NaN();  // 无热模型。
    feedback.stamp = sampled;
    feedback.available = true;
    feedback.stamp_known = true;
    feedback.position_source = Source::kSimulationTruth;
    feedback.velocity_source = Source::kSimulationTruth;
    out.ready = out.ready && truth.enabled;
  }
  return out;
}

}  // namespace sckd::io
