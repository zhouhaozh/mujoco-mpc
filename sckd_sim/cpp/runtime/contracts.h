#pragma once
// 增量模块化的值类型：不含仿真器、socket或求解器对象。
// 这些结构承载事实与有效域，不自行改变当前接纳/锁止/回退策略。
// 旧 BallObs/BaseState/命令仍沿用 core/types.h 的 SI 单位与17轴顺序。
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "core/types.h"

namespace sckd::runtime {

using Sequence = std::uint64_t;

struct Timestamp {
  double seconds = 0;
  // 仿真时钟、相机捕获时钟与steady墙钟必须使用不同id；跨域先显式变换。
  std::string clock_id = "simulation";
};

inline bool SameClock(const Timestamp& a, const Timestamp& b) {
  return !a.clock_id.empty() && a.clock_id == b.clock_id;
}

struct TimeInterval {
  Timestamp begin, end;
  bool Contains(const Timestamp& time) const {
    return SameClock(begin, end) && SameClock(begin, time) &&
      std::isfinite(begin.seconds) && std::isfinite(end.seconds) &&
      std::isfinite(time.seconds) && begin.seconds <= time.seconds && time.seconds <= end.seconds;
  }
};

struct FrameIdentity {
  std::string world = "world";
  std::string body = "base_link";
  std::string racket = "racket_center";
  std::string transform_version;  // 未校准/未提供时为空，不能默认视为已校准。
};

struct ObservationIdentity {
  std::string track_id;
  Sequence observation_seq = 0;
  Timestamp captured;
  Timestamp received;
  FrameIdentity frames;
};

struct BallObservation {
  ObservationIdentity identity;
  std::array<double, 3> position_world{};
  bool valid = false;

  // 显式传入接收钟；不把 BallObs.t 改写为当前接收时刻。
  static BallObservation FromLegacy(const BallObs& observation,
                                    ObservationIdentity identity) {
    identity.captured.seconds = observation.t;
    BallObservation out;
    out.identity = std::move(identity);
    for (int i = 0; i < 3; ++i) out.position_world[i] = observation.p[i];
    out.valid = std::isfinite(observation.t);
    for (double value : out.position_world) out.valid = out.valid && std::isfinite(value);
    return out;
  }
};

enum class RobotStateSource { kSimulationTruth, kMeasured, kEstimated };

struct RobotEstimate {
  Sequence state_seq = 0;
  Timestamp sampled;
  FrameIdentity frames;
  BaseState base;  // world位置/姿态，body vx/vy；沿用core/types.h。
  std::array<JointReport, kNumAxes> joints{};
  RobotStateSource base_source = RobotStateSource::kSimulationTruth;
  // 当前仿真臂q经过量化、dq仍是真值；不能用一个整包标签掩盖混合来源。
  std::array<RobotStateSource, kNumAxes> joint_position_source{};
  std::array<RobotStateSource, kNumAxes> joint_velocity_source{};
  bool valid = false;
};

struct ActuatorState {
  Timestamp sampled;
  Sequence state_seq = 0;
  std::array<MotorCmd, kNumAxes> issued{};
  std::array<MotorCmd, kNumAxes> effective{};
  // nmpc6执行器/底盘预测候选使用的已有执行状态；不能以全0冒充可观测。
  bool commands_available = false;
  std::array<double, 3> ramped_twist_body{};  // vx,vy,wz。
  std::array<double, kNumCorners> steer_angles_rad{};
  std::array<double, 2 * kNumCorners> wheel_targets_rad_s{};  // 每角L/R，同core轴序。
  bool chassis_state_available = false;
  double command_transport_s = 0.009;
};

struct HitTaskSnapshot {
  std::string track_id;
  Sequence target_version = 0;
  Sequence forecast_seq = 0;
  Timestamp forecast_captured;
  Timestamp published;
  Timestamp hit_time;
  FrameIdentity frames;
  std::array<double, 3> position_world{}, incoming_velocity_world{};
  std::array<double, 3> racket_velocity_world{}, face_normal_world{};
  std::array<double, 3> outgoing_velocity_world{};
  std::array<double, 3> base_target_world{};  // x,y,yaw。
  std::array<double, 2> base_velocity_at_hit_world{};  // 原v_base_star，初规划冻结量。
  bool feasible = false;
  bool locked = false;
};

struct BaseReferenceAnchor {
  // 原 BaseRefGen::Schedule 的起始锚；精化终点不能重置它。
  Timestamp started;
  std::array<double, 2> position_world{};
};

struct PlanIdentity {
  Sequence plan_id = 0;
  Sequence target_version = 0;
  Sequence state_seq = 0;
  std::string track_id;
  std::string algorithm;
};

struct TrajectoryTiming {
  Timestamp created;
  TimeInterval sample_domain;
  Timestamp desired_contact;
  // wbmpc接触节点离散化后时刻；与任务击球时刻/随挥视界终点各自记录。
  std::optional<Timestamp> discrete_contact;
  double follow_through_max_s = 0.15;
};

struct SolveReport {
  PlanIdentity candidate;
  Timestamp started, finished;
  std::string solver_status;
  bool solver_success = false;
  bool internal_accepted = false;
  bool upper_accepted = false;
  bool applied = false;
  std::string reject_reason;
  double solve_ms = 0;
  std::optional<double> max_constraint_violation;
  std::optional<double> position_residual_m;
  // status成功、内部接纳、上层接纳和实际应用为不同事件，不互相自动推导。
};

enum class PlanDecision { kKeepPrevious, kAdoptCandidate, kFallback, kHold, kControlledStop };

struct PlanDecisionEvent {
  Timestamp decided;
  PlanIdentity candidate, executing;
  PlanDecision decision = PlanDecision::kKeepPrevious;
  std::string reason;
};

struct WholeBodyReference {
  PlanIdentity plan;
  Timestamp sampled;
  FrameIdentity frames;
  std::array<double, 3> base_pose_world{};  // x,y,yaw。
  std::array<double, 3> base_velocity_world{};  // vx,vy,wz；执行适配器再转body。
  std::array<double, kNumArm> arm_q{}, arm_dq{}, arm_ddq{};
  bool valid = false;
};

inline void to_json(nlohmann::json& out, const Timestamp& time) {
  out = {{"seconds", time.seconds}, {"clock_id", time.clock_id}};
}

inline void to_json(nlohmann::json& out, const PlanIdentity& plan) {
  out = {{"plan_id", plan.plan_id}, {"target_version", plan.target_version},
         {"state_seq", plan.state_seq}, {"track_id", plan.track_id}, {"algorithm", plan.algorithm}};
}

inline void to_json(nlohmann::json& out, const SolveReport& report) {
  out = {{"schema", "SCKD_SOLVE_REPORT/1"}, {"candidate", report.candidate},
         {"started", report.started}, {"finished", report.finished},
         {"solver_status", report.solver_status}, {"solver_success", report.solver_success},
         {"internal_accepted", report.internal_accepted}, {"upper_accepted", report.upper_accepted},
         {"applied", report.applied}, {"reject_reason", report.reject_reason}, {"solve_ms", report.solve_ms},
         {"max_constraint_violation", report.max_constraint_violation ? nlohmann::json(*report.max_constraint_violation) : nlohmann::json(nullptr)},
         {"position_residual_m", report.position_residual_m ? nlohmann::json(*report.position_residual_m) : nlohmann::json(nullptr)}};
}

inline void to_json(nlohmann::json& out, const PlanDecisionEvent& event) {
  const char* decision = "keep_previous";
  switch (event.decision) {
    case PlanDecision::kKeepPrevious: break;
    case PlanDecision::kAdoptCandidate: decision = "adopt_candidate"; break;
    case PlanDecision::kFallback: decision = "fallback"; break;
    case PlanDecision::kHold: decision = "hold"; break;
    case PlanDecision::kControlledStop: decision = "controlled_stop"; break;
  }
  out = {{"schema", "SCKD_PLAN_DECISION/1"}, {"decided", event.decided},
         {"candidate", event.candidate}, {"executing", event.executing},
         {"decision", decision}, {"reason", event.reason}};
}

}  // namespace sckd::runtime
