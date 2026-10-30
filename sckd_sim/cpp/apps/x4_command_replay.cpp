// Dynamics-only replay: measured initial state + recorded commands -> X4 MuJoCo.
// No measured state is copied after initialization and no MPC code is linked.
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>

#include "control/chassis_servo.h"
#include "hal/sim_backend.h"
#include "robot/swerve_kinematics.h"

namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace sckd;

namespace {

struct ReplayCommand {
  double t = 0;
  std::array<double, 5> q{}, dq{};
  Twist base{};
};

template <std::size_t N>
std::array<double, N> FiniteArray(const Json& value, const char* name) {
  if (!value.is_array() || value.size() != N)
    throw std::runtime_error(std::string(name) + " must contain " + std::to_string(N) + " values");
  std::array<double, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = value.at(i).get<double>();
    if (!std::isfinite(out[i])) throw std::runtime_error(std::string(name) + " must be finite");
  }
  return out;
}

double FiniteNumber(const Json& value, const char* name) {
  const double out = value.get<double>();
  if (!std::isfinite(out)) throw std::runtime_error(std::string(name) + " must be finite");
  return out;
}

int Named(const mjModel* model, mjtObj kind, const char* name) {
  const int id = mj_name2id(model, kind, name);
  if (id < 0) throw std::runtime_error(std::string("missing model object: ") + name);
  return id;
}

fs::path RelativeProjectPath(const fs::path& project, const Json& config, const char* key) {
  const fs::path relative = config.at(key).get<std::string>();
  if (relative.empty() || relative.is_absolute())
    throw std::runtime_error(std::string(key) + " must be a nonempty project-relative path");
  return fs::weakly_canonical(project / relative);
}

Json VectorJson(const double* values, int count) {
  return std::vector<double>(values, values + count);
}

Json ArrayJson(const std::array<double, 5>& value) {
  return std::vector<double>(value.begin(), value.end());
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc != 3) throw std::runtime_error("usage: x4_command_replay INPUT_JSON OUTPUT_DIR");
  const fs::path input_path = fs::absolute(argv[1]);
  const fs::path output = fs::absolute(argv[2]);
  std::ifstream input_stream(input_path);
  if (!input_stream) throw std::runtime_error("cannot open input: " + input_path.string());
  Json config;
  input_stream >> config;
  if (config.at("schema") != "X4_COMMAND_REPLAY/1")
    throw std::runtime_error("unexpected input schema");

  const fs::path project = config.at("project_root").get<std::string>();
  if (!project.is_absolute() || !fs::is_directory(project))
    throw std::runtime_error("project_root must be an existing absolute directory");
  const fs::path scene = RelativeProjectPath(project, config, "scene_path");
  const fs::path motors = RelativeProjectPath(project, config, "motors_path");
  const fs::path variant = RelativeProjectPath(project, config, "robot_variant_path");
  const fs::path chassis = RelativeProjectPath(project, config, "chassis_path");

  const auto initial_xy = FiniteArray<2>(config.at("initial").at("base_xy_m"), "initial.base_xy_m");
  const double initial_yaw = FiniteNumber(config.at("initial").at("base_yaw_rad"), "initial.base_yaw_rad");
  const auto initial_base_velocity =
      FiniteArray<3>(config.at("initial").at("base_world_velocity"), "initial.base_world_velocity");
  const auto initial_q = FiniteArray<5>(config.at("initial").at("q_rad"), "initial.q_rad");
  const auto initial_dq = FiniteArray<5>(config.at("initial").at("dq_rad_s"), "initial.dq_rad_s");
  const double duration = FiniteNumber(config.at("duration_s"), "duration_s");
  if (duration < 0 || duration > 1e6)
    throw std::runtime_error("duration_s must be nonnegative and no greater than 1e6 seconds");
  const std::string base_frame = config.at("base_command_frame").get<std::string>();
  if (base_frame != "world" && base_frame != "body")
    throw std::runtime_error("base_command_frame must be 'world' or 'body'");
  const std::string arm_mode = config.at("arm_mode").get<std::string>();
  if (arm_mode != "hybrid" && arm_mode != "j5_velocity")
    throw std::runtime_error("arm_mode must be 'hybrid' or 'j5_velocity'");
  const std::string arm_start_policy = config.value("arm_start_policy", "immediate");
  if (arm_start_policy != "immediate" && arm_start_policy != "hold_until_fresh_command")
    throw std::runtime_error(
        "arm_start_policy must be 'immediate' or 'hold_until_fresh_command'");
  const bool gravity_feedforward = config.at("gravity_feedforward").get<bool>();

  const Json& command_json = config.at("commands");
  if (!command_json.is_array() || command_json.empty())
    throw std::runtime_error("commands must be a nonempty array");
  std::vector<ReplayCommand> commands;
  commands.reserve(command_json.size());
  for (std::size_t i = 0; i < command_json.size(); ++i) {
    const Json& row = command_json.at(i);
    ReplayCommand command;
    command.t = FiniteNumber(row.at("t_s"), "commands[].t_s");
    command.q = FiniteArray<5>(row.at("q_rad"), "commands[].q_rad");
    command.dq = FiniteArray<5>(row.at("dq_rad_s"), "commands[].dq_rad_s");
    const auto base = FiniteArray<3>(row.at("base_twist"), "commands[].base_twist");
    command.base = {base[0], base[1], base[2]};
    if ((i == 0 && std::abs(command.t) > 1e-12) ||
        (i > 0 && command.t <= commands.back().t))
      throw std::runtime_error("command clock must start at zero and strictly increase");
    if (command.t > duration + 1e-12)
      throw std::runtime_error("command time exceeds duration_s");
    commands.push_back(command);
  }

  if (fs::exists(output)) {
    if (!fs::is_directory(output) || !fs::is_empty(output))
      throw std::runtime_error("output directory must be new or empty: " + output.string());
  } else {
    fs::create_directories(output);
  }
  std::ofstream frames(output / "frames.jsonl");
  if (!frames) throw std::runtime_error("cannot create frames.jsonl");
  std::ofstream(output / "resolved_input.json") << config.dump(2) << '\n';

  SimBackend backend(scene.string(), motors.string(), variant.string(), chassis.string());
  if (!backend.ok() || !backend.reset()) throw std::runtime_error("X4 backend initialization failed");
  const mjModel* model = backend.model();
  mjData* data = backend.data_mut();
  if (backend.motor_config().robot_variant != "x4_950")
    throw std::runtime_error("this replay requires X4_950");
  if (std::abs(model->opt.timestep - 0.001) > 1e-12)
    throw std::runtime_error("X4 command replay requires a 1 ms physics timestep");
  if (!backend.motor_config().arm_servo.enabled)
    throw std::runtime_error("X4 arm servo configuration is required");

  const int root_joint = Named(model, mjOBJ_JOINT, "root");
  if (model->jnt_type[root_joint] != mjJNT_FREE) throw std::runtime_error("root must be a free joint");
  const int root_q = model->jnt_qposadr[root_joint];
  const int root_v = model->jnt_dofadr[root_joint];
  std::array<int, 5> arm_q{}, arm_v{};
  for (int j = 0; j < 5; ++j) {
    const std::string name = "Joint" + std::to_string(j + 1);
    const int joint = Named(model, mjOBJ_JOINT, name.c_str());
    arm_q[j] = model->jnt_qposadr[joint];
    arm_v[j] = model->jnt_dofadr[joint];
  }
  const int racket = Named(model, mjOBJ_GEOM, "racket_face");

  // This is the only state assignment after reset. Missing wheel states are
  // reconstructed as the no-slip planar state matching the measured base twist.
  data->qpos[root_q] = initial_xy[0];
  data->qpos[root_q + 1] = initial_xy[1];
  data->qpos[root_q + 3] = std::cos(0.5 * initial_yaw);
  data->qpos[root_q + 4] = 0;
  data->qpos[root_q + 5] = 0;
  data->qpos[root_q + 6] = std::sin(0.5 * initial_yaw);
  data->qvel[root_v] = initial_base_velocity[0];
  data->qvel[root_v + 1] = initial_base_velocity[1];
  data->qvel[root_v + 5] = initial_base_velocity[2];
  for (int j = 0; j < 5; ++j) {
    data->qpos[arm_q[j]] = initial_q[j];
    data->qvel[arm_v[j]] = initial_dq[j];
  }
  const double cy = std::cos(initial_yaw), sy = std::sin(initial_yaw);
  const Twist initial_body{cy * initial_base_velocity[0] + sy * initial_base_velocity[1],
                           -sy * initial_base_velocity[0] + cy * initial_base_velocity[1],
                           initial_base_velocity[2]};
  const ChassisConfig& chassis_config = backend.chassis_config();
  for (int corner = 0; corner < kNumCorners; ++corner) {
    const double vx = initial_body.vx - initial_body.wz * chassis_config.corner_y[corner];
    const double vy = initial_body.vy + initial_body.wz * chassis_config.corner_x[corner];
    const double steer = std::hypot(vx, vy) > chassis_config.coord.zero_speed
                             ? std::atan2(vy, vx)
                             : 0.0;
    const int steer_axis = AxisSteer(corner);
    const int steer_joint = Named(model, mjOBJ_JOINT, kAxisJointName[steer_axis]);
    data->qpos[model->jnt_qposadr[steer_joint]] = steer;
    const double module_speed = std::hypot(vx, vy);
    double wheel_speed[2];
    SwerveKinematics::DualWheel(module_speed, initial_body.wz, 0.0,
                                chassis_config.dual_offset, wheel_speed);
    const int axes[2] = {AxisWheelL(corner), AxisWheelR(corner)};
    for (int side = 0; side < 2; ++side) {
      const int joint = Named(model, mjOBJ_JOINT, kAxisJointName[axes[side]]);
      const int sign_index = 2 * corner + side;
      data->qvel[model->jnt_dofadr[joint]] =
          chassis_config.wheel_sign[sign_index] * wheel_speed[side] / chassis_config.wheel_radius;
    }
  }
  mj_forward(model, data);

  backend.enable(-1, true);
  ChassisServo chassis_servo(chassis_config, &backend);
  bool servo_state_initialized = false;
  Twist initial_ramped = initial_body;
  if (config.at("initial").contains("chassis_servo_state")) {
    const Json& state = config.at("initial").at("chassis_servo_state");
    if (!state.is_object())
      throw std::runtime_error("initial.chassis_servo_state must be an object");
    if (state.contains("ramped_body_velocity")) {
      const auto ramped = FiniteArray<2>(state.at("ramped_body_velocity"),
                                         "initial.chassis_servo_state.ramped_body_velocity");
      initial_ramped.vx = ramped[0];
      initial_ramped.vy = ramped[1];
    }
    chassis_servo.initialize_consistent_state(initial_ramped);
    servo_state_initialized = true;
  }
  const ArmServoConfig& arm_servo = backend.motor_config().arm_servo;
  const double dt = model->opt.timestep;
  const int physics_steps = static_cast<int>(std::floor(duration / dt + 1e-12));
  const double expected_end = physics_steps * dt;
  std::size_t command_index = 0;
  bool finite = true;
  int time_mismatch_count = 0;
  std::array<int, 3> warning_start = {
      data->warning[mjWARN_BADQPOS].number,
      data->warning[mjWARN_BADQVEL].number,
      data->warning[mjWARN_BADQACC].number};
  std::size_t previous_command_index = 0;
  constexpr double kFreshArmPositionRad = 0.5 * std::acos(-1.0) / 180.0;
  constexpr double kFreshArmVelocityRadS = 0.05;
  constexpr double kStaleArmPositionRad = 2.0 * std::acos(-1.0) / 180.0;
  std::array<bool, 5> arm_axis_hold_candidate{}, arm_axis_pending_initial{};
  std::array<bool, 5> arm_axis_command_active{};
  std::array<double, 5> arm_axis_activation_time{};
  for (int j = 0; j < 5; ++j) {
    arm_axis_hold_candidate[j] =
        std::abs(initial_dq[j]) < 0.1 &&
        std::abs(commands.front().dq[j]) < kFreshArmVelocityRadS &&
        std::abs(commands.front().q[j] - initial_q[j]) > kStaleArmPositionRad;
    arm_axis_pending_initial[j] =
        arm_start_policy == "hold_until_fresh_command" && arm_axis_hold_candidate[j];
    arm_axis_command_active[j] = !arm_axis_pending_initial[j];
    arm_axis_activation_time[j] = arm_axis_command_active[j] ? 0.0 : -1.0;
  }
  const bool arm_hold_eligible =
      std::any_of(arm_axis_hold_candidate.begin(), arm_axis_hold_candidate.end(),
                  [](bool value) { return value; });
  const bool arm_hold_entered =
      std::any_of(arm_axis_pending_initial.begin(), arm_axis_pending_initial.end(),
                  [](bool value) { return value; });

  for (int step = 0; step <= physics_steps; ++step) {
    // mj_step leaves derived geometry at the pre-integration configuration.
    // Refresh kinematics for the state being recorded without changing state.
    mj_forward(model, data);
    const double t = data->time;
    if (std::abs(t - step * dt) > 1e-9) ++time_mismatch_count;
    while (command_index + 1 < commands.size() && commands[command_index + 1].t <= t + 1e-12)
      ++command_index;
    const ReplayCommand& command = commands[command_index];

    for (int j = 0; j < 5; ++j) {
      if (!arm_axis_command_active[j] &&
          (std::abs(command.q[j] - commands.front().q[j]) > kFreshArmPositionRad ||
           std::abs(command.dq[j]) > kFreshArmVelocityRadS)) {
        arm_axis_command_active[j] = true;
        arm_axis_activation_time[j] = t;
      }
    }
    const bool arm_command_active =
        std::all_of(arm_axis_command_active.begin(), arm_axis_command_active.end(),
                    [](bool value) { return value; });

    ArmHybridCmd arm_command;
    for (int j = 0; j < 5; ++j) {
      arm_command.q[j] = arm_axis_command_active[j] ? command.q[j] : initial_q[j];
      arm_command.dq[j] = arm_axis_command_active[j] ? command.dq[j] : 0.0;
      arm_command.kp[j] = arm_servo.kp[j];
      arm_command.kd[j] = arm_servo.kd[j];
    }
    if (arm_axis_command_active[4] && arm_mode == "j5_velocity") arm_command.kp[4] = 0;
    if (gravity_feedforward && !backend.arm_gravity(arm_command.q, arm_command.tau_ff))
      throw std::runtime_error("gravity feedforward failed");
    backend.set_arm_hybrid(arm_command);

    Twist body_command = command.base;
    const BaseState base = backend.base_state();
    if (base_frame == "world") {
      const double c = std::cos(base.yaw), s = std::sin(base.yaw);
      body_command.vx = c * command.base.vx + s * command.base.vy;
      body_command.vy = -s * command.base.vx + c * command.base.vy;
    }
    chassis_servo.set_twist(body_command);
    chassis_servo.update(t);

    std::array<double, 5> q{}, dq{}, raw{}, applied{}, cap{};
    for (int j = 0; j < 5; ++j) {
      const JointReport state = backend.joint(AxisArm(j));
      q[j] = state.q;
      dq[j] = state.dq;
    }
    backend.arm_effort(raw.data(), applied.data(), cap.data());
    const BaseState recorded_base = backend.base_state();
    const double world_vx = std::cos(recorded_base.yaw) * recorded_base.vx -
                            std::sin(recorded_base.yaw) * recorded_base.vy;
    const double world_vy = std::sin(recorded_base.yaw) * recorded_base.vx +
                            std::cos(recorded_base.yaw) * recorded_base.vy;
    const Twist& ramped = chassis_servo.ramped();
    frames << Json{{"t", t},
                   {"qpos", VectorJson(data->qpos, model->nq)},
                   {"qvel", VectorJson(data->qvel, model->nv)},
                   {"base", {recorded_base.x, recorded_base.y, recorded_base.yaw}},
                   {"base_world_velocity", {world_vx, world_vy, recorded_base.wz}},
                   {"q", ArrayJson(q)}, {"dq", ArrayJson(dq)},
                   {"racket_xyz", {data->geom_xpos[3 * racket], data->geom_xpos[3 * racket + 1],
                                      data->geom_xpos[3 * racket + 2]}},
                   {"command_index", command_index}, {"command_t_s", command.t},
                   {"command_age_s", t - command.t},
                   {"torque_interval_start_s", step == 0 ? Json(nullptr) : Json(t - dt)},
                   {"torque_command_index", step == 0 ? Json(nullptr) : Json(previous_command_index)},
                   {"q_command", ArrayJson(command.q)}, {"dq_command", ArrayJson(command.dq)},
                   {"applied_q_command", VectorJson(arm_command.q, 5)},
                   {"applied_dq_command", VectorJson(arm_command.dq, 5)},
                   {"arm_command_active", arm_command_active},
                   {"arm_axis_command_active", arm_axis_command_active},
                   {"base_twist_input", {command.base.vx, command.base.vy, command.base.wz}},
                   {"base_twist_ramped", {ramped.vx, ramped.vy, ramped.wz}},
                   {"tau_raw", ArrayJson(raw)}, {"tau_applied", ArrayJson(applied)},
                   {"tau_cap", ArrayJson(cap)}}.dump() << '\n';

    for (int i = 0; i < model->nq; ++i) finite &= std::isfinite(data->qpos[i]);
    for (int i = 0; i < model->nv; ++i) finite &= std::isfinite(data->qvel[i]);
    if (step < physics_steps) {
      previous_command_index = command_index;
      if (!backend.run(1)) throw std::runtime_error("simulation step failed");
    }
  }
  frames.close();

  const std::array<int, 3> warning_count = {
      data->warning[mjWARN_BADQPOS].number - warning_start[0],
      data->warning[mjWARN_BADQVEL].number - warning_start[1],
      data->warning[mjWARN_BADQACC].number - warning_start[2]};
  const bool time_ok = time_mismatch_count == 0 && std::abs(data->time - expected_end) <= 1e-9;
  Json arm_axis_activation_times = Json::array();
  for (double activation_time : arm_axis_activation_time)
    arm_axis_activation_times.push_back(
        activation_time >= 0 ? Json(activation_time) : Json(nullptr));
  const bool all_arm_axes_active =
      std::all_of(arm_axis_command_active.begin(), arm_axis_command_active.end(),
                  [](bool value) { return value; });
  const double last_arm_activation =
      *std::max_element(arm_axis_activation_time.begin(), arm_axis_activation_time.end());
  const Json summary = {
      {"schema", "X4_COMMAND_REPLAY_RESULT/1"}, {"physics_steps", physics_steps},
      {"integration_dt", dt}, {"duration", data->time}, {"requested_duration", duration},
      {"finite", finite}, {"time_advancement_sane", time_ok},
      {"time_mismatch_count", time_mismatch_count},
      {"warnings", {{"BADQPOS", warning_count[0]}, {"BADQVEL", warning_count[1]},
                     {"BADQACC", warning_count[2]}}},
      {"initial_state_copied", true}, {"wheel_state_inferred", true},
      {"servo_history_unknown", true},
      {"servo_state_initialized_consistently", servo_state_initialized},
      {"servo_initialization",
       {{"source", servo_state_initialized ? "measured_twist_or_input_override" : "cold_start"},
        {"initial_ramped_body_twist", servo_state_initialized
            ? Json{initial_ramped.vx, initial_ramped.vy, initial_ramped.wz} : Json(nullptr)},
        {"wheel_integral_source", servo_state_initialized ? "zero" : "cold_start"}}},
      {"future_measured_state_copied", false},
      {"base_command_frame", base_frame}, {"arm_mode", arm_mode},
      {"arm_start_policy", arm_start_policy},
      {"arm_hold_eligible", arm_hold_eligible},
      {"arm_hold_entered", arm_hold_entered},
      {"arm_axis_hold_candidate", arm_axis_hold_candidate},
      {"arm_axis_pending_initial", arm_axis_pending_initial},
      {"arm_axis_command_active_final", arm_axis_command_active},
      {"arm_axis_activation_times_s", arm_axis_activation_times},
      {"arm_activation_time_s", all_arm_axes_active ? Json(last_arm_activation) : Json(nullptr)},
      {"gravity_feedforward", gravity_feedforward},
      {"notes", {"Wheel steer angles and wheel speeds were reconstructed from the initial planar base twist.",
                   servo_state_initialized
                       ? "Chassis ramp, wheel targets, and steering history were initialized consistently at the recording boundary; the true controller history remains unknown."
                       : "Chassis ramp and controller history before the first recorded command are unavailable.",
                   "Commands use zero-order hold; no measured state is copied after initialization.",
                   arm_start_policy == "hold_until_fresh_command"
                       ? "The arm start hold is an experimental shared control-semantics assumption, not a confirmed hardware mode."
                       : "Arm commands are applied immediately using the legacy replay policy.",
                   "Torque fields describe the preceding physics interval; tick zero has no preceding interval."}}};
  std::ofstream(output / "summary.json") << summary.dump(2) << '\n';
  std::cout << summary.dump(2) << '\n';
  if (!finite || !time_ok || warning_count[0] || warning_count[1] || warning_count[2]) return 2;
  return 0;
} catch (const std::exception& error) {
  std::cerr << "x4_command_replay: " << error.what() << '\n';
  return 1;
}
