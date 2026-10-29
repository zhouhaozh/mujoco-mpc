// Independent X4 path: measured shuttle -> ball-mlp process -> supplied MPC ->
// existing torque-limited simulation backend. No legacy NMPC or impact model.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>
#include "control/chassis_servo.h"
#include "control/native_mpc_bridge.h"
#include "estimation/shuttle_prediction_client.h"
#include "hal/sim_backend.h"

namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace sckd;

namespace {
struct Row { double t; std::array<double, 3> p; };
std::vector<Row> ReadReference(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open trajectory: " + path.string());
  std::vector<Row> rows;
  std::string line;
  double origin = 0;
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::istringstream fields(line);
    Row r;
    if (!(fields >> r.t >> r.p[0] >> r.p[1] >> r.p[2]))
      throw std::runtime_error("invalid trajectory position row");
    if (!std::isfinite(r.t)) throw std::runtime_error("nonfinite trajectory time");
    for (double& p : r.p) {
      if (!std::isfinite(p)) throw std::runtime_error("nonfinite trajectory position");
      p *= .01;
    }
    if (rows.empty()) origin = r.t;
    r.t = (r.t - origin) * .001;
    if (!rows.empty() && r.t <= rows.back().t)
      throw std::runtime_error("trajectory clock must increase");
    rows.push_back(r);
  }
  if (rows.size() < 2) throw std::runtime_error("trajectory needs two or more points");
  return rows;
}

void SetReference(const std::vector<Row>& rows, double t, mjData* d, int qa, int va) {
  const auto it = std::upper_bound(rows.begin(), rows.end(), t,
      [](double x, const Row& r) { return x < r.t; });
  const size_t i = std::clamp<size_t>(it - rows.begin(), 1, rows.size() - 1);
  const Row& a = rows[i - 1]; const Row& b = rows[i];
  const double u = std::clamp((t - a.t) / (b.t - a.t), 0.0, 1.0);
  for (int k = 0; k < 3; ++k) {
    d->qpos[qa + k] = a.p[k] + u * (b.p[k] - a.p[k]);
    d->qvel[va + k] = t >= 0 && t < rows.back().t ? (b.p[k] - a.p[k]) / (b.t - a.t) : 0;
    d->qvel[va + 3 + k] = 0;
  }
}

int Named(const mjModel* m, mjtObj kind, const char* name) {
  const int id = mj_name2id(m, kind, name);
  if (id < 0) throw std::runtime_error(std::string("missing model object: ") + name);
  return id;
}

double SurfaceGap(const mjModel* m, const mjData* d, int face, int ball_qa, double radius) {
  double squared = 0;
  for (int axis = 0; axis < 3; ++axis) {
    double local = 0;
    for (int k = 0; k < 3; ++k)
      local += d->geom_xmat[9 * face + 3 * k + axis] *
          (d->qpos[ball_qa + k] - d->geom_xpos[3 * face + k]);
    const double outside = std::max(std::abs(local) - m->geom_size[3 * face + axis], 0.0);
    squared += outside * outside;
  }
  return std::sqrt(squared) - radius;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 4 && argc != 5)
    throw std::runtime_error(
        "usage: x4_native_replay CONFIG OUTPUT_DIR PREDICTOR_SOCKET [REALTIME_SPEED]");
  const fs::path config_path = fs::absolute(argv[1]);
  const fs::path output = fs::absolute(argv[2]);
  const std::string socket = argv[3];
  const bool realtime = argc == 5;
  double realtime_speed = 1.0;
  if (realtime) {
    const std::string value = argv[4];
    std::size_t consumed = 0;
    realtime_speed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(realtime_speed) || realtime_speed <= 0.0)
      throw std::runtime_error("REALTIME_SPEED must be a finite positive number");
  }
  std::ifstream config_in(config_path);
  Json cfg; config_in >> cfg;
  if (cfg.at("schema") != "X4_NATIVE_MPC_REPLAY/1") throw std::runtime_error("unexpected config schema");
  const fs::path root = fs::weakly_canonical(config_path.parent_path() / cfg.at("project_root").get<std::string>());
  auto path = [&](const char* key) { return fs::absolute(root / cfg.at(key).get<std::string>()).string(); };
  const auto rows = ReadReference(path("trajectory_path"));
  fs::create_directories(output);
  // The supplied library generates nlp.c during construction. Keep it with this run.
  fs::current_path(output);
  std::ofstream trace(output / "frames.jsonl"), events(output / "events.jsonl");
  if (!trace || !events) throw std::runtime_error("cannot open run outputs");
  std::ofstream(output / "resolved_config.json") << cfg.dump(2) << '\n';

  SimBackend be(path("scene_path"), path("motors_path"), path("robot_variant_path"), path("chassis_path"));
  if (!be.ok() || !be.reset()) throw std::runtime_error("X4 backend initialization failed");
  const mjModel* m = be.model(); mjData* d = be.data_mut();
  if (be.motor_config().robot_variant != "x4_950") throw std::runtime_error("this entry requires X4_950");
  const int ball_j = Named(m, mjOBJ_JOINT, "ball_free");
  const int ball_qa = m->jnt_qposadr[ball_j], ball_va = m->jnt_dofadr[ball_j];
  const int ball_g = Named(m, mjOBJ_GEOM, "ball_geom"), face = Named(m, mjOBJ_GEOM, "racket_face");
  if (m->geom_contype[ball_g] || m->geom_conaffinity[ball_g])
    throw std::runtime_error("measured ball must have physical collision disabled");
  const auto initial_base = cfg.at("initial_base_xy_m").get<std::array<double, 2>>();
  const auto signs = cfg.at("mpc_joint_sign").get<std::array<double, 5>>();
  const auto offsets = cfg.at("mpc_joint_offset_rad").get<std::array<double, 5>>();
  const auto vmax = cfg.at("arm_velocity_limit_rad_s").get<std::array<double, 5>>();
  const auto amax = cfg.at("arm_acceleration_limit_rad_s2").get<std::array<double, 5>>();
  const auto window = cfg.at("unlimited_joint_position_window_rad").get<std::array<double, 2>>();
  const auto& arm = be.motor_config().arm_servo;
  if (!arm.enabled) throw std::runtime_error("independent X4 servo config required");
  std::array<int, 5> joints{};
  std::array<double, 5> qcmd = arm.standby_q, dqcmd{}, lo{}, hi{}, qstart{};
  for (int j = 0; j < 5; ++j) {
    if (std::abs(signs[j]) != 1 || !std::isfinite(offsets[j]) || !(vmax[j] > 0) || !(amax[j] > 0))
      throw std::runtime_error("invalid joint mapping or execution limits");
    joints[j] = Named(m, mjOBJ_JOINT, kAxisJointName[AxisArm(j)]);
    lo[j] = m->jnt_limited[joints[j]] ? m->jnt_range[2*joints[j]] : window[0];
    hi[j] = m->jnt_limited[joints[j]] ? m->jnt_range[2*joints[j]+1] : window[1];
    if (qcmd[j] < lo[j] || qcmd[j] > hi[j]) throw std::runtime_error("standby outside joint limits");
    d->qpos[m->jnt_qposadr[joints[j]]] = qcmd[j];
    qstart[j] = qcmd[j];
  }
  d->qpos[0] = initial_base[0]; d->qpos[1] = initial_base[1];
  SetReference(rows, -1, d, ball_qa, ball_va);
  mj_forward(m, d);
  ChassisServo servo(be.chassis_config(), &be);
  be.enable(-1, true);
  ShuttlePredictionClient predictor(socket);
  NativeMpcBridge controller(path("native_mpc_library"));
  controller.SetHitParams({cfg.at("hit_speed_m_s"), cfg.at("hit_yaw_deg"), cfg.at("hit_pitch_deg"), cfg.at("hit_vertical_speed_m_s")});

  const double dt = m->opt.timestep;
  const double settle = cfg.at("settle_s"), stop = cfg.at("stop_s");
  const int predict_ticks = cfg.at("prediction_period_ms"), mpc_ticks = cfg.at("mpc_period_ms");
  if (std::abs(dt - .001) > 1e-12 || predict_ticks < 1 || mpc_ticks < 1 || settle < 0 || stop < 0)
    throw std::runtime_error("invalid schedule");
  const double end = settle + rows.back().t + stop;
  const int ticks = static_cast<int>(std::ceil(end / dt));
  const double minimum_remaining = cfg.at("minimum_remaining_s");
  const double base_speed = cfg.at("base_speed_limit_m_s");
  NativeMpcPlan plan;
  double plan_start = 0, plan_end = -1, min_gap = std::numeric_limits<double>::infinity();
  double max_joint_motion = 0, max_mpc_joint_motion = 0, max_solve_ms = 0, max_prediction_ms = 0;
  std::array<double, 5> first_execution_q{};
  int attempts = 0, successes = 0, command_ticks = 0, clipped = 0, predictions = 0;
  bool geometric_contact = false;
  Json forecast = {{"target", {{"valid", false}}},
                   {"predicted_trajectory", Json::array()},
                   {"stats", Json::object()}};
  const auto wall_start = std::chrono::steady_clock::now();
  for (int tick = 0; tick <= ticks; ++tick) {
    const double t = d->time, replay_t = t - settle;
    const bool active = replay_t >= -1e-10 && replay_t <= rows.back().t + 1e-10;
    SetReference(rows, replay_t, d, ball_qa, ball_va);
    mj_forward(m, d);
    if (active && (tick % predict_ticks == 0 || replay_t + dt > rows.back().t)) {
      forecast = predictor.Update(std::clamp(replay_t, 0.0, rows.back().t));
      if (forecast.at("target").at("valid").get<bool>()) ++predictions;
      max_prediction_ms = std::max(max_prediction_ms, forecast.at("target").value("prediction_ms", 0.0));
    }
    const BaseState base = be.base_state();
    const auto& current_target = forecast.at("target");
    const bool target_current = active && current_target.at("valid").get<bool>() &&
        current_target.at("t_hit_s").get<double>() > replay_t &&
        current_target.at("t_hit_s").get<double>() <= rows.back().t;
    // A failed solve may retain a valid prior plan, but an invalid/expired
    // observation must stop that plan. Execution limits ramp velocity to zero.
    if (!target_current) plan.success = false;
    if (target_current && tick % mpc_ticks == 0) {
      const auto& target = forecast.at("target");
      const double hit_t = target.at("t_hit_s");
      const double remaining = hit_t - replay_t;
      if (remaining >= minimum_remaining && hit_t <= rows.back().t) {
        std::array<double, 14> state{};
        state[0] = base.x - initial_base[0]; state[1] = base.y - initial_base[1];
        state[7] = std::cos(base.yaw)*base.vx - std::sin(base.yaw)*base.vy;
        state[8] = std::sin(base.yaw)*base.vx + std::cos(base.yaw)*base.vy;
        for (int j = 0; j < 5; ++j) {
          const auto actual = be.joint_measured(AxisArm(j));
          state[2+j] = signs[j]*actual.q + offsets[j]; state[9+j] = signs[j]*actual.dq;
        }
        auto goal = target.at("p_hit_m").get<std::array<double, 3>>();
        goal[0] -= initial_base[0]; goal[1] -= initial_base[1];
        auto next = controller.Solve(state, goal, remaining);
        ++attempts; max_solve_ms = std::max(max_solve_ms, next.solve_ms);
        events << Json{{"kind", "solve"}, {"t", t}, {"replay_t", replay_t}, {"target", target},
            {"state", state}, {"goal_local", goal}, {"success", next.success},
            {"solver_success", next.solver_success}, {"accuracy_success", next.accuracy_success},
            {"control_semantics", "acceleration"}, {"accelerations", next.accelerations},
            {"segment_dt_s", next.segment_dt_s},
            {"solve_ms", next.solve_ms}, {"error", next.error}, {"velocities", next.velocities}}.dump() << '\n';
        if (next.success) { plan = next; plan_start = t; plan_end = settle + hit_t; ++successes; }
      }
    }
    std::array<double, 7> desired{};
    const bool executing = target_current && plan.success && t < plan_end;
    if (executing) {
      if (!command_ticks) for (int j = 0; j < 5; ++j) first_execution_q[j] = be.joint(AxisArm(j)).q;
      desired = plan.VelocityAt(t-plan_start); ++command_ticks;
    }
    ArmHybridCmd cmd;
    for (int j = 0; j < 5; ++j) {
      const double raw = signs[j] * desired[2+j];
      double target_dq = std::clamp(raw, -vmax[j], vmax[j]);
      if (raw != target_dq) ++clipped;
      dqcmd[j] += std::clamp(target_dq-dqcmd[j], -amax[j]*dt, amax[j]*dt);
      const double next_q = std::clamp(qcmd[j] + dqcmd[j]*dt, lo[j], hi[j]);
      dqcmd[j] = (next_q-qcmd[j])/dt; qcmd[j] = next_q;
      cmd.q[j] = qcmd[j]; cmd.dq[j] = dqcmd[j]; cmd.kp[j] = arm.kp[j]; cmd.kd[j] = arm.kd[j];
      max_joint_motion = std::max(max_joint_motion, std::abs(be.joint(AxisArm(j)).q-qstart[j]));
      if (executing) max_mpc_joint_motion = std::max(max_mpc_joint_motion,
          std::abs(be.joint(AxisArm(j)).q-first_execution_q[j]));
    }
    if (!be.arm_gravity(cmd.q, cmd.tau_ff)) throw std::runtime_error("gravity feedforward failed");
    be.set_arm_hybrid(cmd);
    const double vx = std::clamp(desired[0], -base_speed, base_speed);
    const double vy = std::clamp(desired[1], -base_speed, base_speed);
    servo.set_twist({std::cos(base.yaw)*vx + std::sin(base.yaw)*vy,
        -std::sin(base.yaw)*vx + std::cos(base.yaw)*vy, std::clamp(-base.yaw, -.2, .2)});
    servo.update(t);
    const double gap = SurfaceGap(m, d, face, ball_qa, m->geom_size[3*ball_g]);
    if (active) { min_gap = std::min(min_gap, gap); geometric_contact |= gap <= 0; }
    for (int j = 0; j < m->nq; ++j) if (!std::isfinite(d->qpos[j])) throw std::runtime_error("nonfinite qpos");
    for (int j = 0; j < m->nv; ++j) if (!std::isfinite(d->qvel[j])) throw std::runtime_error("nonfinite qvel");
    if (tick % 5 == 0 || tick == ticks) {
      trace << Json{{"t", t}, {"replay_t", replay_t}, {"phase", active ? "replay" : replay_t < 0 ? "settle" : "stop"},
          {"qpos", std::vector<double>(d->qpos, d->qpos+m->nq)}, {"qvel", std::vector<double>(d->qvel, d->qvel+m->nv)},
          {"target", forecast.at("target")}, {"predicted_trajectory", forecast.at("predicted_trajectory")},
          {"input", forecast.at("stats")}, {"executing", executing},
          {"q_command", qcmd}, {"dq_command", dqcmd}, {"surface_gap_m", gap}}.dump() << '\n';
      if (realtime) trace.flush();
    }
    if (tick < ticks && !be.run(1)) throw std::runtime_error("simulation step failed");
    if (realtime && tick < ticks) {
      const auto deadline = wall_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(d->time / realtime_speed));
      std::this_thread::sleep_until(deadline);
    }
  }
  be.enable(-1, false); predictor.Close();
  const bool passed = successes > 0 && command_ticks > 0 && predictions > 0 && max_mpc_joint_motion > 1e-5;
  const Json summary = {{"schema", "X4_NATIVE_REPLAY_RESULT/1"}, {"pipeline_passed", passed},
      {"scene", path("scene_path")}, {"trajectory", path("trajectory_path")}, {"native_mpc", path("native_mpc_library")},
      {"simulated_s", d->time}, {"physics_steps", be.tick()}, {"solve_attempts", attempts}, {"solve_successes", successes},
      {"valid_predictions", predictions}, {"mpc_execution_ticks", command_ticks}, {"velocity_clip_count", clipped},
      {"max_joint_motion_rad", max_joint_motion}, {"minimum_surface_gap_m", min_gap}, {"geometric_contact", geometric_contact},
      {"max_mpc_joint_motion_rad", max_mpc_joint_motion}, {"input_policy", cfg.value("preassociated_replay", false) ? "preassociated_forced" : "normal_association"},
      {"physical_hit", nullptr}, {"ball_mode", "measured_reference_no_collision_no_bounce"}, {"end_reason", "recording_end_then_stop"},
      {"max_solve_ms", max_solve_ms}, {"max_prediction_ms", max_prediction_ms}, {"input_stats", forecast.at("stats")},
      {"execution_mode", realtime ? "wall_clock_paced" : "unpaced"},
      {"realtime_speed_multiplier", realtime ? Json(realtime_speed) : Json(nullptr)},
      {"wall_s", std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()}};
  std::ofstream(output / "summary.json") << summary.dump(2) << '\n';
  std::cout << summary.dump(2) << '\n';
  return passed ? 0 : 2;
} catch (const std::exception& e) {
  std::cerr << "x4_native_replay: " << e.what() << '\n';
  return 1;
}
