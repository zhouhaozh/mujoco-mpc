// 配置契约检查：不连接设备、不读取模型、不运行物理或求解器。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "runtime/runtime_config.h"
#include "runtime/contracts.h"

using namespace sckd::runtime;

namespace {

Environment Env(std::map<std::string, std::string> values = {}) {
  return [values = std::move(values)](const std::string& key) -> std::optional<std::string> {
    const auto found = values.find(key);
    if (found == values.end()) return std::nullopt;
    return found->second;
  };
}

std::vector<std::string> Args(const std::string& mode = "pfly") {
  return {"m2_intercept", "scene.xml", "1", "qp", "0.25", "2026", mode, "fixed", "xb"};
}

ConfigReader Document(Json document) {
  return [document = std::move(document)](const std::string&) { return document; };
}

}  // namespace

TEST_CASE("runtime defaults retain the old main entry point") {
  const auto config = ResolvedConfig::Resolve({"m2_intercept"}, Env());
  const auto& v = config.values();
  const auto& m = config.modes();
  CHECK(v.scene_path == "models/generated/x4_tennis_demo.mjcf");
  CHECK(v.trials == 10);
  CHECK(v.init_delay_s == doctest::Approx(.25));
  CHECK(v.seed == 2026);
  CHECK(v.base_mode.empty());
  CHECK(v.control_variant == "baseline");
  CHECK(v.steer_rate_mode == "limited");
  CHECK(v.arm_feedforward == "preserve");
  CHECK_FALSE(v.robot_io_log_prefix.has_value());
  CHECK(v.nmpc6_speed_factor == doctest::Approx(.8));
  CHECK(v.nmpc6_nlp_path == "third_party/nmpc6/nlp.so");
  CHECK_FALSE(m.use_qp);
  CHECK_FALSE(m.fly);
  CHECK_FALSE(m.corrected);
  CHECK_FALSE(m.remote_predictor);
  CHECK_FALSE(m.deduplicate_obs);
  CHECK(m.sim_joint23_limits);
  CHECK(m.direction_limited);
  CHECK(m.feedforward_limited);
  CHECK_FALSE(v.control_diagnostics_path.has_value());
  CHECK(config.sources().at("scene_path") == "default");
  CHECK(config.sources().at("arm_feedforward") == "default");
  CHECK(config.ToJson().at("values").at("robot_io_log_prefix").is_null());
  // 配置对象通过const访问器暴露字段，集成处无需散落可变bool。
  static_assert(std::is_same<decltype(config.values()), const RuntimeValues&>::value, "values must be immutable to callers");
}

TEST_CASE("direct baseline and runner explicit limited retain distinct provenance") {
  for (const std::string mode : {"pfly", "nmpc6"}) {
    const auto direct = ResolvedConfig::Resolve(Args(mode), Env());
    CHECK(direct.values().steer_rate_mode == "both_unlimited");
    CHECK_FALSE(direct.modes().direction_limited);
    CHECK_FALSE(direct.modes().feedforward_limited);
    CHECK(direct.sources().at("steer_rate_mode") == "derived:control_variant+base_mode");
    const auto runner = ResolvedConfig::Resolve(Args(mode), Env({{"M2_STEER_RATE_MODE", "limited"}}));
    CHECK(runner.values().steer_rate_mode == "limited");
    CHECK(runner.modes().direction_limited);
    CHECK(runner.modes().feedforward_limited);
    CHECK(runner.sources().at("steer_rate_mode") == "env:M2_STEER_RATE_MODE");
  }
  const auto wb = ResolvedConfig::Resolve(Args("wbmpc"), Env());
  CHECK(wb.values().steer_rate_mode == "limited");
}

TEST_CASE("legacy positional parsing preserves atoi atof and unrecognized selectors") {
  auto args = Args("unrecognized_base_mode");
  args[2] = "2trials";
  args[3] = "QP";
  args[4] = ".35seconds";
  args[5] = "-7seed";
  args[7] = "unrecognized_serve";
  args[8] = "unrecognized_fly";
  args.push_back("historically_ignored");
  const auto config = ResolvedConfig::Resolve(args, Env());
  CHECK(config.values().trials == 2);
  CHECK(config.values().init_delay_s == doctest::Approx(.35));
  CHECK(config.values().seed == -7);
  CHECK_FALSE(config.modes().use_qp);
  CHECK_FALSE(config.modes().fly);
  CHECK_FALSE(config.modes().rand_serve);
  CHECK_FALSE(config.modes().gs_serve);
  CHECK_FALSE(config.modes().cross_bounce);
  CHECK(config.modes().fly_rule == 0);
  CHECK(config.ToJson().at("ignored_positional").at(0) == "historically_ignored");
}

TEST_CASE("environment flags retain presence semantics including zero and empty") {
  const auto config = ResolvedConfig::Resolve(Args(), Env({
    {"M2_VIEW", "0"}, {"BALL_DEDUP", ""}, {"NMPC6_LEGACY_LIMITS", "0"},
    {"N6_TRACE", ""}, {"WB_CHASSIS", "0"}, {"N6_SPD", ".75tail"},
    {"BALL_PREDICTOR_SOCKET", ""}, {"M2_MEASURED_REPLAY", ""},
    {"M2_CONTROL_DIAGNOSTICS", ""}, {"M2_VIDEO_TRACE", ""}, {"M2_TIMING_PATH", ""},
    {"M2_SERVO_IDENTIFICATION", ""}}));
  CHECK(config.values().viewer_enabled);
  CHECK(config.values().nmpc6_trace);
  CHECK(config.values().wb_chassis_trace);
  CHECK(config.modes().deduplicate_obs);
  CHECK_FALSE(config.modes().sim_joint23_limits);
  CHECK_FALSE(config.modes().remote_predictor);
  CHECK_FALSE(config.modes().measured_replay);
  CHECK(config.values().nmpc6_speed_factor == doctest::Approx(.75));
  REQUIRE(config.values().control_diagnostics_path.has_value());
  CHECK(config.values().control_diagnostics_path->empty());
  CHECK(config.values().servo_identification_path.empty());
  CHECK(config.ToJson().at("values").at("video_trace_path") == "");
  CHECK(config.ToJson().at("environment").at("M2_VIEW") == "0");
}

TEST_CASE("JSON selection and overrides follow a single documented precedence") {
  Json document = {{"schema", "SCKD_RUNTIME_CONFIG/1"}, {"values", {
    {"scene_path", "json_scene.xml"}, {"base_mode", "wbmpc"}, {"seed", 41},
    {"trials", 4}, {"control_variant", "feasible"}, {"steer_rate_mode", "both_unlimited"},
    {"predictor_socket_path", "relative/predictor.sock"}, {"viewer_enabled", false}}}};
  auto args = Args("nmpc6");
  args.insert(args.begin() + 3, {"--config", "chosen.json"});
  std::string loaded;
  const auto config = ResolvedConfig::Resolve(args, Env({
    {"M2_CONFIG_PATH", "ignored.json"}, {"M2_CONTROL_VARIANT", "gains"},
    {"M2_STEER_RATE_MODE", "limited"}, {"M2_VIEW", "0"}}),
    [&](const std::string& path) { loaded = path; return document; });
  CHECK(loaded == "chosen.json");
  CHECK(config.config_path_source() == "argv:--config");
  CHECK(config.values().scene_path == "scene.xml");
  CHECK(config.values().trials == 1);
  CHECK(config.values().seed == 2026);
  CHECK(config.values().base_mode == "nmpc6");
  CHECK(config.values().control_variant == "gains");
  CHECK(config.values().steer_rate_mode == "limited");
  CHECK(config.values().viewer_enabled);
  CHECK(config.modes().identified_gains);
  CHECK(config.modes().remote_predictor);
  CHECK(config.modes().deduplicate_obs);
  CHECK(*config.values().predictor_socket_path == "relative/predictor.sock");
  CHECK(config.sources().at("predictor_socket_path") == "json:chosen.json#values.predictor_socket_path");
  CHECK(config.sources().at("scene_path") == "argv[1]");
  CHECK(config.sources().at("seed") == "argv[7]");  // --config占据原argv[3..4]。
  CHECK(config.ToJson().at("path_base") == "process_cwd");
}

TEST_CASE("environment selected JSON can opt in while absent configuration is legacy") {
  const Json document = {{"values", {{"base_mode", "pfly"}, {"fly_mode", "v2"}, {"swing_mode", "qp"},
    {"motors_path", "relative/motors.json"}, {"nmpc6_nlp_path", "pinned/nlp.so"}}}};
  const auto config = ResolvedConfig::Resolve({"m2_intercept"}, Env({{"M2_CONFIG_PATH", "runtime.json"}}), Document(document));
  CHECK(config.config_path_source() == "env:M2_CONFIG_PATH");
  CHECK(config.modes().fly_rule == 1);
  CHECK(config.modes().use_qp);
  CHECK(config.values().steer_rate_mode == "both_unlimited");
  CHECK(config.values().motors_path == "relative/motors.json");
  CHECK(config.values().nmpc6_nlp_path == "pinned/nlp.so");
}

TEST_CASE("resolved JSON can replay final values without replaying captured environment") {
  const auto original = ResolvedConfig::Resolve(Args("nmpc6"), Env({
    {"M2_CONTROL_VARIANT", "servo_mpc"}, {"M2_VIEW", "1"}, {"BALL_DEDUP", "1"},
    {"M2_ARM_FEEDFORWARD", "zero"}, {"M2_ROBOT_IO_LOG_PREFIX", "run/commands"}}));
  const auto document = original.ToJson();
  const auto replayed = ResolvedConfig::Resolve({"m2_intercept", "--config=resolved.json"}, Env(), Document(document));
  CHECK(replayed.ToJson().at("values") == document.at("values"));
  CHECK(replayed.ToJson().at("derived") == document.at("derived"));
  CHECK(replayed.ToJson().at("environment").empty());
  CHECK(replayed.values().nmpc6_nlp_path == "third_party/nmpc6/nlp_servo.so");
  CHECK(replayed.values().arm_feedforward == "zero");
  CHECK(replayed.values().robot_io_log_prefix == "run/commands");
}

TEST_CASE("robot IO settings use JSON then environment and retain provenance") {
  const Json document = {{"values", {{"arm_feedforward", "zero"},
    {"robot_io_log_prefix", "json/commands"}}}};
  const std::vector<std::string> args{"m2_intercept", "--config=runtime.json"};
  const auto from_json = ResolvedConfig::Resolve(args, Env(), Document(document));
  CHECK(from_json.values().arm_feedforward == "zero");
  CHECK(from_json.values().robot_io_log_prefix == "json/commands");
  CHECK(from_json.sources().at("arm_feedforward") == "json:runtime.json#values.arm_feedforward");
  CHECK(from_json.sources().at("robot_io_log_prefix") == "json:runtime.json#values.robot_io_log_prefix");
  // 空日志前缀是显式关闭；空前馈模式不是默认值，另一个用例要求拒绝。
  const auto overridden = ResolvedConfig::Resolve(args, Env({
    {"M2_ARM_FEEDFORWARD", "preserve"}, {"M2_ROBOT_IO_LOG_PREFIX", ""}}), Document(document));
  CHECK(overridden.values().arm_feedforward == "preserve");
  REQUIRE(overridden.values().robot_io_log_prefix.has_value());
  CHECK(overridden.values().robot_io_log_prefix->empty());
  CHECK(overridden.sources().at("arm_feedforward") == "env:M2_ARM_FEEDFORWARD");
  CHECK(overridden.sources().at("robot_io_log_prefix") == "env:M2_ROBOT_IO_LOG_PREFIX");
  CHECK(overridden.ToJson().at("environment").at("M2_ARM_FEEDFORWARD") == "preserve");
  CHECK(overridden.ToJson().at("environment").at("M2_ROBOT_IO_LOG_PREFIX") == "");
}

TEST_CASE("feedforward mode is explicit and changes resolved experiment values") {
  const auto baseline = ResolvedConfig::Resolve(Args(), Env());
  const auto zero = ResolvedConfig::Resolve(Args(), Env({{"M2_ARM_FEEDFORWARD", "zero"}}));
  auto expected = baseline.ToJson().at("values");
  expected["arm_feedforward"] = "zero";
  CHECK(zero.ToJson().at("values") == expected);
  CHECK(zero.ToJson().at("values") != baseline.ToJson().at("values"));
  CHECK(zero.ToJson().at("derived") == baseline.ToJson().at("derived"));
  const auto logged = ResolvedConfig::Resolve(Args(), Env({{"M2_ROBOT_IO_LOG_PREFIX", "run/commands"}}));
  expected = baseline.ToJson().at("values");
  expected["robot_io_log_prefix"] = "run/commands";
  CHECK(logged.ToJson().at("values") == expected);
}

TEST_CASE("robot IO configuration rejects ambiguous modes and invalid types") {
  for (const std::string mode : {"", "ZERO", "disabled", "zero "}) {
    CHECK_THROWS_AS(ResolvedConfig::Resolve(Args(), Env({{"M2_ARM_FEEDFORWARD", mode}})), std::runtime_error);
    const Json document = {{"values", {{"arm_feedforward", mode}}}};
    CHECK_THROWS_AS(ResolvedConfig::Resolve({"m2_intercept", "--config=bad.json"}, Env(), Document(document)), std::runtime_error);
  }
  for (const Json document : {
      Json{{"values", {{"arm_feedforward", nullptr}}}},
      Json{{"values", {{"arm_feedforward", true}}}},
      Json{{"values", {{"robot_io_log_prefix", 42}}}}}) {
    CHECK_THROWS_AS(ResolvedConfig::Resolve({"m2_intercept", "--config=bad.json"}, Env(), Document(document)), std::runtime_error);
  }
  const Json no_log = {{"values", {{"robot_io_log_prefix", nullptr}}}};
  const auto config = ResolvedConfig::Resolve({"m2_intercept", "--config=runtime.json"}, Env(), Document(no_log));
  CHECK_FALSE(config.values().robot_io_log_prefix.has_value());
}

TEST_CASE("chassis candidates and variants preserve derived capabilities") {
  const auto chassis = ResolvedConfig::Resolve(Args("nmpc6"), Env({{"M2_CONTROL_VARIANT", "chassis_guard"}}));
  CHECK(chassis.modes().chassis_model);
  CHECK(chassis.modes().chassis_recovery);
  CHECK(chassis.modes().chassis_guard);
  CHECK(chassis.modes().identified_gains);
  CHECK(chassis.values().nmpc6_nlp_path == "third_party/nmpc6/nlp_chassis.so");
  CHECK_THROWS_AS(ResolvedConfig::Resolve(Args("pfly"), Env({{"M2_CONTROL_VARIANT", "chassis_model"}})), std::runtime_error);
  const auto feasible = ResolvedConfig::Resolve(Args(), Env({{"M2_CONTROL_VARIANT", "feasible"}}));
  CHECK(feasible.modes().corrected);
  CHECK_FALSE(feasible.modes().coherent);
  CHECK_FALSE(feasible.modes().inverse_ff);
  const auto clock = ResolvedConfig::Resolve(Args(), Env({{"M2_CONTROL_VARIANT", "clock"}}));
  CHECK(clock.modes().coherent);
  CHECK_FALSE(clock.modes().inverse_ff);
}

TEST_CASE("old fatal configuration cases and new JSON errors remain explicit") {
  CHECK_THROWS_AS(ResolvedConfig::Resolve({"m2_intercept"}, Env({{"M2_MEASURED_REPLAY", "input.csv"}})), std::runtime_error);
  CHECK_THROWS_AS(ResolvedConfig::Resolve(Args(), Env({{"M2_CONTROL_VARIANT", ""}})), std::runtime_error);
  CHECK_THROWS_AS(ResolvedConfig::Resolve(Args(), Env({{"M2_STEER_RATE_MODE", ""}})), std::runtime_error);
  CHECK_THROWS_AS(ResolvedConfig::Resolve({"m2_intercept", "--config"}, Env()), std::runtime_error);
  CHECK_THROWS_AS(ResolvedConfig::Resolve({"m2_intercept", "--config="}, Env()), std::runtime_error);
  CHECK_THROWS_AS(ResolvedConfig::Resolve({"m2_intercept", "--config=a", "--config=b"}, Env()), std::runtime_error);
  for (const Json document : {
      Json{{"values", {{"unknown", 1}}}},
      Json{{"values", {{"viewer_enabled", "0"}}}},
      Json{{"values", {{"trials", 1.5}}}},
      Json{{"values", {{"trials", 18446744073709551615ull}}}},
      Json{{"values", {{"nmpc6_speed_factor", "0.8"}}}},
      Json{{"schema", "UNKNOWN"}, {"values", Json::object()}}}) {
    CHECK_THROWS_AS(ResolvedConfig::Resolve({"m2_intercept", "--config=bad.json"}, Env(), Document(document)), std::runtime_error);
  }
}

TEST_CASE("time domains and observation capture remain explicit") {
  const TimeInterval interval{{1., "simulation"}, {2., "simulation"}};
  CHECK(interval.Contains({1.5, "simulation"}));
  CHECK_FALSE(interval.Contains({1.5, "camera"}));
  CHECK_FALSE(interval.Contains({2.1, "simulation"}));
  sckd::BallObs old;
  old.t = 1.; old.p[0] = .5;
  ObservationIdentity identity;
  identity.track_id = "ball-1";
  identity.observation_seq = 7;
  identity.captured.clock_id = "capture";
  identity.received = {1.015, "receive"};
  const auto observation = BallObservation::FromLegacy(old, identity);
  CHECK(observation.valid);
  CHECK(observation.identity.captured.seconds == 1.);
  CHECK(observation.identity.captured.clock_id == "capture");
  CHECK(observation.identity.received.seconds == 1.015);
  CHECK(observation.identity.received.clock_id == "receive");
  CHECK(observation.position_world[0] == .5);
}

TEST_CASE("time window experiments are explicit validated and replayable") {
  TimeWindowPolicy old;
  CHECK(old.planner_min_lead_s == .9);
  CHECK_FALSE(old.pfly_init_delay_s.has_value());
  CHECK(old.nmpc6.Durations(1.1)[0] == 1.1 / 3.);
  CHECK(TimeWindowPolicy::FromJson(old.ToJson()).ToJson() == old.ToJson());
  Json policy = old.ToJson();
  policy["planner_min_lead_s"] = .6;
  policy["pfly_init_delay_s"] = .1;
  policy["refine_before_arm_active"] = true;
  policy["nmpc6"]["dt_ratios"] = {.5, .3, .2};
  policy["nmpc6"]["allow_nonuniform_dt"] = true;
  const auto chosen = ResolvedConfig::Resolve({"m2_intercept"},
    Env({{"M2_TIME_WINDOWS_PATH", "experiment.json"}, {"M2_NMPC6_NLP", "isolated/nlp.so"}}),
    [policy](const std::string& path) { CHECK(path == "experiment.json"); return policy; });
  CHECK(chosen.values().time_windows.planner_min_lead_s == .6);
  CHECK(chosen.values().time_windows.refine_before_arm_active);
  CHECK(chosen.values().time_windows.nmpc6.Durations(1.)[2] == .2);
  CHECK(chosen.values().time_windows_path == "experiment.json");
  CHECK(chosen.values().nmpc6_nlp_path == "isolated/nlp.so");
  CHECK(chosen.ToJson()["values"]["time_windows"] == policy);
  CHECK(chosen.sources().at("time_windows") == "file:experiment.json");
}

TEST_CASE("time policy rejects misspellings unsafe ratios and invalid windows") {
  const auto valid = TimeWindowPolicy().ToJson();
  for (const auto& change : std::vector<Json>{
      Json{{"target_lock_s", -1}}, Json{{"planner_min_lead_s", 3}},
      Json{{"pfly_duration_min_s", 0}}, Json{{"pfly_ready_transition_s", -1}},
      Json{{"pfly_init_delay_s", -1}}, Json{{"refine_before_arm_active", 1}},
      Json{{"typo", .1}}, Json{{"nmpc6", {{"dt_ratios", {.5, .3, .2}}}}},
      Json{{"nmpc6", {{"dt_ratios", {.5, .3, .1}}, {"allow_nonuniform_dt", true}}}},
      Json{{"nmpc6", {{"stop_s", -1}}}}}) {
    auto policy = valid;
    policy.merge_patch(change);
    CHECK_THROWS_AS(TimeWindowPolicy::FromJson(policy), std::runtime_error);
  }
}

TEST_CASE("solve report separates success acceptance and application") {
  SolveReport report;
  report.solver_status = "Solve_Succeeded";
  report.solver_success = true;
  report.internal_accepted = true;
  report.reject_reason = "late_target_version";
  const Json json = report;
  CHECK(json.at("solver_success") == true);
  CHECK(json.at("internal_accepted") == true);
  CHECK(json.at("upper_accepted") == false);
  CHECK(json.at("applied") == false);
  CHECK(json.at("max_constraint_violation").is_null());
  CHECK(json.at("reject_reason") == "late_target_version");
  PlanDecisionEvent event;
  event.decision = PlanDecision::kFallback;
  event.reason = "no_accepted_plan";
  const Json decision = event;
  CHECK(decision.at("decision") == "fallback");
}
