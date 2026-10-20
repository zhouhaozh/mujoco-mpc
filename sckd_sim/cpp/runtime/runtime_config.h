#pragma once
// m2_intercept 的运行配置边界；本头不访问 MuJoCo，也不修改环境或资源文件。
// 优先级：内置默认 < JSON values < 环境变量 < 旧位置参数。
// 配置文件由 --config PATH（或 --config=PATH）选择；它优先于 M2_CONFIG_PATH。
// --config 可放在任意位置，剥离后旧 argv[1..8] 的位置/atoi/atof 语义不变。
// JSON 中的相对资源路径仍相对于进程 cwd，不能悄悄改为配置文件所在目录。
// ResolvedConfig 只暴露 const 访问器；解析完以后不得逐项改配置。
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include "runtime/time_window_policy.h"

namespace sckd::runtime {

using Json = nlohmann::json;
using Environment = std::function<std::optional<std::string>(const std::string&)>;
using ConfigReader = std::function<Json(const std::string&)>;

inline std::optional<std::string> ProcessEnvironment(const std::string& key) {
  const char* value = std::getenv(key.c_str());
  if (!value) return std::nullopt;
  return std::string(value);  // 存在且为空与不存在必须区分。
}

inline Json ReadConfigFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read runtime config: " + path);
  Json value;
  try {
    value = Json::parse(input);
  } catch (const std::exception& error) {
    throw std::runtime_error("invalid runtime config " + path + ": " + error.what());
  }
  return value;
}

struct RuntimeValues {
  std::string scene_path = "models/generated/x4_tennis_demo.mjcf";
  int trials = 10;
  std::string swing_mode = "quintic";  // 只有字面值 qp 启用 ArmSwing。
  double init_delay_s = 0.25;
  int seed = 2026;  // 旧 atoi 后传 mt19937；负数的无符号转换由原调用保留。
  std::string base_mode;
  std::string serve_family = "fixed";
  std::string fly_mode;  // 只有 xb/v2 有特殊含义；不新增旧位置参数校验。
  std::string control_variant = "baseline";
  std::string steer_rate_mode;  // 未显式设置时按最终 variant/base_mode 推导。

  // nullopt=未指定；空字符串=明确指定空路径。diagnostics/video 的空路径
  // 在原 main 仍会尝试 fopen，socket/replay 的空路径则不启用对应模块。
  std::optional<std::string> measured_replay_path;
  std::optional<std::string> predictor_socket_path;
  std::optional<std::string> control_diagnostics_path;
  std::optional<std::string> video_trace_path;
  std::optional<std::string> timing_path;
  bool viewer_enabled = false;
  bool deduplicate_ball_observations = false;
  bool legacy_nmpc6_limits = false;
  double nmpc6_speed_factor = 0.8;
  bool nmpc6_trace = false;
  bool wb_chassis_trace = false;

  std::string motors_path = "models/config/motors.json";
  std::string robot_variant_path = "models/config/x4_tennis.json";
  std::string chassis_path = "models/config/chassis_m1.json";
  std::string servo_identification_path = "models/config/servo_identified_20261015.json";
  std::string nmpc6_nlp_path;  // 未显式设置时随预测模型候选选择生成库。
  std::string viewer_state_path = "/tmp/sckd_demo_state.bin";
  std::string viewer_lock_path = "/tmp/sckd_demo_state.lock";
  std::optional<std::string> time_windows_path;
  TimeWindowPolicy time_windows;
};

struct RuntimeModes {
  bool use_qp = false;
  bool measured_replay = false;
  bool remote_predictor = false;
  bool deduplicate_obs = false;
  bool wb_mode = false, n6_mode = false;
  bool corrected = false, coherent = false, inverse_ff = false;
  bool chassis_model = false, chassis_recovery = false, chassis_guard = false;
  bool identified_gains = false;
  bool direction_limited = true, feedforward_limited = true;
  bool fly = false, ocp_mode = false, use_mpc = false;
  bool rand_serve = false, gs_serve = false, opposite_serve = false;
  bool cross_bounce = false;
  int fly_rule = 0;
  bool sim_joint23_limits = true;
  bool actuator_prediction = false;
};

class ResolvedConfig {
 public:
  ResolvedConfig(const ResolvedConfig&) = default;
  ResolvedConfig(ResolvedConfig&&) = default;
  const RuntimeValues& values() const { return values_; }
  const RuntimeModes& modes() const { return modes_; }
  const std::map<std::string, std::string>& sources() const { return sources_; }
  const std::optional<std::string>& config_path() const { return config_path_; }
  const std::string& config_path_source() const { return config_path_source_; }

  // argv 含程序名；可注入 env/reader，以便 Ubuntu 测试不污染真实进程环境。
  static ResolvedConfig Resolve(const std::vector<std::string>& argv,
                                Environment env = ProcessEnvironment,
                                ConfigReader reader = ReadConfigFile) {
    ResolvedConfig config;
    config.original_argv_ = argv;
    config.SetDefaultSources();
    config.CaptureEnvironment(env);
    std::vector<std::string> positional;
    std::vector<size_t> positional_indices;
    bool options = true, cli_config_seen = false;
    for (size_t i = argv.empty() ? 0 : 1; i < argv.size(); ++i) {
      const auto& arg = argv[i];
      if (options && arg == "--") { options = false; continue; }
      if (options && (arg == "--config" || arg.rfind("--config=", 0) == 0)) {
        if (cli_config_seen) throw std::runtime_error("duplicate --config option");
        cli_config_seen = true;
        std::string path;
        if (arg == "--config") {
          if (++i >= argv.size()) throw std::runtime_error("--config requires a path");
          path = argv[i];
        } else path = arg.substr(std::string("--config=").size());
        if (path.empty()) throw std::runtime_error("--config requires a nonempty path");
        config.config_path_ = path;
        config.config_path_source_ = "argv:--config";
      } else {
        positional.push_back(arg);
        positional_indices.push_back(i);
      }
    }
    if (!cli_config_seen) {
      auto selected = config.EnvironmentValue("M2_CONFIG_PATH");
      if (selected) {
        if (selected->empty()) throw std::runtime_error("M2_CONFIG_PATH must not be empty");
        config.config_path_ = *selected;
        config.config_path_source_ = "env:M2_CONFIG_PATH";
      }
    }
    if (config.config_path_) config.ApplyJson(reader(*config.config_path_));
    config.ApplyEnvironment();
    if (config.values_.time_windows_path) {
      config.values_.time_windows = TimeWindowPolicy::FromJson(reader(*config.values_.time_windows_path));
      config.sources_["time_windows"] = "file:" + *config.values_.time_windows_path;
    }
    config.ApplyPositional(positional, positional_indices);
    config.ResolveModes();
    return config;
  }

  static ResolvedConfig Load(int argc, const char* const* argv,
                            Environment env = ProcessEnvironment,
                            ConfigReader reader = ReadConfigFile) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i] ? argv[i] : "");
    return Resolve(args, std::move(env), std::move(reader));
  }

  Json ToJson() const {
    const auto& v = values_;
    Json values = {
      {"scene_path", v.scene_path}, {"trials", v.trials}, {"swing_mode", v.swing_mode},
      {"init_delay_s", v.init_delay_s}, {"seed", v.seed}, {"base_mode", v.base_mode},
      {"serve_family", v.serve_family}, {"fly_mode", v.fly_mode},
      {"control_variant", v.control_variant}, {"steer_rate_mode", v.steer_rate_mode},
      {"viewer_enabled", v.viewer_enabled},
      {"deduplicate_ball_observations", v.deduplicate_ball_observations},
      {"legacy_nmpc6_limits", v.legacy_nmpc6_limits},
      {"nmpc6_speed_factor", v.nmpc6_speed_factor}, {"nmpc6_trace", v.nmpc6_trace},
      {"wb_chassis_trace", v.wb_chassis_trace}, {"motors_path", v.motors_path},
      {"robot_variant_path", v.robot_variant_path}, {"chassis_path", v.chassis_path},
      {"servo_identification_path", v.servo_identification_path},
      {"nmpc6_nlp_path", v.nmpc6_nlp_path}, {"viewer_state_path", v.viewer_state_path},
      {"viewer_lock_path", v.viewer_lock_path}};
    values["time_windows"] = v.time_windows.ToJson();
    auto path = [&](const char* name, const std::optional<std::string>& value) {
      values[name] = value ? Json(*value) : Json(nullptr);
    };
    path("measured_replay_path", v.measured_replay_path);
    path("predictor_socket_path", v.predictor_socket_path);
    path("control_diagnostics_path", v.control_diagnostics_path);
    path("video_trace_path", v.video_trace_path);
    path("timing_path", v.timing_path);
    path("time_windows_path", v.time_windows_path);
    const auto& m = modes_;
    Json modes = {
      {"use_qp", m.use_qp}, {"measured_replay", m.measured_replay},
      {"remote_predictor", m.remote_predictor}, {"deduplicate_obs", m.deduplicate_obs},
      {"wb_mode", m.wb_mode}, {"n6_mode", m.n6_mode}, {"corrected", m.corrected},
      {"coherent", m.coherent}, {"inverse_ff", m.inverse_ff},
      {"chassis_model", m.chassis_model}, {"chassis_recovery", m.chassis_recovery},
      {"chassis_guard", m.chassis_guard}, {"identified_gains", m.identified_gains},
      {"direction_limited", m.direction_limited}, {"feedforward_limited", m.feedforward_limited},
      {"fly", m.fly}, {"ocp_mode", m.ocp_mode}, {"use_mpc", m.use_mpc},
      {"rand_serve", m.rand_serve}, {"gs_serve", m.gs_serve},
      {"opposite_serve", m.opposite_serve}, {"cross_bounce", m.cross_bounce},
      {"fly_rule", m.fly_rule}, {"sim_joint23_limits", m.sim_joint23_limits},
      {"actuator_prediction", m.actuator_prediction}};
    modes["effective_init_delay_s"] = v.time_windows.pfly_init_delay_s.value_or(v.init_delay_s);
    return {{"schema", "SCKD_RUNTIME_CONFIG/1"}, {"values", values},
            {"derived", modes}, {"sources", sources_},
            {"config_path", config_path_ ? Json(*config_path_) : Json(nullptr)},
            {"config_path_source", config_path_source_},
            {"path_base", "process_cwd"}, {"precedence", "defaults < JSON < environment < positional_argv"},
            {"argv", original_argv_}, {"environment", environment_},
            {"ignored_positional", ignored_positional_}};
  }

 private:
  ResolvedConfig() = default;
  RuntimeValues values_;
  RuntimeModes modes_;
  std::map<std::string, std::string> sources_;
  std::optional<std::string> config_path_;
  std::string config_path_source_ = "none";
  std::vector<std::string> original_argv_, ignored_positional_;
  std::map<std::string, std::string> environment_;

  void SetDefaultSources() {
    for (const char* key : {
      "scene_path", "trials", "swing_mode", "init_delay_s", "seed", "base_mode",
      "serve_family", "fly_mode", "control_variant", "steer_rate_mode",
      "measured_replay_path", "predictor_socket_path", "control_diagnostics_path",
      "video_trace_path", "timing_path", "viewer_enabled", "deduplicate_ball_observations",
      "legacy_nmpc6_limits", "nmpc6_speed_factor", "nmpc6_trace", "wb_chassis_trace",
      "motors_path", "robot_variant_path", "chassis_path", "servo_identification_path",
      "nmpc6_nlp_path", "viewer_state_path", "viewer_lock_path", "time_windows_path", "time_windows"}) sources_[key] = "default";
  }

  void CaptureEnvironment(const Environment& env) {
    for (const char* key : {
      "M2_CONFIG_PATH", "M2_CONTROL_VARIANT", "M2_STEER_RATE_MODE", "M2_MEASURED_REPLAY",
      "BALL_PREDICTOR_SOCKET", "M2_CONTROL_DIAGNOSTICS", "M2_VIDEO_TRACE", "M2_TIMING_PATH",
      "M2_VIEW", "BALL_DEDUP", "NMPC6_LEGACY_LIMITS", "N6_SPD", "N6_TRACE", "WB_CHASSIS",
      "M2_SERVO_IDENTIFICATION", "M2_TIME_WINDOWS_PATH", "M2_NMPC6_NLP",
      // 下列仅记录现有求解器日志开关，不通过JSON改动其getenv调用。
      "NMPC6_DEBUG", "WM_DEBUG", "WBO_DEBUG"}) {
      auto value = env(key);
      if (value) environment_[key] = *value;
    }
  }

  std::optional<std::string> EnvironmentValue(const std::string& key) const {
    auto found = environment_.find(key);
    if (found == environment_.end()) return std::nullopt;
    return found->second;
  }

  void ApplyJson(const Json& document) {
    if (!document.is_object()) throw std::runtime_error("runtime config must be an object");
    for (auto it = document.begin(); it != document.end(); ++it) {
      const auto& key = it.key();
      // ToJson() 产物可作为配置输入；审计元信息不反向决定运行值。
      if (key != "schema" && key != "values" && key != "derived" && key != "sources" &&
          key != "config_path" && key != "config_path_source" && key != "path_base" &&
          key != "precedence" && key != "argv" && key != "environment" && key != "ignored_positional")
        throw std::runtime_error("unknown runtime config root field: " + key);
    }
    if (document.contains("schema") && document.at("schema") != "SCKD_RUNTIME_CONFIG/1")
      throw std::runtime_error("unsupported runtime config schema");
    if (document.contains("path_base") && document.at("path_base") != "process_cwd")
      throw std::runtime_error("runtime config supports only process_cwd paths");
    if (!document.contains("values") || !document.at("values").is_object())
      throw std::runtime_error("runtime config requires object field values");
    for (auto it = document.at("values").begin(); it != document.at("values").end(); ++it) {
      const auto& key = it.key();
      const auto& value = it.value();
      if (!sources_.count(key)) throw std::runtime_error("unknown runtime config field: " + key);
      bool assigned = false;
      auto string_field = [&](const char* name, std::string& field) {
        if (key != name) return;
        if (!value.is_string()) throw std::runtime_error("runtime config string expected: " + key);
        field = value.get<std::string>(); assigned = true;
      };
      auto int_field = [&](const char* name, int& field) {
        if (key != name) return;
        if (!value.is_number_integer()) throw std::runtime_error("runtime config integer expected: " + key);
        if (value.is_number_unsigned() && value.get<unsigned long long>() >
            static_cast<unsigned long long>(std::numeric_limits<int>::max()))
          throw std::runtime_error("runtime config integer out of range: " + key);
        const auto number = value.get<long long>();
        if (number < std::numeric_limits<int>::min() || number > std::numeric_limits<int>::max())
          throw std::runtime_error("runtime config integer out of range: " + key);
        field = static_cast<int>(number); assigned = true;
      };
      auto double_field = [&](const char* name, double& field) {
        if (key != name) return;
        if (!value.is_number()) throw std::runtime_error("runtime config number expected: " + key);
        field = value.get<double>();
        if (!std::isfinite(field)) throw std::runtime_error("runtime config finite number expected: " + key);
        assigned = true;
      };
      auto bool_field = [&](const char* name, bool& field) {
        if (key != name) return;
        if (!value.is_boolean()) throw std::runtime_error("runtime config boolean expected: " + key);
        field = value.get<bool>(); assigned = true;
      };
      auto path_field = [&](const char* name, std::optional<std::string>& field) {
        if (key != name) return;
        if (value.is_null()) field.reset();
        else if (value.is_string()) field = value.get<std::string>();
        else throw std::runtime_error("runtime config string or null expected: " + key);
        assigned = true;
      };
      auto& v = values_;
      string_field("scene_path", v.scene_path); int_field("trials", v.trials);
      string_field("swing_mode", v.swing_mode); double_field("init_delay_s", v.init_delay_s);
      int_field("seed", v.seed); string_field("base_mode", v.base_mode);
      string_field("serve_family", v.serve_family); string_field("fly_mode", v.fly_mode);
      string_field("control_variant", v.control_variant); string_field("steer_rate_mode", v.steer_rate_mode);
      path_field("measured_replay_path", v.measured_replay_path);
      path_field("predictor_socket_path", v.predictor_socket_path);
      path_field("control_diagnostics_path", v.control_diagnostics_path);
      path_field("video_trace_path", v.video_trace_path); path_field("timing_path", v.timing_path);
      bool_field("viewer_enabled", v.viewer_enabled);
      bool_field("deduplicate_ball_observations", v.deduplicate_ball_observations);
      bool_field("legacy_nmpc6_limits", v.legacy_nmpc6_limits);
      double_field("nmpc6_speed_factor", v.nmpc6_speed_factor);
      bool_field("nmpc6_trace", v.nmpc6_trace); bool_field("wb_chassis_trace", v.wb_chassis_trace);
      string_field("motors_path", v.motors_path); string_field("robot_variant_path", v.robot_variant_path);
      string_field("chassis_path", v.chassis_path);
      string_field("servo_identification_path", v.servo_identification_path);
      string_field("nmpc6_nlp_path", v.nmpc6_nlp_path);
      string_field("viewer_state_path", v.viewer_state_path); string_field("viewer_lock_path", v.viewer_lock_path);
      path_field("time_windows_path", v.time_windows_path);
      if (key == "time_windows") { v.time_windows = TimeWindowPolicy::FromJson(value); assigned = true; }
      if (!assigned) throw std::runtime_error("unhandled runtime config field: " + key);
      sources_[key] = "json:" + *config_path_ + "#values." + key;
    }
  }

  void ApplyEnvironment() {
    auto& v = values_;
    auto string_field = [&](const char* env, const char* key, std::string& field) {
      auto value = EnvironmentValue(env);
      if (value) { field = *value; sources_[key] = std::string("env:") + env; }
    };
    auto path_field = [&](const char* env, const char* key, std::optional<std::string>& field) {
      auto value = EnvironmentValue(env);
      if (value) { field = *value; sources_[key] = std::string("env:") + env; }
    };
    auto presence_flag = [&](const char* env, const char* key, bool& field) {
      if (EnvironmentValue(env)) { field = true; sources_[key] = std::string("env-presence:") + env; }
    };
    string_field("M2_CONTROL_VARIANT", "control_variant", v.control_variant);
    string_field("M2_STEER_RATE_MODE", "steer_rate_mode", v.steer_rate_mode);
    string_field("M2_SERVO_IDENTIFICATION", "servo_identification_path", v.servo_identification_path);
    string_field("M2_NMPC6_NLP", "nmpc6_nlp_path", v.nmpc6_nlp_path);
    path_field("M2_TIME_WINDOWS_PATH", "time_windows_path", v.time_windows_path);
    path_field("M2_MEASURED_REPLAY", "measured_replay_path", v.measured_replay_path);
    path_field("BALL_PREDICTOR_SOCKET", "predictor_socket_path", v.predictor_socket_path);
    path_field("M2_CONTROL_DIAGNOSTICS", "control_diagnostics_path", v.control_diagnostics_path);
    path_field("M2_VIDEO_TRACE", "video_trace_path", v.video_trace_path);
    path_field("M2_TIMING_PATH", "timing_path", v.timing_path);
    presence_flag("M2_VIEW", "viewer_enabled", v.viewer_enabled);
    presence_flag("BALL_DEDUP", "deduplicate_ball_observations", v.deduplicate_ball_observations);
    presence_flag("NMPC6_LEGACY_LIMITS", "legacy_nmpc6_limits", v.legacy_nmpc6_limits);
    presence_flag("N6_TRACE", "nmpc6_trace", v.nmpc6_trace);
    presence_flag("WB_CHASSIS", "wb_chassis_trace", v.wb_chassis_trace);
    if (auto value = EnvironmentValue("N6_SPD")) {
      v.nmpc6_speed_factor = std::atof(value->c_str());
      sources_["nmpc6_speed_factor"] = "env:N6_SPD";
    }
  }

  void ApplyPositional(const std::vector<std::string>& args, const std::vector<size_t>& indices) {
    auto& v = values_;
    auto source = [&](const char* key, int index) {
      sources_[key] = "argv[" + std::to_string(indices.at(index - 1)) + "]";
    };
    if (args.size() > 0) { v.scene_path = args[0]; source("scene_path", 1); }
    if (args.size() > 1) { v.trials = std::atoi(args[1].c_str()); source("trials", 2); }
    if (args.size() > 2) { v.swing_mode = args[2]; source("swing_mode", 3); }
    if (args.size() > 3) { v.init_delay_s = std::atof(args[3].c_str()); source("init_delay_s", 4); }
    if (args.size() > 4) { v.seed = std::atoi(args[4].c_str()); source("seed", 5); }
    if (args.size() > 5) { v.base_mode = args[5]; source("base_mode", 6); }
    if (args.size() > 6) { v.serve_family = args[6]; source("serve_family", 7); }
    if (args.size() > 7) { v.fly_mode = args[7]; source("fly_mode", 8); }
    if (args.size() > 8) ignored_positional_.assign(args.begin() + 8, args.end());
  }

  void ResolveModes() {
    auto& v = values_;
    auto& m = modes_;
    m.use_qp = v.swing_mode == "qp";
    m.measured_replay = v.measured_replay_path && !v.measured_replay_path->empty();
    if (m.measured_replay && v.trials != 1)
      throw std::runtime_error("measured replay requires exactly one trial");
    m.remote_predictor = v.predictor_socket_path && !v.predictor_socket_path->empty();
    m.deduplicate_obs = m.remote_predictor || v.deduplicate_ball_observations;
    m.wb_mode = v.base_mode == "wbmpc";
    m.n6_mode = v.base_mode == "nmpc6";
    m.corrected = v.control_variant != "baseline" && (m.n6_mode || v.base_mode == "pfly");
    m.coherent = m.corrected && v.control_variant != "feasible";
    m.inverse_ff = m.coherent && v.control_variant != "clock";
    m.chassis_model = v.control_variant == "chassis_model" || v.control_variant == "chassis_recovery" || v.control_variant == "chassis_guard";
    m.chassis_recovery = m.chassis_model && v.control_variant != "chassis_model";
    m.chassis_guard = v.control_variant == "chassis_guard";
    if (m.chassis_model && !m.n6_mode) throw std::runtime_error("chassis variants require nmpc6");
    m.identified_gains = m.inverse_ff && (v.control_variant == "gains" || v.control_variant == "base_feedback" || v.control_variant == "servo_mpc" || m.chassis_model);
    if (v.control_variant != "baseline" && v.control_variant != "feasible" && v.control_variant != "clock" &&
        v.control_variant != "feedforward" && v.control_variant != "gains" && v.control_variant != "base_feedback" &&
        v.control_variant != "servo_mpc" && !m.chassis_model)
      throw std::runtime_error("unknown control variant");
    if (sources_.at("steer_rate_mode") == "default") {
      v.steer_rate_mode = v.control_variant == "baseline" && (m.n6_mode || v.base_mode == "pfly") ? "both_unlimited" : "limited";
      sources_["steer_rate_mode"] = "derived:control_variant+base_mode";
    }
    if (v.steer_rate_mode != "limited" && v.steer_rate_mode != "direction_unlimited" &&
        v.steer_rate_mode != "feedforward_unlimited" && v.steer_rate_mode != "both_unlimited")
      throw std::runtime_error("unknown steer rate mode");
    m.direction_limited = v.steer_rate_mode != "direction_unlimited" && v.steer_rate_mode != "both_unlimited";
    m.feedforward_limited = v.steer_rate_mode != "feedforward_unlimited" && v.steer_rate_mode != "both_unlimited";
    m.fly = v.base_mode == "mpcfly" || v.base_mode == "pfly" || v.base_mode == "ocpfly" || m.wb_mode || m.n6_mode;
    m.ocp_mode = v.base_mode == "ocpfly";
    m.use_mpc = v.base_mode == "mpc" || v.base_mode == "mpcfly";
    m.rand_serve = v.serve_family == "rand";
    m.gs_serve = v.serve_family == "gs";
    m.opposite_serve = v.serve_family == "opposite";
    m.cross_bounce = v.fly_mode == "xb";
    m.fly_rule = v.fly_mode == "v2" ? 1 : 0;
    m.sim_joint23_limits = !v.legacy_nmpc6_limits;
    // 原 main 对 servo_mpc 的判据未额外检查 base_mode；保留这一细节。
    m.actuator_prediction = v.control_variant == "servo_mpc";
    if (!v.time_windows.nmpc6.IsUniform()) {
      if (m.actuator_prediction || m.chassis_model)
        throw std::runtime_error("nonuniform time experiments require the baseline prediction model");
      if (sources_.at("nmpc6_nlp_path") == "default" || v.nmpc6_nlp_path.empty())
        throw std::runtime_error("nonuniform time experiments require an explicit regenerated nmpc6_nlp_path");
    }
    if (sources_.at("nmpc6_nlp_path") == "default") {
      v.nmpc6_nlp_path = m.actuator_prediction ? "third_party/nmpc6/nlp_servo.so" :
        m.chassis_model ? "third_party/nmpc6/nlp_chassis.so" : "third_party/nmpc6/nlp.so";
      sources_["nmpc6_nlp_path"] = "derived:control_variant";
    }
  }
};

}  // namespace sckd::runtime
