#include "core/config.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

namespace sckd {
namespace {

nlohmann::json ReadJson(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open config: " + path);
  nlohmann::json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    throw std::runtime_error("JSON parse error in " + path + ": " + e.what());
  }
  return j;
}

MotorTypeConfig ParseMotor(const std::string& name, const nlohmann::json& j) {
  MotorTypeConfig m;
  m.name = name;
  m.peak_torque_nm = j.at("peak_torque_nm");
  m.continuous_torque_nm = j.at("continuous_torque_nm");
  m.braking_torque_nm = j.at("braking_torque_nm");
  m.no_load_speed_rad_s = j.at("no_load_speed_rad_s");
  m.hybrid_speed_limit_rad_s = j.at("hybrid_speed_limit_rad_s");
  m.hybrid_position_limit_rad = j.at("hybrid_position_limit_rad");
  m.kp_min = j.at("kp_range")[0];
  m.kp_max = j.at("kp_range")[1];
  m.kd_min = j.at("kd_range")[0];
  m.kd_max = j.at("kd_range")[1];
  for (const auto& p : j.at("motoring_points_rad_s_nm")) {
    m.motoring.push_back({p[0], p[1]});
  }
  if (m.motoring.empty()) throw std::runtime_error("motor " + name + ": empty envelope");
  for (size_t i = 1; i < m.motoring.size(); ++i) {
    if (m.motoring[i].speed_rad_s <= m.motoring[i - 1].speed_rad_s)
      throw std::runtime_error("motor " + name + ": envelope speeds must be ascending");
  }
  return m;
}

}  // namespace

MotorSystemConfig LoadMotorConfig(const std::string& motors_json_path,
                                  const std::string& variant_json_path) {
  const auto mj = ReadJson(motors_json_path);
  const auto vj = ReadJson(variant_json_path);

  MotorSystemConfig cfg;
  cfg.robot_variant = vj.value("variant", "");
  cfg.truth_feedback = mj.value("truth_feedback", false);
  if (cfg.robot_variant == "x4_950" || mj.value("variant", "") == "x4_950") {
    if (cfg.robot_variant != "x4_950" || mj.value("variant", "") != "x4_950")
      throw std::runtime_error("X4_950 motor and robot variants must match");
    const auto reference = std::filesystem::path(vj.at("motor_config").get<std::string>()).lexically_normal();
    const auto variant_dir = std::filesystem::path(variant_json_path).parent_path();
    // Directory bundles use a sibling filename. Preserve the models-relative
    // convention of older configs (e.g. config/motors_x4_950.json).
    const auto expected = (reference.has_parent_path() ? variant_dir.parent_path() : variant_dir)
                          / reference;
    if (mj.value("variant", "") != cfg.robot_variant ||
        std::filesystem::weakly_canonical(expected) !=
            std::filesystem::weakly_canonical(motors_json_path))
      throw std::runtime_error("X4_950 motor config does not match variant.motor_config");
  }
  cfg.command_delay_ms = mj.value("command_delay_ms", 9.0);
  cfg.control_dt_s = mj.value("control_dt_s", 0.001);
  cfg.encoder_bits = mj.value("encoder_bits", 14);
  cfg.calibrated = mj.value("calibrated", false);
  cfg.profile_peak = (mj.value("torque_profile", "peak") == "peak");

  std::map<std::string, MotorTypeConfig> motors;
  for (auto it = mj.at("motors").begin(); it != mj.at("motors").end(); ++it)
    motors[it.key()] = ParseMotor(it.key(), it.value());
  for (const auto& m : motors) cfg.motors.push_back(m.second);

  // 变体关节按契约轴序对齐(缺失即配置错误)
  std::map<std::string, const nlohmann::json*> by_joint;
  for (const auto& jv : vj.at("joints")) by_joint[jv.at("joint").get<std::string>()] = &jv;
  for (int axis = 0; axis < kNumAxes; ++axis) {
    const std::string name = kAxisJointName[axis];
    auto it = by_joint.find(name);
    if (it == by_joint.end())
      throw std::runtime_error("variant missing joint (contract axis " + std::to_string(axis) +
                               "): " + name);
    const auto& jv = *it->second;
    JointConfig jc;
    jc.joint = name;
    jc.motor = jv.at("motor");
    jc.mechanism_limit_nm = jv.at("mechanism_limit_nm");
    if (!motors.count(jc.motor))
      throw std::runtime_error("joint " + name + " references unknown motor " + jc.motor);
    if (!(jc.mechanism_limit_nm > 0))
      throw std::runtime_error("joint " + name + ": mechanism_limit_nm must be > 0");
    cfg.joints.push_back(jc);
  }
  if (cfg.robot_variant == "x4_950") {
    const auto& servo = vj.at("arm_servo");
    for (const char* key : {"kp", "kd", "standby_q_rad"})
      if (!servo.at(key).is_array() || servo.at(key).size() != kNumArm)
        throw std::runtime_error("X4_950 arm servo requires five values per field");
    cfg.arm_servo.kp = servo.at("kp").get<std::array<double, kNumArm>>();
    cfg.arm_servo.kd = servo.at("kd").get<std::array<double, kNumArm>>();
    cfg.arm_servo.standby_q = servo.at("standby_q_rad").get<std::array<double, kNumArm>>();
    for (int j = 0; j < kNumArm; ++j) {
      const auto& m = motors.at(cfg.joints[AxisArm(j)].motor);
      const double kp = cfg.arm_servo.kp[j], kd = cfg.arm_servo.kd[j];
      const double q = cfg.arm_servo.standby_q[j];
      if (!std::isfinite(kp) || !std::isfinite(kd) || !std::isfinite(q) ||
          kp < m.kp_min || kp > m.kp_max || kd < m.kd_min || kd > m.kd_max ||
          std::fabs(q) > m.hybrid_position_limit_rad)
        throw std::runtime_error("X4_950 arm servo exceeds motor command range");
      if (j == 1 || j == 2) {
        const auto& range = vj.at("confirmed_joint_position_limits_rad").at("Joint" + std::to_string(j+1));
        if (q < range.at(0).get<double>() || q > range.at(1).get<double>())
          throw std::runtime_error("X4_950 standby exceeds confirmed joint range");
      }
    }
    cfg.arm_servo.enabled = true;
  }
  return cfg;
}

ChassisConfig LoadChassisConfig(const std::string& chassis_json_path) {
  const auto j = ReadJson(chassis_json_path);
  ChassisConfig c;
  c.robot_variant = j.value("variant", "");
  const auto& g = j.at("geometry");
  int i = 0;
  for (const auto& xy : g.at("corner_xy_m")) {
    c.corner_x[i] = xy[0];
    c.corner_y[i] = xy[1];
    ++i;
  }
  if (i != 4) throw std::runtime_error("geometry.corner_xy_m must have 4 corners");
  c.wheel_radius = g.at("wheel_radius_m");
  c.dual_offset = g.at("dual_wheel_offset_m");
  i = 0;
  for (const double v : g.at("wheel_sign")) c.wheel_sign[i++] = v;
  if (i != 8) throw std::runtime_error("geometry.wheel_sign must have 8 values");
  c.robot_mass_kg = g.value("robot_mass_kg", 51.26);

  const auto& co = j.at("coordination");
  c.coord.enable_cos_projection = co.value("enable_cos_projection", true);
  c.coord.cos_deadband = co.value("cos_deadband_rad", 0.10);
  c.coord.cos_floor = co.value("cos_floor", 0.0);
  c.coord.reverse_threshold = co.value("reverse_threshold_rad", 1.5708);
  c.coord.reverse_hysteresis = co.value("reverse_hysteresis_rad", 0.20);
  c.coord.zero_speed = co.value("zero_speed_mps", 1e-3);

  const auto& r = j.at("ramp");
  c.ramp.accel_limit_mps2 = r.at("accel_limit_mps2");
  c.ramp.ang_accel_limit_rad_s2 = r.at("ang_accel_limit_rad_s2");
  c.ramp.dir_rate_limit_rad_s = r.at("dir_rate_limit_rad_s");

  const auto& s = j.at("servo");
  c.servo.period_ms = s.value("period_ms", 4.0);
  c.servo.steer_kp = s.at("steer_kp");
  c.servo.steer_kd = s.at("steer_kd");
  c.servo.wheel_kp = s.at("wheel_kp");
  c.servo.wheel_ki = s.at("wheel_ki");
  c.servo.wheel_int_max_rad = s.value("wheel_int_max_rad", 8.0);
  c.servo.wheel_target_accel_mps2 = s.value("wheel_accel_mps2", 8.0);

  const auto& o = j.at("odom");
  i = 0;
  for (const double v : o.at("steer_offset_rad")) c.odom.steer_offset_rad[i++] = v;
  if (i != 4) throw std::runtime_error("odom.steer_offset_rad must have 4 values");
  c.odom.wheel_speed_noise_rad_s = o.value("wheel_speed_noise_rad_s", 0.01);
  c.odom.imu_rate_hz = o.value("imu_rate_hz", 200.0);
  c.odom.imu_latency_ms = o.value("imu_latency_ms", 2.0);
  c.odom.gyro_bias_rad_s = o.value("gyro_bias_rad_s", 0.002);
  c.odom.gyro_noise_rad_s = o.value("gyro_noise_rad_s", 0.002);
  c.odom.acc_noise_mps2 = o.value("acc_noise_mps2", 0.05);
  c.odom.gyro_fallback_ms = o.value("gyro_fallback_ms", 300.0);
  return c;
}

}  // namespace sckd
