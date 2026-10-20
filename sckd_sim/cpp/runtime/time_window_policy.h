#pragma once
// 独立时间窗口实验；默认保持当前控制行为，不改变物理/限矩/速度约束。
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>
#include "control/nmpc_time_policy.h"

namespace sckd::runtime {
struct TimeWindowPolicy {
  double planner_min_lead_s = .9;
  double planner_max_horizon_s = 2.5;
  double target_lock_s = .15;
  bool refine_before_arm_active = false;
  std::optional<double> pfly_init_delay_s;
  double pfly_duration_min_s = .30;
  double pfly_duration_max_s = 1.8;
  double pfly_legacy_lead_s = .020;
  double pfly_ready_transition_s = .25;
  NmpcTimePolicy nmpc6;

  nlohmann::json ToJson() const {
    return {{"schema", "SCKD_TIME_WINDOWS/1"},
      {"planner_min_lead_s", planner_min_lead_s},
      {"planner_max_horizon_s", planner_max_horizon_s},
      {"target_lock_s", target_lock_s},
      {"refine_before_arm_active", refine_before_arm_active},
      {"pfly_init_delay_s", pfly_init_delay_s ? nlohmann::json(*pfly_init_delay_s) : nlohmann::json(nullptr)},
      {"pfly_duration_min_s", pfly_duration_min_s},
      {"pfly_duration_max_s", pfly_duration_max_s},
      {"pfly_legacy_lead_s", pfly_legacy_lead_s},
      {"pfly_ready_transition_s", pfly_ready_transition_s},
      {"nmpc6", {{"dt_ratios", nmpc6.dt_ratios},
        {"freeze_window_s", nmpc6.freeze_window_s}, {"stop_s", nmpc6.stop_s},
        {"strict_window_s", nmpc6.strict_window_s},
        {"allow_nonuniform_dt", nmpc6.allow_nonuniform_dt}}}};
  }

  static TimeWindowPolicy FromJson(const nlohmann::json& document) {
    using Json = nlohmann::json;
    if (!document.is_object() || document.value("schema", std::string()) != "SCKD_TIME_WINDOWS/1")
      throw std::runtime_error("SCKD_TIME_WINDOWS/1 object required");
    const std::set<std::string> fields{"schema", "planner_min_lead_s", "planner_max_horizon_s",
      "target_lock_s", "refine_before_arm_active", "pfly_init_delay_s", "pfly_duration_min_s",
      "pfly_duration_max_s", "pfly_legacy_lead_s", "pfly_ready_transition_s", "nmpc6"};
    for (auto i = document.begin(); i != document.end(); ++i)
      if (!fields.count(i.key())) throw std::runtime_error("unknown time window: " + i.key());
    TimeWindowPolicy p;
    auto number = [](const Json& j, const char* name, double& out) {
      if (!j.contains(name)) return;
      if (!j.at(name).is_number()) throw std::runtime_error(std::string("finite time window required: ") + name);
      out = j.at(name).get<double>();
      if (!std::isfinite(out)) throw std::runtime_error(std::string("finite time window required: ") + name);
    };
    auto boolean = [](const Json& j, const char* name, bool& out) {
      if (!j.contains(name)) return;
      if (!j.at(name).is_boolean()) throw std::runtime_error(std::string("boolean required: ") + name);
      out = j.at(name).get<bool>();
    };
    number(document, "planner_min_lead_s", p.planner_min_lead_s);
    number(document, "planner_max_horizon_s", p.planner_max_horizon_s);
    number(document, "target_lock_s", p.target_lock_s);
    number(document, "pfly_duration_min_s", p.pfly_duration_min_s);
    number(document, "pfly_duration_max_s", p.pfly_duration_max_s);
    number(document, "pfly_legacy_lead_s", p.pfly_legacy_lead_s);
    number(document, "pfly_ready_transition_s", p.pfly_ready_transition_s);
    boolean(document, "refine_before_arm_active", p.refine_before_arm_active);
    if (document.contains("pfly_init_delay_s") && !document.at("pfly_init_delay_s").is_null()) {
      double delay = 0; number(document, "pfly_init_delay_s", delay); p.pfly_init_delay_s = delay;
    }
    if (document.contains("nmpc6")) {
      const auto& n = document.at("nmpc6");
      if (!n.is_object()) throw std::runtime_error("nmpc6 time policy must be object");
      const std::set<std::string> nf{"dt_ratios", "freeze_window_s", "stop_s", "strict_window_s", "allow_nonuniform_dt"};
      for (auto i = n.begin(); i != n.end(); ++i)
        if (!nf.count(i.key())) throw std::runtime_error("unknown nmpc6 time window: " + i.key());
      if (n.contains("dt_ratios")) {
        const auto& ratios = n.at("dt_ratios");
        if (!ratios.is_array() || ratios.size() != 3) throw std::runtime_error("three dt_ratios required");
        for (int i = 0; i < 3; ++i) {
          if (!ratios[i].is_number()) throw std::runtime_error("numeric dt_ratios required");
          p.nmpc6.dt_ratios[i] = ratios[i].get<double>();
        }
      }
      number(n, "freeze_window_s", p.nmpc6.freeze_window_s);
      number(n, "stop_s", p.nmpc6.stop_s);
      number(n, "strict_window_s", p.nmpc6.strict_window_s);
      boolean(n, "allow_nonuniform_dt", p.nmpc6.allow_nonuniform_dt);
    }
    if (p.planner_min_lead_s <= 0 || p.planner_max_horizon_s <= p.planner_min_lead_s ||
        p.target_lock_s < 0 || p.pfly_duration_min_s <= 0 ||
        p.pfly_duration_max_s < p.pfly_duration_min_s || p.pfly_legacy_lead_s < 0 ||
        p.pfly_ready_transition_s < 0 || (p.pfly_init_delay_s && *p.pfly_init_delay_s < 0))
      throw std::runtime_error("invalid time window range");
    p.nmpc6.Validate();
    return p;
  }
};
}  // namespace sckd::runtime
