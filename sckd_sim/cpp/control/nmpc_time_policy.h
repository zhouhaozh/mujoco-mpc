#pragma once
// 三节点时间网格及驱动层时间门；不读取环境、JSON或运行资源。
// 默认值保持旧等分网格；非均匀网格仅用于独立重生成的baseline库。
#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>

namespace sckd {

struct NmpcTimePolicy {
  std::array<double, 3> dt_ratios{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  double freeze_window_s = 1.0;  // 0关闭冻结；正值=remaining超过该值时允许冻结。
  double stop_s = .012;
  double strict_window_s = .100;
  bool allow_nonuniform_dt = false;  // 必须配套--nonuniform-time生成的独立nlp库。

  bool IsUniform() const {
    return dt_ratios[0] == dt_ratios[1] && dt_ratios[1] == dt_ratios[2];
  }

  void Validate() const {
    double sum = 0;
    for (double ratio : dt_ratios) {
      if (!std::isfinite(ratio) || ratio <= 0)
        throw std::runtime_error("nmpc6 dt_ratios must contain three finite positive values");
      sum += ratio;
    }
    if (std::fabs(sum - 1.0) > 1e-9)
      throw std::runtime_error("nmpc6 dt_ratios must sum to one");
    for (double window : {freeze_window_s, stop_s, strict_window_s})
      if (!std::isfinite(window) || window < 0)
        throw std::runtime_error("nmpc6 time windows must be finite and nonnegative");
    if (!IsUniform() && !allow_nonuniform_dt)
      throw std::runtime_error("nonuniform nmpc6 dt requires allow_nonuniform_dt and a regenerated library");
  }

  std::array<double, 3> Durations(double remaining_s) const {
    // 保留原T/3运算，避免默认路径因乘倒数引入不同浮点舍入。
    if (IsUniform()) return {remaining_s / 3.0, remaining_s / 3.0, remaining_s / 3.0};
    return {remaining_s * dt_ratios[0], remaining_s * dt_ratios[1],
            remaining_s * dt_ratios[2]};
  }
};

}  // namespace sckd
