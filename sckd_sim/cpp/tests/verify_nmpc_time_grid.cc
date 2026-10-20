// Ubuntu独立验收工具：argv[1]=旧等分probe.so，argv[2]=逐段dt probe.so。
// 不调用IPOPT；用任意控制检查生成动力学、位置约束、加速度约束及等分等价。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <casadi/casadi.hpp>

namespace {
void RequireNear(double actual, double expected, const char* label) {
  const double error = std::fabs(actual - expected);
  if (!std::isfinite(actual) || error > 1e-10 * (1 + std::fabs(expected))) {
    std::fprintf(stderr, "%s actual=%.17g expected=%.17g error=%.3g\n",
                 label, actual, expected, error);
    throw std::runtime_error(label);
  }
}

std::vector<casadi::DM> Evaluate(const casadi::Function& function,
                               const std::vector<double>& controls,
                               const std::vector<double>& parameters) {
  return function(std::vector<casadi::DM>{casadi::DM(controls), casadi::DM(parameters)});
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: verify_nmpc_time_grid legacy_probe.so stage_probe.so");
    const auto legacy = casadi::external("time_grid_probe", argv[1]);
    const auto staged = casadi::external("time_grid_probe", argv[2]);
    std::vector<double> controls(21), parameters(27, 0);
    const std::array<double, 7> inverse_acc{1./8, 1./8, 1./30, 1./30, 1./180, 1./400, 1./200};
    const std::array<double, 14> initial{.5, -.1, 0, -.61, -.61, 0, 0,
                                       .2, -.1, .1, -.2, .3, -.4, .1};
    std::copy(initial.begin(), initial.end(), parameters.begin());
    parameters[14] = 1.; parameters[15] = .2; parameters[16] = 1.1;
    parameters[20] = 10; parameters[21] = 6; parameters[22] = 2;
    parameters[23] = 20; parameters[24] = 5; parameters[25] = 1;
    for (int axis = 0; axis < 7; ++axis)
      for (int node = 0; node < 3; ++node)
        controls[axis * 3 + node] = (node == 1 ? -.03 : .02) * (axis + 1);

    // 多个等分时长逐项比较f/g/X，保证开关默认输入下为同一数值模型。
    for (double h : {.031, .11, .327}) {
      parameters[17] = parameters[18] = parameters[19] = h;
      const auto old_values = Evaluate(legacy, controls, parameters);
      const auto new_values = Evaluate(staged, controls, parameters);
      if (old_values.size() != 4 || new_values.size() != 4)
        throw std::runtime_error("probe output schema mismatch");
      RequireNear(static_cast<double>(old_values[3]), 0, "legacy schema");
      RequireNear(static_cast<double>(new_values[3]), 1, "per-stage schema");
      for (int output = 0; output < 3; ++output) {
        const auto a = std::vector<double>(old_values[output]);
        const auto b = std::vector<double>(new_values[output]);
        if (a.size() != b.size()) throw std::runtime_error("probe output size mismatch");
        for (size_t i = 0; i < a.size(); ++i) RequireNear(b[i], a[i], "uniform equivalence");
      }
    }

    // 独立逐段算式使用不同dt，避免“只改rq.dt，模型仍重复dt[0]”假通过。
    for (const auto& dt : {std::array<double, 3>{.4, .2, .1},
                          std::array<double, 3>{.08, .24, .48}}) {
      for (int k = 0; k < 3; ++k) parameters[17 + k] = dt[k];
      const auto result = Evaluate(staged, controls, parameters);
      const auto states = std::vector<double>(result[0]);
      const auto constraints = std::vector<double>(result[1]);
      if (states.size() != 56 || constraints.size() != 87)
        throw std::runtime_error("baseline state/constraint layout changed");
      auto state = initial;
      for (int node = 0; node < 3; ++node) {
        for (int axis = 0; axis < 7; ++axis) {
          const double u = controls[axis * 3 + node];
          const double smooth_abs = std::sqrt(u * u + 1e-4);
          state[axis] += (state[7 + axis] + u) * dt[node] - .5 * u * smooth_abs * inverse_acc[axis];
          state[7 + axis] += u;
          RequireNear(states[(node + 1) * 14 + axis], state[axis], "nonuniform position");
          RequireNear(states[(node + 1) * 14 + 7 + axis], state[7 + axis], "nonuniform velocity");
          RequireNear(constraints[21 + node * 7 + axis], state[axis], "position constraint node");
          RequireNear(constraints[63 + node * 7 + axis], dt[node] / inverse_acc[axis] - smooth_abs,
                      "acceleration constraint dt");
        }
      }
    }
    std::puts("nmpc time grid: equal-grid equivalence and independent nonuniform node checks passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "nmpc time-grid verification failed: %s\n", error.what());
    return 1;
  }
}
