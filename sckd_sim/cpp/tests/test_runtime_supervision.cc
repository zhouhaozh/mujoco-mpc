// 只验证现行守门/仿真调度契约；不运行求解器或物理引擎。
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <cmath>
#include <limits>

#include "runtime/plan_supervisor.h"
#include "runtime/synchronous_schedule.h"
#include "control/nmpc_time_policy.h"

using namespace sckd;
using namespace sckd::runtime;

TEST_CASE("nmpc time policy preserves exact default division and validates stage timing") {
  NmpcTimePolicy policy;
  REQUIRE_NOTHROW(policy.Validate());
  const double remaining = std::nextafter(.73, 0.);
  const auto defaults = policy.Durations(remaining);
  for (double h : defaults) CHECK(h == remaining / 3.0);
  policy.dt_ratios = {.2, .3, .5};
  CHECK_THROWS(policy.Validate()); // 旧单dt库不得接受非均匀请求。
  policy.allow_nonuniform_dt = true;
  REQUIRE_NOTHROW(policy.Validate());
  const auto stages = policy.Durations(remaining);
  CHECK(stages[0] == doctest::Approx(remaining * .2));
  CHECK(stages[1] == doctest::Approx(remaining * .3));
  CHECK(stages[2] == doctest::Approx(remaining * .5));
  CHECK(stages[0] + stages[1] + stages[2] == doctest::Approx(remaining));
  policy.dt_ratios = {0., .5, .5};
  CHECK_THROWS(policy.Validate());
  policy.dt_ratios = {.2, .3, .6};
  CHECK_THROWS(policy.Validate());
  policy.dt_ratios = {.2, .3, .5};
  policy.freeze_window_s = 0;
  CHECK_NOTHROW(policy.Validate()); // 0为显式关闭冻结。
  policy.stop_s = -1;
  CHECK_THROWS(policy.Validate());
}

TEST_CASE("configurable terminal acceptance window changes only the near-hit gate") {
  LegacyNmpcResult result;
  result.status = LegacyNmpcStatus::kLoose;
  result.solver_ok = true;
  result.solve_ms = 20;
  CHECK_FALSE(PlanSupervisor::CanApplyNmpc(result, false, .15, .2));
  CHECK(PlanSupervisor::CanApplyNmpc(result, false, .2, .2));
  CHECK(PlanSupervisor::CanApplyNmpc(result, false, .001, 0));
  CHECK(PlanSupervisor::CanApplyNmpc(result, true, .001, .2));
}

TEST_CASE("last 100ms applies strict successful solutions below but not at 10ms") {
  LegacyNmpcResult result;
  result.status = LegacyNmpcStatus::kOk;
  result.solver_ok = true;
  result.solve_ms = std::nextafter(10., 0.);
  CHECK(PlanSupervisor::CanApplyNmpc(result, false, .099));
  result.solve_ms = 10.;
  CHECK_FALSE(PlanSupervisor::CanApplyNmpc(result, false, .099));
  result.solve_ms = std::nextafter(10., std::numeric_limits<double>::infinity());
  CHECK_FALSE(PlanSupervisor::CanApplyNmpc(result, false, .099));
  result.solve_ms = 1.;
  result.solver_ok = false;
  CHECK_FALSE(PlanSupervisor::CanApplyNmpc(result, false, .099));
  result.solver_ok = true;
  result.status = LegacyNmpcStatus::kLoose;
  CHECK_FALSE(PlanSupervisor::CanApplyNmpc(result, false, .099));
}

TEST_CASE("100ms boundary and coherent clock bypass preserve caller-owned acceptance") {
  LegacyNmpcResult result;
  result.status = LegacyNmpcStatus::kLoose;
  result.solver_ok = true;
  result.solve_ms = 20.;
  CHECK(PlanSupervisor::CanApplyNmpc(result, false, .100));
  CHECK_FALSE(PlanSupervisor::CanApplyNmpc(result, false, std::nextafter(.100, 0.)));
  CHECK(PlanSupervisor::CanApplyNmpc(result, true, .001));
  // CanApplyNmpc不是内部求解验收器，调用方必须先检查Solve返回的accepted。
  result.status = LegacyNmpcStatus::kSolverFail;
  result.solver_ok = false;
  const bool internal_accepted = false;
  CHECK(PlanSupervisor::CanApplyNmpc(result, true, .001));
  CHECK_FALSE((internal_accepted && PlanSupervisor::CanApplyNmpc(result, true, .001)));
}

TEST_CASE("nmpc first-plan failures fall back on failure 16 and existing plans do not count") {
  int failures = 0;
  for (int failure = 1; failure <= 15; ++failure) {
    CHECK_FALSE(PlanSupervisor::NmpcFallback(false, failures, false));
    CHECK(failures == failure);
  }
  CHECK(PlanSupervisor::NmpcFallback(false, failures, false));
  CHECK(failures == 16);
  failures = 4;
  for (int rejected = 0; rejected < 100; ++rejected)
    CHECK_FALSE(PlanSupervisor::NmpcFallback(true, failures, false));
  CHECK(failures == 4);
  CHECK(PlanSupervisor::NmpcFallback(true, failures, true));
  CHECK(failures == 4);  // stale独立触发，已有解不增加“从未有解”计数。
  failures = 0;
  CHECK(PlanSupervisor::NmpcFallback(false, failures, true));
  CHECK(failures == 1);  // 保留原短路表达式先累计再检查stale的顺序。
}

TEST_CASE("wbmpc no-plan and existing-plan guards trigger on failure 16 and 21") {
  for (const bool have : {false, true}) {
    int failures = 0;
    const int allowed = have ? 20 : 15;
    for (int failure = 1; failure <= allowed; ++failure) {
      CHECK_FALSE(PlanSupervisor::WbmpcFallback(have, failures));
      CHECK(failures == failure);
    }
    CHECK(PlanSupervisor::WbmpcFallback(have, failures));
    CHECK(failures == allowed + 1);
  }
}

TEST_CASE("wbmpc face-normal rejection remains caller managed") {
  // 当前调用方对Solve成功但n_dot<.75只累计，不调用fallback分支；守卫必须
  // 接受这种已经超过阈值的输入，不能假定调用方总从0逐次调用到恰好21。
  int failures = 30;
  // 以后真正的Solve失败才调用本守卫；不要在提取时把两个分支合并。
  CHECK(PlanSupervisor::WbmpcFallback(true, failures));
  CHECK(failures == 31);
  // 正常接受解的计数复位也由调用方管理。
  failures = 0;
  CHECK_FALSE(PlanSupervisor::WbmpcFallback(true, failures));
  CHECK(failures == 1);
}

TEST_CASE("due schedules are simulation tick predicates without wall clock state") {
  CHECK(SynchronousSchedule::kMaxTicks == 6000);
  CHECK(SynchronousSchedule::Time(6000) == doctest::Approx(6.));
  CHECK_FALSE(SynchronousSchedule::Replan(19));
  CHECK(SynchronousSchedule::Replan(20));
  CHECK_FALSE(SynchronousSchedule::Replan(21));
  CHECK(SynchronousSchedule::StateTrace(5));
  CHECK_FALSE(SynchronousSchedule::StateTrace(6));
  CHECK(SynchronousSchedule::Viewer(17));
  CHECK_FALSE(SynchronousSchedule::Viewer(18));
  // 重复询问同一仿真tick仍然due；它不是消费一次的计时器，也不统计deadline。
  CHECK(SynchronousSchedule::Replan(20));
  CHECK(SynchronousSchedule::Replan(20));
  CHECK(SynchronousSchedule::Time(20) == doctest::Approx(.020));
  int replans = 0, traces = 0, views = 0;
  for (int tick = 1; tick <= SynchronousSchedule::kMaxTicks; ++tick) {
    replans += SynchronousSchedule::Replan(tick);
    traces += SynchronousSchedule::StateTrace(tick);
    views += SynchronousSchedule::Viewer(tick);
  }
  CHECK(replans == 300);
  CHECK(traces == 1200);
  CHECK(views == 352);
}
