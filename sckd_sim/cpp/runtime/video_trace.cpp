#include "runtime/intercept_modules.h"
#include "runtime/controller_adapters.h"
#include "runtime/plan_supervisor.h"
#include "runtime/synchronous_schedule.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <algorithm>
namespace sckd::runtime {

namespace {
nlohmann::json BuiltinForecast(const BallPredictionInterface& predictor,
                               double time_s) {
  if (!predictor.initialized()) return nullptr;

  constexpr double kStepSeconds = 0.02;
  constexpr double kHorizonSeconds = 2.5;
  const double start = std::max(time_s, predictor.t_last());
  const double bounce = predictor.NextBounceTime();
  nlohmann::json grid = nlohmann::json::array();
  nlohmann::json state_grid = nlohmann::json::array();
  for (double t = start; t <= start + kHorizonSeconds + 1e-12;
       t += kStepSeconds) {
    double p[3], v[3];
    if (!predictor.PredictFull(t, p, v)) break;
    const bool bounced = bounce >= 0.0 && t >= bounce;
    grid.push_back({t, p[0], p[1], p[2], v[2], bounced});
    state_grid.push_back(
        {t, p[0], p[1], p[2], v[0], v[1], v[2], bounced});
  }
  return {{"seq", static_cast<long long>(
                      std::llround(predictor.t_last() * 1e9))},
          {"grid", std::move(grid)},
          {"state_grid", std::move(state_grid)}};
}
}  // namespace

void RecordVideoTrace(VideoTraceContext c, double time_s) {
  auto& plan = c.plan;
  auto& measured = c.measured;
  auto& predictor = c.predictor;
  auto& remote_predictor = c.remote_predictor;
  auto& m = c.m;
  auto& d = c.d;
  auto& video_trace = c.video_trace;
  auto& base_mode = c.base_mode;
  auto& n6_mode = c.n6_mode;
  auto& wb_mode = c.wb_mode;
  auto& measured_replay = c.measured_replay;
  auto& wb_dead = c.wb_dead;
  auto& n6_dead = c.n6_dead;
  auto& ball_hit = c.ball_hit;
  auto& video_ground_contacts = c.video_ground_contacts;
  auto& ball_qadr = c.ball_qadr;
  auto& trial = c.trial;
  if (!video_trace) return;
  nlohmann::json row={{"trial",trial},{"t",time_s},{"mode",base_mode},{"measurement_replay",measured_replay},{"received_observations",measured ? measured->received() : 0},
    {"qpos",std::vector<double>(d->qpos,d->qpos+m->nq)},
    {"qvel",std::vector<double>(d->qvel,d->qvel+m->nv)},
    {"ball",{d->qpos[ball_qadr],d->qpos[ball_qadr+1],d->qpos[ball_qadr+2]}},
    {"ground_contacts",video_ground_contacts},{"hit",ball_hit},{"fallback",(wb_mode&&wb_dead)||(n6_mode&&n6_dead)},
    {"target",plan.feasible ? nlohmann::json{plan.p_hit[0],plan.p_hit[1],plan.p_hit[2]} : nlohmann::json(nullptr)}};
  row["forecast"] = remote_predictor
      ? remote_predictor->CachedVisualization()
      : BuiltinForecast(predictor, time_s);
  if (std::fprintf(video_trace, "%s\n", row.dump().c_str()) < 0 ||
      std::fflush(video_trace) != 0)
    throw std::runtime_error("cannot write video trace");

}
}
