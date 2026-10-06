#include "planning/hit_planner.h"
#include <cmath>
#include <algorithm>

namespace sckd {

HitPlanner::HitPlanner(const PlannerParams& pp, const BallAeroParams& aero)
    : pp_(pp), aero_(aero) {}

void HitPlanner::RefineAt(const BallPredictor& predictor, const BaseState& base,
                          HitPlan* hp, double t_now) {
  if (!hp || !hp->feasible) return;
  double p[3], v_in[3];
  if (!predictor.PredictFull(hp->t_hit, p, v_in)) return;
  for (int i = 0; i < 3; ++i) hp->p_hit[i] = p[i];
  double bc[3] = {base.x, base.y, base.yaw};
  BaseTargetFromHit(hp->p_hit, bc, hp->base_target);
  hp->time_margin = hp->t_hit - t_now - pp_.swing_time - 0.20;  // 服务窗口径
}

void HitPlanner::ReboundInverse(const double v_in[3], const double v_out[3],
                                double v_racket[3], double n_face[3]) const {
  const double vo = std::sqrt(v_out[0]*v_out[0]+v_out[1]*v_out[1]+v_out[2]*v_out[2]);
  for (int i = 0; i < 3; ++i) n_face[i] = v_out[i] / std::max(vo, 0.01);
  const double e = pp_.restitution;
  const double v_in_n = v_in[0]*n_face[0] + v_in[1]*n_face[1] + v_in[2]*n_face[2];
  const double vr_n = (vo - (1.0 - e) * v_in_n) / (1.0 + e);
  for (int i = 0; i < 3; ++i) v_racket[i] = n_face[i] * vr_n;
}

void HitPlanner::BaseTargetFromHit(const double p_hit[3], const double base_cur[3],
                                   double target[3]) const {
  const double yaw = std::atan2(p_hit[1]-base_cur[1], p_hit[0]-base_cur[0]);
  const double back = 0.8;
  target[0] = p_hit[0] - back * std::cos(yaw);
  target[1] = p_hit[1] - back * std::sin(yaw);
  target[2] = yaw;
}

double HitPlanner::Score(const HitPlan& hp, const BaseState& base, double t_now) const {
  if (!hp.feasible) return -1e9;
  // 裕度只需够启动挥拍(封顶);越早接触越好——晚等累积反弹/模型误差风险
  // (旧版 10·log1p(margin) 奖励晚击,规划器偏爱反弹后远期目标,闭环 0 命中主因)
  double s = 3.0 * std::min(hp.time_margin, 0.30);
  s -= 1.5 * (hp.t_hit - t_now);
  const double h_mid = 0.5 * (pp_.height_lo + pp_.height_hi);
  s += 5.0 * (1.0 - std::fabs(hp.p_hit[2] - h_mid) / (0.5*(pp_.height_hi - pp_.height_lo)));
  const double travel = std::hypot(hp.base_target[0]-base.x, hp.base_target[1]-base.y);
  if (travel > pp_.max_base_travel) return -1e9;
  s -= 2.0 * travel;
  return s;
}

HitPlan HitPlanner::Plan(const BallPredictor& predictor, const BaseState& base,
                         double t_now, const HitIntent& intent, double min_lead_s) {
  HitPlan best; best.feasible = false; best.score = -1e9;
  // 基线策略:只打反弹前(反弹模型未标定,误差 ~0.8m;对齐 trobot MLP 域边界)
  const double t_bounce = predictor.NextBounceTime();
  const double t_end = (t_bounce > 0.0) ? std::min(t_bounce, t_now + 2.5) : t_now + 2.5;
  const double lead = (min_lead_s > 0.0) ? min_lead_s : pp_.prep_time + pp_.swing_time;
  for (double t = t_now + lead; t < t_end; t += 0.02) {
    double p[3], v_in[3];
    if (!predictor.PredictFull(t, p, v_in)) continue;
    if (p[2] < pp_.height_lo || p[2] > pp_.height_hi) continue;
    const double margin = t - t_now - lead;  // 相对有效提前量(重规划时 lead=服务窗)
    if (margin < 0) continue;
    double bt[3]; double bc[3] = {base.x, base.y, base.yaw};
    BaseTargetFromHit(p, bc, bt);
    if (std::hypot(bt[0]-base.x, bt[1]-base.y) > pp_.max_base_travel) continue;
    const double yaw = intent.v_out_yaw_deg * M_PI / 180.0;
    const double el = intent.v_out_elev_deg * M_PI / 180.0;
    double v_out[3] = {intent.v_out_speed*std::cos(el)*std::cos(yaw),
                       intent.v_out_speed*std::cos(el)*std::sin(yaw),
                       intent.v_out_speed*std::sin(el)};
    double vr[3], nf[3];
    ReboundInverse(v_in, v_out, vr, nf);
    HitPlan hp;
    for (int i=0;i<3;++i) { hp.p_hit[i]=p[i]; hp.base_target[i]=bt[i]; hp.v_racket[i]=vr[i]; hp.n_face[i]=nf[i]; }
    hp.t_hit=t; hp.time_margin=margin; hp.feasible=true;
    hp.score = Score(hp, base, t_now);
    if (hp.score > best.score) best = hp;
  }
  return best;
}

}  // namespace sckd
