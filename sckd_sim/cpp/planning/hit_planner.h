#pragma once
// 命中点规划(§6.5):从球预测器采样候选 (P_hit, t*),
// 评分 = 可达性 + 时间裕度 + 击球高度区;输出预击球位姿 + 反弹逆解 (v*, n*)。
#include "estimation/ball_predictor.h"
#include "core/types.h"
#include "sim/ball_aero.h"

namespace sckd {

struct HitIntent {
  double p_hit[3] = {0.9, 0.0, 1.3};
  double v_out_speed = 8.0;
  double v_out_yaw_deg = 0.0;
  double v_out_elev_deg = 15.0;
  double t_remain = 3.0;
  int hit_type = 0;
};

struct HitPlan {
  double p_hit[3] = {0, 0, 0};
  double t_hit = 0;
  double base_target[3] = {0, 0, 0};
  double v_racket[3] = {0, 0, 0};
  double n_face[3] = {0, 0, 1};
  bool feasible = false;
  double score = 0;
  double time_margin = 0;
};

struct PlannerParams {
  double height_lo = 0.5, height_hi = 2.0;
  double arm_reach_min = 0.4, arm_reach_max = 1.1;
  double swing_time = 0.35, prep_time = 0.55;
  double max_base_travel = 5.0;
  double restitution = 0.60;
};

class HitPlanner {
 public:
  HitPlanner(const PlannerParams& pp, const BallAeroParams& aero);
  // min_lead_s:候选击球时刻相对 t_now 的最小提前量;-1(默认)取 prep+swing
  // (首次规划,底盘需就位时间);重规划传挥拍服务窗(约 0.55s,底盘已在途)
  HitPlan Plan(const BallPredictor& predictor, const BaseState& base,
               double t_now, const HitIntent& intent, double min_lead_s = -1.0);
  // 供重规划与在位计划在同一 t_now 下公平比分
  double ScoreOf(const HitPlan& hp, const BaseState& base, double t_now) const {
    return Score(hp, base, t_now);
  }
  // 重规划精化:t_hit 锚定不动,仅用最新预测更新 p_hit/base_target/margin。
  // (用评分比较新旧计划会系统性拒绝精化——高度偏好偏爱陈旧的高 z 计划,
  //  2026-10-06 定位;新鲜信息无条件更优,不需要比分)
  void RefineAt(const BallPredictor& predictor, const BaseState& base,
                HitPlan* hp, double t_now);
 private:
  void ReboundInverse(const double v_in[3], const double v_out[3],
                      double v_racket[3], double n_face[3]) const;
  void BaseTargetFromHit(const double p_hit[3], const double base_cur[3],
                         double target[3]) const;
  double Score(const HitPlan& hp, const BaseState& base, double t_now) const;
  PlannerParams pp_;
  BallAeroParams aero_;
};

}  // namespace sckd
