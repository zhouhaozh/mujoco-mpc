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
  // 精化合法性:预测噪声不得把锚定击球点拖出高度窗——早期(起拍后 ~0.3s)预测
  // 会给出 z≈0.25 的地下候选,IK 必败 → 起挥 Init 失败(2026-10-07 随机族定位);
  // 越窗即保持旧值,等预测收敛后再精化
  if (p[2] < pp_.height_lo || p[2] > pp_.height_hi) return;
  for (int i = 0; i < 3; ++i) { hp->p_hit[i] = p[i]; hp->v_in[i] = v_in[i]; }
  double bc[3] = {base.x, base.y, base.yaw};
  BaseTargetFromHit(hp->p_hit, bc, hp->base_target);
  hp->time_margin = hp->t_hit - t_now - pp_.swing_time - 0.20;  // 服务窗口径
  // v_base_star 不动:fly-through 可行性取决于初规划时跑道 D₀(锚点→目标),
  // 用途中位置重算会使 D 随接近收缩、v_base* 在触球前塌到 0 —— 意图与参考
  // 起锚一样属"冻结量"(方案 B′ 交接契约,2026-10-07)。
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
  // 基线策略:只打反弹前(反弹模型未标定,误差 ~0.8m;对齐 trobot MLP 域边界)。
  // 反弹时刻保护带:早期(起拍 ~0.1s)预测的反弹时刻误差 ±0.15s(vz 预热 σ0.7),
  // 贴着 t_bounce 选锚会把 t_hit 锚进"真值已反弹"区——RefineAt 锚定不可移,
  // 事后无法自救(随机族实测 plan_err 0.68/0.99,2026-10-07)
  const double t_bounce = predictor.NextBounceTime();
  const double t_end = (t_bounce > 0.0)
      ? std::min(t_bounce - pp_.bounce_guard, t_now + 2.5) : t_now + 2.5;
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
    // 时间可达性:跑道须在 t_hit 前以 v_max 均速跑完(必要条件;无加速/跟踪裕度,
    // 边界候选由 Score 的 −2·travel 再压)。随机族实测:1.9-2.2m 跑道配 0.9s 时窗
    // (均速 2.1-2.4 m/s)被旧判据误收 → 底盘物理到不了,必 miss(2026-10-07)
    if (std::hypot(bt[0]-base.x, bt[1]-base.y) > pp_.base_v_max * (t - t_now)) continue;
    const double yaw = intent.v_out_yaw_deg * M_PI / 180.0;
    const double el = intent.v_out_elev_deg * M_PI / 180.0;
    double v_out[3] = {intent.v_out_speed*std::cos(el)*std::cos(yaw),
                       intent.v_out_speed*std::cos(el)*std::sin(yaw),
                       intent.v_out_speed*std::sin(el)};
    double vr[3], nf[3];
    ReboundInverse(v_in, v_out, vr, nf);
    HitPlan hp;
    for (int i=0;i<3;++i) { hp.p_hit[i]=p[i]; hp.base_target[i]=bt[i]; hp.v_in[i]=v_in[i]; hp.v_racket[i]=vr[i]; hp.n_face[i]=nf[i]; }
    // v_base*:接近方向 d̂ 上 v*_xy 投影(≤0 不飞);上限含 Hermite 可行域
    // v·Ta/D₀ ≤ 2.2(base_ref_gen 终端速度剖面单调性约束),D₀=本次跑道
    {
      const double ddx = bt[0]-base.x, ddy = bt[1]-base.y;
      const double D0 = std::hypot(ddx, ddy);
      const double va = (vr[0]*ddx + vr[1]*ddy) / std::max(D0, 1e-6);
      if (D0 > 0.30 && va > 0.05) {
        // 上限:规则 0(B′ 原)= min(1.2, 2.2x) 标量封顶;规则 1(方案 B v0
        // 蒸馏,wholebody_ocp sweep 2026-10-07)= 2.2x/(1+0.55x) 饱和曲线——
        // 联合最优的 d̂ 投影拟合(40 工况 OCP:小 x 退化为原 Hermite 渐近,
        // 长 x 长到 ~2.0 顶每轴钳;OCP 显示长跑道最优把出球方向分量顶满每轴
        // clamp,旧 1.2 封顶丢掉该红利;w=v·T/D=2.2/(1+0.55x)≤2.2 恒满足
        // Hermite 单调域)。每轴伺服 clamp 1.5 仍是物理兜底
        const double x = D0 / std::max(t - t_now, 0.3);
        const double cap = (pp_.fly_rule == 1)
            ? 2.2 * x / (1.0 + pp_.fly_sat * x)
            : std::min(pp_.fly_speed_max, 2.2 * x);
        const double vf = std::min(va, cap);
        hp.v_base_star[0] = vf * ddx / D0;
        hp.v_base_star[1] = vf * ddy / D0;
      }
    }
    hp.t_hit=t; hp.time_margin=margin; hp.feasible=true;
    hp.score = Score(hp, base, t_now);
    if (hp.score > best.score) best = hp;
  }
  return best;
}

}  // namespace sckd
