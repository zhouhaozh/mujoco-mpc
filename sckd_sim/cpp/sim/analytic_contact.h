#pragma once
// 拍-球解析冲量(§3.5 拍面支"已知碰撞模型",2026-10-07)。
// 背景:软接触在合速 >10 m/s 处于穿透态——停球距离 v·τ(τ≥2dt 稳定界)≫
// 面厚+球半径的几何预算,球埋穿拍面,受控台架与真实跑动的出射均非物理
// (R²<0,rebound_calib.py 留档)。故排除求解器接触对(patch_demo §5),由本
// 头在真值侧每拍施加参数化冲量;planner ReboundInverse 用同一映射逆解。
// 模型(法向,拍面局部 y):v_out = v_face + k·(v_face − v_ball),即
//   v_out·n = a·vr_n + b·v_in_n,a = 1+k,b = −k;
// k 为回弹系数(拍面有效质量并入,初始 0.50,实机实测直接替换 —— 同一参数
// 两处生效,自洽)。切向/旋转 v1 不变(简化,地面支标定时一并处理)。
#include <cmath>
#include <mujoco/mujoco.h>

namespace sckd {

struct RacketImpulseParams {
  double k = 0.50;             // 回弹系数(实机标定替换点)
  double half_l = 0.100;       // 拍面半长(局部 x;racket_face size[0])
  double half_h = 0.130;       // 拍面半高(局部 z;size[2])
  double half_t = 0.015;       // 半厚(法向;size[1])
  double ball_r = 0.033;
  double min_closing = 0.30;   // 触发阈(m/s;冲量后 v_close 变号,天然防重触)
  double depth_back = 0.080;   // 已穿透深度容限(迟一拍仍可救)
};

struct RacketImpulseEvent {
  bool fired = false;
  double v_close = 0;          // 冲量前沿法向接近速度(<0)
  double vn_out = 0;           // 冲量后球速法向分量
  double off_l = 0, off_h = 0; // 命中偏移(面内)
  double n[3] = {0, 0, 0};     // 作用法向(世界系,指向球侧)
  double v_in[3] = {0, 0, 0};  // 冲量前球速(标定数据)
  double v_face[3] = {0, 0, 0};
};

// 每物理拍调用(拍面/球运动学须为当前步状态)。命中且施加冲量 → fired。
inline RacketImpulseEvent ApplyRacketImpulse(const mjModel* m, mjData* d,
                                             int face_gid, int ball_qadr,
                                             int ball_vadr,
                                             const RacketImpulseParams& pp) {
  RacketImpulseEvent ev;
  const double* c = d->geom_xpos + face_gid * 3;
  const double* R = d->geom_xmat + face_gid * 9;   // 行主;列 j = 局部 j 轴世界向
  const double ax[3] = {R[0], R[3], R[6]};
  const double ay[3] = {R[1], R[4], R[7]};
  const double az[3] = {R[2], R[5], R[8]};
  const double* p = d->qpos + ball_qadr;
  const double* vb = d->qvel + ball_vadr;
  double rel[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
  const double sd = rel[0] * ay[0] + rel[1] * ay[1] + rel[2] * ay[2];
  for (int i = 0; i < 3; ++i) ev.n[i] = (sd >= 0 ? 1.0 : -1.0) * ay[i];
  const double sdt = std::fabs(sd);
  ev.off_l = rel[0] * ax[0] + rel[1] * ax[1] + rel[2] * ax[2];
  ev.off_h = rel[0] * az[0] + rel[1] * az[1] + rel[2] * az[2];
  if (std::fabs(ev.off_l) > pp.half_l + pp.ball_r) return ev;   // 面外(长边)
  if (std::fabs(ev.off_h) > pp.half_h + pp.ball_r) return ev;   // 面外(高边)
  if (sdt > pp.half_t + pp.ball_r + pp.depth_back) return ev;   // 太远/太深
  double rv[6];
  mj_objectVelocity(m, d, mjOBJ_GEOM, face_gid, rv, /*flg_local=*/0);
  for (int i = 0; i < 3; ++i) {
    ev.v_face[i] = rv[3 + i];
    ev.v_in[i] = vb[i];
  }
  const double vc = (vb[0] - rv[3]) * ev.n[0] + (vb[1] - rv[4]) * ev.n[1] +
                    (vb[2] - rv[5]) * ev.n[2];
  if (vc > -pp.min_closing) return ev;                          // 未接近
  ev.v_close = vc;
  const double dv = -(1.0 + pp.k) * vc;                         // 法向增量(>0)
  double* qv = d->qvel + ball_vadr;
  for (int i = 0; i < 3; ++i) qv[i] += dv * ev.n[i];
  ev.vn_out = qv[0] * ev.n[0] + qv[1] * ev.n[1] + qv[2] * ev.n[2];
  ev.fired = true;
  return ev;
}

}  // namespace sckd
