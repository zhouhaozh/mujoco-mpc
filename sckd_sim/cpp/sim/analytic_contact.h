#pragma once
// 解析冲量("已知碰撞模型",§3.5;拍面支 2026-10-07,地面支 2026-10-08)。
// 拍面支背景:软接触在合速 >10 m/s 处于穿透态——停球距离 v·τ(τ≥2dt 稳定界)
// ≫ 面厚+球半径的几何预算,球埋穿拍面,受控台架与真实跑动的出射均非物理
// (R²<0,rebound_calib.py 留档)。故排除求解器接触对(patch_demo §5),由本
// 头在真值侧每拍施加参数化冲量;planner ReboundInverse 用同一映射逆解。
// 模型(法向,拍面局部 y):v_out = v_face + k·(v_face − v_ball),即
//   v_out·n = a·vr_n + b·v_in_n,a = 1+k,b = −k;
// k 为回弹系数(拍面有效质量并入,初始 0.50,实机实测直接替换 —— 同一参数
// 两处生效,自洽)。切向/旋转 v1 不变(简化,地面支标定时一并处理)。
//
// 地面支(2026-10-08,ground_calib.py 台架):原始求解器 ball↔floor 为
// solref(0.02,1) 临界阻尼软接触,无旋落球 e_z≈0.09 ≈ 死球(与网球 ITF
// ~0.73 差 8 倍;速度档 4.6-7.5 一致,法向残差 2.5%)→ 不拟合计入真值,
// 排除求解器对(patch_demo §6),真值侧改为本头参数化冲量:
//   法向 vz' = −e_z·vz;切向 Coulomb+stick(接触点表面速度 u=v_t;ω 耦合
//   后置——2026-10-08 用户定向:平抽球先行,旋转球后续迭代局部模块):
//   粘滞 P=(2/7)·|u|,滑动 P=μ·(1+e_z)·|vz|,取小者。
// e_z=0.73(ITF 文献初值,待实机替换)、μ=0.20(台架无旋拟合初值);
// 预测器 BounceParams 同参数(自洽,反弹段残差=估计误差而非模型失配)。
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

// ---------- 地面支(平抽 v1:法向恢复 + Coulomb+stick 切向;ω 不变) ----------
struct GroundImpulseParams {
  double e_z = 0.73;          // 法向恢复系数(ITF 文献初值;实机实测替换点)
  double mu = 0.20;           // Coulomb 滑动摩擦(ground_calib.py 无旋拟合初值)
  double ball_r = 0.033;
  double v_rest = 0.30;       // 法向反弹低于此 → 静止钳位(收敛微弹)
  double roll_decay = 0.995;  // 静止滚动每步衰减(占位耗散:排除对后无滚动摩擦)
};

struct GroundImpulseEvent {
  bool fired = false;
  bool resting = false;       // 进入静止钳位(不再弹)
  double v_in_z = 0, v_out_z = 0;
};

// 每物理拍调用(在 mj_step 前)。z≤R 且 vz<0 → 反弹冲量(vz 反号后 z 回升,
// 天然防重触);低能弹 → 静止钳位(z=R、vz=0、水平衰减)。ω 平抽不耦合。
inline GroundImpulseEvent ApplyGroundImpulse(mjData* d, int ball_qadr, int ball_vadr,
                                             const GroundImpulseParams& gp) {
  GroundImpulseEvent ev;
  double* q = d->qpos + ball_qadr;
  double* v = d->qvel + ball_vadr;
  if (q[2] > gp.ball_r) return ev;
  if (v[2] < 0) {
    const double vz_in = v[2];
    const double vt = std::hypot(v[0], v[1]);
    const double jn = (1.0 + gp.e_z) * (-vz_in);          // 法向冲量(单位质量)
    if (vt > 1e-9) {
      const double p_stick = (2.0 / 7.0) * vt;            // 粘滞:表面速度归零
      const double p = std::min(p_stick, gp.mu * jn);     // 滑动上限
      v[0] -= p * v[0] / vt;
      v[1] -= p * v[1] / vt;
    }
    if (-vz_in * gp.e_z < gp.v_rest) {                    // 低能 → 静止
      q[2] = gp.ball_r;
      v[2] = 0.0;
      v[0] *= gp.roll_decay;                              // 占位滚动耗散
      v[1] *= gp.roll_decay;
      ev.resting = true;
    } else {
      v[2] = -vz_in * gp.e_z;
    }
    ev.fired = true;
    ev.v_in_z = vz_in;
    ev.v_out_z = v[2];
  }
  return ev;
}

}  // namespace sckd
