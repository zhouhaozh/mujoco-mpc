#pragma once
// 臂拍心逆运动学(有界 DLS):给定拍心世界目标 + 底盘位姿 → 5 关节角。
// 从 demo_hit 提取为正式模块;兼作臂 NLP 的终端角求解器。
#include <mujoco/mujoco.h>
#include "hal/sim_backend.h"

namespace sckd {

struct IkResult {
  double q[5] = {0};
  double err = 1e9;       // 收敛残差(m)
  bool converged = false;
};

class ArmIk {
 public:
  ArmIk(mjModel* m, mjData* d_scratch);
  // target:拍心世界位置;base_x/y/yaw:底盘位姿(IK 期间底盘视为固定)
  // Joint2/Joint3 限位缺失时 Solve 抛异常;FK 接口仍可用。
  IkResult Solve(const double target[3], double base_x, double base_y,
                 double base_yaw, const double q_seed[5], int max_iter = 300);
  bool WithinLimits(const double q[5]) const;  // 未确认 J2/J3 限位也返回 false
  // 拍面法向(世界系):racket_face 盒的正薄轴;此接口固定 yaw=0
  void FaceNormal(const double q[5], double base_x, double base_y, double n_out[3]);
  // 拍心位置(世界系)FK:整机 OCP 终端流形线性化用(2026-10-08,v2 在线 OCP)
  void SitePos(const double q[5], double base_x, double base_y, double p_out[3]);
  // 拍心位置雅可比(3×5,世界系,yaw=0):dq_hit=J⁺v* 用(臂轨迹 QP 阶段1)
  void SiteJacobian(const double q[5], double base_x, double base_y, double J[15]);

 private:
  void SetBasePose(double x, double y, double yaw);
  mjModel* m_;
  mjData* d_;
  int base_qadr_ = -1;
  double base_height_ = 0;
  int site_id_ = -1;
  int face_id_ = -1;
  int face_axis_ = -1;
  int joint_id_[5] = {};
  int qadr_[5] = {};
  int vadr_[5] = {};
};

}  // namespace sckd
