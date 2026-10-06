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
  IkResult Solve(const double target[3], double base_x, double base_y,
                 double base_yaw, const double q_seed[5], int max_iter = 300);
  // 拍面法向(世界系)FK:给定 q 与底盘位姿(yaw=0 约定同 Solve)
  void FaceNormal(const double q[5], double base_x, double base_y, double n_out[3]);

 private:
  mjModel* m_;
  mjData* d_;
  int site_id_ = -1;
  int qadr_[5] = {};
  int vadr_[5] = {};
};

}  // namespace sckd
