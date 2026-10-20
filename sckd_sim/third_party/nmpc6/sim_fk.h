#pragma once
// sim(x4_tennis)精确臂 FK —— 生成器(gen_nlp.cc,SX 符号)与驱动
// (legacy_nmpc.cpp,数值 rollout)同源。常数逐项取自
// models/generated/x4_tennis_demo.mjcf(全精度,2026-10-09);公式对 MuJoCo
// 300 随机位形验证 max|err|≈4e-4(即 body_pos 打印截断级)。
//
// 链(base→tip,yaw=0 契约,base 原点高 0.209):
//   J1 yaw(−z)@Link1 → J2 pitch(+y)@Link2 → J3 pitch(−y)@Link3 →
//   J4 roll(−z)@Link4 → J5 spin(+x)@Link5;拍心/racket_face geom 恒等
//   安装于 Link5 (−0.606,0,0),拍面法向 = R·e_y(盒薄轴 y,§m2 已证)。
//
// 旧仓库 FK 差异留档:d1=0.685/a2=0.705 与本机一致;拍长 d5=0.85(羽毛球)
// vs 本机腕滚半径 0.584(网球)——旧"网球化"未改 FK,误差 0.1-0.3m 级。
#include <cmath>

namespace sckd_nmpc6 {

constexpr double kBaseZ = 0.208700;                 // base_link 原点高(地面起算)
constexpr double kL1[3] = {0.0, 0.0, 0.373000};     // Link1(J1 yaw −z)
constexpr double kL2[3] = {0.000023, 0.029000, 0.105000};   // Link2(J2 pitch +y)
constexpr double kL3[3] = {0.705000, -0.003054, 0.001284};  // Link3(J3 pitch −y)
constexpr double kL4[3] = {0.001197, -0.026501, 0.121990};  // Link4(J4 roll −z)
constexpr double kL5[3] = {0.025656, 0.000561, 0.066247};   // Link5(J5 spin +x)
constexpr double kSite[3] = {-0.606000, 0.0, 0.0};  // racket_center/@Link5

// 拍心世界位置。base=[x,y],q=[q1..q5](sim 关节,恒等映射),出 p=[x,y,z]
template <typename T>
void SimArmPos(const T base[2], const T q[5], T p[3]) {
  using std::cos;
  using std::sin;
  T x = T(kSite[0]), y = T(kSite[1]), z = T(kSite[2]);
  {  // J5 spin(+x)
    const T c = cos(q[4]), s = sin(q[4]);
    const T y2 = c * y - s * z, z2 = s * y + c * z;
    y = y2; z = z2;
  }
  x += T(kL5[0]); y += T(kL5[1]); z += T(kL5[2]);
  {  // J4 roll(−z)
    const T c = cos(-q[3]), s = sin(-q[3]);
    const T x2 = c * x - s * y, y2 = s * x + c * y;
    x = x2; y = y2;
  }
  x += T(kL4[0]); y += T(kL4[1]); z += T(kL4[2]);
  {  // J3 pitch(−y) = Ry(−q3)
    const T c = cos(-q[2]), s = sin(-q[2]);
    const T x2 = c * x + s * z, z2 = -s * x + c * z;
    x = x2; z = z2;
  }
  x += T(kL3[0]); y += T(kL3[1]); z += T(kL3[2]);
  {  // J2 pitch(+y) = Ry(q2)
    const T c = cos(q[1]), s = sin(q[1]);
    const T x2 = c * x + s * z, z2 = -s * x + c * z;
    x = x2; z = z2;
  }
  x += T(kL2[0]); y += T(kL2[1]); z += T(kL2[2]);
  {  // J1 yaw(−z)
    const T c = cos(-q[0]), s = sin(-q[0]);
    const T x2 = c * x - s * y, y2 = s * x + c * y;
    x = x2; y = y2;
  }
  p[0] = x + T(kL1[0]) + base[0];
  p[1] = y + T(kL1[1]) + base[1];
  p[2] = z + T(kL1[2]) + T(kBaseZ);
}

// 拍面法向(世界系)= R_chain·e_y(racket_face 盒薄轴;±号在代价 |n·d| 下不敏感)
template <typename T>
void SimArmNormal(const T q[5], T n[3]) {
  using std::cos;
  using std::sin;
  T x = T(0), y = T(1), z = T(0);
  {  // J5 spin(+x)
    const T c = cos(q[4]), s = sin(q[4]);
    const T y2 = c * y - s * z, z2 = s * y + c * z;
    y = y2; z = z2;
  }
  {  // J4 roll(−z)
    const T c = cos(-q[3]), s = sin(-q[3]);
    const T x2 = c * x - s * y, y2 = s * x + c * y;
    x = x2; y = y2;
  }
  {  // J3 pitch(−y)
    const T c = cos(-q[2]), s = sin(-q[2]);
    const T x2 = c * x + s * z, z2 = -s * x + c * z;
    x = x2; z = z2;
  }
  {  // J2 pitch(+y)
    const T c = cos(q[1]), s = sin(q[1]);
    const T x2 = c * x + s * z, z2 = -s * x + c * z;
    x = x2; z = z2;
  }
  {  // J1 yaw(−z)
    const T c = cos(-q[0]), s = sin(-q[0]);
    const T x2 = c * x - s * y, y2 = s * x + c * y;
    x = x2; y = y2;
  }
  n[0] = x; n[1] = y; n[2] = z;
}

}  // namespace sckd_nmpc6
