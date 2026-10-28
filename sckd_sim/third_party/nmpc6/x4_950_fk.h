#pragma once
// X4_950.urdf / x4_950_scene.mjcf 几何;五轴,yaw=0,底座 z=0。
// 有效拍长 .865；加上 Link5 的 -.065 Y 偏置补偿，Link6 位于 (0,.930,0)。
#include <cmath>

namespace sckd_nmpc6 {

template <typename T>
void X4950RotateJoint(int j, const T& q, T v[3]) {
  using std::cos;
  using std::sin;
  const T angle = j == 1 ? q : -q;
  const T c = cos(angle), s = sin(angle);
  if (j == 0 || j == 3) {  // Joint1/4: -Z
    const T x = c*v[0] - s*v[1], y = s*v[0] + c*v[1];
    v[0] = x; v[1] = y;
  } else {                 // Joint2: +Y; Joint3/5: -Y
    const T x = c*v[0] + s*v[2], z = -s*v[0] + c*v[2];
    v[0] = x; v[2] = z;
  }
}

template <typename T>
void X4950ArmPos(const T base[2], const T q[5], T p[3]) {
  constexpr double offsets[5][3] = {
    {0, 0, .7954}, {0, -.0105, .105}, {.705, .0445, 0},
    {.000035678, -.034015, .0685}, {0, -.065, .102}};
  p[0] = T(0); p[1] = T(.865 + .065); p[2] = T(0);
  for (int j = 4; j >= 0; --j) {
    X4950RotateJoint(j, q[j], p);
    for (int r = 0; r < 3; ++r) p[r] += T(offsets[j][r]);
  }
  p[0] += base[0]; p[1] += base[1];
}

template <typename T>
void X4950ArmNormal(const T q[5], T n[3]) {
  // racket_face geom 的正 Z 薄轴,含 MJCF 导出四元数的归一化。
  constexpr double w=.500398, x=.5, y=.499602, z=.5;
  constexpr double norm2=w*w+x*x+y*y+z*z;
  n[0] = T(2*(x*z+w*y)/norm2);
  n[1] = T(2*(y*z-w*x)/norm2);
  n[2] = T((w*w-x*x-y*y+z*z)/norm2);
  for (int j = 4; j >= 0; --j) X4950RotateJoint(j, q[j], n);
}

}  // namespace sckd_nmpc6
