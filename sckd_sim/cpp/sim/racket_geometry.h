#pragma once

// 拍面盒的纯几何视图。尺寸和坐标轴直接来自当前 MuJoCo 模型，既支持网球
// racket_face 的局部 Y 薄轴，也支持 X4_950 的局部 Z 薄轴。
#include <cmath>
#include <stdexcept>

#include <mujoco/mujoco.h>

namespace sckd {

struct RacketFaceGeometry {
  double center[3] = {0, 0, 0};
  double tangent[2][3] = {{0, 0, 0}, {0, 0, 0}};
  double normal[3] = {0, 0, 0};
  double half_extent[2] = {0, 0};
  double half_thickness = 0;
  int local_normal_axis = -1;
};

struct RacketPointProjection {
  double tangent[2] = {0, 0};
  double normal = 0;
};

inline RacketFaceGeometry ReadRacketFaceGeometry(const mjModel* m,
                                                 const mjData* d,
                                                 int face_gid) {
  if (!m || !d || face_gid < 0 || face_gid >= m->ngeom ||
      m->geom_type[face_gid] != mjGEOM_BOX)
    throw std::invalid_argument("racket_face must be a valid MuJoCo box geom");

  RacketFaceGeometry face;
  const mjtNum* size = m->geom_size + 3 * face_gid;
  int normal_axis = 0;
  if (size[1] < size[normal_axis]) normal_axis = 1;
  if (size[2] < size[normal_axis]) normal_axis = 2;
  const int tangent_axis[2] = {(normal_axis + 1) % 3, (normal_axis + 2) % 3};
  // 面内轴保持局部轴编号递增，使网球继续使用 local X/Z。
  const int t0 = tangent_axis[0] < tangent_axis[1] ? tangent_axis[0] : tangent_axis[1];
  const int t1 = tangent_axis[0] < tangent_axis[1] ? tangent_axis[1] : tangent_axis[0];
  const mjtNum* R = d->geom_xmat + 9 * face_gid;  // 行主；列为局部轴的世界表示。
  const mjtNum* center = d->geom_xpos + 3 * face_gid;
  for (int i = 0; i < 3; ++i) {
    face.center[i] = center[i];
    face.tangent[0][i] = R[3 * i + t0];
    face.tangent[1][i] = R[3 * i + t1];
    face.normal[i] = R[3 * i + normal_axis];
  }
  face.half_extent[0] = size[t0];
  face.half_extent[1] = size[t1];
  face.half_thickness = size[normal_axis];
  face.local_normal_axis = normal_axis;
  return face;
}

inline RacketPointProjection ProjectToRacketFace(const RacketFaceGeometry& face,
                                                 const double point[3]) {
  RacketPointProjection projection;
  double relative[3];
  for (int i = 0; i < 3; ++i) relative[i] = point[i] - face.center[i];
  for (int i = 0; i < 3; ++i) {
    projection.tangent[0] += relative[i] * face.tangent[0][i];
    projection.tangent[1] += relative[i] * face.tangent[1][i];
    projection.normal += relative[i] * face.normal[i];
  }
  return projection;
}

inline bool RacketProjectionWithin(const RacketFaceGeometry& face,
                                   const RacketPointProjection& projection,
                                   double radius, double depth_back) {
  return std::fabs(projection.tangent[0]) <= face.half_extent[0] + radius &&
         std::fabs(projection.tangent[1]) <= face.half_extent[1] + radius &&
         std::fabs(projection.normal) <= face.half_thickness + radius + depth_back;
}

}  // namespace sckd
