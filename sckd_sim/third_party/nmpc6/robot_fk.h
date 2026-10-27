#pragma once
#include <string>
#include <stdexcept>
#include "sim_fk.h"
#include "x4_950_fk.h"

namespace sckd_nmpc6 {

enum class RobotGeometry { kTennis, kX4950 };

inline RobotGeometry ParseRobotGeometry(const std::string& name) {
  if (name == "x4_tennis" || name == "tennis_robot") return RobotGeometry::kTennis;
  if (name == "x4_950" || name == "X4_950") return RobotGeometry::kX4950;
  throw std::invalid_argument("unsupported NMPC robot geometry: " + name);
}

template <typename T>
void RobotArmPos(RobotGeometry robot, const T base[2], const T q[5], T p[3]) {
  if (robot == RobotGeometry::kX4950) X4950ArmPos(base, q, p);
  else SimArmPos(base, q, p);
}

template <typename T>
void RobotArmNormal(RobotGeometry robot, const T q[5], T n[3]) {
  if (robot == RobotGeometry::kX4950) X4950ArmNormal(q, n);
  else SimArmNormal(q, n);
}

}  // namespace sckd_nmpc6
