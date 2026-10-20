#!/usr/bin/env bash
# M1 验收:单测 + M0 回归 + 三件套(轨迹跟踪/臂重力前馈/里程计对照)。
# 用法: bash eval/run_m1.sh     (在 sckd_sim/ 下执行)
# 输出: eval/reports/m1_report.txt;退出码 = 失败项数
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
CFG="$ROOT/models/config"
GEN="$ROOT/models/generated"
LOG="$ROOT/eval/reports/m1_report.txt"
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

pass=0; fail=0
note() { echo "$1" | tee -a "$LOG"; }
run() {
  local name="$1"; shift
  echo "--- $name" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then note "[PASS] $name"; pass=$((pass+1));
  else note "[FAIL] $name"; fail=$((fail+1)); fi
}

note "=== M1 acceptance $(date '+%F %T') ==="
note "边界: T-N 读图估值(calibrated=false);连杆惯量未辨识;地面摩擦名义值;增益=仿真口径(实机作初值,行为对齐)"

run unit_motor   "$BUILD/test_motor_model"
run unit_swerve  "$BUILD/test_swerve"
run m0_determinism "$BUILD/m0_determinism" "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json" "$CFG/chassis_m1.json"
run m0_static    "$BUILD/m0_static_contact" "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json" "$CFG/chassis_m1.json"
run m1_traj      "$BUILD/m1_traj_tracking"  "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json" "$CFG/chassis_m1.json"
run m1_gravity   "$BUILD/m1_arm_gravity"    "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json" "$CFG/chassis_m1.json"
run m1_odom      "$BUILD/m1_odom"           "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json" "$CFG/chassis_m1.json"
run m1_mpc       "$BUILD/m1_mpc_tracking"   "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json" "$CFG/chassis_m1.json"

note "=== summary: pass=$pass fail=$fail ==="
exit "$fail"
