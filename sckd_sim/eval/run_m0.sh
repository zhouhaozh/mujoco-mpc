#!/usr/bin/env bash
# M0 验收:模型管线 + 单元测试 + 四项验收(确定性/吞吐/静止接触/臂阶跃)。
# 用法: bash eval/run_m0.sh          (在 sckd_sim/ 下执行)
# 输出: eval/reports/m0_report.txt   退出码 = 失败项数
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SCKD_PY:-/home/ubuntu/miniconda3/envs/sim/bin/python}"
BUILD="$ROOT/build"
CFG="$ROOT/models/config"
GEN="$ROOT/models/generated"
LOG="$ROOT/eval/reports/m0_report.txt"
mkdir -p "$BUILD" "$(dirname "$LOG")"
: > "$LOG"

pass=0; fail=0
note() { echo "$1" | tee -a "$LOG"; }
run() {
  local name="$1"; shift
  echo "--- $name" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then note "[PASS] $name"; pass=$((pass+1));
  else note "[FAIL] $name"; fail=$((fail+1)); fi
}

note "=== M0 acceptance $(date '+%F %T') ==="
note "T-N 包络为手册读图估值(calibrated=false),数值结论以此为边界"

(cd "$ROOT" && "$PY" models/build_model.py) >>"$LOG" 2>&1 \
  && { note "[PASS] model_pipeline"; pass=$((pass+1)); } \
  || { note "[FAIL] model_pipeline"; fail=$((fail+1)); }

if cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >>"$LOG" 2>&1 \
   && cmake --build "$BUILD" -j >>"$LOG" 2>&1; then
  note "[PASS] build"; pass=$((pass+1))
else
  note "[FAIL] build(见日志)"; fail=$((fail+1)); note "pass=$pass fail=$fail"; exit "$fail"
fi

run unit_tests        "$BUILD/test_motor_model"
run determinism       "$BUILD/m0_determinism"       "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json"
run step_rate         "$BUILD/m0_step_rate"         "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json"
run static_contact    "$BUILD/m0_static_contact"    "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json"
run arm_step_response "$BUILD/m0_arm_step_response" "$GEN/x4_tennis_scene.mjcf" "$CFG/motors.json" "$CFG/x4_tennis.json"

note "=== summary: pass=$pass fail=$fail ==="
exit "$fail"
