#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIM_ROOT="$PROJECT_ROOT/sckd_sim"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
VIEWER="$SIM_ROOT/python/tools/demo_viewer.py"
INTERCEPT="$SIM_ROOT/build/m2_intercept"
VIEWER_LOG="${TMPDIR:-/tmp}/sckd_demo_viewer.log"

STRATEGY="${1:-pfly}"
TRIALS="${2:-3}"

: "${DISPLAY:?DISPLAY is not set; run this from the graphical desktop session}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -x "$INTERCEPT" ]]; then
  echo "Simulation executable not found: $INTERCEPT" >&2
  exit 1
fi

viewer_pid=""
cleanup() {
  if [[ -n "$viewer_pid" ]] && kill -0 "$viewer_pid" 2>/dev/null; then
    kill "$viewer_pid" 2>/dev/null || true
  fi
}
trap cleanup ERR
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

cd "$SIM_ROOT"
nohup setsid "$PYTHON" "$VIEWER" </dev/null >"$VIEWER_LOG" 2>&1 &
viewer_pid=$!

sleep 2
if ! kill -0 "$viewer_pid" 2>/dev/null; then
  echo "MuJoCo viewer failed to start; see $VIEWER_LOG" >&2
  wait "$viewer_pid" || true
  exit 1
fi

if M2_VIEW=1 "$INTERCEPT" \
    models/generated/x4_tennis_demo.mjcf "$TRIALS" qp 0.25 2026 "$STRATEGY"; then
  simulation_status=0
else
  simulation_status=$?
fi

trap - ERR INT TERM
if (( simulation_status != 0 && simulation_status != 1 )); then
  cleanup
  echo "Simulation returned status $simulation_status; viewer closed." >&2
  exit "$simulation_status"
fi

if (( simulation_status == 1 )); then
  echo "Simulation returned 1; check the output above and the hit-rate threshold." >&2
fi
echo "Viewer remains open (PID $viewer_pid)."
echo "Viewer log: $VIEWER_LOG"
exit "$simulation_status"
