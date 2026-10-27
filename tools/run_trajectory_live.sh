#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$PROJECT_ROOT/.venv/bin/python" \
  "$PROJECT_ROOT/sckd_sim/python/tools/live_trajectory_viewer.py" "$@"
