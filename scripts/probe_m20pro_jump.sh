#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_vla_env.sh"
TERM=xterm timeout --signal=INT --kill-after=10s 120s \
  python "${SCRIPT_DIR}/probe_m20pro_jump.py" --headless "$@"
