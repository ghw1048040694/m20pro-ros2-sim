#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_vla_env.sh"
TERM=xterm timeout --signal=INT --kill-after=15s 180s \
  python "${SCRIPT_DIR}/play_m20pro_ppo.py" --headless "$@"
