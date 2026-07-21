#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_vla_env.sh"
exec env TERM=xterm python "${SCRIPT_DIR}/train_m20pro_ppo.py" --headless "$@"
