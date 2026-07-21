#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_vla_env.sh"
exec python "${SCRIPT_DIR}/play_m20_retargeted.py" --headless "$@"
