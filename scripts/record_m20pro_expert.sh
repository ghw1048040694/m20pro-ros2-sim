#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_vla_env.sh"
exec python "${SCRIPT_DIR}/record_m20pro_expert.py" --headless "$@"
