#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_smolvla_env.sh"

exec "${M20PRO_SMOLVLA_PYTHON}" "${SCRIPT_DIR}/audit_m20_smolvla_data.py" "$@"
