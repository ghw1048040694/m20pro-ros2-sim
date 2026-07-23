#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_smolvla_env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /tmp
exec "${M20PRO_SMOLVLA_PYTHON}" "${SCRIPT_DIR}/check_smolvla.py" "$@"
