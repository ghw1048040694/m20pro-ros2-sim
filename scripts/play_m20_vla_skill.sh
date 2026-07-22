#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env TERM=xterm python "${SCRIPT_DIR}/play_m20_vla_skill.py" --headless --video "$@"
