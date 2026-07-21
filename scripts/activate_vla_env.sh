#!/usr/bin/env bash

_M20PRO_VLA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export M20PRO_SIM_WS="$(cd "${_M20PRO_VLA_SCRIPT_DIR}/.." && pwd)"
export ISAACLAB_PATH="${M20PRO_SIM_WS}/.deps/IsaacLab"
export M20PRO_VLA_ENV="${M20PRO_VLA_ENV:-m20pro-vla}"
export OMNI_KIT_ACCEPT_EULA=Y

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${M20PRO_VLA_ENV}"

# Keep simulator caches inside the sim project boundary.
export OMNI_KIT_CACHE_PATH="${M20PRO_SIM_WS}/.isaac-cache/kit"
export OV_CACHE_ROOT="${M20PRO_SIM_WS}/.isaac-cache/ov"

# Training logs/checkpoints/videos live on the mounted 2 TB disk via workspace symlinks.
export M20PRO_OUTPUT_ROOT="${M20PRO_SIM_WS}/logs"

unset _M20PRO_VLA_SCRIPT_DIR
echo "Activated ${M20PRO_VLA_ENV} (Isaac Lab: ${ISAACLAB_PATH})"
