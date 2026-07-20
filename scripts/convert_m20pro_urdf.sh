#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/activate_vla_env.sh"

URDF="${WS_DIR}/src/m20pro_description/urdf/M20.urdf"
USD="${WS_DIR}/assets/m20pro/m20pro.usd"

mkdir -p "$(dirname "${USD}")"
export ROS_PACKAGE_PATH="${WS_DIR}/src${ROS_PACKAGE_PATH:+:${ROS_PACKAGE_PATH}}"

cd "${ISAACLAB_PATH}"
TERM=xterm ./isaaclab.sh -p scripts/tools/convert_urdf.py \
  "${URDF}" "${USD}" \
  --joint-target-type none \
  --headless

echo "Generated ${USD}"
