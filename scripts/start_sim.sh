#!/usr/bin/env bash
set -euo pipefail

RVIZ="${1:-true}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
elif [[ -n "${ROS_DISTRO:-}" ]]; then
  :
else
  echo "ROS 2 environment is not sourced. Source Humble first or install /opt/ros/humble." >&2
  exit 2
fi

if [[ -f install/setup.bash ]]; then
  set +u
  source install/setup.bash
  set -u
fi

exec ros2 launch m20pro_bringup m20pro_sim.launch.py rviz:="${RVIZ}"
