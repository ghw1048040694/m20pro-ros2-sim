#!/usr/bin/env bash
set -euo pipefail

echo "[status_sim] processes:"
ps -eo pid,args | awk '/m20pro_sim|dual_lidar|dynamic_obstacle|pointcloud_fusion|web_dashboard|rviz2/ && !/awk/ {print}' || true

echo
echo "[status_sim] web health:"
curl -fsS http://127.0.0.1:8080/healthz 2>/dev/null || true

if command -v ros2 >/dev/null 2>&1; then
  echo
  echo "[status_sim] selected topics:"
  ros2 topic list 2>/dev/null | grep -E '^/(cloud_nav|scan|map|odom|tf|local_costmap/costmap|global_costmap/costmap|m20pro/current_floor|m20pro/stair_status)$' || true
fi
