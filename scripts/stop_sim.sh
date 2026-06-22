#!/usr/bin/env bash
set -euo pipefail

patterns=(
  "m20pro_sim.launch.py"
  "m20pro_dual_lidar_simulator"
  "m20pro_dynamic_obstacle_simulator"
  "m20pro_sim_health_monitor"
  "m20pro_pointcloud_fusion"
  "m20pro_web_dashboard"
  "rviz2"
)

for pattern in "${patterns[@]}"; do
  mapfile -t pids < <(ps -eo pid=,args= | awk -v pat="${pattern}" '$0 ~ pat && $0 !~ /awk/ {print $1}')
  if [[ "${#pids[@]}" -gt 0 ]]; then
    echo "[stop_sim] stopping ${pattern}: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
  fi
done

sleep 1
ps -eo pid,args | awk '/m20pro_sim|dual_lidar|dynamic_obstacle|pointcloud_fusion|web_dashboard|rviz2/ && !/awk/ {print}' || true
