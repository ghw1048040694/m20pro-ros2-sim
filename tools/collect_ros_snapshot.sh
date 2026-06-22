#!/usr/bin/env bash
set -u

STAMP="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname 2>/dev/null || echo unknown_host)"
OUT="${1:-m20pro_ros_snapshot_${HOST}_${STAMP}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "$OUT"/{system,ros,topics,nodes,params,tf,maps}

run() {
  local name="$1"
  shift
  {
    echo "\$ $*"
    echo
    timeout 20 bash -lc "$*"
    local rc=$?
    echo
    echo "[exit_code=$rc]"
  } > "$OUT/$name.txt" 2>&1
}

run_long() {
  local name="$1"
  shift
  {
    echo "\$ $*"
    echo
    timeout 60 bash -lc "$*"
    local rc=$?
    echo
    echo "[exit_code=$rc]"
  } > "$OUT/$name.txt" 2>&1
}

source_ros_if_needed() {
  if command -v ros2 >/dev/null 2>&1; then
    return
  fi
  if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
    # shellcheck disable=SC1090
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    return
  fi
  for distro in foxy humble; do
    if [ -f "/opt/ros/${distro}/setup.bash" ]; then
      # shellcheck disable=SC1090
      source "/opt/ros/${distro}/setup.bash"
      return
    fi
  done
}

safe_name() {
  echo "$1" | sed 's#^/##; s#[^A-Za-z0-9_.-]#_#g'
}

topic_exists() {
  local topic="$1"
  [ -s "$topic_file" ] && awk '{print $1}' "$topic_file" | grep -Fxq "$topic"
}

source_ros_if_needed

run system/summary "date; hostname; uname -a; whoami; pwd; uptime"
run system/env_ros "printenv | sort | grep -E '^(ROS|RMW|CYCLONE|FAST|DDS|AMENT|COLCON)_' || true"
run system/network "ip addr; echo; ip route"
run system/processes "ps -eo pid,ppid,stat,pcpu,pmem,cmd --sort=cmd"
run system/sim_processes "ps -eo pid,ppid,stat,pcpu,pmem,cmd | grep -Ei 'ros|map|nav|local|lidar|dds|rviz|m20pro' | grep -v grep || true"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ros2 command not found. Source the robot ROS environment before running this script." > "$OUT/ros/ROS2_NOT_FOUND.txt"
  tar -czf "${OUT}.tar.gz" "$OUT"
  echo "Created ${OUT}.tar.gz"
  exit 0
fi

run ros/doctor_report "ros2 doctor --report"
run ros/topic_list "ros2 topic list -t"
run ros/node_list "ros2 node list"
run ros/service_list "ros2 service list -t"
run ros/action_list "ros2 action list -t"
run ros/pkg_list "ros2 pkg list"
run ros/interface_packages "ros2 interface packages"
run ros/param_list "ros2 param list"

topic_file="$OUT/ros/topic_list.txt"
if [ -s "$topic_file" ]; then
  awk '{print $1}' "$topic_file" | grep '^/' | sort -u | while read -r topic; do
    name="$(safe_name "$topic")"
    run "topics/${name}_info" "ros2 topic info -v '$topic'"
  done
fi

for topic in \
  /cloud_nav \
  /LIDAR/FRONT/POINTS \
  /LIDAR/REAR/POINTS \
  /scan \
  /map \
  /plan \
  /local_costmap/costmap \
  /global_costmap/costmap \
  /dynamic_obstacle_markers \
  /odom \
  /tf \
  /tf_static \
  /m20pro_tcp_bridge/map_pose \
  /m20pro_tcp_bridge/localization_ok \
  /m20pro_tcp_bridge/obstacle_active \
  /m20pro_tcp_bridge/navigation_status; do
  name="$(safe_name "$topic")"
  if topic_exists "$topic"; then
    run "topics/${name}_hz" "ros2 topic hz '$topic' --window 10"
    run "topics/${name}_bw" "ros2 topic bw '$topic' --window 10"
  else
    echo "Topic not present in ros2 topic list: $topic" > "$OUT/topics/${name}_missing.txt"
  fi
done

for topic in \
  /m20pro_tcp_bridge/map_pose \
  /m20pro_tcp_bridge/localization_ok \
  /m20pro_tcp_bridge/obstacle_active \
  /m20pro_tcp_bridge/navigation_status \
  /odom \
  /tf_static; do
  name="$(safe_name "$topic")"
  if topic_exists "$topic"; then
    run "topics/${name}_echo_once" "ros2 topic echo --once '$topic'"
  else
    echo "Topic not present in ros2 topic list: $topic" > "$OUT/topics/${name}_echo_missing.txt"
  fi
done

node_file="$OUT/ros/node_list.txt"
if [ -s "$node_file" ]; then
  grep '^/' "$node_file" | sort -u | while read -r node; do
    name="$(safe_name "$node")"
    run "nodes/${name}_info" "ros2 node info '$node'"
    run "params/${name}_dump" "ros2 param dump '$node'"
  done
fi

(
  cd "$OUT/tf" || exit 0
  timeout 20 ros2 run tf2_tools view_frames > view_frames_stdout.txt 2> view_frames_stderr.txt || true
)
run tf/tf_echo_map_base "ros2 run tf2_ros tf2_echo map base_link"
run tf/tf_echo_odom_base "ros2 run tf2_ros tf2_echo odom base_link"

if [ -d "${WS_DIR}/src/m20pro_bringup/maps" ]; then
  run maps/project_maps "find '${WS_DIR}/src/m20pro_bringup/maps' -maxdepth 4 -type f -o -type l | sort"
fi
if [ -f "${WS_DIR}/src/m20pro_bringup/config/map_manifest.yaml" ]; then
  cp "${WS_DIR}/src/m20pro_bringup/config/map_manifest.yaml" "$OUT/maps/map_manifest.yaml" 2>/dev/null || true
fi

run_long system/recent_logs "journalctl -n 300 --no-pager 2>/dev/null || true"

tar -czf "${OUT}.tar.gz" "$OUT"
echo "Created ${OUT}.tar.gz"
