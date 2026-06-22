# M20 Pro ROS 2 Sim

这是 M20 Pro 仿真项目，只服务上位机本地仿真、RViz 调参、地图/楼层/任务逻辑验证。真机 104 启动、自检、开机自启、原始雷达 relay 和现场脚本已经拆到独立 real 仓库。

## 编译

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 启动

带 RViz：

```bash
./scripts/start_sim.sh true
```

不带 RViz：

```bash
./scripts/start_sim.sh false
```

直接 launch：

```bash
ros2 launch m20pro_bringup m20pro_sim.launch.py
```

网页默认地址：

```text
http://127.0.0.1:8080
```

## 停止和状态

```bash
./scripts/status_sim.sh
./scripts/stop_sim.sh
```

## 仿真链路

仿真启动会拉起：

- `sim_bridge`，以 `m20pro_tcp_bridge` 节点名发布仿真位姿/odom/TF；
- `dual_lidar_simulator`，从 PCD 地图裁剪局部点云；
- `pointcloud_fusion`，生成 `/scan`；
- Nav2 map server、planner、controller、BT navigator；
- `floor_manager` 和 `floor_goal_bridge`；
- 可选 `dynamic_obstacle_simulator`；
- 可选网页前端和 RViz。

## 关键文件

```text
src/m20pro_bringup/launch/m20pro_sim.launch.py
src/m20pro_bringup/config/m20pro.yaml
src/m20pro_bringup/config/nav2_params_sim.yaml
src/m20pro_bringup/config/map_manifest.yaml
src/m20pro_bringup/config/inspection_waypoints.yaml
src/m20pro_bringup/rviz/m20pro_sim.rviz
src/m20pro_navigation/m20pro_navigation/sim_bridge_node.py
src/m20pro_navigation/m20pro_navigation/dual_lidar_simulator.py
src/m20pro_navigation/m20pro_navigation/dynamic_obstacle_simulator.py
```

## 注意

- 本仓库不控制真实机器狗。
- 本仓库不安装 systemd 自启动。
- 本仓库不访问 103/104/106 的真机服务。
- 仿真沿用 `/m20pro_tcp_bridge/...` 话题命名，是为了让网页和任务逻辑与真机接口形状一致。
- `m20pro日志.md` 保留了历史开发记录，其中会提到 real 内容；新的真机开发请到 real 仓库继续。
