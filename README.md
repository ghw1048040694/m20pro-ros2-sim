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

## VLA 具身仿真环境

Isaac Sim/Isaac Lab 环境与 ROS 2 Humble 环境分开管理：

```bash
./scripts/install_vla_env.sh
source ./scripts/activate_vla_env.sh
./scripts/check_vla_env.sh
```

固定版本为 Python 3.11、Isaac Sim 5.1.0、Isaac Lab 2.3.2、PyTorch
2.7.0/CUDA 12.8 和 RSL-RL。本地 Isaac Lab 源码与仿真缓存位于已忽略的
`.deps/` 和 `.isaac-cache/`，不提交到仓库。RTX 3060 12GB 启动 Isaac
Sim 前应先停止其他 GPU 训练进程。

### M20 Pro Isaac Lab 资产

从 URDF 生成本地 USD，然后运行 2 秒无界面物理测试：

```bash
./scripts/convert_m20pro_urdf.sh
./scripts/test_m20pro_asset.sh
```

`assets/m20pro/m20pro_cfg.py` 定义了 Isaac Lab 资产。腿部 12 个关节使用
DC motor 模型，4 个轮关节使用速度型隐式执行器。生成的 USD 及派生
网格位于 `assets/m20pro/` 且不纳入 Git。

### 第一个可训练任务

`tasks/m20pro_locomotion/` 是平地轮腿移动基线：16 维关节力矩动作、
60 维本体观测、+X 方向进度奖励，默认 64 并行环境（适配 RTX 3060）。先用它验证训练链路，再加入
激光雷达、前后相机、障碍物和语言条件：

```bash
source scripts/activate_vla_env.sh
TERM=xterm python scripts/check_m20pro_task.py --headless
./scripts/smoke_m20pro_locomotion.sh --num-envs 1 --steps 4
```

VLA 兴趣项目的持续记录位于 [`m20pro-VLA.md`](m20pro-VLA.md)，每次本项目对话后更新。
