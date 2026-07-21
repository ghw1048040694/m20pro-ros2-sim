# M20 Pro VLA 具身智能仿真项目记录

Last updated: 2026-07-21 CST

## 项目边界

- 本项目只做仿真，不做 sim2real。
- 工作区：`/home/fabu/桌面/M20Pro/m20pro_sim_ros2_ws`。
- 真机工作区：`m20pro_real_ros2_ws`，不在本项目中修改、部署或控制。
- 目标是语言条件下的开放词汇导航：指定地点、指定物体、自主探索，并在需要时跨越 1 m 障碍。
- 长期方向是 VLA（vision-language-action）策略，不把 Nav2/A*/AMCL 作为主要决策器。

## 技术基线

- Conda：`m20pro-vla`
- Python 3.11.15
- PyTorch 2.7.0+cu128
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2 / package 0.54.2
- RSL-RL 3.1.2
- GPU：RTX 3060 12 GB，compute capability 8.6
- NVIDIA EULA 已由用户阅读并接受；激活脚本使用 `OMNI_KIT_ACCEPT_EULA=Y`

## 已完成

### 2026-07-20 仿真环境

- 完成独立 Conda 环境、Isaac Sim/Lab 和 CUDA 验证。
- 完成 `scripts/install_vla_env.sh`、`activate_vla_env.sh` 和 `check_vla_env.sh`。
- 修正 EULA 变量，Isaac Sim headless 启动到 `[INFO]: Setup complete...`。

### 2026-07-20 M20 Pro USD 资产

- 使用 `src/m20pro_description/urdf/M20.urdf` 转换 USD。
- 结构为 17 个刚体、16 个关节：12 个腿关节、4 个轮关节。
- 所有 STL 视觉网格和 URDF 中的简化碰撞体均已找到。
- [m20pro_cfg.py](assets/m20pro/m20pro_cfg.py) 已配置 DC motor 腿执行器和轮速度执行器。
- 240 个物理步测试通过：根高度稳定在约 0.590 m，无 NaN。

### 2026-07-20 第一个 RL 任务

- 新增 `tasks/m20pro_locomotion/`。
- 平地 +X 移动基线：16 维动作、60 维本体观测、1024 并行环境、20 s episode。
- [check_m20pro_task.py](scripts/check_m20pro_task.py) 已通过 headless 配置校验。
- 用户实际运行后复核输出为 `configuration valid`，`action_space=16`、`observation_space=60`、`num_envs=1024`、`episode_length_s=20.0`。
- 同次输出的 CPU powersave 和 IOMMU 为 Isaac Sim 运行环境警告，不影响本次任务配置校验；后续训练前可将 CPU governor 调整为 performance 以降低时延。
- 新增 `smoke_m20pro_locomotion.py`；Isaac Lab 已成功创建 M20 articulation 和平地场景，日志显示 17 个 body / 16 个 joint。
- 修正连续轮关节的无限位置归一化（避免 `Inf/Inf` 导致轮位置观测 NaN）。现在 1 个环境运行 4 步的 reset/step smoke 测试通过：`obs=(1, 60)`、`reward_mean=0.9000`、`terminated=0`。
- 已注册 Gymnasium 任务 `M20Pro-Locomotion-Flat-v0`，并加入 RSL-RL PPO runner 配置。`check_m20pro_gym_registration.py` 已在 headless Kit 中验证 entry point 和 registry。
- 根据 RTX 3060 资源调低初始训练设置：默认任务并行环境现调为 128，PPO 脚本默认为 64。
- 用户已完成首次 PPO smoke 训练：16 个并行环境、2 次迭代。本地已产生 `logs/rsl_rl/m20pro_locomotion_smoke/model_0.pt` 和 `model_1.pt`，并有 TensorBoard event 文件；训练进程已退出、显存已释放。
- 完成并行环境基准：16、32、64、128、256 均通过 4 步 reset/step，结果保存在 `logs/m20pro_env_benchmark/summary.tsv`。因为这是短测试，首轮长训练先用 64，再升到 128/256。
- 完成 64 环境、100 次 PPO 训练，生成 `logs/rsl_rl/m20pro_locomotion_64_env_100/model_99.pt`。TensorBoard 中 `mean_reward` 从约 104.9 到 91.9，`mean_episode_length` 从 54.5 到 42.3；PPO 确实更新，但当前策略有摔倒趋势，需先回放和调整奖励/初始姿态。
- 新增 `play_m20pro_ppo.sh`，用于无头回放 checkpoint，并输出前进位移、最低根高度和终止步数。
- 回放 `model_99.pt` 得到：1000 步平均 +X 位移仅 `0.0038 m`，最低根高度 `0.3594 m`，终止步数 0。结论是策略学会了原地站立，没有学会前进。
- 针对奖励漏洞完成第二版环境：改用 1 m/s 前向速度指数跟踪奖励，降低静止存活奖励，增加侧向/垂向/角速度和动作代价。轮关节改为显式 DC motor 力矩执行器，PPO rollout 由 smoke 的 4 步恢复到 24 步，初始探索噪声由 1.0 降到 0.5。
- 新环境已通过 4 环境 × 24 步 smoke：`obs=(4, 60)`、`reward_mean=0.5866`、`terminated=0`。
- v2 `model_299.pt` 旧回放统计曾报告位移仅 `0.0010 m`，但该统计只计算最终世界坐标减初始坐标，且没有正确累计 done，不能用于自动 reset 环境。修正为速度积分和真实 done 计数后，4 环境 × 200 步结果为：平均积分前进 `3.7245 m`、平均前向速度 `0.9311 m/s`、最低根高度 `0.3546 m`、`done_count=0`、平均绝对动作 `0.6841`。策略已学会接近 1 m/s 目标的前进，但根高度距 0.35 m 终止阈值仅约 4.6 mm，需增加姿态/高度裕量。
- 回放脚本新增 `--video`，可使用 headless rendering 录制单环境 MP4，避免有头 Isaac Sim GUI 的持续开销。已生成 `videos/m20pro_locomotion_v2/model_299-step-0.mp4`（400 步，473833 bytes）；`videos/` 为本地产物，不提交 Git。
- 用户从视频观察到机器人像在无动作平移。增加关节动作/速度诊断后确认：机身速度 `0.9311 m/s`，轮速 `10.3702 rad/s`，按 0.09 m 轮半径计算表面速度为 `0.9333 m/s`，二者仅差约 0.2%，因此不是接触滑移漏洞，而是正常轮式滚动。腿关节平均速度仅 `0.1617 rad/s`，轮子轴对称且无可视标记，所以视频中像无动作平移。
- 范围修正：当前 v2 只是平地轮式移动基线，不是最终轮腿/跳跃策略。平地奖励下策略必然倾向最省力的轮式方案；跳跃阶段必须单独建立禁用/锁定轮驱动的腿部训练课程，再与轮式策略组合。
- 视频还显示 v2 机身低贴、腿姿不对称。关节统计确认策略通过收腿降低重心，这是缺少机身高度/站立构型约束造成的奖励漏洞。v3 加入 0.59 m 机身高度指数跟踪、腿关节零位姿态代价，并把终止高度从 0.35 m 提高到 0.45 m。4 环境 × 24 步 smoke 通过：`reward_mean=1.5866`、`terminated=0`。
- 已记录 PPO 指标含义：`Loss/value_function` 是 critic 价值回归误差，`Loss/surrogate` 是 PPO 裁剪策略损失，`Policy/mean_noise_std` 是 actor 输出动作分布的平均探索标准差。

## 常用验证

```bash
cd /home/fabu/桌面/M20Pro/m20pro_sim_ros2_ws
source scripts/activate_vla_env.sh
./scripts/convert_m20pro_urdf.sh
./scripts/test_m20pro_asset.sh
TERM=xterm python scripts/check_m20pro_task.py --headless
```

`isaaclab.sh` 运行时需要 `TERM=xterm`，以避免 `TERM=dumb` 下 `tabs` 命令失败。

## PPO / TensorBoard 指标词典

### 当前 locomotion 奖励函数

实现位于 `tasks/m20pro_locomotion/m20pro_locomotion_env.py::_get_rewards()`。它不是云深处官方奖励，也不是从某个现成机器人任务原样复制；它使用常见的速度跟踪 locomotion 结构，并针对旧策略“原地站立即获得高奖励”的回放结果设定了第一组工程初值。

```text
reward =
    2.0   * exp(-(vx - 1.0)^2 / 0.25)   # 跟踪 1 m/s 前向速度
  + 1.0   * exp(-(base_z - 0.59)^2/0.0025) # 跟踪正常机身高度
  + 0.5   * upright                      # 保持机身直立
  + 0.05                                 # 小额存活奖励
  - 0.20  * ||leg_joint_position||^2     # 保持对称零位站姿
  - 0.10  * vy^2                         # 抑制侧滑
  - 0.20  * vz^2                         # 抑制垂向抖动
  - 0.02  * ||angular_velocity||^2       # 抑制机身旋转
  - 0.005 * ||action||^2                 # 抑制过大动作
```

根高度低于 `0.45 m` 终止 episode，终止步奖励为 `-2.0`。这组系数仍是工程基线，必须根据回放中的前向速度、倒地率、姿态和能耗继续调整。

### `Loss/value_function`

价值网络（critic）的回归误差。critic 学习预测 `V(s)`，即从当前状态开始未来可获得的累计奖励。该损失衡量预测回报与实际回报的差异。

- 通常越低越好，但训练早期或状态分布快速变化时可能暂时升高。
- 持续很高表明 critic 无法稳定区分好状态和差状态，常见原因是奖励尺度过大、episode 频繁终止或 rollout 过短。
- 本次 64 环境 / 100 iterations 从约 `2.9` 升到 `399`，明显偏高，需检查摔倒、奖励尺度和 `num_steps_per_env`。

### `Loss/surrogate`

PPO 的 actor 策略损失。它根据新旧策略对同一动作的概率比率和 advantage 更新策略，并使用 clipping 防止单次更新过大。

- 它不是简单的“越低越好”，正常情况下在 0 附近小幅波动。
- 绝对值很大可能表示策略更新过猛；长期非常接近 0 可能表示已收敛，也可能是 advantage 太小或没有有效改善方向。
- 本次从约 `-0.072` 到 `-0.0026`，策略更新逐渐变小；由于 mean reward 同时下降，不能解读为成功收敛。

### `Policy/mean_noise_std`

Actor 输出的动作高斯分布的平均标准差。策略从 `Normal(mean_action, noise_std)` 采样动作，标准差决定探索强度。

- 数值高表示动作更随机、探索更强；数值低表示策略更确定。
- 过高可能产生关节抖动和摔倒；过低可能过早停止探索，陷入差策略。
- 本次从约 `0.999` 到 `0.995`，几乎没有下降，表明策略仍在强随机探索；对当前容易摔倒的 M20 任务可能过高。

### 当前综合判断

当前曲线组合是 critic 估值趋于不稳定、actor 更新趋于变小、探索噪声仍很高，同时 mean reward 和 episode length 下降。因此暂不认定平地移动策略已收敛，需先回放 checkpoint，检查倒地率、根高度、前向速度和奖励设计。

## 待办路线

1. 将 `M20ProLocomotionEnv` 注册成 Gym 任务，接入 RSL-RL PPO，完成平地轮腿移动训练。
2. 回放 `model_99.pt`，统计倒地率、根高度和 +X 速度，调整平地站立/前进奖励。
3. 加入相机、激光雷达和语言 embedding 观测，保持一个统一 policy 接口。
3. 加入随机障碍和 1 m 跳跃课程，再做跨障与导航联合训练。
4. 加入开放词汇物体搜索、地图/无地图切换和语言任务评测。

### 并行环境容量基准

`16` 是初始保守值，不是硬限制。`16–256` 短测试均通过，长训练按 64 → 128 → 256 逐级升级。

## 维护约定

每次本项目对话结束前，更新本文件的 `Last updated`、完成事项、验证结果、待办和当轮的重要决策。该文件只记录本 VLA 仿真兴趣项目，不记录真机部署或与本项目无关的 ROS 维修。

## GitHub 学习记录同步

- 同步仓库：`git@github.com:ghw1048040694/VLA-Learning.git`
- 本地克隆路径：`/home/fabu/桌面/VLA-Learning`
- 本文件在仓库中作为独立的 `m20pro-VLA.md`，不覆盖仓库原有的 `lerobot/Lerobot.md`。
- 后续每次更新本文件后，同步复制到该仓库并提交推送。
