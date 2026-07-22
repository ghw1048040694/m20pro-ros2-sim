# M20 Pro VLA 具身智能仿真项目记录

Last updated: 2026-07-22 CST

## 项目边界

- 本项目只做仿真，不做 sim2real。
- 工作区：`/home/fabu/桌面/M20Pro/m20pro_sim_ros2_ws`。
- 真机工作区：`m20pro_real_ros2_ws`，不在本项目中修改、部署或控制。
- 目标是语言条件下的开放词汇导航：指定地点、指定物体、自主探索，并在需要时跨越 1 m 障碍。
- 长期方向是 VLA（vision-language-action）策略，不把 Nav2/A*/AMCL 作为主要决策器。

## 训练存储

- 系统盘为 `/dev/sda3`，可用约 60 GB；Isaac Sim/环境本体仍留在系统盘。
- 2 TB 硬盘为 `/dev/sdb2`，UUID `b9cbb43d-5119-4328-99d9-10f7c0d91e37`，当前挂载于 `/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37`，剩余约 715 GB。
- 训练输出根目录：`/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA`。工作区中 `logs` 和 `videos` 是指向该目录的软链接，旧训练产物已迁移，没有再占用系统盘。
- 重启后先运行 `./scripts/prepare_output_storage.sh`；如未挂载，运行 `udisksctl mount -b /dev/disk/by-uuid/b9cbb43d-5119-4328-99d9-10f7c0d91e37`。

## 技术基线

- Conda：`m20pro-vla`
- Python 3.11.15
- PyTorch 2.7.0+cu128
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2 / package 0.54.2
- RSL-RL 3.1.2
- GPU：RTX 3060 12 GB，compute capability 8.6
- Python 运行环境确认：Isaac Sim、Isaac Lab、PyTorch 和 RSL-RL 均运行在 Conda 环境 `m20pro-vla`，解析器为 `/home/fabu/miniconda3/envs/m20pro-vla/bin/python`。
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
- 用户已完成首次 PPO smoke 训练：16 个并行环境、2 次迭代。相关 checkpoint 已在后续空间清理中删除，指标和结论保留在本日志；训练进程已退出、显存已释放。
- 完成并行环境基准：16、32、64、128、256 均通过 4 步 reset/step，结果保存在 `logs/m20pro_env_benchmark/summary.tsv`。因为这是短测试，首轮长训练先用 64，再升到 128/256。
- 完成 64 环境、100 次 PPO 训练；checkpoint 已在后续空间清理中删除。TensorBoard 中 `mean_reward` 从约 104.9 到 91.9，`mean_episode_length` 从 54.5 到 42.3；PPO 确实更新，但当前策略有摔倒趋势，需先回放和调整奖励/初始姿态。
- 新增 `play_m20pro_ppo.sh`，用于无头回放 checkpoint，并输出前进位移、最低根高度和终止步数。
- 回放 `model_99.pt` 得到：1000 步平均 +X 位移仅 `0.0038 m`，最低根高度 `0.3594 m`，终止步数 0。结论是策略学会了原地站立，没有学会前进。
- 针对奖励漏洞完成第二版环境：改用 1 m/s 前向速度指数跟踪奖励，降低静止存活奖励，增加侧向/垂向/角速度和动作代价。轮关节改为显式 DC motor 力矩执行器，PPO rollout 由 smoke 的 4 步恢复到 24 步，初始探索噪声由 1.0 降到 0.5。
- 新环境已通过 4 环境 × 24 步 smoke：`obs=(4, 60)`、`reward_mean=0.5866`、`terminated=0`。
- v2 `model_299.pt` 旧回放统计曾报告位移仅 `0.0010 m`，但该统计只计算最终世界坐标减初始坐标，且没有正确累计 done，不能用于自动 reset 环境。修正为速度积分和真实 done 计数后，4 环境 × 200 步结果为：平均积分前进 `3.7245 m`、平均前向速度 `0.9311 m/s`、最低根高度 `0.3546 m`、`done_count=0`、平均绝对动作 `0.6841`。策略已学会接近 1 m/s 目标的前进，但根高度距 0.35 m 终止阈值仅约 4.6 mm，需增加姿态/高度裕量。
- 回放脚本新增 `--video`，可使用 headless rendering 录制单环境 MP4，避免有头 Isaac Sim GUI 的持续开销。已生成 `videos/m20pro_locomotion_v2/model_299-step-0.mp4`（400 步，473833 bytes）；`videos/` 为本地产物，不提交 Git。
- 用户从视频观察到机器人像在无动作平移。增加关节动作/速度诊断后确认：机身速度 `0.9311 m/s`，轮速 `10.3702 rad/s`，按 0.09 m 轮半径计算表面速度为 `0.9333 m/s`，二者仅差约 0.2%，因此不是接触滑移漏洞，而是正常轮式滚动。腿关节平均速度仅 `0.1617 rad/s`，轮子轴对称且无可视标记，所以视频中像无动作平移。
- 范围修正：当前 v2 只是平地轮式移动基线，不是最终轮腿/跳跃策略。平地奖励下策略必然倾向最省力的轮式方案；跳跃阶段必须单独建立禁用/锁定轮驱动的腿部训练课程，再与轮式策略组合。
- 视频还显示 v2 机身低贴、腿姿不对称。关节统计确认策略通过收腿降低重心，这是缺少机身高度/站立构型约束造成的奖励漏洞。v3 加入 0.59 m 机身高度指数跟踪、腿关节零位姿态代价，并把终止高度从 0.35 m 提高到 0.45 m。4 环境 × 24 步 smoke 通过：`reward_mean=1.5866`、`terminated=0`。
- v3 训练受旧 `train_m20pro_ppo.sh` 的 180 s 硬超时影响，实际完成 279/300 次更新，最后定期 checkpoint 为 `model_250.pt`。已移除训练脚本的固定超时，后续长训练不再被 180 s 截断。
- v3 `model_250.pt` 在 16 环境 × 1000 步评估中：平均前向速度 `1.0432 m/s`、最低根高度 `0.5459 m`、轮表面速度 `1.0484 m/s`、16 个 done 全部对应 1000 步/20 s time limit，无提前倒地。相比 v2，机身最低高度由约 0.355 m 提高到 0.546 m。
- v3 已录制 `videos/m20pro_locomotion_v3/model_250-step-0.mp4`（400 步，415325 bytes）用于姿态视觉检查。
- v3 视频复核显示：虽然机身高度已改善，但腿部仍维持僵硬支撑，整体视觉上像平移。这不是继续调整平地速度奖励能解决的问题：对轮腿机器人，无障碍平地上的最优低能耗行为本来就是静态支腿+轮式滚动；人为奖励摆腿只会导致无效抖动。

### 轮腿技能分层决策

- `rolling` 巡航技能：只输出 4 个轮关节动作，12 个腿关节由学习到的/固定站姿控制器保持；用于平地高效移动。
- `jump` 跳跃技能：锁定或禁用轮驱动，只训练 12 个腿关节，使用起跳、腾空高度、越障、着地稳定奖励和逐级障碍课程。
- VLA 高层策略：根据语言目标、相机和激光雷达判断何时巡航、何时切换跳跃，并输出技能条件/子目标。
- 不再把“平地上腿必须摆动”作为训练目标。下一实现里程碑是将现有 v3 重构为 4 维轮式巡航基线，并新建独立 12 维腿部跳跃环境。
- 已新建 `M20ProJumpEnv` / `M20ProJumpEnvCfg`：12 维腿力矩动作、56 维观测，轮关节在 `_apply_action()` 中固定为零力矩；当前目标高度为 0.80 m，还未加入障碍物。
- 跳跃环境已通过 1 环境 × 1 步和 4 环境 × 24 步 smoke（配置接口、刚体和零轮力矩均通过）；当前还没有声称跳跃已训练成功。
- 已将 `M20Pro-Jump-Direct-v0` 注册到 Gym 并接入 `M20ProJumpPPORunnerCfg`；同一个 `train_m20pro_ppo.sh` 可通过 `--task` 切换平地巡航与腿部跳跃。跳跃任务现在还是原地跳跃，尚未加障碍物和 1 m 课程。
- 首版 jump 训练已完成 300 次迭代并生成 `model_299.pt`，但 16 环境 × 200 步回放的最低高度仅 `0.5874 m`，腿动作平均 `0.1088`，基本是原地站立而非起跳。原因是初始高度 0.62 m 在旧高度奖励中已能获得过高收益。
- 已改跳跃奖励：收紧 0.80 m 峰值高度奖励，增大起跳高度进度、向上速度和最高高度进度奖励；新版还未重新训练。
- jump v2 回放表明腿部会动但不起跳：最低高度 `0.4978 m`、腿速 `1.6128 rad/s`、平均前向速度 `0.3375 m/s`，表明不对称腿动作在制造水平漂移。
- jump v3 加入 57 维 phase 观测（下蹲、起跳、腾空三阶段），并将 episode 缩短到 2 s 以增强时序学习。新版已通过 4 环境 × 24 步 smoke，但尚未重新训练；旧 jump checkpoint 与 57 维观测不兼容。
- jump v3 重训后仍不起跳：200 步回放最低高度 `0.5158 m`，说明纯力矩探索仍很难学会协调伸腿。
- 历史记录曾写入开环探针 `max_root_height=0.8223 m`，但在本轮重新验证中无法复现：带前后相机和 LiDAR 的场景下，`squat_minus_thrust_plus` 为 `max=0.7106 m/min=0.0700 m`，反向符号为 `max=0.7130 m/min=0.0700 m`，两者都倒地。因此 0.8223 m 暂时降级为“未复现历史结果”，不能作为成功专家。
- jump v4 重训仍然没有学到时序：200 步回放最低高度 `0.4178 m`、`done_count=32`，但开环探针已证明 PD 目标姿态可以跳。
- 已加入 jump v5 reference bootstrap 奖励：phase 前 30% 学习收腿目标，中间阶段学习伸腿目标，同时保留高度、上向速度和腾空奖励。这是用已验证的开环序列帮助 PPO 先学会可行跳跃时序，后续再降低 reference 权重。
- 已记录 PPO 指标含义：`Loss/value_function` 是 critic 价值回归误差，`Loss/surrogate` 是 PPO 裁剪策略损失，`Policy/mean_noise_std` 是 actor 输出动作分布的平均探索标准差。
- `jump_env.py` 中 reference 项权重为 `8.0` 的改动作为 jump v5 实验保留并单独提交；本轮公开专家/模仿学习主路线不依赖该奖励，也不再以它继续盲调 PPO。

### 公开专家与模仿学习切换（2026-07-21）

- 当前判断：继续调 M20 jump PPO 奖励没有证据能得到可靠动作，正式切换到“公开专家 → 轨迹采集 → M20 动作重定向 → 模仿学习”的路线。PPO 保留为失败对照实验，不再作为主要技能来源。
- Isaac Lab 官方公开的 Go2 rough-terrain RSL-RL checkpoint 已成功下载到 2 TB 盘：`.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt`，同时生成 TorchScript 和 ONNX 导出。来源仓库：[IsaacLab](https://github.com/isaac-sim/IsaacLab)，遵循其 BSD-3-Clause 许可和 checkpoint 发布条款。
- `scripts/record_public_go2_expert.sh` 已采集 400 步固定 `0.8 m/s` 前向命令：`datasets/public_go2_rough_v0/episode_0000.h5`，观测维度 `235`（含 187 维 height scan），动作维度 `12`，额外记录实际 `joint_position_target`、命令、状态和 episode done。`done_count=0`。
- 公共专家视频已生成：`videos/public_go2_rough_v0/public-go2-rough-expert-step-0.mp4`，400 步、50 Hz；视频已通过抽帧确认能看到 Go2 行走。所有采集/回放命令均保留 `--video` 和独立 `--video-dir`。
- `scripts/retarget_go2_to_m20.py` 已把 Go2 `hip/thigh/calf` 映射到 M20 `hipx/hipy/knee`，输出 `m20_retarget_v0.h5`。当前只做名称、站立偏置和范围映射，`validated=False`；腿关节符号、偏置和 M20 形态缩放必须通过 M20 视频回放校准后才能训练。
- 重定向第三人称视频验证结果：幅度 `1.0` 时 `min_root_height=0.2710 m`、向后 `0.6189 m`；幅度 `0.50` 时 `min=0.2997 m`、向后 `0.6470 m`；幅度 `0.35` 时 `min=0.5617 m`、位移 `+0.0070 m`；幅度 `0.25` 时 `min=0.5756 m`、位移 `+0.0016 m`。因此 `0.25–0.35` 是当前稳定但几乎不前进的安全区，Go2 足式步态不能直接成为 M20 轮腿巡航动作。
- 对应视频均写入 `videos/m20_retarget_amp025_v0/`、`videos/m20_retarget_amp035_v0/` 和 `videos/m20_retarget_amp050_v0/`，回放器为 `scripts/play_m20_retargeted.sh`；这些 MP4 是校准证据，不是最终 VLA 成果。
- 本轮新增的 M20 传感器采集器 `scripts/record_m20pro_expert.sh` 已验证前后 `160x96` RGB、72 线环形 LiDAR、45 维状态和 HDF5/MP4 同步写入；其开环 jump 样本因倒地标记为诊断数据，不进入成功专家集合。

### 公开 Go1 parkour 专家协议验证（2026-07-22）

- 找到并整理了 Robot Parkour Learning（CoRL 2023，MIT）公开 Go1 checkpoint：`public_experts/parkour_go1/skill/model_674000.pt`（视觉 crawl/jump/leap）和 `walk/model_107500.pt`（本体 walk）。Extreme Parkour 源码曾以 CC BY-NC 4.0 临时保存，已在空间清理中删除；本轮没有把它作为可直接下载的 checkpoint。
- 新增 `scripts/validate_public_parkour_checkpoint.py`。它按 checkpoint 的真实协议构造 `48 维本体 + 1x48x64 深度图`，使用 `weights_only=True` 加载并验证 GRU、动作形状和记忆复位。验证结果：`observation_shape=[4,3120]`、`action_shape=[8,4,12]`、动作范围约 `[-0.8223,0.3638]`，平面/障碍深度会改变动作。
- 新增 `scripts/play_public_go1_parkour.py` 和 `play_public_go1_parkour.sh`，原生加载 Isaac Lab Go1 USD，使用公开 `Kp=40/Kd=0.5`、0.5 action scale、前向深度预处理和第三人称 MP4。所有回放命令强制带 `--video`。
- 公开策略在当前 Isaac Lab 5.1/2.3.2 适配器中没有形成可靠步态：skill checkpoint、0.45 m 障碍、200 步回放为 `x_displacement=-2.3574 m`、`min_root_height=0.0604 m`；无障碍仍倒地。walk checkpoint 作为无视觉对照也失败：200 步为 `x_displacement=-1.9766 m`、`min_root_height=0.0612 m`。
- 已核对并修正过两项输入偏差：深度从米制转换为公开部署使用的 `[0,1]`（0–2 m 范围、裁剪后 resize），Go1 执行器改为显式 `Kp=40/Kd=0.5` DCMotor；修正后 walk 对照仍失败。
- 结论：当前失败不是 M20 关节重定向，也不是“继续训练几轮”能解决的奖励问题，而是公开 IsaacGym parkour checkpoint 与 Isaac Lab USD/执行器/坐标协议尚未完成等价适配。Go1 checkpoint 不能标记为 M20 成功专家，不能直接用于 VLA 训练；当前视频属于协议诊断证据。

下一步调整为：先冻结这些公开 checkpoint，优先做原始 parkour 环境的动作/坐标/PD 逐项等价测试；若无法复现原环境，再只使用公开轨迹作为离线行为克隆数据，不把异构策略硬迁移到 M20。M20-specific rolling/jump 仍需独立采集可验证专家轨迹。

公开参考路线（用于数据/接口，不直接把异构机器人 checkpoint 当成 M20 策略）：

- [LeRobot](https://github.com/huggingface/lerobot)：数据集、动作 chunk 和 ACT/Diffusion/SmolVLA 训练接口。
- [Open X-Embodiment / OpenVLA](https://github.com/openvla/openvla)：开放 VLA 数据和架构参考；7B 全量训练不适合本机 RTX 3060。
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)：embodiment/action schema 参考；本机不做 N1.7 全量微调。
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)：公开四足 RL 训练/部署参考，后续用于动作语义核对。

此前的 Go2→M20 重定向路线已冻结为失败对照；当前优先使用原生 M20 专家采集同构数据，再把前后相机、LiDAR、语言任务标签合并进 LeRobot-compatible 数据集，最后训练小型 BC/ACT 基线。

### 原生 M20 专家协议与首次有效回放（2026-07-22）

- 在 GitHub 仓库 [AI-DA-STC/M20-autonomy-sim](https://github.com/AI-DA-STC/M20-autonomy-sim) 中找到原生 M20 `policy.onnx`。模型输入为 `obs[1,57]`，输出为 `actions[1,16]`；当前 Conda 环境已安装 `onnxruntime==1.27.0`，CPU 推理足够支撑单环境 50 Hz。
- 官方观测顺序为 `base_omega*0.25(3) + projected_gravity(3) + command(3) + joint_pos-default(16) + joint_vel*0.05(16) + last_action(16)`。动作顺序是 12 个腿关节后接 4 个轮关节，腿缩放 `[0.125,0.25,0.25]`、轮速度缩放 `5.0`，官方策略协议的轮 `Kd=0.6`；Isaac Lab 适配器默认平移使用 `Kd=0.6`，转向命令自动使用 Gazebo bridge `wheel_kd_scale=6.0` 对应的 `Kd=3.6`，腿仍为 `Kp=80/Kd=2`。
- 官方 M20 的后腿髋/膝姿态必须镜像：默认策略姿态为 `FL/FR=[0,-0.6,1.0]`、`HL/HR=[0,0.6,-1.0]`。此前开环 jump 和 Go2 重定向把后腿符号处理错，导致“平移、僵硬、倒地”的错误现象；新适配器严格按官方 policy order 和 USD joint order 验证。
- 原生 rolling 回放（第三人称 MP4）结果：`500` 个策略步、命令 `[0.5,0,0]`，`x_displacement=14.7179 m`，`mean_forward_speed=0.3684 m/s`，`min_root_height=0.5154 m`，`max_root_height=0.5295 m`，`terminated_steps=0`。这证明 M20 USD、轮执行器和关节协议已经能产生持续滚动；腿保持支撑姿态是该 rolling policy 的预期行为，不应再把“腿不摆”当作平地轮式专家失败。
- 近景复核视频：`videos/public_m20_native_close_v1/m20-native-x+0.50-y+0.00-yaw+0.00-step-0.mp4`，250 步得到 `x_displacement=7.3376 m`、`mean_forward_speed=0.3678 m/s`、`min_root_height=0.5154 m`、`terminated_steps=0`。
- 新增 [play_public_m20_policy.py](scripts/play_public_m20_policy.py) / `play_public_m20_policy.sh`，每次回放强制带 `--video`；新增 [record_public_m20_expert.py](scripts/record_public_m20_expert.py) / `record_public_m20_expert.sh`，记录前后 RGB、72 线 360° LiDAR、57 维原生 proprio、45 维全状态、16 维动作、command 和自然语言标签。
- 采集器 5 步 smoke 已通过：HDF5 形状为 `front/rear=(5,96,160,3)`、`lidar=(5,72)`、`proprio=(5,57)`、`state=(5,45)`、`action=(5,16)`，`terminated_steps=0`，同步 MP4 正常写入。该 smoke 仅是格式验证，正式数据需使用完整 episode。
- 第一批正式 rolling 数据已采集到 `datasets/public_m20_native_v1/`：4 个 episode、每条 500 帧/10 秒、每条均 `success=true`。逐文件检查结果：位移 `14.7162–14.7326 m`，最低根高 `0.5152–0.5154 m`，`terminated_steps=0`，所有 RGB/LiDAR/proprio/state/action 数值有限；对应 4 个 MP4 均为 500 帧、50 Hz、`480x288`。这批数据是后续语言条件 action-chunk BC/VLA 的 rolling 正样本，不包含 jump 标签。
- 额外命令覆盖结果：`向后走`（`command_x=-0.5`、`Kd=0.6`）250 步保持稳定，`x_displacement=-6.3242 m`、`min_root_height=0.5152 m`、`terminated_steps=0`，可作为反向 rolling 正样本；旧 `Kd=0.6` 的 `向左转` 产生 `yaw_delta=1.4632 rad` 同时 `x=-4.9744 m/y=-4.0087 m` 漂移，已标记 `success=false`。转向使用 `Kd=3.6` 后，250 步为 `yaw_delta=0.6105 rad`、`x=-1.3562 m`、`y=-0.3536 m`、`min_root_height=0.5168 m`、`success=true`；统一 `Kd=3.6` 的 forward 对照反而倒退 `x=-1.3055 m`，因此代码采用按 command 自适应，不覆盖已验证的平移数据。
- 采集器已加入命令有效性判据：有 forward command 必须位移方向正确；有 yaw command 除转向方向外还要求平移漂移小于 2 m。仅“不跌倒”不再自动算成功。
- 上游仓库未发现可供重新分发的根目录 LICENSE 文件；本项目只在本机保留来源说明和个人研究验证，不把外部源码或权重提交到仿真仓库或 VLA-Learning 仓库。

当前判断：原生 M20 policy 已经是合格的 rolling 专家，但它没有跳跃能力，也没有语言、相机和 LiDAR 决策能力。因此“自然语言找物体/导航 + 1 m 障碍跳跃”仍未完成；下一步先采集多命令 rolling 正样本，再加入可物理验证的 jump skill，最后训练小型语言条件 action-chunk BC/VLA，而不是继续盲目增加 PPO 迭代。

### 语言条件多模态 VLA/BC v1（2026-07-22）

- 新增 [train_m20_vla_bc.py](scripts/train_m20_vla_bc.py)：输入为前后 RGB（下采样为 `6x48x80`）、72 线 LiDAR、原生 57 维 proprio 和 UTF-8 语言字节序列，输出连续 `8x16` action chunk。它是公开专家模仿学习，不使用奖励函数或 PPO。
- 训练数据只读取 HDF5 `success=true` episode：forward 4 条、backward 1 条、turn v2 1 条，共 `6` 条 episode、`985` 个训练窗口、`247` 个验证窗口。checkpoint 位于 `checkpoints/m20_vla_bc_v1/best.pt`，约 `1.8 MB`，训练最佳验证损失 `9.741e-5`。
- 新增 [play_m20_vla_bc.py](scripts/play_m20_vla_bc.py)：默认每 4 个控制周期执行一个预测 chunk，再重新读取传感器，真正使用 action chunk 而不是只取第一步；每次强制写 MP4 和 JSON 指标。
- 闭环 forward chunk4 250 步结果：`x_displacement=7.3347 m`、`mean_forward_speed=0.3676 m/s`、`yaw_delta=0.026 rad`、`min_root_height=0.5152 m`、`terminated_steps=0`。50 步 backward 结果为 `x=-1.2321 m`、平均 `-0.3112 m/s`、`terminated_steps=0`；50 步 turn 结果为 `yaw_delta=0.114 rad`、`x=0.4440 m`、`min_root_height=0.5179 m`、`terminated_steps=0`。
- 当前 VLA 只学会语言条件下的 rolling/backward/turn 低层动作，语言标签仍是运动指令；它尚未学习“寻找某物体”、地图/无地图探索或 jump skill，不能把 v1 宣称为完整任务完成。

### 视觉目标到达 VLA/BC v2-v4（2026-07-22）

- `record_public_m20_expert.py` 已支持红/蓝/绿非碰撞方块、目标坐标、语言标签和 `--stop-on-target`。模拟器目标真值只用于生成专家停车标签和计算验收指标；VLA 回放控制器不读取目标坐标或成功半径，控制输入仍是前后 RGB、LiDAR、proprio 和语言。
- 修正了一处关键的 VLA 输入泄漏：原生 57 维 proprio 的 `6:9` 是专家速度 command，训练和回放现均将这三维清零。模型必须从语言和视觉推断任务，不再直接复制专家 command。
- 训练拆分改为按语言分组；只有重复语言才留出验证 episode，单条颜色数据不会被错误分到纯验证集。训练窗口使用 `WeightedRandomSampler` 按任务文本均衡采样，避免四条长 forward 数据淹没颜色目标样本。
- v2 使用固定帧数停车，红色闭环通过，但蓝/绿仍穿过目标。数据复核确认固定 `stop_after` 发生时目标已经离开前相机，因此标签与视觉时序不一致。
- v3 改为专家首次进入 `0.8 m` 目标半径后立刻输出全零动作；红色通过，但旧图像编码器的全局平均池化会抹掉目标位置和尺度，蓝色仍失败。该失败没有靠继续堆 epoch 掩盖。
- v4 新增 `spatial_v2` 图像编码：保留 `3x5` 空间网格并投影到 128 维，同时继续兼容旧 `global_v1` checkpoint。训练使用 9 条成功 episode、`1206` 个训练窗口和 `247` 个验证窗口，80 epochs 最终训练损失 `0.002181`，最佳 forward 验证损失 `1.03e-4`。
- v4 checkpoint：`/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_bc_v4/best.pt`。
- 红色目标闭环：`x=1.3046 m`、最小目标距离 `0.6575 m`、到达后平均速度 `0.0280 m/s`、最终平面速度 `0.0010 m/s`、`terminated_steps=0`、`success=true`。
- 蓝色目标闭环：`x=2.3656 m`、最小目标距离 `0.6779 m`、到达后平均速度 `0.0317 m/s`、最终平面速度 `0.0013 m/s`、`terminated_steps=0`、`success=true`。
- 绿色目标闭环：`x=2.0100 m`、最小目标距离 `0.7086 m`、到达后平均速度 `0.0287 m/s`、最终平面速度 `0.0026 m/s`、`terminated_steps=0`、`success=true`。
- 普通“向前走”回归测试仍通过：250 步位移 `7.3197 m`、平均速度 `0.3669 m/s`、最低根高 `0.5152 m`、`terminated_steps=0`。空间视觉能力没有破坏原生 rolling 动作质量。
- 对应 MP4 和 JSON 位于 `videos/m20_vla_bc_v4/{red,blue,green,forward}/` 与 `logs/m20_vla_bc_v4_*.json`。这只是固定场景下的“看到指定颜色目标并停车”里程碑；尚未覆盖随机目标位置、主动转向搜索、开放词汇、地图/无地图探索、障碍物和跳跃，不能宣称完整任务已完成。

### 主动视觉搜索专家边界（2026-07-22 续）

- 公开 `AI-DA-STC/M20-autonomy-sim` README 明确显示其 M20 ONNX 是 `/cmd_vel -> RL locomotion` 低层控制器；公开仓库的导航部分是 CMU/FAR/local planner，没有 M20 jump policy 或视觉目标搜索专家 checkpoint。该公开经验可以作为 VLA 的同构 rolling 专家，但不能直接提供完整 ObjectNav/jump 示范。
- 尝试用目标真值生成离线专家：连续 bearing command、差速轮覆盖、固定 turn→forward 技能串联都做了带 MP4 的 smoke。它们分别出现异常角速度、命中后惯性回滑、目标未到达等问题，全部 `success=false`，已删除对应 HDF5/MP4，未进入训练集。`command_y=0.5` 协议测试也不是侧移，而是倒退并伴随转向，不能当横向导航专家。
- 当前可复用的数据仍只包括固定直线目标和公开 rolling/backward/turn 正样本；下一次搜索必须先解决原生 turn/forward 技能的物理切换和刹停，再采集随机方位目标，不能用失败轨迹硬训模型。

### 导航回放诊断与动力学一致性（2026-07-22）

- 用户回放观察到机器狗仍然像平移，且腿部动作僵硬。这个判断与指标一致：公开 M20 ONNX 是轮式 rolling 低层策略，不是四足步态或完整 ObjectNav 专家；继续训练 v4/v5 不能凭空产生转向和跳跃能力。
- 原生 turn skill 的最大机身角速度约 `12.8–13.2 rad/s`；把 command/Kd 分段平滑到 60 步仍约 `13.36 rad/s`，因此没有把它用于随机目标专家。
- 直接轮速 pattern（幅值 `1.0/0.5/0.25`）全部出现塌机身，最低根高约 `0.24–0.39 m`，不作为低层控制器。
- 目标 `(3,-1)` 的失败验收记录：旧 override 在进入 `0.8 m` 后又滑到 `1.874 m`、末速度 `0.612 m/s`；连续闭环 v1 末距 `1.109 m` 且 `yaw_delta=0.033 rad`；滑移增益 v2 把目标转到错误的 `y=+0.791 m`；修正符号 v3 仍末距 `1.044 m` 且几乎不转。公开 `cmd_vel` 通道的 combined yaw 末 `yaw_delta=-0.006 rad`，纯 yaw `0.05` 也只有 `0.0015 rad`。这些 HDF5/MP4 已清理，全部 `success=false`，没有进入训练集。
- 对照官方 `AI-DA-STC/M20-autonomy-sim`：ONNX 的 MuJoCo `M20.xml` 使用不同的 link 惯量和 `friction="1 0.01 0.001"` 接触模型；当前 URDF→USD 只保留简化碰撞和各向同性 PhysX 摩擦，动力学并不等价。官方 Gazebo 控制桥的轮子 `Kd=0.6`，launch 默认把 wheel Kd scale 设为 `1.0`；此前为 yaw 临时放大的 `3.6` 已确认会造成角速度尖峰。
- 新增 [convert_m20_mjcf.py](scripts/convert_m20_mjcf.py) / [convert_m20_mjcf.sh](scripts/convert_m20_mjcf.sh)，显式启用 Isaac Sim 5.1 的 MJCF importer，并删除 importer 生成的第二 articulation root `/M20/worldBody`；`assets/m20_mjcf_official_nofloor_v6.usd` 已在 2 TB 盘生成。它通过了 `17 bodies / 16 joints` 和 240 步静态 smoke（最终根高约 `0.560 m`），但尚未替换默认资产。
- candidate 前进回放：150 步位移 `4.389 m`、最低根高 `0.5151 m`、最大角速度 `0.1571 rad/s`，带 MP4 保存在 `videos/public_m20_mjcf_forward_v2/`。
- candidate 转向回放：`wheel Kd=0.2`、正 yaw `0.5` 得到 `yaw_delta=+0.3846 rad`、最大角速度 `0.5406 rad/s`；负 yaw `-0.3` 只有 `-0.0459 rad`，负 yaw `-0.5` 在 Kd `0.6` 下为 `-0.1325 rad`。这表明当前转换资产/公开网络仍有单向转向不对称，不能直接作为任意方位导航专家。
- candidate recorder 已根据 `M20PRO_USD_PATH` 自动把相机/LiDAR 挂到 `base_link/base_link`；固定负转向 300 步后前进的目标串联仍 `success=false`（最终距目标 `2.823 m`、路径 `10.491 m`），失败 HDF5/MP4 已删除。
- 当前决策：暂停随机方位专家采集和 VLA 继续训练；先让官方 MJCF robot-only USD 通过拓扑、直行、转向三项带视频回归，再采集成功的随机目标轨迹。跳跃仍需独立的落地控制器，不能用当前 rolling policy 代替。

### Jump 专家搜索边界（2026-07-22）

- GitHub/API 搜索了 M20/DeepRobotics/wheel-legged parkour 公开仓库，没有找到可复用的 M20 jump checkpoint；保留的原生 M20公开专家只有 rolling `policy.onnx`。
- 新增 [search_m20_jump_expert.py](scripts/search_m20_jump_expert.py)，显式使用官方镜像姿态和前后腿镜像膝关节，采用 `Kp=200/Kd=4`、短下蹲/起跳目标序列，并行执行 `324` 个纯物理候选；v2 最高 `max_root_height=0.6618 m`，v3 扩展时序后最高 `0.6738 m`，所有候选均未满足 `min_root_height>=0.45 m` 的稳定判据。
- 这证明当前 M20 USD/执行器和简单关节目标序列还不能产生可验证 1 m 跳跃；失败序列不进入 VLA 数据集。搜索结果只保留在 2 TB 盘 `logs/m20_jump_expert_search_v1/v2/v3.json`，不再继续盲目 PPO 跳跃训练。
- 继续扩大无奖励物理搜索：腿执行器力矩 `150` 时最高 `0.882 m`，力矩 `300`、`Kp=500/Kd=8` 时最高 `0.920 m`；所有最高候选最低根高约 `0.16–0.19 m`，`survived=false`。这说明“跳得高但摔倒”不能作为专家，当前仍没有可用于 VLA 的 1 m jump action chunk。

### 数据清理与存储（2026-07-22）

- 已删除旧 `logs/rsl_rl` PPO checkpoint、失败 jump HDF5/视频、重复诊断视频，以及本轮失败目标搜索 HDF5/MP4；失败 MJCF v1 candidate 也已删除。
- 只保留：成功的 M20 rolling/目标数据、v4 checkpoint、必要的 M20 `policy.onnx`/协议文件、公开 Go1 checkpoint、跳跃搜索 JSON，以及用于动力学核对的官方 M20 模型源码快照和 no-floor MJCF candidate，全部在 2 TB 盘。
- 官方源码快照只用于个人仿真协议核对，不复制进本 Git 仓库，也不重新分发外部权重；candidate USD 尚未成为默认资产。

### 公开 VLA 架构检索与 M20 兼容性探针（2026-07-22）

- 检索并核对了公开的 [NaVILA](https://github.com/AnjieCheng/NaVILA)、[NaVILA-Bench](https://github.com/yang-zj1026/NaVILA-Bench) 和 [legged-loco](https://github.com/yang-zj1026/legged-loco)。它们采用“高层视觉语言导航动作 + 独立低层 locomotion policy”的两层协议；公开 NaVILA Llama-3 8B checkpoint 约需 24 GB 以上显存，当前 RTX 3060 12 GB 不适合直接部署。
- 已下载公开 NaVILA Go2 vision locomotion checkpoint：`public_experts/navila_go2_vision/policy.jit`，协议为 `909 -> 12` 个 Go2 腿关节目标。它不是 M20 专家，也不是完整 ObjectNav 或 jump VLA。
- 新增 [play_public_navila_go2_on_m20.py](scripts/play_public_navila_go2_on_m20.py) 做物理兼容性探针：固定 M20 关节顺序、官方镜像站立姿态，输入 909 维观测，保留第三人称 MP4 和 JSON 指标。`100` 步 smoke 实测 `x_displacement=0.1189 m`、`min_root_height=0.2131 m`、`terminated_steps=97`，说明 Go2 权重直接套用 M20 会倒地，不能进入训练集或宣称为 M20 低层专家。
- 探针结果确认两层 VLA 思路值得借鉴，但低层必须换成 M20 原生 rolling/未来的 M20 jump expert；当前不再继续把 Go2/Go1 异构关节动作硬重定向到 M20。
- 本轮清理后 2 TB 输出根目录约 `289 MB`，失败目标搜索数据、旧 checkpoint 和重复视频已移除；保留的每条诊断结果仍有 MP4、JSON 和来源说明。

## 常用验证

```bash
cd /home/fabu/桌面/M20Pro/m20pro_sim_ros2_ws
source scripts/activate_vla_env.sh
./scripts/convert_m20pro_urdf.sh
./scripts/test_m20pro_asset.sh
TERM=xterm python scripts/check_m20pro_task.py --headless
```

官方 M20 MJCF→USD 动力学基线（输出在 2 TB 盘，当前仍是实验 candidate）：

```bash
./scripts/convert_m20_mjcf.sh \
  --output /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/assets/m20_mjcf_official_nofloor_v6.usd

M20PRO_USD_PATH=/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/assets/m20_mjcf_official_nofloor_v6.usd \
  ./scripts/test_m20pro_asset.sh --steps 240
```

`isaaclab.sh` 运行时需要 `TERM=xterm`，以避免 `TERM=dumb` 下 `tabs` 命令失败。

回放命令默认应包含 `--video` 和独立 `--video-dir`，以 headless rendering 生成 MP4；不再只给无视频的回放命令。

公开 Go2 专家采集（自动包含视频）：

```bash
./scripts/record_public_go2_expert.sh \
  --steps 400 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_go2_rough_v0
```

Go2→M20 目标动作重定向（此命令不启动 Isaac Sim）：

```bash
python scripts/retarget_go2_to_m20.py \
  /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0/episode_0000.h5 \
  /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0/m20_retarget_v0.h5
```

M20 第三人称重定向回放（自动录制视频）：

```bash
./scripts/play_m20_retargeted.sh \
  --actions-h5 /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0/m20_retarget_v0.h5 \
  --steps 200 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_retargeted_v0
```

原生 M20 专家回放（自动包含近景第三人称视频）：

```bash
./scripts/play_public_m20_policy.sh \
  --steps 500 --warmup-steps 75 \
  --command-x 0.5 --command-y 0.0 --command-yaw 0.0 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_m20_native_v1
```

训练带空间视觉编码的语言条件 action-chunk VLA/BC：

```bash
./scripts/train_m20_vla_bc.sh \
  --architecture spatial_v2 \
  --epochs 80 --batch-size 64 --horizon 8 --stride 2 \
  --device cuda:0 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_bc_v4
```

VLA 红色目标闭环回放（wrapper 自动带 `--headless --video`，并写 MP4/JSON）：

```bash
./scripts/play_m20_vla_bc.sh \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_bc_v4/best.pt \
  --task-text "到红色方块去" \
  --target-color red --target-x 2.0 --target-y 0.0 \
  --steps 160 --chunk-execution 4 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_bc_v4/red \
  --metrics /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_vla_bc_v4_red.json
```

采集原生 M20 rolling 正样本（每个 episode 自动写 HDF5 和 MP4）：

```bash
./scripts/record_public_m20_expert.sh \
  --episodes 4 --steps 500 --warmup-steps 75 \
  --task-text "向前走" --command-x 0.5 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_m20_native_v1 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_m20_native_v1
```

训练两层高层 VLA 技能选择器（监督模仿，不使用奖励函数）：

```bash
./scripts/train_m20_vla_skill.sh \
  --epochs 60 --batch-size 64 --post-reach-steps 20 --device cuda:0 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_skill_v8
```

两层 VLA 红色目标回放（高层 CPU 推理、低层公开 M20 ONNX、自动写视频）：

```bash
export M20PRO_USD_PATH=/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/assets/m20_mjcf_official_nofloor_v6.usd
./scripts/play_m20_vla_skill.sh \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_skill_v8/best.pt \
  --policy /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/public_experts/m20_native/policy.onnx \
  --task-text "到红色方块去" --target-color red --target-x 2.5 --target-y 0.0 \
  --steps 300 --warmup-steps 75 --model-device cpu \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_skill_v8/red \
  --metrics /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_vla_skill_v8_red.json
```

两层 VLA search 技能回放（当前 candidate 的物理 yaw 适配为单向扫描）：

```bash
./scripts/play_m20_vla_skill.sh \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_skill_v8/best.pt \
  --policy /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/public_experts/m20_native/policy.onnx \
  --task-text "寻找目标" --target-color none --steps 120 --warmup-steps 75 \
  --search-yaw-command 0.5 --model-device cpu \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_skill_v8/search \
  --metrics /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_vla_skill_v8_search.json
```

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

### VLA 闭环修复与 v14-v20 诊断（2026-07-22）

- 修复了 `play_m20_vla_bc.py` 的两个闭环错误：第三人称相机必须在物理步前设置，否则 MP4 会录到机器人内部的旧相机姿态；`horizon=1` 每次重新预测后必须把 `cached_index` 归零，否则第二步访问越界并被 Isaac Sim 退出流程掩盖成“卡死”。
- VLA 回放的推理默认移到 CPU（`--model-device cpu`），Isaac Sim 仍使用 GPU；这避免相机/PhysX 与小型 PyTorch 网络共享 CUDA 上下文。所有回放仍强制 `--video`，新增 `--debug-steps` 可定位传感器、推理和物理阶段。
- v14 对蓝色目标 `(3,-1)` 的有效 300 步回放：第 `139` 步进入目标半径，最小距离 `0.111 m`，但旧数据在到达后仍标注滚动动作，最终距离 `6.585 m`、最低根高 `0.291 m`、`terminated_steps=213`。这解释了“能找到但冲过/摔倒”。
- v15 将成功轨迹到达后的 40 帧重标为零动作，最佳训练损失 `0.000648`，但闭环提前失稳，未进入目标；说明直接把停车动作混入 16 维腿/轮连续回归会损伤协变量鲁棒性。
- 新增 DAgger 采集支持并修正 `stop_on_target`：`m20_dagger_stop_blue_v1` 在第 `178` 步达到最小距离 `0.242 m`，到达后 522 帧为零动作；训练加载器现对 `success` 和 `dagger` 轨迹统一截断为到达后 40 帧，避免零动作尾段淹没滚动样本。
- v17 加入截断后的 DAgger 数据后保持稳定（300 步 `terminated_steps=0`），但只前进 `0.467 m`、最小目标距离 `2.497 m`；仍不足以称为导航成功。
- 新增独立 learned stop head：它和动作头共享 RGB/LiDAR/proprio/语言特征，训练时用 `target_reached_step` 生成 BCE 停止标签，回放通过 `--stop-threshold` 触发停止；v18 从头联合训练会误触发或改变动作分布，v20 改为冻结 v14 动作头、只训练 stop head，但受 GPU PhysX 非确定性影响，本次最小距离 `0.890 m`，仍未成功。
- 停止触发后新增短时基于机身速度的轮刹车，仅负责耗散滚动惯性，不负责目标选择或路径规划。当前结果仍是诊断里程碑：VLA rolling 能力和可学习停车接口已建立，但随机方位 ObjectNav、稳定停车和 1 m 跳跃均未完成。

当前最有价值的可复现视频/指标：

```text
videos/v14_fixed_valid_300/        # v14 到达目标后冲过并倒地
videos/v17_dagger_trimmed_blue_300/ # DAgger 截断后稳定但未到达
videos/v20_frozen_stop_blue_300/  # 冻结动作头 + learned stop
```

下一轮应优先采集多个初始姿态、目标距离和目标横向位置的 DAgger 轨迹，并以“到达后 0.8 m 内保持 2 s、根高 >= 0.45 m、无倒地”为验收；在此之前不再增加 epoch，也不进入 1 m jump 集成。

### 两层 VLA 技能策略 v1-v11（2026-07-22）

- 新增 [m20_vla_skill_model.py](scripts/m20_vla_skill_model.py)、[train_m20_vla_skill.py](scripts/train_m20_vla_skill.py) 和 [play_m20_vla_skill.py](scripts/play_m20_vla_skill.py)。高层模型输入前后 RGB、72 线 LiDAR、去除专家 command/action history 的 57 维 proprio 和语言，输出 `forward/backward/left/right/stop/search/jump` 技能以及给 M20 原生 ONNX 的 3 维 command；低层仍是公开 `AI-DA-STC/M20-autonomy-sim` 的 `57 -> 16` 专家，不使用 PPO 奖励。
- 修正了两项数据/闭环问题：到达后只保留 20 帧停车样本，避免静止尾段淹没滚动数据；慢速接近命令（约 `0.045 m/s`）不能按 stop 标注，只有明确全零命令才是 stop。当前 v8 checkpoint 位于 `checkpoints/m20_vla_skill_v8/best.pt`，19 条 episode、2254 个训练帧，验证技能准确率约 `0.98`；标签覆盖 `forward/backward/left/right/stop/search`，`jump` 仍为零标签。v8 新增 3 条物理稳定的 `寻找目标` 扫描轨迹；它们因 candidate yaw 符号与公开 command 约定相反而不标作导航 success，但作为 search skill 数据保留。
- 红色目标 `(2.5, 0.0)` 两层闭环通过：`target_reached=true`、最小距离 `0.7376 m`、停止后稳定、`terminated_steps=0`。关键视频：`videos/m20_vla_skill_v4/red_center/`。
- 绿色目标 `(2.5, 0.4)` 两层闭环通过：`target_reached=true`、最小距离 `0.6346 m`、停止步 `121`、`terminated_steps=0`。关键视频：`videos/m20_vla_skill_v5/green_left04/`。
- 蓝色正前方目标能识别并接近，但当前固定停止确认窗口的指标为 `0.8005–0.8482 m`，尚未按严格 `0.8 m` 验收；蓝色横向目标能进入半径但最终不停车，且 yaw/横向转向不足。当前结论是两层接口和 RGB 目标识别已建立，主动搜索、任意横向方位和稳定停车尚未完成。
- `寻找目标` search 回放：120 步 `skill_counts={'search':120}`、`min_root_height=0.515 m`、`terminated_steps=0`、`yaw_delta=-0.132 rad`，视频位于 `videos/m20_vla_skill_v8/search_v2/`。当前 candidate 只能稳定单向扫描，尚未把目标重新获取和 stop 串成成功 ObjectNav。
- 新增成功专家数据：红色 `(2.2,0.25)`、绿色 `(2.8,0.8)`、绿色近距 `(1.5,0.4)`、蓝色负向镜像 `(3.0,-0.75)`、蓝色近距 `(1.5,-0.4)`；失败的蓝色负转弯和红色远距轨迹已删除，不进入训练集。
- 当前两层回放命令必须包含 `--headless --video`。停止候选使用连续确认/投票，只负责耗散接近目标时的惯性；它没有读取目标坐标，目标坐标只用于离线指标。

### v11 隔离 search head 复测（2026-07-22）

- v11 从 v7 初始化，冻结视觉、LiDAR、本体状态、融合、command 和导航主干，只增加独立的语言 `search_intent_head`；这样 search 语义不会直接改写已验证的 rolling 主干。
- 红色目标 `(2.5, 0.0)` 回放通过：`target_reached=true`、第 `100` 步进入目标判定、最小距离 `0.4663 m`、第 `120` 步停止、`terminated_steps=0`。视频：`videos/m20_vla_skill_v11/red_center/`。
- 绿色目标 `(2.5, 0.4)` 复测失败：最小距离 `1.1601 m`，第 `99` 步误触发停止，`terminated_steps=0`；技能计数为 `left=26`、`forward_recovery=63`、`stop=211`。因此 v11 不能作为全场景导航成功版本，需先区分 GPU PhysX 非确定性和 search head/输入变化造成的行为偏移。
- v11 search 扫描本身稳定：`120` 步全为 `search`，`min_root_height=0.515 m`、`terminated_steps=0`、`yaw_delta=-0.132 rad`；这仍不代表发现物体或完成 ObjectNav。

### v12-v13 learned target stop 与视频兼容性（2026-07-22）

- v7/v11 在当前闭环代码下都可重复复现绿色目标提前停车：第 `99` 步停止、最小距离 `1.1601 m`。因此该问题不是 search head 或单次 PhysX 随机性，而是旧 stop/command 输出在目标尚未进入 `0.8 m` 半径时过早衰减。
- v12 从 v7 初始化并冻结导航主干，只训练独立 `target-stop head`；它仍在约 `1.12 m` 处误停。v13 将 head 扩为停止概率和归一化视觉距离联合监督，目标坐标和根位置只用于离线标签，推理仍只看前后 RGB、LiDAR、本体状态和语言。
- v13 使用理论距离阈值 `0.16`（约 `0.8 m / 5 m`）时在 `1.1539 m` 误停；阈值降到 `0.10` 后不再触发 stop，但 command 回归头把前进速度降到接近零，最终仍停在 `1.1543 m`。因此不能把 v13 标记为成功。
- 为尚未满足 learned stop 的 `forward` 技能设置 `0.08 m/s` 最小接近速度后，阈值 `0.10` 和 `0.14` 两次回放都进入目标半径，最小距离分别为 `0.3439 m` 和 `0.3438 m`，且 `terminated_steps=0`；但两次均为 `stop_step=None`，机器人穿过目标而没有停车。因此最小接近速度只解决了约 `1.15 m` 处速度衰减，learned stop 仍未校准，下一步必须记录完整预测曲线而不是继续猜阈值。
- 定位到此前全部 MP4 使用 OpenCV `mp4v`（MPEG-4 Part 2），文件本身完整但桌面播放器兼容性差。新增 [video_utils.py](scripts/video_utils.py) 和 [convert_videos_to_h264.py](scripts/convert_videos_to_h264.py)，所有 9 个录制/回放脚本在关闭视频后自动原子转为 `H.264/yuv420p + faststart`。
- 2 TB 盘上的视频已再次全量扫描，并将 `logs/videos/` 中最后一个遗漏的 MPEG-4 文件原位转码。当前已有 `55/55` 为 H.264（含 v13 trace 新回放），使用打包的 FFmpeg 从第一帧到最后一帧完整解码，失败数为 `0`；视频主目录约 `11 MB`，桌面播放器不会再依赖旧 `mp4v` 解码器。
- 新录制链路另以 `videos/h264_writer_smoke_v1/` 完成 30 帧实测：输出为 `H.264/yuv420p`、`50 FPS`、`480x288`，首帧和末帧均可解码；桌面 `video/mp4` 默认关联已统一到 Microsoft Edge。

### v13 trace 与 v14 独立目标视觉分支（2026-07-22）

- `play_m20_vla_skill.py` 的指标格式升级为 `m20_vla_two_layer_replay_v2`：每次高层推理保存真实评估距离、预测距离、target/generic stop 概率、原始/选中技能、命令和 stop latch 状态，并汇总预测距离最小值、stop 概率最大值及对应步数。目标坐标只用于评估 trace，仍未输入策略。
- v13 绿色目标 `(2.5, 0.4)` 的 300 步 trace 复测再次进入目标半径，真实最小距离 `0.3348 m`、`terminated_steps=0`，但 `stop_step=None`。预测距离全程仅在 `2.1833–2.6376 m` 之间，真实/预测距离相关系数只有 `0.148`；在真实距离 `<=0.8 m` 的 69 个推理样本中仍预测 `2.1833–2.2021 m`，最大 target-stop 概率仅 `0.0036`。因此已确认是表征/泛化失败，不是继续猜 `0.10/0.14/0.16` 阈值能解决的问题。
- v14 新增与动作主干隔离的 `visual_v2` target encoder，用独立 RGB 空间特征和语言特征预测停止概率/距离；v7 已验证的动作主干完全冻结。stop BCE 只在有目标帧上计算，标签严格定义为离线真实距离 `<=0.8 m`；冻结目标训练采用正负目标样本平衡，并允许只为目标感知使用物理稳定且曾进入目标半径的 DAgger 帧，不使用其失败后的动作作为控制专家。
- `m20_vla_skill_v14_target_visual` 使用 20 条 episode，训练/验证分别为 `2350/775` 帧，训练目标标签为 `not_reached=1029`、`reached=71`；最佳 checkpoint 在第 36 轮，`val_target_loss=0.1170`。训练刚结束时这还只是离线改善，随后才执行下述绿色、红色和蓝色闭环带视频验收。
- v14 验证集目标距离回归相关系数为 `0.963`、MAE 为 `0.116 m`。单一绝对预测距离门限没有三色共同可靠区间：归一化 `0.32` 可让绿/红停车但蓝色不触发，`0.34` 又会让蓝色在真实约 `0.98 m` 处提前形成候选。因此停止确认改为 learned distance 时序门控：最近 20 次预测的最小值低于 `1.7 m`，随后回升至少 `0.03 m`，再使用原有 8 帧/投票确认；该门控只读取模型输出，不读取目标坐标。
- 同一组时序参数完成三色带视频闭环：绿色 `(2.5,0.4)` 第 `109` 步停车，最小/最终距离 `0.4806 m`；红色 `(2.5,0.0)` 第 `134` 步停车，最小/最终距离 `0.1084 m`；蓝色 `(3.0,-0.75)` 第 `161` 步停车，最小/最终距离 `0.7894 m`。三条均 `target_reached=true`、`terminated_steps=0`，停车后分别保持 `111/86/119` 步。不过蓝色横向位移仅 `0.0358 m`，主要是在目标半径边缘停车，说明 learned stop 已通过这三条代表性场景，而任意横向位置的目标导向仍未完成。
- 最终视频保存在 `videos/m20_vla_skill_v14/{green_left04_turnaround_v3,red_center_turnaround_v2,blue_right075_turnaround_v2}/`。删除 4 组被替代的 v14 中间视频/JSON 和 10 个与 `best.pt` 重复的 `last.pt` 后，checkpoint 从约 `27 MB` 降为 `14 MB`；2 TB 盘现有 `58/58` 个 MP4 均为 H.264，完整逐帧解码失败数为 `0`。

v14 绿色目标复现命令（显式无头并录制视频）：

```bash
source scripts/activate_vla_env.sh
python scripts/play_m20_vla_skill.py --headless --video \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_skill_v14_target_visual/best.pt \
  --task-text "到绿色方块去" --target-color green --target-x 2.5 --target-y 0.4 \
  --steps 220 --target-distance-threshold 0.34 \
  --target-turnaround-window 20 --target-turnaround-rise-m 0.03 \
  --min-forward-command 0.08 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_skill_v14/green_replay \
  --metrics /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_vla_skill_v14_green_replay.json
```

## 待办路线

1. 为高层增加真实 `search` 轨迹和后视相机闭环：目标放在侧后方，训练转向/扫描/重新获取目标，而不是把横向进入半径当作成功。
2. 解决 jump skill 的物理能力问题：优先查找官方动作/仿真协议或调整 M20 USD/执行器，未通过高度、越障、着地稳定和视频检查不得进入 VLA 数据。
3. 将成功 rolling/search/jump 数据扩展为 LeRobot-compatible episode/skill schema，保留专家来源、传感器时间戳和视频索引。
4. 在具备成功 jump expert 后加入随机障碍、1 m 越障、地图/无地图和开放词汇物体搜索评测。
5. 训练高层语言/视觉/LiDAR skill selector，并用闭环视频和任务成功率验收，而不是只看离线 loss。

### 并行环境容量基准

`16` 是初始保守值，不是硬限制。`16–256` 短测试均通过，长训练按 64 → 128 → 256 逐级升级。

## 维护约定

每次本项目对话结束前，更新本文件的 `Last updated`、完成事项、验证结果、待办和当轮的重要决策。该文件只记录本 VLA 仿真兴趣项目，不记录真机部署或与本项目无关的 ROS 维修。

## GitHub 学习记录同步

- 同步仓库：`git@github.com:ghw1048040694/VLA-Learning.git`
- 本地克隆路径：`/home/fabu/桌面/VLA-Learning`
- 本文件在仓库中作为独立的 `m20pro-VLA.md`，不覆盖仓库原有的 `lerobot/Lerobot.md`。
- 后续每次更新本文件后，同步复制到该仓库并提交推送。
