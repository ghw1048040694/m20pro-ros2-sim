"""Replay a trained M20 Pro PPO checkpoint in headless Isaac Sim."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="M20Pro-Locomotion-Flat-v0")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--video", action="store_true", help="Record an MP4 while replaying.")
parser.add_argument("--video-dir", default="videos/m20pro_ppo", help="Video output directory.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True

app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from tasks.m20pro_locomotion import M20ProJumpEnvCfg, M20ProLocomotionEnvCfg  # noqa: E402
from tasks.m20pro_locomotion.agents import M20ProJumpPPORunnerCfg, M20ProLocomotionPPORunnerCfg  # noqa: E402

env = None
try:
    is_jump = args.task == "M20Pro-Jump-Direct-v0"
    env_cfg = M20ProJumpEnvCfg() if is_jump else M20ProLocomotionEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device or "cuda:0"
    agent_cfg = M20ProJumpPPORunnerCfg() if is_jump else M20ProLocomotionPPORunnerCfg()
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.steps,
            name_prefix=Path(args.checkpoint).stem,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(Path(args.checkpoint).parent), device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observations = env.get_observations()
    min_height = env.unwrapped.robot.data.root_pos_w[:, 2].clone()
    max_height = env.unwrapped.robot.data.root_pos_w[:, 2].clone()
    done_count = torch.zeros(args.num_envs, dtype=torch.int32, device=env.unwrapped.device)
    integrated_x = torch.zeros(args.num_envs, device=env.unwrapped.device)
    action_magnitude = torch.zeros(args.num_envs, device=env.unwrapped.device)
    leg_action_magnitude = torch.zeros(args.num_envs, device=env.unwrapped.device)
    wheel_action_magnitude = torch.zeros(args.num_envs, device=env.unwrapped.device)
    leg_velocity_magnitude = torch.zeros(args.num_envs, device=env.unwrapped.device)
    wheel_velocity_magnitude = torch.zeros(args.num_envs, device=env.unwrapped.device)
    leg_position_sum = torch.zeros((args.num_envs, 12), device=env.unwrapped.device)
    step_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(observations)
        observations, _, dones, _ = env.step(actions)
        root_pos = env.unwrapped.robot.data.root_pos_w
        forward_velocity = env.unwrapped.robot.data.root_lin_vel_b[:, 0]
        min_height = torch.minimum(min_height, root_pos[:, 2])
        max_height = torch.maximum(max_height, root_pos[:, 2])
        done_count += dones.to(torch.int32)
        integrated_x += forward_velocity * step_dt
        action_magnitude += torch.mean(torch.abs(actions), dim=-1)
        leg_action_magnitude += torch.mean(torch.abs(actions[:, :12]), dim=-1)
        if actions.shape[1] > 12:
            wheel_action_magnitude += torch.mean(torch.abs(actions[:, 12:]), dim=-1)
        leg_velocity_magnitude += torch.mean(torch.abs(env.unwrapped.robot.data.joint_vel[:, :12]), dim=-1)
        wheel_velocity_magnitude += torch.mean(torch.abs(env.unwrapped.robot.data.joint_vel[:, 12:]), dim=-1)
        leg_position_sum += env.unwrapped.robot.data.joint_pos[:, :12]
    print(f"[M20PRO-PLAY] checkpoint={args.checkpoint}", flush=True)
    print(f"[M20PRO-PLAY] integrated_forward_distance={integrated_x.mean().item():.4f} m", flush=True)
    print(
        f"[M20PRO-PLAY] mean_forward_velocity={(integrated_x / (args.steps * step_dt)).mean().item():.4f} m/s",
        flush=True,
    )
    print(f"[M20PRO-PLAY] min_root_height={min_height.min().item():.4f} m", flush=True)
    print(f"[M20PRO-PLAY] max_root_height={max_height.max().item():.4f} m", flush=True)
    print(f"[M20PRO-PLAY] done_count={int(done_count.sum())}", flush=True)
    print(f"[M20PRO-PLAY] mean_abs_action={(action_magnitude / args.steps).mean().item():.4f}", flush=True)
    print(f"[M20PRO-PLAY] mean_abs_leg_action={(leg_action_magnitude / args.steps).mean().item():.4f}", flush=True)
    print(f"[M20PRO-PLAY] mean_abs_wheel_action={(wheel_action_magnitude / args.steps).mean().item():.4f}", flush=True)
    print(
        f"[M20PRO-PLAY] mean_abs_leg_velocity={(leg_velocity_magnitude / args.steps).mean().item():.4f} rad/s",
        flush=True,
    )
    mean_wheel_velocity = (wheel_velocity_magnitude / args.steps).mean()
    print(f"[M20PRO-PLAY] mean_abs_wheel_velocity={mean_wheel_velocity.item():.4f} rad/s", flush=True)
    print(f"[M20PRO-PLAY] wheel_surface_speed={0.09 * mean_wheel_velocity.item():.4f} m/s", flush=True)
    print(
        f"[M20PRO-PLAY] mean_leg_joint_positions={(leg_position_sum / args.steps).mean(dim=0).tolist()}",
        flush=True,
    )
finally:
    if env is not None:
        env.close()
    app.close()
