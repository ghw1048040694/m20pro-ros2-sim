"""Replay a trained M20 Pro PPO checkpoint in headless Isaac Sim."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from tasks.m20pro_locomotion import M20ProLocomotionEnvCfg  # noqa: E402
from tasks.m20pro_locomotion.agents import M20ProLocomotionPPORunnerCfg  # noqa: E402

env = None
try:
    env_cfg = M20ProLocomotionEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device or "cuda:0"
    agent_cfg = M20ProLocomotionPPORunnerCfg()
    env = gym.make("M20Pro-Locomotion-Flat-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(Path(args.checkpoint).parent), device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    observations = env.get_observations()
    min_height = env.unwrapped.robot.data.root_pos_w[:, 2].clone()
    done_count = torch.zeros(args.num_envs, dtype=torch.int32, device=env.unwrapped.device)
    integrated_x = torch.zeros(args.num_envs, device=env.unwrapped.device)
    action_magnitude = torch.zeros(args.num_envs, device=env.unwrapped.device)
    step_dt = env.unwrapped.cfg.sim.dt * env.unwrapped.cfg.decimation
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(observations)
        observations, _, dones, _ = env.step(actions)
        root_pos = env.unwrapped.robot.data.root_pos_w
        forward_velocity = env.unwrapped.robot.data.root_lin_vel_b[:, 0]
        min_height = torch.minimum(min_height, root_pos[:, 2])
        done_count += dones.to(torch.int32)
        integrated_x += forward_velocity * step_dt
        action_magnitude += torch.mean(torch.abs(actions), dim=-1)
    print(f"[M20PRO-PLAY] checkpoint={args.checkpoint}", flush=True)
    print(f"[M20PRO-PLAY] integrated_forward_distance={integrated_x.mean().item():.4f} m", flush=True)
    print(
        f"[M20PRO-PLAY] mean_forward_velocity={(integrated_x / (args.steps * step_dt)).mean().item():.4f} m/s",
        flush=True,
    )
    print(f"[M20PRO-PLAY] min_root_height={min_height.min().item():.4f} m", flush=True)
    print(f"[M20PRO-PLAY] done_count={int(done_count.sum())}", flush=True)
    print(f"[M20PRO-PLAY] mean_abs_action={(action_magnitude / args.steps).mean().item():.4f}", flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
