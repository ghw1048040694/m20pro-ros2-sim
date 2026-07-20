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
    start_x = env.unwrapped.robot.data.root_pos_w[:, 0].clone()
    min_height = env.unwrapped.robot.data.root_pos_w[:, 2].clone()
    falls = torch.zeros(args.num_envs, dtype=torch.int32, device=env.unwrapped.device)
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(observations)
        observations, _, dones, _ = env.step(actions)
        root_pos = env.unwrapped.robot.data.root_pos_w
        min_height = torch.minimum(min_height, root_pos[:, 2])
        falls += (root_pos[:, 2] < env.unwrapped.cfg.termination_height).to(torch.int32)
    displacement = env.unwrapped.robot.data.root_pos_w[:, 0] - start_x
    print(f"[M20PRO-PLAY] checkpoint={args.checkpoint}")
    print(f"[M20PRO-PLAY] mean_x_displacement={displacement.mean().item():.4f} m")
    print(f"[M20PRO-PLAY] min_root_height={min_height.min().item():.4f} m")
    print(f"[M20PRO-PLAY] terminated_steps={int(falls.sum())}")
finally:
    if env is not None:
        env.close()
    app.close()
