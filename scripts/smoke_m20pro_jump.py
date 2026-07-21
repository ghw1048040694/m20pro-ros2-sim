"""Instantiate and step the M20 Pro leg-only jump environment."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from tasks.m20pro_locomotion import M20ProJumpEnv, M20ProJumpEnvCfg  # noqa: E402

env = None
try:
    cfg = M20ProJumpEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = M20ProJumpEnv(cfg)
    observations, _ = env.reset()
    if observations["policy"].shape != (args.num_envs, cfg.observation_space):
        raise RuntimeError(f"Unexpected observation shape: {observations['policy'].shape}")
    for _ in range(args.steps):
        actions = torch.zeros((args.num_envs, cfg.action_space), device=env.device)
        observations, rewards, _, _, _ = env.step(actions)
    print(f"[M20PRO-JUMP] reset/step passed: obs={tuple(observations['policy'].shape)}")
    print(f"[M20PRO-JUMP] reward_mean={rewards.mean().item():.4f}")
finally:
    if env is not None:
        env.close()
    app.close()
