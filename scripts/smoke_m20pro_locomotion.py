"""Instantiate and step the M20 Pro locomotion RL environment."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from tasks.m20pro_locomotion import M20ProLocomotionEnv, M20ProLocomotionEnvCfg  # noqa: E402


try:
    print("[M20PRO-RL] creating environment", flush=True)
    cfg = M20ProLocomotionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = M20ProLocomotionEnv(cfg)
    print("[M20PRO-RL] environment created; resetting", flush=True)
    reset_result = env.reset()
    print(f"[M20PRO-RL] reset returned type={type(reset_result).__name__}", flush=True)
    observations, _ = reset_result
    policy_obs = observations["policy"]
    if policy_obs.shape != (args.num_envs, cfg.observation_space):
        raise RuntimeError(f"Unexpected observation shape: {policy_obs.shape}")
    for _ in range(args.steps):
        print(f"[M20PRO-RL] stepping {_ + 1}/{args.steps}", flush=True)
        actions = torch.zeros((args.num_envs, cfg.action_space), device=env.device)
        observations, rewards, terminated, truncated, _ = env.step(actions)
        if not torch.isfinite(observations["policy"]).all():
            bad = ~torch.isfinite(observations["policy"])
            bad_cols = torch.where(bad.any(dim=0))[0].tolist()
            print(f"[M20PRO-RL] non-finite observation columns={bad_cols}", flush=True)
            print(f"[M20PRO-RL] root_pos={env.robot.data.root_pos_w.tolist()}", flush=True)
            print(f"[M20PRO-RL] root_quat={env.robot.data.root_quat_w.tolist()}", flush=True)
            print(f"[M20PRO-RL] joint_pos={env.robot.data.joint_pos.tolist()}", flush=True)
            raise RuntimeError("Non-finite policy observation")
    print(f"[M20PRO-RL] reset/step passed: obs={tuple(policy_obs.shape)}")
    print(f"[M20PRO-RL] reward_mean={rewards.mean().item():.4f} terminated={int(terminated.sum())}")
except BaseException as exc:
    print(f"[M20PRO-RL] FAILED: {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    print("[M20PRO-RL] closing", flush=True)
    if "env" in locals():
        env.close()
    app.close()
