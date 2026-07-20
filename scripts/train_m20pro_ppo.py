"""Run a small RSL-RL PPO smoke training run for the M20 Pro."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=2)
parser.add_argument("--log-dir", type=str, default="logs/rsl_rl/m20pro_locomotion_smoke")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym  # noqa: E402
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
    agent_cfg.max_iterations = args.iterations
    agent_cfg.num_steps_per_env = 4
    agent_cfg.experiment_name = "m20pro_locomotion_smoke"

    env = gym.make("M20Pro-Locomotion-Flat-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=args.log_dir, device=agent_cfg.device)
    print(f"[M20PRO-PPO] training envs={args.num_envs} iterations={args.iterations}", flush=True)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
    print(f"[M20PRO-PPO] training smoke passed; logs={args.log_dir}", flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
