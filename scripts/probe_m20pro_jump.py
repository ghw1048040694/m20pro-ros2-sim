"""Probe open-loop leg torque phases before training the jump policy."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--pattern", choices=["squat_minus_thrust_plus", "squat_plus_thrust_minus"], default="squat_minus_thrust_plus")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
from tasks.m20pro_locomotion import M20ProJumpEnv, M20ProJumpEnvCfg  # noqa: E402


def run_probe(name, squat, thrust):
    cfg = M20ProJumpEnvCfg()
    cfg.scene.num_envs = 1
    env = M20ProJumpEnv(cfg)
    env.reset()
    max_z = env.robot.data.root_pos_w[0, 2].clone()
    for step in range(args.steps):
        actions = torch.zeros((1, 12), device=env.device)
        if step < args.steps // 3:
            actions[:, 4:12] = squat
        elif step < 2 * args.steps // 3:
            actions[:, 4:12] = thrust
        observations, _, _, _, _ = env.step(actions)
        max_z = torch.maximum(max_z, env.robot.data.root_pos_w[0, 2])
    print(f"[M20PRO-PROBE] {name}: max_root_height={max_z.item():.4f} m final_z={env.robot.data.root_pos_w[0, 2].item():.4f} m")
    print(f"[M20PRO-PROBE] {name}: final_leg_pos={env.robot.data.joint_pos[0, :12].tolist()}")
    env.close()


try:
    if args.pattern == "squat_minus_thrust_plus":
        run_probe(args.pattern, -1.0, 1.0)
    else:
        run_probe(args.pattern, 1.0, -1.0)
finally:
    app.close()
