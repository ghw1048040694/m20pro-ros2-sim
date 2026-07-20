"""Validate the M20 Pro task configuration without launching a simulator."""

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(_parser)
_args = _parser.parse_args()
_app = AppLauncher(_args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tasks.m20pro_locomotion import M20ProLocomotionEnvCfg  # noqa: E402


cfg = M20ProLocomotionEnvCfg()
assert cfg.action_space == 16
assert cfg.observation_space == 60
assert len(cfg.joint_gears) == 16
assert cfg.robot.spawn.usd_path.endswith("assets/m20pro/m20pro.usd")
print("[M20PRO-TASK] configuration valid")
print(f"[M20PRO-TASK] action_space={cfg.action_space} observation_space={cfg.observation_space}")
print(f"[M20PRO-TASK] num_envs={cfg.scene.num_envs} episode_length_s={cfg.episode_length_s}")
_app.close()
