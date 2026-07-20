"""Check that the custom M20 Pro task is present in Gymnasium."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym  # noqa: E402
import tasks.m20pro_locomotion  # noqa: F401, E402

assert "M20Pro-Locomotion-Flat-v0" in gym.registry
spec = gym.spec("M20Pro-Locomotion-Flat-v0")
print(f"[M20PRO-GYM] registered: {spec.id}")
print(f"[M20PRO-GYM] entry_point: {spec.entry_point}")
app.close()
