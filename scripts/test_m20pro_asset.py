"""Load the M20 Pro asset and run a short headless physics smoke test."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))
from assets.m20pro import M20PRO_CFG  # noqa: E402


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )

    robot = Articulation(M20PRO_CFG.replace(prim_path="/World/M20Pro"))
    sim.reset()

    print(f"[M20PRO] bodies={robot.num_bodies} joints={robot.num_joints}")
    print(f"[M20PRO] joint_names={robot.joint_names}")
    if robot.num_bodies != 17 or robot.num_joints != 16:
        raise RuntimeError("Unexpected M20 Pro topology")

    target = robot.data.default_joint_pos.clone()
    for _ in range(args.steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    root_pos = robot.data.root_pos_w[0]
    if not torch.isfinite(root_pos).all():
        raise RuntimeError(f"Non-finite root pose: {root_pos}")
    print(f"[M20PRO] final_root_xyz={root_pos.tolist()}")
    print("[M20PRO] asset smoke test passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
