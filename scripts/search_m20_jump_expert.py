"""Search mirrored M20 squat->thrust joint primitives by physics rollout.

The public M20 release has no jump checkpoint. This fallback creates a small
verified expert library directly from USD dynamics, with no PPO reward
optimization. Front/rear hip and knee targets are explicitly mirrored to
match the native M20 kinematic convention.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics-steps", type=int, default=90, help="50 Hz control steps per candidate.")
parser.add_argument("--top-k", type=int, default=12)
parser.add_argument("--leg-kp", type=float, default=200.0)
parser.add_argument("--leg-kd", type=float, default=4.0)
parser.add_argument("--json-output", type=Path, default=Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_jump_expert_search_v1.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.physics_steps < 40:
    parser.error("--physics-steps must be at least 40")
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

from assets.m20pro import M20PRO_JUMP_CFG  # noqa: E402


LEG_NAMES = [
    "fl_hipx_joint", "fr_hipx_joint", "hl_hipx_joint", "hr_hipx_joint",
    "fl_hipy_joint", "fr_hipy_joint", "hl_hipy_joint", "hr_hipy_joint",
    "fl_knee_joint", "fr_knee_joint", "hl_knee_joint", "hr_knee_joint",
]
WHEEL_NAMES = ["fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint"]
STAND_POSE = [0.0, 0.0, 0.0, 0.0, -0.6, -0.6, 0.6, 0.6, 1.0, 1.0, -1.0, -1.0]


@dataclass(frozen=True)
class Candidate:
    squat_hip: float
    squat_knee: float
    thrust_hip: float
    thrust_knee: float
    squat_steps: int
    thrust_steps: int
    settle_steps: int


def make_candidates() -> list[Candidate]:
    result = []
    for squat_hip in (0.8, 1.0, 1.2):
        for squat_knee in (1.5, 2.0, 2.4):
            for thrust_hip in (0.0, 0.15, 0.3):
                for thrust_knee in (0.0, 0.25, 0.5):
                    for squat_steps, thrust_steps in ((3, 15), (5, 20), (8, 25)):
                        result.append(Candidate(squat_hip, squat_knee, thrust_hip, thrust_knee, squat_steps, thrust_steps, 30))
    return result


def mirrored_pose(hip: float, knee: float, count: int, device: torch.device) -> torch.Tensor:
    pose = torch.zeros((count, 12), device=device)
    pose[:, 4:6] = -hip
    pose[:, 6:8] = hip
    pose[:, 8:10] = knee
    pose[:, 10:12] = -knee
    return pose


@configclass
class JumpSearchSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = M20PRO_JUMP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=M20PRO_JUMP_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.54),
            joint_pos={name: value for name, value in zip(LEG_NAMES + WHEEL_NAMES, STAND_POSE + [0.0] * 4)},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*_(hipx|hipy|knee)_joint"], effort_limit=76.4, saturation_effort=76.4,
                velocity_limit=22.4, stiffness=args.leg_kp, damping=args.leg_kd,
            ),
            "wheels_locked": ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_joint"], stiffness=120.0, damping=12.0,
                effort_limit_sim=21.6, velocity_limit_sim=79.3,
            ),
        },
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)))


def main() -> int:
    specs = make_candidates()
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=args.device or "cuda:0"))
    scene = InteractiveScene(JumpSearchSceneCfg(num_envs=len(specs), env_spacing=2.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot = scene["robot"]
    leg_ids, names = robot.find_joints(LEG_NAMES, preserve_order=True)
    wheel_ids, wheel_names = robot.find_joints(WHEEL_NAMES, preserve_order=True)
    if names != LEG_NAMES or wheel_names != WHEEL_NAMES:
        raise RuntimeError(f"joint order mismatch: legs={names} wheels={wheel_names}")
    count = len(specs)
    device = robot.device
    wheel_targets = robot.data.default_joint_pos[:, wheel_ids].clone()
    initial_pos = robot.data.root_pos_w[:, :3].clone()
    max_height = initial_pos[:, 2].clone()
    min_height = initial_pos[:, 2].clone()
    max_abs_pitch = torch.zeros(count, device=device)
    for step in range(args.physics_steps):
        targets = torch.zeros((count, 12), device=device)
        for index, spec in enumerate(specs):
            if step < spec.squat_steps:
                targets[index] = mirrored_pose(spec.squat_hip, spec.squat_knee, 1, device)[0]
            elif step < spec.squat_steps + spec.thrust_steps:
                targets[index] = mirrored_pose(spec.thrust_hip, spec.thrust_knee, 1, device)[0]
            else:
                targets[index] = torch.tensor(STAND_POSE, device=device)
        robot.set_joint_position_target(targets, joint_ids=leg_ids)
        robot.set_joint_position_target(wheel_targets, joint_ids=wheel_ids)
        for _ in range(4):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        root = robot.data.root_pos_w[:, :3]
        max_height = torch.maximum(max_height, root[:, 2])
        min_height = torch.minimum(min_height, root[:, 2])
        gravity = quat_apply_inverse(robot.data.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=device).expand(count, -1))
        max_abs_pitch = torch.maximum(max_abs_pitch, torch.abs(torch.atan2(gravity[:, 0], -gravity[:, 2])))
    final_pos = robot.data.root_pos_w[:, :3]
    rows = []
    for index, spec in enumerate(specs):
        row = asdict(spec)
        row.update({
            "candidate_id": index,
            "max_root_height": float(max_height[index].item()),
            "min_root_height": float(min_height[index].item()),
            "final_x_displacement": float((final_pos[index, 0] - initial_pos[index, 0]).item()),
            "max_abs_pitch_rad": float(max_abs_pitch[index].item()),
            "survived": bool(min_height[index].item() >= 0.45 and max_abs_pitch[index].item() < 1.2),
        })
        rows.append(row)
    rows.sort(key=lambda row: (row["survived"], row["max_root_height"], -abs(row["final_x_displacement"])), reverse=True)
    payload = {
        "format": "m20_mirrored_jump_expert_search_v1",
        "candidate_count": len(rows),
        "physics_steps": args.physics_steps,
        "source": "mirrored joint-target physics rollouts; no PPO reward optimization",
        "native_pose": {"front": [0.0, -0.6, 1.0], "rear": [0.0, 0.6, -1.0]},
        "results": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[M20PRO-JUMP-EXPERT-SEARCH] candidates={len(rows)} steps={args.physics_steps}", flush=True)
    for row in rows[: args.top_k]:
        print(
            f"[M20PRO-JUMP-EXPERT-SEARCH] id={row['candidate_id']} survived={row['survived']} "
            f"max_z={row['max_root_height']:.4f} min_z={row['min_root_height']:.4f} "
            f"dx={row['final_x_displacement']:.4f} pitch={row['max_abs_pitch_rad']:.3f} "
            f"squat={row['squat_hip']:.2f}/{row['squat_knee']:.2f} "
            f"thrust={row['thrust_hip']:.2f}/{row['thrust_knee']:.2f} phase={row['squat_steps']}/{row['thrust_steps']}",
            flush=True,
        )
    scene.reset()
    sim.clear_instance()
    return 0


try:
    raise SystemExit(main())
finally:
    app.close()
