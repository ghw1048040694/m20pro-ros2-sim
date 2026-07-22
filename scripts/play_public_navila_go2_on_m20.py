"""Replay NaVILA's public Go2 vision locomotion expert on the M20 morphology.

This is a compatibility probe, not a claim that Go2 weights are already
valid for M20.  The public policy is a 909->12 joint-position controller;
height-map features are zeroed for the first flat-ground smoke test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from isaaclab.app import AppLauncher

DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_POLICY = DATA_ROOT / "public_experts/navila_go2_vision/policy.jit"
DEFAULT_VIDEO_DIR = DATA_ROOT / "videos/navila_go2_on_m20_probe_v1"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--warmup-steps", type=int, default=50)
parser.add_argument("--command-x", type=float, default=0.5)
parser.add_argument("--command-y", type=float, default=0.0)
parser.add_argument("--command-yaw", type=float, default=0.0)
parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
parser.add_argument("--video", action="store_true", help="Required: record a third-person MP4.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.video:
    parser.error("--video is required")
if not args.policy.is_file():
    parser.error(f"public policy not found: {args.policy}")

args.enable_cameras = True
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

from assets.m20pro import M20PRO_CFG  # noqa: E402


POLICY_JOINT_NAMES = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
DEFAULT_POSE = torch.tensor(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0], dtype=torch.float32
)
ACTION_SCALE = 0.25
HEIGHT_FEATURES = 864


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = M20PRO_CFG.replace(
        init_state=M20PRO_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.54),
            joint_pos={name: float(value) for name, value in zip(POLICY_JOINT_NAMES, DEFAULT_POSE)},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hipx": DCMotorCfg(
                joint_names_expr=[".*_hipx_joint"], effort_limit=32.4, saturation_effort=32.4,
                velocity_limit=45.0, stiffness=80.0, damping=2.0,
            ),
            "hipy_knee": DCMotorCfg(
                joint_names_expr=[".*_(hipy|knee)_joint"], effort_limit=76.4, saturation_effort=76.4,
                velocity_limit=22.4, stiffness=80.0, damping=2.0,
            ),
        },
    ).replace(prim_path="{ENV_REGEX_NS}/Robot")
    third_person = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPerson", update_period=0.02, height=288, width=480, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    )


def base_rpy(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat.unbind(-1)
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = torch.where(sin_pitch.abs() < 1.0, torch.asin(sin_pitch), torch.sign(sin_pitch) * (torch.pi / 2.0))
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return torch.stack((roll, pitch, yaw), dim=-1)


def make_observation(robot: Articulation, joint_ids: list[int], last_action: torch.Tensor) -> torch.Tensor:
    command = torch.tensor([[args.command_x, args.command_y, args.command_yaw]], device=robot.device)
    joint_pos = robot.data.joint_pos[:, joint_ids] - DEFAULT_POSE.to(robot.device)
    joint_vel = robot.data.joint_vel[:, joint_ids]
    height = torch.zeros((1, HEIGHT_FEATURES), device=robot.device)
    return torch.cat(
        (robot.data.root_ang_vel_b, base_rpy(robot.data.root_quat_w), command, joint_pos, joint_vel, last_action, height),
        dim=-1,
    )


def reset_scene(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    robot.reset()
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()


def main() -> None:
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device="cuda:0"))
    scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    joint_ids, names = robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
    if names != POLICY_JOINT_NAMES:
        raise RuntimeError(f"M20 joint order mismatch: {names}")
    default_pose = DEFAULT_POSE.to(robot.device).unsqueeze(0)
    third = scene["third_person"]
    reset_scene(scene)
    for _ in range(args.warmup_steps):
        robot.set_joint_position_target(default_pose, joint_ids=joint_ids)
        scene.write_data_to_sim()
        for _ in range(4):
            sim.step()
            scene.update(sim.get_physics_dt())
    last_action = torch.zeros((1, 12), device=robot.device)
    start_pos = robot.data.root_pos_w[0].clone()
    min_height = max_height = float(start_pos[2].item())
    terminated_steps = 0
    video_dir = args.video_dir
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "navila-go2-on-m20-step-0.mp4"
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
    if not video.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        for _step in range(args.steps):
            observation = make_observation(robot, joint_ids, last_action).cpu()
            with torch.inference_mode():
                action = policy(observation).to(robot.device).clamp(-4.0, 4.0)
            robot.set_joint_position_target(default_pose + action * ACTION_SCALE, joint_ids=joint_ids)
            camera_target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.1]], device=robot.device)
            third.set_world_poses_from_view(
                camera_target + torch.tensor([[-1.4, 1.4, 0.85]], device=robot.device), camera_target
            )
            last_action = action
            for _ in range(4):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())
            frame = third.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            height = float(robot.data.root_pos_w[0, 2].item())
            min_height, max_height = min(min_height, height), max(max_height, height)
            gravity_z = float(
                quat_apply_inverse(
                    robot.data.root_quat_w,
                    torch.tensor([[0.0, 0.0, -1.0]], device=robot.device),
                )[0, 2].item()
            )
            terminated_steps += int(height < 0.45 or gravity_z > -0.5)
    finally:
        video.release()
    displacement = float(robot.data.root_pos_w[0, 0].item() - start_pos[0].item())
    metrics = {
        "policy": str(args.policy), "command": [args.command_x, args.command_y, args.command_yaw],
        "steps": args.steps, "x_displacement": displacement,
        "min_root_height": min_height, "max_root_height": max_height,
        "terminated_steps": terminated_steps, "video": str(video_path),
        "observation_shape": [1, 909], "height_features_zeroed": True,
    }
    metrics_path = video_dir / "navila-go2-on-m20-step-0.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[M20PRO-NAVILA-PROBE] x_displacement={displacement:.4f} m")
    print(f"[M20PRO-NAVILA-PROBE] min_root_height={min_height:.4f} m terminated_steps={terminated_steps}")
    print(f"[M20PRO-NAVILA-PROBE] video={video_path}")
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
