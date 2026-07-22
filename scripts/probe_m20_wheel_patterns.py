"""Calibrate direct M20 wheel signs before building a navigation expert."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from isaaclab.app import AppLauncher
from video_utils import finalize_h264_video

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--wheel-action", type=float, default=1.0, help="Normalized wheel action magnitude; velocity target is action*5.")
parser.add_argument("--video-dir", type=Path, default=Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_wheel_pattern_probe_v1"))
parser.add_argument("--json-output", type=Path, default=Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_wheel_pattern_probe_v1.json"))
parser.add_argument("--video", action="store_true", help="Required: write one MP4 per tested pattern.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.video:
    parser.error("--video is required")
if args.steps <= 0 or args.wheel_action <= 0:
    parser.error("--steps and --wheel-action must be positive")
args.enable_cameras = True
app = AppLauncher(args).app

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

from assets.m20pro import M20PRO_CFG  # noqa: E402


JOINT_NAMES = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]
DEFAULT_POSE = torch.tensor(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)
PATTERNS = [
    (1, -1, 1, -1),
    (-1, 1, -1, 1),
    (1, -1, -1, 1),
    (-1, 1, 1, -1),
    (1, 1, -1, -1),
    (-1, -1, 1, 1),
    (1, 1, 1, 1),
    (-1, -1, -1, -1),
]


@configclass
class WheelPatternSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = M20PRO_CFG.replace(
        init_state=M20PRO_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.54),
            joint_pos={name: float(value) for name, value in zip(JOINT_NAMES, DEFAULT_POSE)},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "hipx": DCMotorCfg(joint_names_expr=[".*_hipx_joint"], effort_limit=32.4, saturation_effort=32.4, velocity_limit=45.0, stiffness=80.0, damping=2.0),
            "hipy_knee": DCMotorCfg(joint_names_expr=[".*_(hipy|knee)_joint"], effort_limit=76.4, saturation_effort=76.4, velocity_limit=22.4, stiffness=80.0, damping=2.0),
            "wheels": DCMotorCfg(joint_names_expr=[".*_wheel_joint"], effort_limit=21.6, saturation_effort=21.6, velocity_limit=79.3, stiffness=0.0, damping=0.6),
        },
    ).replace(prim_path="{ENV_REGEX_NS}/Robot")
    third_person = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPerson", update_period=0.02, height=288, width=480, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)))


def yaw_from_quaternion(quat: torch.Tensor) -> torch.Tensor:
    return torch.atan2(2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]), 1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))


def main() -> None:
    args.video_dir.mkdir(parents=True, exist_ok=True)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    device = args.device or "cuda:0"
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=device))
    scene = InteractiveScene(WheelPatternSceneCfg(num_envs=len(PATTERNS), env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    joint_ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if names != JOINT_NAMES:
        raise RuntimeError(f"joint order mismatch: {names}")
    leg_ids, wheel_ids = joint_ids[:12], joint_ids[12:]
    camera = scene["third_person"]
    default_pose = DEFAULT_POSE.to(robot.device).unsqueeze(0).expand(len(PATTERNS), -1)
    zero_wheels = torch.zeros((len(PATTERNS), 4), device=robot.device)
    wheel_targets = torch.tensor(PATTERNS, dtype=torch.float32, device=robot.device) * args.wheel_action * 5.0
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(default_pose, torch.zeros_like(default_pose))
    scene.reset()
    for _ in range(75):
        robot.set_joint_position_target(default_pose[:, :12], joint_ids=leg_ids)
        robot.set_joint_velocity_target(zero_wheels, joint_ids=wheel_ids)
        scene.write_data_to_sim()
        for _ in range(4):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
    start_xy = robot.data.root_pos_w[:, :2].clone()
    start_yaw = yaw_from_quaternion(robot.data.root_quat_w).clone()
    min_height = robot.data.root_pos_w[:, 2].clone()
    max_abs_angular_velocity = torch.zeros(len(PATTERNS), device=robot.device)
    videos = []
    video_paths = []
    for index in range(len(PATTERNS)):
        path = args.video_dir / f"pattern_{index:02d}.mp4"
        video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
        if not video.isOpened():
            raise RuntimeError(f"Unable to open video writer: {path}")
        videos.append(video)
        video_paths.append(path)
    try:
        for _ in range(args.steps):
            robot.set_joint_position_target(default_pose[:, :12], joint_ids=leg_ids)
            robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_ids)
            scene.write_data_to_sim()
            for _ in range(4):
                sim.step()
                scene.update(sim.get_physics_dt())
            target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.1]], device=robot.device)
            camera.set_world_poses_from_view(target + torch.tensor([[-1.4, 1.4, 0.85]], device=robot.device), target)
            frames = camera.data.output["rgb"][..., :3].detach().cpu().numpy().astype(np.uint8)
            for index, video in enumerate(videos):
                video.write(cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR))
            min_height = torch.minimum(min_height, robot.data.root_pos_w[:, 2])
            max_abs_angular_velocity = torch.maximum(max_abs_angular_velocity, robot.data.root_ang_vel_b.abs().max(dim=1).values)
    finally:
        for video, path in zip(videos, video_paths):
            finalize_h264_video(video, path)
    final_yaw = yaw_from_quaternion(robot.data.root_quat_w)
    yaw_delta = torch.atan2(torch.sin(final_yaw - start_yaw), torch.cos(final_yaw - start_yaw))
    rows = []
    for index, pattern in enumerate(PATTERNS):
        rows.append({
            "pattern_id": index, "wheel_signs": pattern, "wheel_action": args.wheel_action,
            "x_displacement": float((robot.data.root_pos_w[index, 0] - start_xy[index, 0]).item()),
            "y_displacement": float((robot.data.root_pos_w[index, 1] - start_xy[index, 1]).item()),
            "yaw_delta_rad": float(yaw_delta[index].item()), "min_root_height": float(min_height[index].item()),
            "max_abs_angular_velocity": float(max_abs_angular_velocity[index].item()),
            "video": str(args.video_dir / f"pattern_{index:02d}.mp4"),
        })
        print(f"[M20PRO-WHEEL-PATTERN] {rows[-1]}", flush=True)
    args.json_output.write_text(json.dumps({"format": "m20_direct_wheel_pattern_probe_v1", "steps": args.steps, "results": rows}, indent=2) + "\n")
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
