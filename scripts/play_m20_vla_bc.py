"""Closed-loop replay of the compact M20 language-conditioned BC policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from isaaclab.app import AppLauncher


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_CHECKPOINT = DATA_ROOT / "checkpoints/m20_vla_bc_v4/best.pt"
DEFAULT_VIDEO_ROOT = Path(os.environ.get("M20PRO_OUTPUT_ROOT", str(DATA_ROOT))) / "videos/m20_vla_bc_v4"
COMMAND_ALIASES = {
    "向前走": (0.5, 0.0, 0.0),
    "前进": (0.5, 0.0, 0.0),
    "向后走": (-0.5, 0.0, 0.0),
    "后退": (-0.5, 0.0, 0.0),
    "向左转": (0.0, 0.0, 0.5),
    "左转": (0.0, 0.0, 0.5),
}
TARGET_COLORS = {
    "red": (0.9, 0.05, 0.03),
    "blue": (0.03, 0.15, 0.95),
    "green": (0.04, 0.8, 0.08),
}
SENSOR_LINK = "base_link/base_link" if "m20_mjcf" in os.environ.get("M20PRO_USD_PATH", "") else "base_link"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument("--task-text", default="向前走")
parser.add_argument("--command-x", type=float, default=None)
parser.add_argument("--command-y", type=float, default=None)
parser.add_argument("--command-yaw", type=float, default=None)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--warmup-steps", type=int, default=75)
parser.add_argument("--chunk-execution", type=int, default=4, help="How many consecutive control steps to execute from each predicted chunk.")
parser.add_argument("--wheel-damping", type=float, default=None)
parser.add_argument("--target-color", choices=["none", "red", "blue", "green"], default="none")
parser.add_argument("--target-x", type=float, default=3.0)
parser.add_argument("--target-y", type=float, default=0.0)
parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_ROOT)
parser.add_argument("--metrics", type=Path, default=None, help="Optional JSON metrics path.")
parser.add_argument("--video", action="store_true", help="Required: record an inspectable third-person MP4.")
parser.add_argument("--debug-first-action", action="store_true", help="Print the first live observation/action for alignment checks.")
parser.add_argument("--debug-steps", type=int, default=0, help="Print stage timings for the first N control steps.")
parser.add_argument("--model-device", default="cpu", help="Device for VLA inference; CPU avoids CUDA/renderer contention.")
parser.add_argument("--stop-threshold", type=float, default=0.5, help="Sigmoid threshold for the learned VLA stop head.")
parser.add_argument("--stop-brake-gain", type=float, default=2.0, help="Body-speed gain for the learned-stop wheel brake.")
parser.add_argument("--stop-wheel-damping", type=float, default=3.6, help="Wheel damping while the learned stop brake is active.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.video:
    parser.error("--video is required")
if not args.checkpoint.is_file():
    parser.error(f"BC checkpoint not found: {args.checkpoint}")
if args.steps <= 0:
    parser.error("--steps must be positive")
if args.chunk_execution <= 0:
    parser.error("--chunk-execution must be positive")
if args.command_x is None or args.command_y is None or args.command_yaw is None:
    command = COMMAND_ALIASES.get(args.task_text, (0.5, 0.0, 0.0))
    args.command_x = command[0] if args.command_x is None else args.command_x
    args.command_y = command[1] if args.command_y is None else args.command_y
    args.command_yaw = command[2] if args.command_yaw is None else args.command_yaw
args.enable_cameras = True
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from m20_vla_model import M20VLAActionChunk  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns  # noqa: E402
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
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]
DEFAULT_POSE = torch.tensor(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)
LEG_SCALE = torch.tensor([0.125, 0.25, 0.25] * 4, dtype=torch.float32)
WHEEL_SCALE = 5.0
WHEEL_DAMPING = args.wheel_damping if args.wheel_damping is not None else (3.6 if abs(args.command_yaw) >= 0.05 else 0.6)


@configclass
class VLAReplaySceneCfg(InteractiveSceneCfg):
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
            "hipx": DCMotorCfg(joint_names_expr=[".*_hipx_joint"], effort_limit=32.4, saturation_effort=32.4,
                                velocity_limit=45.0, stiffness=80.0, damping=2.0),
            "hipy_knee": DCMotorCfg(joint_names_expr=[".*_(hipy|knee)_joint"], effort_limit=76.4, saturation_effort=76.4,
                                    velocity_limit=22.4, stiffness=80.0, damping=2.0),
            "wheels": DCMotorCfg(joint_names_expr=[".*_wheel_joint"], effort_limit=21.6, saturation_effort=21.6,
                                 velocity_limit=79.3, stiffness=0.0, damping=WHEEL_DAMPING),
        },
    ).replace(prim_path="{ENV_REGEX_NS}/Robot")
    target = (
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=sim_utils.CuboidCfg(
                size=(0.42, 0.42, 0.84),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TARGET_COLORS[args.target_color]),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(args.target_x, args.target_y, 0.42)),
        )
        if args.target_color != "none" else None
    )
    front_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}/front_camera", update_period=0.02, height=96, width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.38, 0.0, 0.12), convention="world"),
    )
    rear_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}/rear_camera", update_period=0.02, height=96, width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(-0.38, 0.0, 0.12), convention="world"),
    )
    lidar = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}", update_period=0.02, ray_alignment="base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.16)),
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0),
                                              horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0),
        max_distance=20.0, mesh_prim_paths=["/World/ground"],
    )
    third_person = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPerson", update_period=0.02, height=288, width=480, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)))


def encode_text(text: str, max_length: int = 32) -> torch.Tensor:
    token_bytes = np.frombuffer(text.encode("utf-8")[:max_length], dtype=np.uint8).astype(np.int64) + 1
    tokens = np.zeros(max_length, dtype=np.int64)
    tokens[: len(token_bytes)] = token_bytes
    return torch.from_numpy(tokens)


def make_proprio(
    robot: Articulation, joint_ids: list[int], last_action: torch.Tensor, mask_privileged_command: bool
) -> torch.Tensor:
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=robot.device)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity)
    command = (
        torch.zeros((1, 3), device=robot.device)
        if mask_privileged_command
        else torch.tensor([[args.command_x, args.command_y, args.command_yaw]], device=robot.device)
    )
    joint_pos = robot.data.joint_pos[:, joint_ids].clone()
    joint_pos[:, 12:] = 0.0
    joint_pos -= DEFAULT_POSE.to(robot.device)
    joint_vel = robot.data.joint_vel[:, joint_ids] * 0.05
    return torch.cat((robot.data.root_ang_vel_b * 0.25, projected_gravity, command, joint_pos, joint_vel, last_action), dim=-1)


def rgb_small(camera) -> torch.Tensor:
    image = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    image = cv2.resize(image, (80, 48), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image)


def lidar_scan(lidar) -> torch.Tensor:
    hits = lidar.data.ray_hits_w[0]
    origin = lidar.data.pos_w[0].unsqueeze(0)
    values = torch.linalg.vector_norm(hits - origin, dim=-1)
    values = torch.nan_to_num(values, nan=20.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0)
    return values.detach().cpu().float().div_(20.0)


def apply_stop_brake(robot: Articulation, action: torch.Tensor) -> torch.Tensor:
    """Keep VLA's stop decision, but dissipate the rolling body's inertia."""
    body_velocity = robot.data.root_lin_vel_b[0]
    body_angular_velocity = robot.data.root_ang_vel_b[0]
    forward = float(torch.clamp(-args.stop_brake_gain * body_velocity[0], -0.35, 0.35).item())
    yaw_rate = float(torch.clamp(-1.5 * body_angular_velocity[2], -0.5, 0.5).item())
    half_track = 0.24
    yaw_differential = half_track * yaw_rate * 4.0
    left_velocity = -(forward + yaw_differential) / 0.09
    right_velocity = -(forward - yaw_differential) / 0.09
    brake = torch.tensor(
        [[left_velocity, right_velocity, left_velocity, right_velocity]],
        dtype=action.dtype,
        device=action.device,
    ).div_(5.0).clamp_(-2.5, 2.5)
    action = action.clone()
    action[:, 12:] = brake
    return action


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
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    horizon = int(payload["horizon"])
    checkpoint_config = payload.get("config", {})
    architecture = checkpoint_config.get("architecture", "global_v1")
    mask_privileged_command = bool(checkpoint_config.get("proprio_command_mask"))
    stop_head = bool(checkpoint_config.get("stop_head", False))
    model = M20VLAActionChunk(horizon, architecture=architecture, stop_head=stop_head)
    model.load_state_dict(payload["model_state_dict"])
    sim_device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model_device = torch.device(
        args.model_device if torch.cuda.is_available() or not args.model_device.startswith("cuda") else "cpu"
    )
    model.to(model_device).eval()
    args.video_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=str(sim_device)))
    scene = InteractiveScene(VLAReplaySceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    joint_ids, names = robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
    if names != POLICY_JOINT_NAMES:
        raise RuntimeError(f"M20 joint order mismatch: {names}")
    leg_ids, wheel_ids = joint_ids[:12], joint_ids[12:]
    front, rear, lidar, third = scene["front_camera"], scene["rear_camera"], scene["lidar"], scene["third_person"]
    default_pose = DEFAULT_POSE.to(robot.device).unsqueeze(0)
    leg_scale = LEG_SCALE.to(robot.device).unsqueeze(0)
    zero_wheels = torch.zeros((1, 4), device=robot.device)
    reset_scene(scene)
    for _ in range(args.warmup_steps):
        robot.set_joint_position_target(default_pose[:, :12], joint_ids=leg_ids)
        robot.set_joint_velocity_target(zero_wheels, joint_ids=wheel_ids)
        scene.write_data_to_sim()
        for _ in range(4):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
    last_action = torch.zeros((1, 16), device=robot.device)
    start_x = float(robot.data.root_pos_w[0, 0].item())
    start_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    start_yaw = float(np.arctan2(2.0 * (start_quat[0] * start_quat[3] + start_quat[1] * start_quat[2]), 1.0 - 2.0 * (start_quat[2] ** 2 + start_quat[3] ** 2)))
    yaw_delta = 0.0
    min_height = max_height = float(robot.data.root_pos_w[0, 2].item())
    forward_speed_sum = 0.0
    action_sum = 0.0
    terminated_steps = 0
    target_reached = args.target_color == "none"
    target_reached_step = None
    min_target_distance = float("inf")
    post_reach_speed_sum = 0.0
    post_reach_steps = 0
    final_planar_speed = 0.0
    video_path = args.video_dir / f"bc-{args.task_text}-step-0.mp4"
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video writer: {video_path}")
    language = encode_text(args.task_text).to(model_device).unsqueeze(0)
    cached_chunk = None
    cached_index = 0
    stop_latched = False
    stop_triggered_step = None
    try:
        for step in range(args.steps):
            debug_started = time.perf_counter()
            if step < args.debug_steps:
                print(f"[M20PRO-VLA-TIMING] step={step} stage=begin", flush=True)
            if cached_chunk is None or cached_index >= min(args.chunk_execution, horizon):
                proprio = make_proprio(robot, joint_ids, last_action, mask_privileged_command)
                front_image = rgb_small(front)
                rear_image = rgb_small(rear)
                rgb = torch.cat((front_image, rear_image), dim=-1).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(model_device)
                scan = lidar_scan(lidar).unsqueeze(0).to(model_device)
                cached_index = 0
            if step < args.debug_steps:
                print(
                    f"[M20PRO-VLA-TIMING] step={step} stage=sensors elapsed={time.perf_counter() - debug_started:.3f}s",
                    flush=True,
                )
            with torch.inference_mode():
                model_output = model(rgb, scan, proprio.to(model_device), language, return_stop=stop_head)
                if stop_head:
                    cached_chunk, stop_logit = model_output
                    stop_probability = float(torch.sigmoid(stop_logit[0]).item())
                    if stop_probability >= args.stop_threshold:
                        stop_latched = True
                        if stop_triggered_step is None:
                            stop_triggered_step = step
                else:
                    cached_chunk = model_output
                    stop_probability = None
            if step < args.debug_steps:
                print(
                    f"[M20PRO-VLA-TIMING] step={step} stage=inference elapsed={time.perf_counter() - debug_started:.3f}s",
                    flush=True,
                )
            if args.debug_first_action and step == 0:
                print(
                    f"[M20PRO-VLA-DEBUG] proprio={proprio[0, :9].detach().cpu().numpy().round(5).tolist()} "
                    f"rgb_mean={rgb.mean().item():.5f} lidar_mean={scan.mean().item():.5f} "
                    f"chunk0={cached_chunk[0, 0].detach().cpu().numpy().round(5).tolist()}",
                    flush=True,
                )
                cached_index = 0
            action = cached_chunk[:, cached_index].clamp(-4.0, 4.0).to(robot.device)
            cached_index += 1
            if stop_latched:
                action.zero_()
                action = apply_stop_brake(robot, action)
                robot.actuators["wheels"].damping.fill_(args.stop_wheel_damping)
            robot.set_joint_position_target(default_pose[:, :12] + action[:, :12] * leg_scale, joint_ids=leg_ids)
            robot.set_joint_velocity_target(action[:, 12:] * WHEEL_SCALE, joint_ids=wheel_ids)
            last_action = action
            # Move the inspection camera before stepping so the rendered
            # frame uses the pose for this control interval.  Setting it
            # after the step leaves one stale frame from the camera's spawn
            # pose (inside the MJCF root), which makes the replay misleading.
            camera_target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.1]], device=robot.device)
            camera_eye = camera_target + torch.tensor([[-1.4, 1.4, 0.85]], device=robot.device)
            if step < args.debug_steps:
                print(
                    f"[M20PRO-VLA-TIMING] step={step} stage=camera_begin elapsed={time.perf_counter() - debug_started:.3f}s",
                    flush=True,
                )
            third.set_world_poses_from_view(camera_eye, camera_target)
            if step < args.debug_steps:
                print(
                    f"[M20PRO-VLA-TIMING] step={step} stage=camera_end elapsed={time.perf_counter() - debug_started:.3f}s",
                    flush=True,
                )
            for _ in range(4):
                if step < args.debug_steps and _ == 0:
                    print(
                        f"[M20PRO-VLA-TIMING] step={step} stage=sim_begin elapsed={time.perf_counter() - debug_started:.3f}s",
                        flush=True,
                    )
                scene.write_data_to_sim()
                # SimulationCfg.render_interval already renders every fourth
                # physics step.  Explicit False/True toggling here can
                # deadlock the next CUDA model inference in headless mode.
                sim.step()
                scene.update(sim.get_physics_dt())
            if step < args.debug_steps:
                print(
                    f"[M20PRO-VLA-TIMING] step={step} stage=simulation elapsed={time.perf_counter() - debug_started:.3f}s",
                    flush=True,
                )
            frame = third.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if step < args.debug_steps:
                print(
                    f"[M20PRO-VLA-TIMING] step={step} stage=video elapsed={time.perf_counter() - debug_started:.3f}s",
                    flush=True,
                )
            height = float(robot.data.root_pos_w[0, 2].item())
            min_height, max_height = min(min_height, height), max(max_height, height)
            forward_speed_sum += float(robot.data.root_lin_vel_b[0, 0].item())
            action_sum += float(action.abs().mean().item())
            final_planar_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[0, :2]).item())
            if args.target_color != "none":
                target_xy = torch.tensor([args.target_x, args.target_y], device=robot.device)
                target_distance = float(torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - target_xy).item())
                min_target_distance = min(min_target_distance, target_distance)
                if not target_reached and target_distance <= 0.8:
                    target_reached = True
                    target_reached_step = step
                if target_reached:
                    post_reach_speed_sum += final_planar_speed
                    post_reach_steps += 1
            gravity_z = float(quat_apply_inverse(robot.data.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=robot.device))[0, 2].item())
            terminated_steps += int(height < 0.45 or gravity_z > -0.5)
            quat = robot.data.root_quat_w[0].detach().cpu().numpy()
            current_yaw = float(np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)))
            yaw_delta = float(np.arctan2(np.sin(current_yaw - start_yaw), np.cos(current_yaw - start_yaw)))
    finally:
        video.release()
    displacement = float(robot.data.root_pos_w[0, 0].item()) - start_x
    final_target_distance = (
        float(torch.linalg.vector_norm(
            robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
        ).item())
        if args.target_color != "none" else None
    )
    post_reach_mean_speed = post_reach_speed_sum / post_reach_steps if post_reach_steps else None
    stable = terminated_steps == 0 and min_height >= 0.45
    stopped_after_reach = bool(post_reach_mean_speed is not None and post_reach_mean_speed < 0.15)
    success = stable and (args.target_color == "none" or (target_reached and stopped_after_reach))
    metrics_path = args.metrics or args.video_dir / f"bc-{args.task_text}-step-0.json"
    metrics = {
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "task_text": args.task_text,
        "command": [args.command_x, args.command_y, args.command_yaw],
        "privileged_command_used_by_model": not mask_privileged_command,
        "target_color": args.target_color,
        "target_xy": [args.target_x, args.target_y] if args.target_color != "none" else None,
        "steps": args.steps,
        "chunk_execution": args.chunk_execution,
        "x_displacement": displacement,
        "yaw_delta": yaw_delta,
        "mean_forward_speed": forward_speed_sum / args.steps,
        "min_root_height": min_height,
        "max_root_height": max_height,
        "terminated_steps": terminated_steps,
        "mean_abs_action": action_sum / args.steps,
        "stop_head": stop_head,
        "stop_threshold": args.stop_threshold if stop_head else None,
        "stop_triggered_step": stop_triggered_step,
        "target_reached": target_reached,
        "target_reached_step": target_reached_step,
        "min_target_distance": None if args.target_color == "none" else min_target_distance,
        "final_target_distance": final_target_distance,
        "post_reach_mean_speed": post_reach_mean_speed,
        "final_planar_speed": final_planar_speed,
        "stopped_after_reach": stopped_after_reach,
        "success": success,
        "wheel_damping": WHEEL_DAMPING,
        "video": str(video_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(f"[M20PRO-VLA-PLAY] checkpoint={args.checkpoint}")
    print(f"[M20PRO-VLA-PLAY] task={args.task_text} command=[{args.command_x:.3f}, {args.command_y:.3f}, {args.command_yaw:.3f}]")
    print(f"[M20PRO-VLA-PLAY] steps={args.steps} x_displacement={displacement:.4f} m")
    print(f"[M20PRO-VLA-PLAY] yaw_delta={yaw_delta:.4f} rad")
    print(f"[M20PRO-VLA-PLAY] mean_forward_speed={forward_speed_sum / args.steps:.4f} m/s")
    print(f"[M20PRO-VLA-PLAY] min_root_height={min_height:.4f} m max_root_height={max_height:.4f} m")
    print(f"[M20PRO-VLA-PLAY] terminated_steps={terminated_steps} mean_abs_action={action_sum / args.steps:.4f}")
    if args.target_color != "none":
        print(
            f"[M20PRO-VLA-PLAY] target_reached={target_reached} reached_step={target_reached_step} "
            f"min_distance={min_target_distance:.4f} m final_distance={final_target_distance:.4f} m"
        )
        print(
            f"[M20PRO-VLA-PLAY] post_reach_mean_speed={post_reach_mean_speed} "
            f"final_planar_speed={final_planar_speed:.4f} m/s stopped={stopped_after_reach} success={success}"
        )
    print(f"[M20PRO-VLA-PLAY] video={video_path}")
    print(f"[M20PRO-VLA-PLAY] metrics={metrics_path}")
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
