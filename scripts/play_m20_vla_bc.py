"""Closed-loop replay of the compact M20 language-conditioned BC policy."""

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
DEFAULT_CHECKPOINT = DATA_ROOT / "checkpoints/m20_vla_bc_v1/best.pt"
DEFAULT_VIDEO_ROOT = Path(os.environ.get("M20PRO_OUTPUT_ROOT", str(DATA_ROOT))) / "videos/m20_vla_bc_v1"
COMMAND_ALIASES = {
    "向前走": (0.5, 0.0, 0.0),
    "前进": (0.5, 0.0, 0.0),
    "向后走": (-0.5, 0.0, 0.0),
    "后退": (-0.5, 0.0, 0.0),
    "向左转": (0.0, 0.0, 0.5),
    "左转": (0.0, 0.0, 0.5),
}

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
parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_ROOT)
parser.add_argument("--metrics", type=Path, default=None, help="Optional JSON metrics path.")
parser.add_argument("--video", action="store_true", help="Required: record an inspectable third-person MP4.")
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
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera", update_period=0.02, height=96, width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.38, 0.0, 0.12), convention="world"),
    )
    rear_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_camera", update_period=0.02, height=96, width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(-0.38, 0.0, 0.12), convention="world"),
    )
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link", update_period=0.02, ray_alignment="base",
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


def make_proprio(robot: Articulation, joint_ids: list[int], last_action: torch.Tensor) -> torch.Tensor:
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=robot.device)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity)
    command = torch.tensor([[args.command_x, args.command_y, args.command_yaw]], device=robot.device)
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
    model = M20VLAActionChunk(horizon)
    model.load_state_dict(payload["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model.to(device).eval()
    args.video_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=str(device)))
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
    video_path = args.video_dir / f"bc-{args.task_text}-step-0.mp4"
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video writer: {video_path}")
    language = encode_text(args.task_text).to(robot.device).unsqueeze(0)
    cached_chunk = None
    cached_index = 0
    try:
        for _ in range(args.steps):
            if cached_chunk is None or cached_index >= min(args.chunk_execution, horizon):
                proprio = make_proprio(robot, joint_ids, last_action)
                front_image = rgb_small(front)
                rear_image = rgb_small(rear)
                rgb = torch.cat((front_image, rear_image), dim=-1).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
                scan = lidar_scan(lidar).unsqueeze(0).to(device)
                with torch.inference_mode():
                    cached_chunk = model(rgb, scan, proprio.to(device), language)
                cached_index = 0
            action = cached_chunk[:, cached_index].clamp(-4.0, 4.0).to(robot.device)
            cached_index += 1
            robot.set_joint_position_target(default_pose[:, :12] + action[:, :12] * leg_scale, joint_ids=leg_ids)
            robot.set_joint_velocity_target(action[:, 12:] * WHEEL_SCALE, joint_ids=wheel_ids)
            last_action = action
            for _ in range(4):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())
            target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.1]], device=robot.device)
            third.set_world_poses_from_view(target + torch.tensor([[-1.4, 1.4, 0.85]], device=robot.device), target)
            frame = third.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            height = float(robot.data.root_pos_w[0, 2].item())
            min_height, max_height = min(min_height, height), max(max_height, height)
            forward_speed_sum += float(robot.data.root_lin_vel_b[0, 0].item())
            action_sum += float(action.abs().mean().item())
            gravity_z = float(quat_apply_inverse(robot.data.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=robot.device))[0, 2].item())
            terminated_steps += int(height < 0.45 or gravity_z > -0.5)
            quat = robot.data.root_quat_w[0].detach().cpu().numpy()
            current_yaw = float(np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)))
            yaw_delta = float(np.arctan2(np.sin(current_yaw - start_yaw), np.cos(current_yaw - start_yaw)))
    finally:
        video.release()
    displacement = float(robot.data.root_pos_w[0, 0].item()) - start_x
    metrics_path = args.metrics or args.video_dir / f"bc-{args.task_text}-step-0.json"
    metrics = {
        "checkpoint": str(args.checkpoint),
        "task_text": args.task_text,
        "command": [args.command_x, args.command_y, args.command_yaw],
        "steps": args.steps,
        "chunk_execution": args.chunk_execution,
        "x_displacement": displacement,
        "yaw_delta": yaw_delta,
        "mean_forward_speed": forward_speed_sum / args.steps,
        "min_root_height": min_height,
        "max_root_height": max_height,
        "terminated_steps": terminated_steps,
        "mean_abs_action": action_sum / args.steps,
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
    print(f"[M20PRO-VLA-PLAY] video={video_path}")
    print(f"[M20PRO-VLA-PLAY] metrics={metrics_path}")
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
