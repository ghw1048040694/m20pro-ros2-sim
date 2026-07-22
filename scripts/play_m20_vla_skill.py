"""Replay the two-layer M20 VLA: skill selector plus native M20 expert.

The high-level policy consumes RGB, LiDAR, proprioception and language, then
emits a bounded navigation command.  The released M20 57->16 ONNX policy is
the low-level controller.  Target coordinates are used only for evaluation;
they are never passed to either policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from isaaclab.app import AppLauncher
from video_utils import finalize_h264_video


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_CHECKPOINT = DATA_ROOT / "checkpoints/m20_vla_skill_v1/best.pt"
DEFAULT_POLICY = DATA_ROOT / "public_experts/m20_native/policy.onnx"
DEFAULT_VIDEO_DIR = DATA_ROOT / "videos/m20_vla_skill_v1"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--task-text", default="到红色方块去")
parser.add_argument("--target-color", choices=["none", "red", "blue", "green"], default="red")
parser.add_argument("--target-x", type=float, default=2.5)
parser.add_argument("--target-y", type=float, default=0.0)
parser.add_argument("--target-radius", type=float, default=0.8)
parser.add_argument(
    "--target-hold-steps",
    type=int,
    default=100,
    help="Required consecutive 50 Hz frames inside the target radius after stop latches.",
)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--warmup-steps", type=int, default=75)
parser.add_argument("--command-hold-steps", type=int, default=1)
parser.add_argument("--stop-threshold", type=float, default=0.65)
parser.add_argument("--target-threshold", type=float, default=0.65, help="Target-stop head probability required to latch stop.")
parser.add_argument("--target-distance-threshold", type=float, default=0.16, help="Normalized predicted target distance required to latch stop (0.16 ~= 0.8 m).")
parser.add_argument(
    "--target-turnaround-window",
    type=int,
    default=0,
    help="If positive, require learned distance to rise from a recent minimum before stop confirmation.",
)
parser.add_argument(
    "--target-turnaround-rise-m",
    type=float,
    default=0.03,
    help="Learned-distance rise from the recent minimum required by turnaround mode.",
)
parser.add_argument("--stop-confirm-steps", type=int, default=8, help="Consecutive high-confidence frames required before latching stop.")
parser.add_argument("--stop-vote-window", type=int, default=15)
parser.add_argument("--stop-vote-fraction", type=float, default=0.60)
parser.add_argument("--max-turn-steps", type=int, default=8, help="Maximum consecutive VLA turn frames before forward recovery.")
parser.add_argument("--turn-recovery-steps", type=int, default=20, help="Forward frames used after the turn watchdog trips.")
parser.add_argument("--max-yaw-command", type=float, default=0.25)
parser.add_argument("--max-forward-command", type=float, default=0.35)
parser.add_argument("--min-forward-command", type=float, default=0.08, help="Minimum approach speed until the learned target-stop gate fires.")
parser.add_argument("--search-yaw-command", type=float, default=0.5)
parser.add_argument("--search-threshold", type=float, default=0.5)
parser.add_argument("--model-device", default="cpu")
parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
parser.add_argument("--metrics", type=Path, default=None)
parser.add_argument("--initial-height", type=float, default=0.54)
parser.add_argument("--mirror-negative-yaw", action="store_true")
parser.add_argument("--debug-steps", type=int, default=0)
parser.add_argument("--video", action="store_true", help="Required: record an inspectable MP4.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.video:
    parser.error("--video is required so every replay leaves an inspectable MP4")
if args.steps <= 0 or args.warmup_steps < 0 or args.command_hold_steps <= 0 or args.target_radius <= 0.0 or args.target_hold_steps <= 0 or args.stop_confirm_steps <= 0 or args.stop_vote_window <= 0 or not 0.0 < args.stop_vote_fraction <= 1.0 or args.max_turn_steps <= 0 or args.turn_recovery_steps <= 0 or args.search_yaw_command <= 0.0 or not 0.0 < args.search_threshold < 1.0 or not 0.0 < args.target_threshold < 1.0 or not 0.0 < args.target_distance_threshold < 1.0 or args.target_turnaround_window < 0 or args.target_turnaround_rise_m < 0.0 or not 0.0 < args.min_forward_command <= args.max_forward_command:
    parser.error("--steps must be positive and timing values must be non-negative")
if not args.checkpoint.is_file():
    parser.error(f"skill checkpoint not found: {args.checkpoint}")
if not args.policy.is_file():
    parser.error(f"M20 policy not found: {args.policy}")
args.enable_cameras = True
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from m20_vla_skill_model import (  # noqa: E402
    COMMAND_SCALE,
    SKILL_NAMES,
    TARGET_DISTANCE_SCALE_M,
    M20VLASkillPolicy,
)

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
MIRROR_PERM = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 13, 12, 15, 14], dtype=torch.long)
MIRROR_SIGN = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
MIRROR_ANGULAR_VELOCITY = torch.tensor([-1.0, 1.0, -1.0])
MIRROR_POLAR_VECTOR = torch.tensor([1.0, -1.0, 1.0])
TARGET_COLORS = {"red": (0.9, 0.05, 0.03), "blue": (0.03, 0.15, 0.95), "green": (0.04, 0.8, 0.08)}
SENSOR_LINK = "base_link/base_link" if "m20_mjcf" in str(os.environ.get("M20PRO_USD_PATH", "")) else "base_link"

PUBLIC_M20_CFG = M20PRO_CFG.replace(
    init_state=M20PRO_CFG.init_state.replace(
        pos=(0.0, 0.0, args.initial_height),
        joint_pos={name: float(value) for name, value in zip(POLICY_JOINT_NAMES, DEFAULT_POSE)},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hipx": DCMotorCfg(joint_names_expr=[".*_hipx_joint"], effort_limit=32.4, saturation_effort=32.4, velocity_limit=45.0, stiffness=80.0, damping=2.0),
        "hipy_knee": DCMotorCfg(joint_names_expr=[".*_(hipy|knee)_joint"], effort_limit=76.4, saturation_effort=76.4, velocity_limit=22.4, stiffness=80.0, damping=2.0),
        "wheels": DCMotorCfg(joint_names_expr=[".*_wheel_joint"], effort_limit=21.6, saturation_effort=21.6, velocity_limit=79.3, stiffness=0.0, damping=0.6),
    },
)


@configclass
class SkillSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = PUBLIC_M20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}/front_camera", update_period=0.02, height=96, width=160, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.38, 0.0, 0.12), convention="world"),
    )
    rear_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}/rear_camera", update_period=0.02, height=96, width=160, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(-0.38, 0.0, 0.12), convention="world"),
    )
    lidar = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}", update_period=0.02, ray_alignment="base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.16)),
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0),
        max_distance=20.0, mesh_prim_paths=["/World/ground"],
    )
    third_person = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPerson", update_period=0.02, height=288, width=480, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)))


def encode_text(text: str, max_length: int = 32) -> torch.Tensor:
    values = np.frombuffer(text.encode("utf-8")[:max_length], dtype=np.uint8).astype(np.int64) + 1
    tokens = np.zeros(max_length, dtype=np.int64)
    tokens[:len(values)] = values
    return torch.from_numpy(tokens)


def native_observation(robot: Articulation, joint_ids: list[int], last_action: torch.Tensor, command: tuple[float, float, float], mirror: bool = False) -> torch.Tensor:
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=robot.device)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity)
    command_tensor = torch.tensor([list(command)], device=robot.device)
    joint_pos = robot.data.joint_pos[:, joint_ids].clone()
    joint_pos[:, 12:] = 0.0
    joint_pos -= DEFAULT_POSE.to(robot.device)
    joint_vel = robot.data.joint_vel[:, joint_ids] * 0.05
    observation = torch.cat([robot.data.root_ang_vel_b * 0.25, projected_gravity, command_tensor, joint_pos, joint_vel, last_action], dim=-1)
    if mirror:
        observation[:, 0:3] *= MIRROR_ANGULAR_VELOCITY.to(robot.device)
        observation[:, 3:6] *= MIRROR_POLAR_VECTOR.to(robot.device)
        for start in (9, 25, 41):
            observation[:, start:start + 16] = observation[:, start:start + 16][:, MIRROR_PERM.to(robot.device)] * MIRROR_SIGN.to(robot.device)
    return observation


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    return action[:, MIRROR_PERM.to(action.device)] * MIRROR_SIGN.to(action.device)


def rgb_small(camera) -> torch.Tensor:
    image = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
    image = cv2.resize(image, (80, 48), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image)


def lidar_scan(lidar) -> torch.Tensor:
    hits = lidar.data.ray_hits_w[0]
    origin = lidar.data.pos_w[0].unsqueeze(0)
    values = torch.linalg.vector_norm(hits - origin, dim=-1)
    return torch.nan_to_num(values, nan=20.0, posinf=20.0, neginf=0.0).clamp(0.0, 20.0).detach().cpu().float().div_(20.0)


def high_level_proprio(robot: Articulation, joint_ids: list[int], last_action: torch.Tensor) -> torch.Tensor:
    # The high-level policy is trained without privileged expert command or
    # expert action history.  It must remain closed under its own low-level
    # actions during replay.
    zero_action = torch.zeros_like(last_action)
    return native_observation(robot, joint_ids, zero_action, (0.0, 0.0, 0.0), mirror=False)


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
    config = payload.get("config", {})
    search_head = bool(config.get("search_head", False))
    target_head = bool(config.get("target_head", False))
    target_head_mode = str(config.get("target_head_mode", "shared_v1"))
    target_distance_scale_m = float(config.get("target_distance_scale_m", TARGET_DISTANCE_SCALE_M))
    if target_distance_scale_m <= 0.0:
        raise RuntimeError(f"Invalid target distance scale: {target_distance_scale_m}")
    model = M20VLASkillPolicy(
        config.get("architecture", "spatial_v2"),
        search_head=search_head,
        target_head=target_head,
        target_head_mode=target_head_mode,
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model_device = torch.device(args.model_device if torch.cuda.is_available() or not args.model_device.startswith("cuda") else "cpu")
    model.to(model_device).eval()
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 57] or session.get_outputs()[0].shape != [1, 16]:
        raise RuntimeError("Native M20 policy is not the expected 57->16 model")
    args.video_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics or args.video_dir / "m20-vla-skill-step-0.json"
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=args.device or "cuda:0"))
    scene = InteractiveScene(SkillSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    joint_ids, names = robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
    if names != POLICY_JOINT_NAMES:
        raise RuntimeError(f"M20 policy order mismatch: {names}")
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
    last_low_level_action = torch.zeros((1, 16), device=robot.device)
    language = encode_text(args.task_text).to(model_device).unsqueeze(0)
    start_xy = robot.data.root_pos_w[0, :2].clone()
    start_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    start_yaw = float(np.arctan2(2.0 * (start_quat[0] * start_quat[3] + start_quat[1] * start_quat[2]), 1.0 - 2.0 * (start_quat[2] ** 2 + start_quat[3] ** 2)))
    min_height = max_height = float(robot.data.root_pos_w[0, 2].item())
    min_target_distance = float("inf")
    target_xy_eval = (
        torch.tensor([args.target_x, args.target_y], device=robot.device)
        if args.target_color != "none"
        else None
    )
    target_reached = args.target_color == "none"
    target_reached_step = None
    post_stop_target_hold_streak = 0
    max_post_stop_target_hold_steps = 0
    stop_triggered_step = None
    stop_confidence_streak = 0
    stop_votes: deque[bool] = deque(maxlen=args.stop_vote_window)
    target_distance_history: deque[float] = deque(
        maxlen=max(args.target_turnaround_window, 1)
    )
    terminated_steps = 0
    command_sum = np.zeros(3, dtype=np.float64)
    command_count = 0
    skill_counts: Counter[str] = Counter()
    prediction_trace: list[dict[str, object]] = []
    min_predicted_target_distance = None
    min_predicted_target_distance_step = None
    max_target_stop_probability = None
    max_target_stop_probability_step = None
    target_distance_abs_error_sum = 0.0
    target_distance_error_count = 0
    video_path = args.video_dir / "m20-vla-skill-step-0.mp4"
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video writer: {video_path}")
    cached_command = (0.0, 0.0, 0.0)
    cached_skill = "stop"
    stop_latched = False
    turn_streak = 0
    recovery_remaining = 0
    last_nonstop_command = (args.max_forward_command, 0.0, 0.0)
    search_remaining = 0
    search_sign = -1.0
    try:
        for step in range(args.steps):
            if search_remaining > 0:
                cached_skill = "search"
                cached_command = (0.0, 0.0, search_sign * args.search_yaw_command)
                search_remaining -= 1
                stop_confidence_streak = 0
                stop_votes.clear()
                target_distance_history.clear()
            elif recovery_remaining > 0:
                cached_skill = "forward_recovery"
                cached_command = (args.max_forward_command, 0.0, 0.0)
                recovery_remaining -= 1
                stop_confidence_streak = 0
                stop_votes.clear()
                target_distance_history.clear()
            elif step % args.command_hold_steps == 0:
                front_image = rgb_small(front)
                rear_image = rgb_small(rear)
                rgb = torch.cat((front_image, rear_image), dim=-1).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(model_device)
                scan = lidar_scan(lidar).unsqueeze(0).to(model_device)
                proprio = high_level_proprio(robot, joint_ids, last_low_level_action).to(model_device)
                with torch.inference_mode():
                    outputs = model(rgb, scan, proprio, language)
                    output_index = 2
                    command_norm, skill_logits = outputs[:2]
                    if search_head:
                        search_probability = float(torch.sigmoid(outputs[output_index][0]).item())
                        output_index += 1
                    else:
                        search_probability = 0.0
                    if target_head:
                        target_stop_probability = float(torch.sigmoid(outputs[output_index][0, 0]).item())
                        target_distance_prediction = float(torch.sigmoid(outputs[output_index][0, 1]).item())
                    else:
                        target_stop_probability = 0.0
                        target_distance_prediction = 1.0
                command_np = (command_norm[0].detach().cpu().numpy() * COMMAND_SCALE.numpy()).astype(np.float32)
                skill_index = int(skill_logits[0].argmax().item())
                raw_skill = SKILL_NAMES[skill_index]
                predicted_skill = raw_skill
                target_stop_candidate = False
                target_turnaround_candidate = False
                target_recent_min_distance = None
                if target_head:
                    # The target head is trained from frame-level reached labels;
                    # do not allow the imbalanced generic stop class to stop early.
                    target_distance_gate = (
                        target_distance_prediction <= args.target_distance_threshold
                    )
                    if args.target_turnaround_window > 0:
                        target_distance_history.append(target_distance_prediction)
                        target_recent_min_distance = min(target_distance_history)
                        target_turnaround_candidate = (
                            len(target_distance_history) >= 4
                            and target_recent_min_distance <= args.target_distance_threshold
                            and target_distance_prediction
                            >= target_recent_min_distance
                            + args.target_turnaround_rise_m / target_distance_scale_m
                        )
                        target_distance_gate = target_turnaround_candidate
                    target_stop_candidate = (
                        target_stop_probability >= args.target_threshold
                        and target_distance_gate
                    )
                    predicted_skill = "stop" if target_stop_candidate else predicted_skill
                    if predicted_skill == "stop" and not target_stop_candidate:
                        predicted_skill = "forward"
                cached_skill = "search" if search_probability >= args.search_threshold else predicted_skill
                # Turn/translation are canonicalized before reaching the
                # native expert.  This keeps regression noise in the command
                # head from turning a discrete skill into a curved skid.
                if cached_skill == "left":
                    turn_streak += 1
                    if turn_streak > args.max_turn_steps:
                        recovery_remaining = args.turn_recovery_steps
                        cached_skill = "forward_recovery"
                        cached_command = (args.max_forward_command, 0.0, 0.0)
                        turn_streak = 0
                    else:
                        cached_command = (0.0, 0.0, min(max(abs(float(command_np[2])), 0.12), args.max_yaw_command))
                elif cached_skill == "right":
                    turn_streak += 1
                    if turn_streak > args.max_turn_steps:
                        recovery_remaining = args.turn_recovery_steps
                        cached_skill = "forward_recovery"
                        cached_command = (args.max_forward_command, 0.0, 0.0)
                        turn_streak = 0
                    else:
                        cached_command = (0.0, 0.0, -min(max(abs(float(command_np[2])), 0.12), args.max_yaw_command))
                elif cached_skill == "forward":
                    turn_streak = 0
                    cached_command = (
                        min(max(float(command_np[0]), args.min_forward_command), args.max_forward_command),
                        0.0,
                        0.0,
                    )
                elif cached_skill == "backward":
                    turn_streak = 0
                    cached_command = (max(min(float(command_np[0]), 0.0), -args.max_forward_command), 0.0, 0.0)
                elif cached_skill == "search":
                    turn_streak = 0
                    search_remaining = max(search_remaining, 20)
                    cached_command = (0.0, 0.0, search_sign * args.search_yaw_command)
                elif cached_skill in {"stop", "jump"}:
                    turn_streak = 0
                    # A stop prediction is only a candidate until its
                    # confidence streak is confirmed.  Keep moving with the
                    # last non-stop command during that confirmation window,
                    # otherwise the gate would stop before the target radius.
                    cached_command = (0.0, 0.0, 0.0) if stop_latched else last_nonstop_command
                else:
                    turn_streak = 0
                generic_stop_probability = float(
                    torch.softmax(skill_logits, dim=-1)[0, SKILL_NAMES.index("stop")].item()
                )
                stop_probability = (
                    target_stop_probability
                    if target_head and target_distance_gate
                    else generic_stop_probability
                )
                if stop_probability >= args.stop_threshold and cached_skill == "stop":
                    stop_confidence_streak += 1
                    stop_votes.append(True)
                else:
                    stop_confidence_streak = 0
                    stop_votes.append(False)
                vote_stop = len(stop_votes) == args.stop_vote_window and sum(stop_votes) >= args.stop_vote_fraction * args.stop_vote_window
                if stop_confidence_streak >= args.stop_confirm_steps or vote_stop:
                    stop_latched = True
                    if stop_triggered_step is None:
                        stop_triggered_step = step
                if cached_skill not in {"stop", "search", "jump"} and cached_skill != "forward_recovery":
                    last_nonstop_command = cached_command
                actual_target_distance = (
                    float(torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - target_xy_eval).item())
                    if target_xy_eval is not None
                    else None
                )
                predicted_target_distance_m = (
                    target_distance_prediction * target_distance_scale_m if target_head else None
                )
                target_distance_error_m = (
                    abs(predicted_target_distance_m - actual_target_distance)
                    if predicted_target_distance_m is not None and actual_target_distance is not None
                    else None
                )
                if target_head:
                    if (
                        min_predicted_target_distance is None
                        or target_distance_prediction < min_predicted_target_distance
                    ):
                        min_predicted_target_distance = target_distance_prediction
                        min_predicted_target_distance_step = step
                    if (
                        max_target_stop_probability is None
                        or target_stop_probability > max_target_stop_probability
                    ):
                        max_target_stop_probability = target_stop_probability
                        max_target_stop_probability_step = step
                if target_distance_error_m is not None:
                    target_distance_abs_error_sum += target_distance_error_m
                    target_distance_error_count += 1
                prediction_trace.append(
                    {
                        "step": step,
                        "actual_target_distance_m": actual_target_distance,
                        "predicted_target_distance_normalized": (
                            target_distance_prediction if target_head else None
                        ),
                        "predicted_target_distance_m": predicted_target_distance_m,
                        "target_distance_absolute_error_m": target_distance_error_m,
                        "target_stop_probability": target_stop_probability if target_head else None,
                        "generic_stop_probability": generic_stop_probability,
                        "stop_gate_probability": stop_probability,
                        "search_probability": search_probability,
                        "raw_skill": raw_skill,
                        "selected_skill": cached_skill,
                        "target_stop_candidate": target_stop_candidate,
                        "target_distance_gate": target_distance_gate if target_head else None,
                        "target_turnaround_candidate": (
                            target_turnaround_candidate if target_head else None
                        ),
                        "target_recent_min_distance_m": (
                            None
                            if target_recent_min_distance is None
                            else target_recent_min_distance * target_distance_scale_m
                        ),
                        "stop_latched": stop_latched,
                        "command": list(cached_command),
                    }
                )
                if args.debug_steps and step < args.debug_steps:
                    print(f"[M20PRO-SKILL-DEBUG] step={step} skill={cached_skill} search_p={search_probability:.3f} target_stop_p={target_stop_probability:.3f} target_dist_p={target_distance_prediction:.3f} stop_p={stop_probability:.3f} command={cached_command}", flush=True)
            command = (0.0, 0.0, 0.0) if stop_latched else cached_command
            command_sum += np.asarray(command)
            command_count += 1
            skill_counts[cached_skill] += 1
            low_level_command = command
            mirror = args.mirror_negative_yaw and low_level_command[2] < -1e-6
            policy_command = (low_level_command[0], low_level_command[1], abs(low_level_command[2])) if mirror else low_level_command
            low_obs = native_observation(robot, joint_ids, last_low_level_action, policy_command, mirror=mirror)
            action_np = session.run(["actions"], {"obs": low_obs.detach().cpu().numpy()})[0]
            action = torch.from_numpy(action_np).to(robot.device, dtype=torch.float32)
            if mirror:
                action = mirror_action(action)
            leg_target = default_pose[:, :12] + action[:, :12] * leg_scale
            robot.set_joint_position_target(leg_target, joint_ids=leg_ids)
            robot.set_joint_velocity_target(action[:, 12:] * WHEEL_SCALE, joint_ids=wheel_ids)
            last_low_level_action = action
            camera_target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.1]], device=robot.device)
            third.set_world_poses_from_view(camera_target + torch.tensor([[-1.4, 1.4, 0.85]], device=robot.device), camera_target)
            for _ in range(4):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())
            frame = third.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            height = float(robot.data.root_pos_w[0, 2].item())
            min_height, max_height = min(min_height, height), max(max_height, height)
            gravity_z = float(quat_apply_inverse(robot.data.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=robot.device))[0, 2].item())
            terminated_steps += int(height < 0.45 or gravity_z > -0.5)
            if args.target_color != "none":
                distance = float(
                    torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - target_xy_eval).item()
                )
                min_target_distance = min(min_target_distance, distance)
                if not target_reached and distance <= args.target_radius:
                    target_reached = True
                    target_reached_step = step
                if stop_latched and distance <= args.target_radius:
                    post_stop_target_hold_streak += 1
                    max_post_stop_target_hold_steps = max(
                        max_post_stop_target_hold_steps, post_stop_target_hold_streak
                    )
                elif stop_latched:
                    post_stop_target_hold_streak = 0
    finally:
        finalize_h264_video(video, video_path)
    displacement = (robot.data.root_pos_w[0, :2] - start_xy).detach().cpu().numpy()
    current_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    current_yaw = float(np.arctan2(2.0 * (current_quat[0] * current_quat[3] + current_quat[1] * current_quat[2]), 1.0 - 2.0 * (current_quat[2] ** 2 + current_quat[3] ** 2)))
    yaw_delta = float(np.arctan2(np.sin(current_yaw - start_yaw), np.cos(current_yaw - start_yaw)))
    final_distance = None
    if args.target_color != "none":
        final_distance = float(
            torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - target_xy_eval).item()
        )
    stable = terminated_steps == 0 and min_height >= 0.45
    success = stable and (
        args.target_color == "none"
        or (
            target_reached
            and stop_triggered_step is not None
            and max_post_stop_target_hold_steps >= args.target_hold_steps
        )
    )
    metrics = {
        "format": "m20_vla_two_layer_replay_v2", "checkpoint": str(args.checkpoint), "low_level_policy": str(args.policy),
        "task_text": args.task_text, "target_color": args.target_color,
        "target_xy": [args.target_x, args.target_y] if args.target_color != "none" else None,
        "target_radius": args.target_radius, "target_hold_steps_required": args.target_hold_steps,
        "target_coordinates_used_by_policy": False, "steps": args.steps, "command_hold_steps": args.command_hold_steps,
        "stop_threshold": args.stop_threshold, "target_threshold": args.target_threshold,
        "target_distance_threshold": args.target_distance_threshold,
        "target_turnaround_window": args.target_turnaround_window,
        "target_turnaround_rise_m": args.target_turnaround_rise_m,
        "stop_confirm_steps": args.stop_confirm_steps,
        "stop_vote_window": args.stop_vote_window, "stop_vote_fraction": args.stop_vote_fraction,
        "max_turn_steps": args.max_turn_steps, "turn_recovery_steps": args.turn_recovery_steps,
        "max_yaw_command": args.max_yaw_command, "max_forward_command": args.max_forward_command,
        "min_forward_command": args.min_forward_command,
        "search_yaw_command": args.search_yaw_command,
        "search_threshold": args.search_threshold, "search_head": search_head, "target_head": target_head,
        "target_head_mode": target_head_mode,
        "target_distance_scale_m": target_distance_scale_m,
        "min_predicted_target_distance_normalized": min_predicted_target_distance,
        "min_predicted_target_distance_m": (
            None
            if min_predicted_target_distance is None
            else min_predicted_target_distance * target_distance_scale_m
        ),
        "min_predicted_target_distance_step": min_predicted_target_distance_step,
        "max_target_stop_probability": max_target_stop_probability,
        "max_target_stop_probability_step": max_target_stop_probability_step,
        "mean_absolute_target_distance_error_m": (
            None
            if target_distance_error_count == 0
            else target_distance_abs_error_sum / target_distance_error_count
        ),
        "prediction_trace_sample_count": len(prediction_trace),
        "prediction_trace": prediction_trace,
        "displacement_xy": displacement.tolist(), "yaw_delta": yaw_delta,
        "min_root_height": min_height, "max_root_height": max_height, "terminated_steps": terminated_steps,
        "target_reached": target_reached, "target_reached_step": target_reached_step,
        "max_post_stop_target_hold_steps": max_post_stop_target_hold_steps,
        "stop_triggered_step": stop_triggered_step, "min_target_distance": None if args.target_color == "none" else min_target_distance,
        "final_target_distance": final_distance, "skill_counts": dict(skill_counts),
        "mean_command": (command_sum / max(command_count, 1)).tolist(), "success": success, "video": str(video_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(f"[M20PRO-SKILL-PLAY] displacement_xy={displacement.tolist()} yaw_delta={yaw_delta:.4f} rad", flush=True)
    print(f"[M20PRO-SKILL-PLAY] min_root_height={min_height:.4f} m terminated_steps={terminated_steps}", flush=True)
    print(f"[M20PRO-SKILL-PLAY] target_reached={target_reached} stop_step={stop_triggered_step} min_distance={min_target_distance:.4f} m success={success}", flush=True)
    if target_head:
        print(
            f"[M20PRO-SKILL-PLAY] min_predicted_distance={min_predicted_target_distance * target_distance_scale_m:.4f} m "
            f"at_step={min_predicted_target_distance_step} max_target_stop_p={max_target_stop_probability:.4f} "
            f"at_step={max_target_stop_probability_step}",
            flush=True,
        )
    print(f"[M20PRO-SKILL-PLAY] skill_counts={dict(skill_counts)}", flush=True)
    print(f"[M20PRO-SKILL-PLAY] video={video_path}", flush=True)
    print(f"[M20PRO-SKILL-PLAY] metrics={metrics_path}", flush=True)
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
