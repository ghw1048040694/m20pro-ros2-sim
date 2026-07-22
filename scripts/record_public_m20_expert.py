"""Record trajectories from the released native M20 ONNX locomotion expert.

This is the first M20-specific positive dataset for the VLA project. It keeps
the publisher's 57->16 policy protocol while additionally recording the front
and rear RGB cameras, a 360-degree LiDAR scan, full state, command and task
language. The resulting HDF5 is intentionally LeRobot-like but lightweight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

from isaaclab.app import AppLauncher
from video_utils import finalize_h264_video


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_POLICY = (
    DATA_ROOT
    / "public_experts/m20_native/policy.onnx"
)
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("M20PRO_OUTPUT_ROOT", str(DATA_ROOT)))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--episodes", type=int, default=4)
parser.add_argument("--steps", type=int, default=500, help="50 Hz control steps per episode")
parser.add_argument("--warmup-steps", type=int, default=75)
parser.add_argument("--command-x", type=float, default=0.5)
parser.add_argument("--command-y", type=float, default=0.0)
parser.add_argument("--command-yaw", type=float, default=0.0)
parser.add_argument("--target-color", choices=["none", "red", "blue", "green"], default="none")
parser.add_argument("--target-x", type=float, default=3.0)
parser.add_argument("--target-y", type=float, default=0.0)
parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
parser.add_argument(
    "--initial-yaw-jitter-deg",
    type=float,
    default=0.0,
    help="Uniform per-episode yaw jitter around --initial-yaw-deg.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--stop-after", type=int, default=None, help="Switch the expert command to zero after this control step.")
parser.add_argument("--stop-on-target", action="store_true", help="Stop with zero action when the simulated target enters the success radius.")
parser.add_argument("--navigate-to-target", action="store_true", help="Generate goal-directed steering demonstrations from target bearing.")
parser.add_argument("--override-navigation-wheels", action="store_true", help="Use a geometric differential-wheel override; disabled by default for native-policy fidelity.")
parser.add_argument("--nav-forward-speed", type=float, default=0.5)
parser.add_argument("--nav-heading-gain", type=float, default=1.0)
parser.add_argument("--nav-max-yaw", type=float, default=0.5)
parser.add_argument("--nav-turn-threshold", type=float, default=1.57)
parser.add_argument("--nav-command-hold-steps", type=int, default=1)
parser.add_argument("--nav-fixed-turn-steps", type=int, default=0, help="Use a fixed turn skill before forward; zero keeps bearing control.")
parser.add_argument("--success-radius", type=float, default=0.8)
parser.add_argument("--nav-slow-radius", type=float, default=1.8)
parser.add_argument("--nav-min-distance-scale", type=float, default=0.15, help="Minimum approach speed fraction before entering the success radius.")
parser.add_argument("--nav-wheel-acceleration", type=float, default=18.0, help="Maximum wheel-target slew rate in rad/s^2.")
parser.add_argument("--nav-wheel-yaw-gain", type=float, default=4.0, help="Empirical skid-steer gain applied to yaw differential.")
parser.add_argument("--stop-wheel-damping", type=float, default=3.6, help="Wheel velocity gain used after reaching a target.")
parser.add_argument("--stop-brake-gain", type=float, default=2.0, help="Proportional body-forward braking gain after reaching a target.")
parser.add_argument("--stop-yaw-brake-gain", type=float, default=1.5, help="Proportional yaw braking gain after reaching a target.")
parser.add_argument("--turn-wheel-damping", type=float, default=None, help="Wheel Kd used while a navigation yaw command is active.")
parser.add_argument("--wheel-radius", type=float, default=0.09)
parser.add_argument("--track-width", type=float, default=0.48)
parser.add_argument("--episode-offset", type=int, default=0, help="First output episode index for appending scenario runs.")
parser.add_argument("--wheel-damping", type=float, default=None, help="Isaac-only wheel Kd override; default adapts for yaw commands.")
parser.add_argument(
    "--mirror-negative-yaw",
    action="store_true",
    help="Mirror the public positive-yaw expert for negative yaw commands using M20 left-right symmetry.",
)
parser.add_argument("--task-text", default="向前走")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "datasets/public_m20_native_v1")
parser.add_argument("--video-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "videos/public_m20_native_v1")
parser.add_argument("--image-width", type=int, default=160)
parser.add_argument("--image-height", type=int, default=96)
parser.add_argument("--video", action="store_true", help="Required: write one MP4 per episode.")
parser.add_argument("--dagger-checkpoint", type=Path, default=None, help="Optional VLA checkpoint used for mixed DAgger rollouts.")
parser.add_argument("--dagger-alpha", type=float, default=0.75, help="Fraction of the public expert action used during a DAgger rollout.")
parser.add_argument("--dagger-model-device", default="cpu", help="Device for the optional DAgger VLA model.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.video:
    parser.error("--video is required so every recorded episode has an inspectable MP4")
if args.episodes <= 0 or args.steps <= 0:
    parser.error("--episodes and --steps must be positive")
if args.episode_offset < 0:
    parser.error("--episode-offset must be non-negative")
if args.initial_yaw_jitter_deg < 0.0:
    parser.error("--initial-yaw-jitter-deg must be non-negative")
if args.navigate_to_target and args.target_color == "none":
    parser.error("--navigate-to-target requires a colored target")
if args.nav_forward_speed <= 0.0 or args.nav_heading_gain <= 0.0 or args.nav_max_yaw <= 0.0:
    parser.error("navigation gains and speed must be positive")
if args.wheel_radius <= 0.0 or args.track_width <= 0.0:
    parser.error("wheel geometry must be positive")
if args.nav_command_hold_steps <= 0:
    parser.error("--nav-command-hold-steps must be positive")
if args.nav_turn_threshold <= 0.0:
    parser.error("--nav-turn-threshold must be positive")
if args.nav_fixed_turn_steps < 0:
    parser.error("--nav-fixed-turn-steps must be non-negative")
if args.success_radius <= 0.0 or args.nav_slow_radius <= args.success_radius:
    parser.error("--nav-slow-radius must be greater than the positive --success-radius")
if not 0.0 < args.nav_min_distance_scale <= 1.0:
    parser.error("--nav-min-distance-scale must be in (0, 1]")
if args.nav_wheel_acceleration <= 0.0 or args.nav_wheel_yaw_gain <= 0.0 or args.stop_wheel_damping <= 0.0:
    parser.error("--nav-wheel-acceleration, --nav-wheel-yaw-gain and --stop-wheel-damping must be positive")
if args.stop_brake_gain < 0.0 or args.stop_yaw_brake_gain < 0.0:
    parser.error("--stop-brake-gain and --stop-yaw-brake-gain must be non-negative")
if not args.policy.is_file():
    parser.error(f"M20 policy not found: {args.policy}")
if args.dagger_checkpoint is not None and not args.dagger_checkpoint.is_file():
    parser.error(f"DAgger checkpoint not found: {args.dagger_checkpoint}")
if not 0.0 <= args.dagger_alpha <= 1.0:
    parser.error("--dagger-alpha must be in [0, 1]")
args.enable_cameras = True
app = AppLauncher(args).app

TARGET_COLORS = {
    "red": (0.9, 0.05, 0.03),
    "blue": (0.03, 0.15, 0.95),
    "green": (0.04, 0.8, 0.08),
}
TARGET_RGB = TARGET_COLORS.get(args.target_color)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnxruntime as ort  # noqa: E402
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
MIRROR_PERM = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 13, 12, 15, 14], dtype=torch.long)
MIRROR_SIGN = torch.tensor([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
MIRROR_ANGULAR_VELOCITY = torch.tensor([-1.0, 1.0, -1.0])
MIRROR_POLAR_VECTOR = torch.tensor([1.0, -1.0, 1.0])
DEFAULT_POLICY_POSE = torch.tensor(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)
LEG_ACTION_SCALE = torch.tensor([0.125, 0.25, 0.25] * 4, dtype=torch.float32)
WHEEL_DAMPING = args.wheel_damping if args.wheel_damping is not None else (3.6 if abs(args.command_yaw) >= 0.05 else 0.6)
TURN_WHEEL_DAMPING = args.turn_wheel_damping if args.turn_wheel_damping is not None else WHEEL_DAMPING
SENSOR_LINK = "base_link/base_link" if "m20_mjcf" in str(os.environ.get("M20PRO_USD_PATH", "")) else "base_link"

PUBLIC_M20_CFG = M20PRO_CFG.replace(
    init_state=M20PRO_CFG.init_state.replace(
        pos=(0.0, 0.0, 0.54),
        joint_pos={name: float(value) for name, value in zip(POLICY_JOINT_NAMES, DEFAULT_POLICY_POSE)},
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
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"], effort_limit=21.6, saturation_effort=21.6,
            velocity_limit=79.3, stiffness=0.0, damping=WHEEL_DAMPING,
        ),
    },
)


@configclass
class NativeM20SceneCfg(InteractiveSceneCfg):
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
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TARGET_RGB),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(args.target_x, args.target_y, 0.42)),
        )
        if TARGET_RGB is not None
        else None
    )
    front_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}/front_camera", update_period=0.02,
        height=args.image_height, width=args.image_width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.38, 0.0, 0.12), convention="world"),
    )
    rear_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}/rear_camera", update_period=0.02,
        height=args.image_height, width=args.image_width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(-0.38, 0.0, 0.12), convention="world"),
    )
    lidar = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SENSOR_LINK}", update_period=0.02, ray_alignment="base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.16)),
        pattern_cfg=patterns.LidarPatternCfg(channels=1, vertical_fov_range=(0.0, 0.0),
                                              horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0),
        max_distance=20.0,
        # Keep LiDAR on the ground mesh; the semantic target is camera-visible
        # and intentionally non-colliding, so it is not a ray-cast surface.
        mesh_prim_paths=["/World/ground"],
    )
    third_person = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPerson", update_period=0.02, height=288, width=480, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)))


def native_observation(
    robot: Articulation,
    joint_ids: list[int],
    last_action: torch.Tensor,
    command: tuple[float, float, float],
    mirror: bool = False,
) -> torch.Tensor:
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=robot.device)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity)
    command = torch.tensor([list(command)], device=robot.device)
    joint_pos = robot.data.joint_pos[:, joint_ids].clone()
    joint_pos[:, 12:] = 0.0
    joint_pos -= DEFAULT_POLICY_POSE.to(robot.device)
    joint_vel = robot.data.joint_vel[:, joint_ids] * 0.05
    observation = torch.cat(
        [robot.data.root_ang_vel_b * 0.25, projected_gravity, command, joint_pos, joint_vel, last_action], dim=-1
    )
    if mirror:
        observation[:, 0:3] *= MIRROR_ANGULAR_VELOCITY.to(robot.device)
        observation[:, 3:6] *= MIRROR_POLAR_VECTOR.to(robot.device)
        observation[:, 9:25] = observation[:, 9:25][:, MIRROR_PERM.to(robot.device)] * MIRROR_SIGN.to(robot.device)
        observation[:, 25:41] = observation[:, 25:41][:, MIRROR_PERM.to(robot.device)] * MIRROR_SIGN.to(robot.device)
        observation[:, 41:57] = observation[:, 41:57][:, MIRROR_PERM.to(robot.device)] * MIRROR_SIGN.to(robot.device)
    return observation


def mirror_action(action: torch.Tensor) -> torch.Tensor:
    return action[:, MIRROR_PERM.to(action.device)] * MIRROR_SIGN.to(action.device)


def command_for_step(step: int) -> tuple[float, float, float]:
    if args.stop_after is not None and step >= args.stop_after:
        return (0.0, 0.0, 0.0)
    return (args.command_x, args.command_y, args.command_yaw)


def yaw_from_quaternion(quat: torch.Tensor) -> float:
    values = quat.detach().cpu().numpy()
    return float(
        np.arctan2(
            2.0 * (values[0] * values[3] + values[1] * values[2]),
            1.0 - 2.0 * (values[2] ** 2 + values[3] ** 2),
        )
    )


def navigation_command(robot: Articulation, target_reached: bool) -> tuple[float, float, float]:
    if target_reached:
        return (0.0, 0.0, 0.0)
    delta = torch.tensor([args.target_x, args.target_y], device=robot.device) - robot.data.root_pos_w[0, :2]
    distance = float(torch.linalg.vector_norm(delta).item())
    target_heading = float(torch.atan2(delta[1], delta[0]).item())
    current_heading = yaw_from_quaternion(robot.data.root_quat_w[0])
    heading_error = float(np.arctan2(np.sin(target_heading - current_heading), np.cos(target_heading - current_heading)))
    yaw_command = float(np.clip(args.nav_heading_gain * heading_error, -args.nav_max_yaw, args.nav_max_yaw))
    remaining = max(distance - args.success_radius, 0.0)
    slow_span = args.nav_slow_radius - args.success_radius
    distance_scale = float(np.clip(remaining / slow_span, args.nav_min_distance_scale, 1.0))
    heading_scale = max(float(np.cos(heading_error)), 0.0)
    if abs(heading_error) >= args.nav_turn_threshold:
        # The released M20 policy loses forward speed when a yaw command is
        # mixed with translation. Saturate the turn and wait for alignment,
        # then issue a pure forward command inside the heading deadband.
        return (0.0, 0.0, float(np.sign(heading_error) * args.nav_max_yaw))
    yaw_command = 0.0
    return (args.nav_forward_speed * distance_scale * heading_scale, 0.0, yaw_command)


def override_navigation_wheels(
    action: torch.Tensor,
    command: tuple[float, float, float],
    previous_action: torch.Tensor,
) -> torch.Tensor:
    forward, _, yaw_rate = command
    half_track = 0.5 * args.track_width
    yaw_differential = half_track * yaw_rate * args.nav_wheel_yaw_gain
    # M20's four wheel joints use the negative Y axis, so their signed
    # differential is opposite to the usual planar-drive convention.
    left_velocity = -(forward + yaw_differential) / args.wheel_radius
    right_velocity = -(forward - yaw_differential) / args.wheel_radius
    desired_wheel_action = torch.tensor(
        [[left_velocity, right_velocity, left_velocity, right_velocity]],
        dtype=action.dtype,
        device=action.device,
    ).div_(5.0).clamp_(-2.5, 2.5)
    max_action_delta = args.nav_wheel_acceleration / 50.0 / 5.0
    wheel_action = torch.clamp(
        desired_wheel_action,
        previous_action[:, 12:] - max_action_delta,
        previous_action[:, 12:] + max_action_delta,
    )
    action = action.clone()
    action[:, 12:] = wheel_action
    return action


def target_stop_command(robot: Articulation) -> tuple[float, float, float]:
    """Convert current body velocity into a bounded reverse wheel command."""
    body_velocity = robot.data.root_lin_vel_b[0]
    body_angular_velocity = robot.data.root_ang_vel_b[0]
    forward = float(torch.clamp(-args.stop_brake_gain * body_velocity[0], -0.35, 0.35).item())
    yaw_rate = float(torch.clamp(-args.stop_yaw_brake_gain * body_angular_velocity[2], -0.5, 0.5).item())
    return (forward, 0.0, yaw_rate)


def set_navigation_wheel_damping(
    robot: Articulation,
    command: tuple[float, float, float],
    target_reached: bool,
) -> None:
    if not args.navigate_to_target:
        return
    wheel_actuator = robot.actuators.get("wheels")
    if wheel_actuator is None:
        return
    if args.override_navigation_wheels:
        damping = args.stop_wheel_damping if target_reached else 0.6
    else:
        # Keep the released controller's wheel Kd. The earlier yaw-only 3.6
        # override created a large angular-velocity spike in Isaac PhysX.
        if target_reached:
            damping = args.stop_wheel_damping
        elif abs(command[2]) >= 0.05:
            damping = TURN_WHEEL_DAMPING
        else:
            damping = WHEEL_DAMPING
    wheel_actuator.damping.fill_(damping)


def rgb(camera) -> np.ndarray:
    return camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)


def scan(lidar, max_distance: float = 20.0) -> np.ndarray:
    hits = lidar.data.ray_hits_w[0]
    origin = lidar.data.pos_w[0].unsqueeze(0)
    values = torch.linalg.vector_norm(hits - origin, dim=-1)
    return torch.nan_to_num(values, nan=max_distance, posinf=max_distance, neginf=0.0).clamp(0.0, max_distance).cpu().numpy().astype(np.float32)


def encode_text(text: str, max_length: int = 32) -> torch.Tensor:
    values = np.frombuffer(text.encode("utf-8")[:max_length], dtype=np.uint8).astype(np.int64) + 1
    tokens = np.zeros(max_length, dtype=np.int64)
    tokens[: len(values)] = values
    return torch.from_numpy(tokens)


def reset_scene(scene: InteractiveScene, initial_yaw_rad: float) -> None:
    robot = scene["robot"]
    robot.reset()
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    half_yaw = 0.5 * initial_yaw_rad
    root_state[:, 3:7] = torch.tensor(
        [[np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]],
        dtype=root_state.dtype,
        device=root_state.device,
    )
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 57] or session.get_outputs()[0].shape != [1, 16]:
        raise RuntimeError("Native M20 policy is not the expected 57->16 model")
    dagger_model = None
    rng = np.random.default_rng(args.seed)
    if args.episode_offset:
        rng.uniform(
            -args.initial_yaw_jitter_deg,
            args.initial_yaw_jitter_deg,
            size=args.episode_offset,
        )
    if args.dagger_checkpoint is not None:
        payload = torch.load(args.dagger_checkpoint, map_location="cpu", weights_only=True)
        config = payload.get("config", {})
        dagger_model = M20VLAActionChunk(
            int(payload["horizon"]), architecture=config.get("architecture", "global_v1")
        )
        dagger_model.load_state_dict(payload["model_state_dict"])
        dagger_device = torch.device(
            args.dagger_model_device
            if torch.cuda.is_available() or not args.dagger_model_device.startswith("cuda")
            else "cpu"
        )
        dagger_model.to(dagger_device).eval()
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=args.device or "cuda:0"))
    scene = InteractiveScene(NativeM20SceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    joint_ids, joint_names = robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
    if joint_names != POLICY_JOINT_NAMES:
        raise RuntimeError(f"M20 policy order mismatch: {joint_names}")
    leg_ids, wheel_ids = joint_ids[:12], joint_ids[12:]
    front, rear, lidar, third = scene["front_camera"], scene["rear_camera"], scene["lidar"], scene["third_person"]
    ray_count = int(lidar.data.ray_hits_w.shape[1])
    physics_dt = sim.get_physics_dt()
    metadata = {
        "format": "m20pro_native_expert_hdf5_v1", "expert": "AI-DA-STC/M20-autonomy-sim policy.onnx",
        "policy_protocol": "57 observation -> 16 action; official M20PolicyRunner; no PPO reward",
        "task_text": args.task_text, "command": [args.command_x, args.command_y, args.command_yaw],
        "stop_after": args.stop_after, "stop_on_target": args.stop_on_target,
        "navigate_to_target": args.navigate_to_target, "override_navigation_wheels": args.override_navigation_wheels,
        "target_color": args.target_color,
        "target_xy": [args.target_x, args.target_y], "wheel_damping": WHEEL_DAMPING,
        "initial_yaw": {
            "center_deg": args.initial_yaw_deg,
            "jitter_deg": args.initial_yaw_jitter_deg,
            "seed": args.seed,
        },
        "asset_usd_path": os.environ.get("M20PRO_USD_PATH", str(M20PRO_CFG.spawn.usd_path)),
        "dagger": args.dagger_checkpoint is not None,
        "dagger_checkpoint": None if args.dagger_checkpoint is None else str(args.dagger_checkpoint),
        "dagger_alpha": args.dagger_alpha,
        "navigation": {
            "forward_speed": args.nav_forward_speed, "heading_gain": args.nav_heading_gain,
            "max_yaw": args.nav_max_yaw, "turn_threshold": args.nav_turn_threshold,
            "command_hold_steps": args.nav_command_hold_steps, "fixed_turn_steps": args.nav_fixed_turn_steps,
            "success_radius": args.success_radius, "slow_radius": args.nav_slow_radius,
            "min_distance_scale": args.nav_min_distance_scale,
            "wheel_acceleration": args.nav_wheel_acceleration, "wheel_yaw_gain": args.nav_wheel_yaw_gain,
            "stop_wheel_damping": args.stop_wheel_damping, "turn_wheel_damping": TURN_WHEEL_DAMPING,
            "wheel_radius": args.wheel_radius,
            "track_width": args.track_width,
            "stop_brake_gain": args.stop_brake_gain,
            "stop_yaw_brake_gain": args.stop_yaw_brake_gain,
            "source": "public M20 ONNX leg policy plus target-bearing differential wheel expert",
        },
        "control_hz": 50.0, "joint_names": POLICY_JOINT_NAMES,
        "observation": {"front_rgb": [args.image_height, args.image_width, 3], "rear_rgb": [args.image_height, args.image_width, 3],
                         "lidar": [ray_count], "proprio": [57], "state": [45], "expert_command": [3]},
        "action": {"shape": [16], "leg": "default_pose + output[:12] * [0.125,0.25,0.25]", "wheel": "output[12:] * 5.0"},
        "success_rule": "stable plus command-direction check; target episodes also require reaching target_xy",
    }
    metadata_name = "metadata.json" if args.episode_offset == 0 else f"metadata_run_{args.episode_offset:04d}.json"
    (args.output_dir / metadata_name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    default_pose = torch.tensor([0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0], device=robot.device).unsqueeze(0)
    leg_scale = LEG_ACTION_SCALE.to(robot.device).unsqueeze(0)
    zero_wheels = torch.zeros((1, 4), device=robot.device)

    for episode in range(args.episodes):
        episode_id = args.episode_offset + episode
        episode_initial_yaw_deg = args.initial_yaw_deg + float(
            rng.uniform(-args.initial_yaw_jitter_deg, args.initial_yaw_jitter_deg)
        )
        episode_initial_yaw_rad = float(np.deg2rad(episode_initial_yaw_deg))
        reset_scene(scene, episode_initial_yaw_rad)
        for _ in range(args.warmup_steps):
            robot.set_joint_position_target(default_pose, joint_ids=leg_ids)
            robot.set_joint_velocity_target(zero_wheels, joint_ids=wheel_ids)
            scene.write_data_to_sim()
            for _ in range(4):
                sim.step(render=_ == 3)
                scene.update(physics_dt)
        path = args.output_dir / f"episode_{episode_id:04d}.h5"
        video_path = args.video_dir / f"episode_{episode_id:04d}.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
        if not video.isOpened():
            raise RuntimeError(f"Unable to open video writer: {video_path}")
        last_action = torch.zeros((1, 16), device=robot.device)
        dagger_last_action = torch.zeros((1, 16), device=robot.device)
        min_height = max_height = float(robot.data.root_pos_w[0, 2].item())
        terminated_steps = 0
        start_x = float(robot.data.root_pos_w[0, 0].item())
        start_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
        start_yaw = float(np.arctan2(2.0 * (start_quat[0] * start_quat[3] + start_quat[1] * start_quat[2]), 1.0 - 2.0 * (start_quat[2] ** 2 + start_quat[3] ** 2)))
        yaw_delta = 0.0
        target_reached = TARGET_RGB is None
        target_reached_step = None
        path_length = 0.0
        previous_xy = robot.data.root_pos_w[0, :2].clone()
        min_target_distance = float("inf")
        cached_navigation_command: tuple[float, float, float] | None = None
        navigation_command_until = -1
        initial_target_heading = float(np.arctan2(args.target_y, args.target_x))
        initial_heading_error = float(
            np.arctan2(
                np.sin(initial_target_heading - episode_initial_yaw_rad),
                np.cos(initial_target_heading - episode_initial_yaw_rad),
            )
        )
        target_turn_sign = 1.0 if initial_heading_error >= 0.0 else -1.0
        with h5py.File(path, "w") as h5:
            obs = h5.create_group("observation")
            front_ds = obs.create_dataset("front_rgb", (args.steps, args.image_height, args.image_width, 3), dtype="u1", compression="lzf")
            rear_ds = obs.create_dataset("rear_rgb", (args.steps, args.image_height, args.image_width, 3), dtype="u1", compression="lzf")
            lidar_ds = obs.create_dataset("lidar", (args.steps, ray_count), dtype="f4", compression="lzf")
            proprio_ds = obs.create_dataset("proprio", (args.steps, 57), dtype="f4", compression="lzf")
            state_ds = obs.create_dataset("state", (args.steps, 45), dtype="f4", compression="lzf")
            action_ds = h5.create_dataset("action", (args.steps, 16), dtype="f4", compression="lzf")
            command_ds = h5.create_dataset("expert_command", (args.steps, 3), dtype="f4", compression="lzf")
            done_ds = h5.create_dataset("terminated", (args.steps,), dtype="u1")
            h5.attrs["task_text"] = args.task_text
            h5.attrs["command"] = np.asarray([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
            h5.attrs["stop_after"] = -1 if args.stop_after is None else args.stop_after
            h5.attrs["stop_on_target"] = args.stop_on_target
            h5.attrs["navigate_to_target"] = args.navigate_to_target
            h5.attrs["target_color"] = args.target_color
            h5.attrs["target_xy"] = np.asarray([args.target_x, args.target_y], dtype=np.float32)
            h5.attrs["initial_yaw_deg"] = episode_initial_yaw_deg
            h5.attrs["random_seed"] = args.seed
            h5.attrs["wheel_damping"] = WHEEL_DAMPING
            h5.attrs["expert"] = metadata["expert"]
            h5.attrs["asset_usd_path"] = metadata["asset_usd_path"]
            h5.attrs["dagger"] = metadata["dagger"]
            h5.attrs["dagger_alpha"] = args.dagger_alpha
            for step in range(args.steps):
                target_in_range = False
                if TARGET_RGB is not None:
                    target_delta = robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
                    target_in_range = bool(torch.linalg.vector_norm(target_delta).item() <= args.success_radius)
                    if target_in_range and not target_reached:
                        target_reached = True
                        target_reached_step = step
                if args.navigate_to_target:
                    if args.nav_fixed_turn_steps > 0 and step < args.nav_fixed_turn_steps and not target_reached:
                        cached_navigation_command = (0.0, 0.0, target_turn_sign * args.nav_max_yaw)
                        navigation_command_until = step + 1
                    elif args.nav_fixed_turn_steps > 0 and step >= args.nav_fixed_turn_steps and not target_reached:
                        cached_navigation_command = (args.nav_forward_speed, 0.0, 0.0)
                        navigation_command_until = args.steps + 1
                    elif cached_navigation_command is None or step >= navigation_command_until or target_reached:
                        cached_navigation_command = navigation_command(robot, target_reached)
                        navigation_command_until = step + args.nav_command_hold_steps
                    command = cached_navigation_command
                else:
                    command = (0.0, 0.0, 0.0) if args.stop_on_target and target_reached else command_for_step(step)
                set_navigation_wheel_damping(robot, command, target_reached)
                policy_command = (0.0, 0.0, 0.0) if target_reached else command
                mirror = args.mirror_negative_yaw and policy_command[2] < -1e-6
                if mirror:
                    policy_command = (policy_command[0], policy_command[1], abs(policy_command[2]))
                observation = native_observation(robot, joint_ids, last_action, policy_command, mirror=mirror)
                # Keep every modality on the same pre-action timestamp as the
                # proprioception and expert action. The post-step frame is
                # reserved for the inspectable video only.
                front_observation = rgb(front)
                rear_observation = rgb(rear)
                lidar_observation = scan(lidar)
                # VLA always receives the robot's physical (unmirrored)
                # proprio frame. Mirroring is an implementation detail of
                # the public ONNX expert for negative-yaw commands only.
                vla_observation = native_observation(robot, joint_ids, last_action, command, mirror=False)
                vla_action = None
                if dagger_model is not None:
                    dagger_device = next(dagger_model.parameters()).device
                    vla_observation = native_observation(
                        robot, joint_ids, dagger_last_action, command, mirror=False
                    )
                    vla_observation = vla_observation.clone()
                    vla_observation[:, 6:9] = 0.0
                    rgb_input = np.concatenate(
                        (
                            cv2.resize(front_observation, (80, 48), interpolation=cv2.INTER_AREA),
                            cv2.resize(rear_observation, (80, 48), interpolation=cv2.INTER_AREA),
                        ),
                        axis=-1,
                    ).transpose(2, 0, 1)
                    rgb_input = torch.from_numpy(rgb_input.copy()).float().div_(255.0).unsqueeze(0).to(dagger_device)
                    lidar_input = torch.from_numpy(lidar_observation.copy()).float().div_(20.0).unsqueeze(0).to(dagger_device)
                    language_input = encode_text(args.task_text).to(dagger_device).unsqueeze(0)
                    with torch.inference_mode():
                        vla_action = dagger_model(
                            rgb_input,
                            lidar_input,
                            vla_observation.to(dagger_device),
                            language_input,
                        )[:, 0].to(robot.device)
                # Keep the native leg stabilizer active at the target. The
                # wheel override applies a bounded reverse command based on
                # measured body velocity, which dissipates inertia without
                # replacing the learned posture controller.
                stop_now = (
                    (args.stop_after is not None and step >= args.stop_after)
                    or (args.stop_on_target and target_reached)
                )
                if stop_now:
                    action = torch.zeros((1, 16), device=robot.device)
                else:
                    action_np = session.run(["actions"], {"obs": observation.cpu().numpy()})[0]
                    action = torch.from_numpy(action_np).to(robot.device)
                    if mirror:
                        action = mirror_action(action)
                    if args.navigate_to_target and args.override_navigation_wheels:
                        if target_reached and target_reached_step is not None:
                            action = override_navigation_wheels(action, target_stop_command(robot), last_action)
                        else:
                            action = override_navigation_wheels(action, command, last_action)
                expert_action = action
                execution_action = expert_action
                if vla_action is not None:
                    execution_action = (
                        args.dagger_alpha * expert_action + (1.0 - args.dagger_alpha) * vla_action
                    ).clamp(-4.0, 4.0)
                if stop_now:
                    # Do not let the learner reintroduce motion after the
                    # oracle has declared the visual target reached.
                    execution_action = torch.zeros_like(expert_action)
                robot.set_joint_position_target(default_pose + execution_action[:, :12] * leg_scale, joint_ids=leg_ids)
                robot.set_joint_velocity_target(execution_action[:, 12:] * 5.0, joint_ids=wheel_ids)
                last_action = expert_action
                dagger_last_action = execution_action
                camera_target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.1]], device=robot.device)
                camera_eye = camera_target + torch.tensor([[-1.4, 1.4, 0.85]], device=robot.device)
                third.set_world_poses_from_view(camera_eye, camera_target)
                for _ in range(4):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(physics_dt)
                frame = rgb(third)
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                state = torch.cat((robot.data.root_pos_w[0], robot.data.root_quat_w[0], robot.data.root_lin_vel_w[0], robot.data.root_ang_vel_w[0], robot.data.joint_pos[0], robot.data.joint_vel[0])).cpu().numpy().astype(np.float32)
                front_ds[step], rear_ds[step], lidar_ds[step] = (
                    front_observation,
                    rear_observation,
                    lidar_observation,
                )
                proprio_ds[step], state_ds[step], action_ds[step] = (
                    vla_observation[0].cpu().numpy(),
                    state,
                    expert_action[0].cpu().numpy(),
                )
                command_ds[step] = np.asarray(command, dtype=np.float32)
                current_xy = robot.data.root_pos_w[0, :2]
                path_length += float(torch.linalg.vector_norm(current_xy - previous_xy).item())
                previous_xy = current_xy.clone()
                height = float(robot.data.root_pos_w[0, 2].item())
                min_height, max_height = min(min_height, height), max(max_height, height)
                if TARGET_RGB is not None:
                    target_delta = robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
                    target_distance = float(torch.linalg.vector_norm(target_delta).item())
                    min_target_distance = min(min_target_distance, target_distance)
                    if target_distance <= args.success_radius and not target_reached:
                        target_reached = True
                        target_reached_step = step
                gravity_z = float(quat_apply_inverse(robot.data.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=robot.device))[0, 2].item())
                terminated = int(height < 0.45 or gravity_z > -0.5)
                done_ds[step] = terminated
                terminated_steps += terminated
                quat = robot.data.root_quat_w[0].detach().cpu().numpy()
                current_yaw = float(np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]), 1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)))
                yaw_delta = float(np.arctan2(np.sin(current_yaw - start_yaw), np.cos(current_yaw - start_yaw)))
            displacement = float(robot.data.root_pos_w[0, 0].item()) - start_x
            stable = terminated_steps == 0 and min_height >= 0.45
            final_target_distance = (
                float(torch.linalg.vector_norm(
                    robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
                ).item())
                if TARGET_RGB is not None else None
            )
            final_planar_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[0, :2]).item())
            if args.navigate_to_target:
                command_ok = (
                    target_reached and final_target_distance is not None
                    and final_target_distance <= args.success_radius + 0.1 and final_planar_speed < 0.15
                )
            elif abs(args.command_x) >= 0.05:
                command_ok = args.command_x * displacement > 0.5
            elif abs(args.command_yaw) >= 0.05:
                command_ok = args.command_yaw * yaw_delta > 0.25 and abs(displacement) < 2.0
            else:
                command_ok = True
            success = bool(stable and command_ok and target_reached)
            h5.attrs["x_displacement"] = displacement
            h5.attrs["yaw_delta"] = yaw_delta
            h5.attrs["min_root_height"] = min_height
            h5.attrs["max_root_height"] = max_height
            h5.attrs["terminated_steps"] = terminated_steps
            h5.attrs["command_ok"] = command_ok
            h5.attrs["target_reached"] = target_reached
            h5.attrs["target_reached_step"] = -1 if target_reached_step is None else target_reached_step
            h5.attrs["min_target_distance"] = min_target_distance if TARGET_RGB is not None else -1.0
            h5.attrs["final_target_distance"] = -1.0 if final_target_distance is None else final_target_distance
            h5.attrs["final_planar_speed"] = final_planar_speed
            h5.attrs["path_length"] = path_length
            h5.attrs["success"] = success
        finalize_h264_video(video, video_path)
        displacement = float(robot.data.root_pos_w[0, 0].item()) - start_x
        stable = terminated_steps == 0 and min_height >= 0.45
        final_target_distance = (
            float(torch.linalg.vector_norm(
                robot.data.root_pos_w[0, :2] - torch.tensor([args.target_x, args.target_y], device=robot.device)
            ).item())
            if TARGET_RGB is not None else None
        )
        final_planar_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[0, :2]).item())
        if args.navigate_to_target:
            command_ok = (
                target_reached and final_target_distance is not None
                and final_target_distance <= args.success_radius + 0.1 and final_planar_speed < 0.15
            )
        elif abs(args.command_x) >= 0.05:
            command_ok = args.command_x * displacement > 0.5
        elif abs(args.command_yaw) >= 0.05:
            command_ok = args.command_yaw * yaw_delta > 0.25 and abs(displacement) < 2.0
        else:
            command_ok = True
        success = bool(stable and command_ok and target_reached)
        print(
            f"[M20PRO-NATIVE-EXPERT] episode={episode_id} initial_yaw_deg={episode_initial_yaw_deg:.3f} "
            f"x_displacement={displacement:.4f} m yaw_delta={yaw_delta:.4f} rad "
            f"min_root_height={min_height:.4f} m terminated_steps={terminated_steps} command_ok={command_ok} "
            f"target_reached={target_reached} final_target_distance={final_target_distance} "
            f"reached_step={target_reached_step} path_length={path_length:.4f} m success={success} "
            f"data={path} video={video_path}",
            flush=True,
        )
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
