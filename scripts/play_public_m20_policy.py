"""Replay the public native M20 ONNX locomotion policy in Isaac Lab.

The observation, action ordering, gains, and 50 Hz control rate mirror the
publisher's M20PolicyRunner. A third-person MP4 is mandatory so motion quality
is judged from the robot articulation as well as scalar metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from isaaclab.app import AppLauncher


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_POLICY = (
    DATA_ROOT
    / "public_experts/m20_native/policy.onnx"
)
DEFAULT_VIDEO_DIR = Path(os.environ.get("M20PRO_OUTPUT_ROOT", str(DATA_ROOT))) / "videos/public_m20_native"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
parser.add_argument("--steps", type=int, default=500, help="Number of 50 Hz policy steps to replay.")
parser.add_argument("--warmup-steps", type=int, default=75, help="PD-hold steps before policy takeover.")
parser.add_argument("--command-x", type=float, default=0.5)
parser.add_argument("--command-y", type=float, default=0.0)
parser.add_argument("--command-yaw", type=float, default=0.0)
parser.add_argument("--initial-height", type=float, default=0.54)
parser.add_argument("--wheel-damping", type=float, default=None, help="Isaac-only wheel Kd override; default adapts for yaw commands.")
parser.add_argument(
    "--segment",
    action="append",
    default=[],
    metavar="STEPS,X,Y,YAW,KD",
    help="Optional command segment; repeat and make step counts sum to --steps.",
)
parser.add_argument("--segment-blend-steps", type=int, default=0, help="Linearly blend command and Kd at segment boundaries.")
parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
parser.add_argument("--video", action="store_true", help="Required: record an inspectable third-person MP4.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.video:
    parser.error("--video is required so every replay leaves an inspectable MP4")
if args.steps <= 0 or args.warmup_steps < 0:
    parser.error("--steps must be positive and --warmup-steps must be non-negative")
if not args.policy.is_file():
    parser.error(f"M20 policy not found: {args.policy}")
if args.segment_blend_steps < 0:
    parser.error("--segment-blend-steps must be non-negative")


def parse_segments(values: list[str]) -> list[tuple[int, tuple[float, float, float], float]]:
    if not values:
        damping = args.wheel_damping if args.wheel_damping is not None else (3.6 if abs(args.command_yaw) >= 0.05 else 0.6)
        return [(args.steps, (args.command_x, args.command_y, args.command_yaw), damping)]
    segments = []
    total = 0
    for value in values:
        fields = value.split(",")
        if len(fields) != 5:
            parser.error(f"invalid --segment={value!r}; expected STEPS,X,Y,YAW,KD")
        try:
            count = int(fields[0])
            command = tuple(float(item) for item in fields[1:4])
            damping = float(fields[4])
        except ValueError as exc:
            parser.error(f"invalid --segment={value!r}: {exc}")
        if count <= 0 or damping < 0.0:
            parser.error(f"invalid --segment={value!r}; steps must be positive and Kd non-negative")
        total += count
        segments.append((count, command, damping))
    if total != args.steps:
        parser.error(f"segment step counts sum to {total}, but --steps={args.steps}")
    return segments


SEGMENTS = parse_segments(args.segment)
args.enable_cameras = True
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnxruntime as ort  # noqa: E402
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
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
    "fl_wheel_joint",
    "fr_wheel_joint",
    "hl_wheel_joint",
    "hr_wheel_joint",
]
DEFAULT_POLICY_POSE = torch.tensor(
    [
        0.0,
        -0.6,
        1.0,
        0.0,
        -0.6,
        1.0,
        0.0,
        0.6,
        -1.0,
        0.0,
        0.6,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=torch.float32,
)
LEG_ACTION_SCALE = torch.tensor([0.125, 0.25, 0.25] * 4, dtype=torch.float32)
WHEEL_ACTION_SCALE = 5.0
WHEEL_DAMPING = args.wheel_damping if args.wheel_damping is not None else (3.6 if abs(args.command_yaw) >= 0.05 else 0.6)

PUBLIC_M20_CFG = M20PRO_CFG.replace(
    init_state=M20PRO_CFG.init_state.replace(
        pos=(0.0, 0.0, args.initial_height),
        joint_pos={name: float(value) for name, value in zip(POLICY_JOINT_NAMES, DEFAULT_POLICY_POSE)},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hipx": DCMotorCfg(
            joint_names_expr=[".*_hipx_joint"],
            effort_limit=32.4,
            saturation_effort=32.4,
            velocity_limit=45.0,
            stiffness=80.0,
            damping=2.0,
        ),
        "hipy_knee": DCMotorCfg(
            joint_names_expr=[".*_(hipy|knee)_joint"],
            effort_limit=76.4,
            saturation_effort=76.4,
            velocity_limit=22.4,
            stiffness=80.0,
            damping=2.0,
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=21.6,
            saturation_effort=21.6,
            velocity_limit=79.3,
            stiffness=0.0,
            # Isaac's wheel dynamics need the higher bridge-equivalent gain
            # for differential yaw; straight rolling remains stable at 0.6.
            damping=WHEEL_DAMPING,
        ),
    },
)


@configclass
class PublicM20SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot = PUBLIC_M20_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    third_person = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPerson",
        update_period=0.02,
        height=288,
        width=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8)),
    )


def make_observation(
    robot: Articulation,
    policy_joint_ids: list[int],
    last_action: torch.Tensor,
    command_values: tuple[float, float, float],
) -> torch.Tensor:
    gravity_w = torch.tensor([[0.0, 0.0, -1.0]], device=robot.device)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity_w)
    command = torch.tensor(
        [list(command_values)],
        device=robot.device,
        dtype=torch.float32,
    )
    joint_pos = robot.data.joint_pos[:, policy_joint_ids].clone()
    joint_pos[:, 12:] = 0.0
    joint_pos -= DEFAULT_POLICY_POSE.to(robot.device)
    joint_vel = robot.data.joint_vel[:, policy_joint_ids] * 0.05
    observation = torch.cat(
        [robot.data.root_ang_vel_b * 0.25, projected_gravity, command, joint_pos, joint_vel, last_action],
        dim=-1,
    )
    if observation.shape != (1, 57):
        raise RuntimeError(f"M20 observation shape mismatch: {tuple(observation.shape)}")
    return observation


def main() -> None:
    device = args.device or "cuda:0"
    args.video_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.name != "obs" or input_meta.shape != [1, 57]:
        raise RuntimeError(f"Unexpected policy input: name={input_meta.name} shape={input_meta.shape}")
    if output_meta.name != "actions" or output_meta.shape != [1, 16]:
        raise RuntimeError(f"Unexpected policy output: name={output_meta.name} shape={output_meta.shape}")

    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=device))
    scene = InteractiveScene(PublicM20SceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    policy_joint_ids, policy_joint_names = robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
    if policy_joint_names != POLICY_JOINT_NAMES:
        raise RuntimeError(f"M20 joint order mismatch: {policy_joint_names}")
    leg_ids = policy_joint_ids[:12]
    wheel_ids = policy_joint_ids[12:]
    default_pose = DEFAULT_POLICY_POSE.to(robot.device).unsqueeze(0)
    leg_scale = LEG_ACTION_SCALE.to(robot.device).unsqueeze(0)
    camera = scene["third_person"]
    segment_ranges = []
    cursor = 0
    for count, command, damping in SEGMENTS:
        segment_ranges.append((cursor, cursor + count, command, damping))
        cursor += count

    def command_for_step(step: int) -> tuple[tuple[float, float, float], float]:
        segment_index = next(index for index, (start, end, _, _) in enumerate(segment_ranges) if start <= step < end)
        start, _, command, damping = segment_ranges[segment_index]
        if args.segment_blend_steps > 0 and segment_index > 0 and step < start + args.segment_blend_steps:
            previous_command, previous_damping = segment_ranges[segment_index - 1][2:]
            blend = min(max(float(step - start + 1) / args.segment_blend_steps, 0.0), 1.0)
            command = tuple((1.0 - blend) * old + blend * new for old, new in zip(previous_command, command))
            damping = (1.0 - blend) * previous_damping + blend * damping
        return command, damping

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(default_pose, torch.zeros_like(default_pose), joint_ids=policy_joint_ids)
    scene.reset()

    zero_wheel_velocity = torch.zeros((1, 4), device=robot.device)
    for _ in range(args.warmup_steps):
        robot.set_joint_position_target(default_pose[:, :12], joint_ids=leg_ids)
        robot.set_joint_velocity_target(zero_wheel_velocity, joint_ids=wheel_ids)
        scene.write_data_to_sim()
        for _ in range(4):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())

    last_action = torch.zeros((1, 16), device=robot.device)
    start_x = float(robot.data.root_pos_w[0, 0].item())
    min_height = max_height = float(robot.data.root_pos_w[0, 2].item())
    forward_velocity_sum = 0.0
    leg_action_sum = 0.0
    wheel_action_sum = 0.0
    terminated_steps = 0
    max_abs_angular_velocity = 0.0
    video_name = (
        "m20-native-sequence-step-0.mp4"
        if args.segment
        else f"m20-native-x{args.command_x:+.2f}-y{args.command_y:+.2f}-yaw{args.command_yaw:+.2f}-step-0.mp4"
    )
    video_path = args.video_dir / video_name
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video writer: {video_path}")

    try:
        for step in range(args.steps):
            command, damping = command_for_step(step)
            robot.actuators["wheels"].damping.fill_(damping)
            observation = make_observation(robot, policy_joint_ids, last_action, command)
            action_np = session.run(["actions"], {"obs": observation.detach().cpu().numpy()})[0]
            if action_np.shape != (1, 16) or not np.isfinite(action_np).all():
                raise RuntimeError("M20 policy returned an invalid action")
            action = torch.from_numpy(action_np).to(device=robot.device, dtype=torch.float32)
            leg_target = default_pose[:, :12] + action[:, :12] * leg_scale
            wheel_velocity_target = action[:, 12:] * WHEEL_ACTION_SCALE
            robot.set_joint_position_target(leg_target, joint_ids=leg_ids)
            robot.set_joint_velocity_target(wheel_velocity_target, joint_ids=wheel_ids)
            last_action = action

            camera_target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.1]], device=robot.device)
            # Keep the robot large enough in the MP4 to inspect leg contacts.
            camera_eye = camera_target + torch.tensor([[-1.4, 1.4, 0.85]], device=robot.device)
            camera.set_world_poses_from_view(camera_eye, camera_target)
            for _ in range(4):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())

            frame = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            height = float(robot.data.root_pos_w[0, 2].item())
            min_height = min(min_height, height)
            max_height = max(max_height, height)
            forward_velocity_sum += float(robot.data.root_lin_vel_b[0, 0].item())
            leg_action_sum += float(action[:, :12].abs().mean().item())
            wheel_action_sum += float(action[:, 12:].abs().mean().item())
            max_abs_angular_velocity = max(
                max_abs_angular_velocity,
                float(robot.data.root_ang_vel_b[0].abs().max().item()),
            )
            gravity_z = float(
                quat_apply_inverse(
                    robot.data.root_quat_w,
                    torch.tensor([[0.0, 0.0, -1.0]], device=robot.device),
                )[0, 2].item()
            )
            terminated_steps += int(height < 0.25 or gravity_z > -0.5)
    finally:
        video.release()

    displacement = float(robot.data.root_pos_w[0, 0].item()) - start_x
    print(f"[M20PRO-NATIVE-PLAY] policy={args.policy}", flush=True)
    print(
        f"[M20PRO-NATIVE-PLAY] command=[{args.command_x:.3f}, {args.command_y:.3f}, {args.command_yaw:.3f}] segments={SEGMENTS}",
        flush=True,
    )
    print(f"[M20PRO-NATIVE-PLAY] steps={args.steps} x_displacement={displacement:.4f} m", flush=True)
    print(
        f"[M20PRO-NATIVE-PLAY] mean_forward_speed={forward_velocity_sum / args.steps:.4f} m/s",
        flush=True,
    )
    print(
        f"[M20PRO-NATIVE-PLAY] min_root_height={min_height:.4f} m max_root_height={max_height:.4f} m",
        flush=True,
    )
    print(f"[M20PRO-NATIVE-PLAY] terminated_steps={terminated_steps}", flush=True)
    print(f"[M20PRO-NATIVE-PLAY] max_abs_angular_velocity={max_abs_angular_velocity:.4f} rad/s", flush=True)
    print(
        f"[M20PRO-NATIVE-PLAY] mean_abs_leg_action={leg_action_sum / args.steps:.4f} "
        f"mean_abs_wheel_action={wheel_action_sum / args.steps:.4f}",
        flush=True,
    )
    print(f"[M20PRO-NATIVE-PLAY] video={video_path}", flush=True)
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
