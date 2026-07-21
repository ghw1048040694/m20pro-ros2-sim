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


DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_POLICY = (
    DATA_ROOT
    / "public_experts/sources/M20-autonomy-sim/src/M20_sdk_deploy/policy/policy.onnx"
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
parser.add_argument("--task-text", default="向前走")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "datasets/public_m20_native_v1")
parser.add_argument("--video-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "videos/public_m20_native_v1")
parser.add_argument("--image-width", type=int, default=160)
parser.add_argument("--image-height", type=int, default=96)
parser.add_argument("--video", action="store_true", help="Required: write one MP4 per episode.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.video:
    parser.error("--video is required so every recorded episode has an inspectable MP4")
if args.episodes <= 0 or args.steps <= 0:
    parser.error("--episodes and --steps must be positive")
if not args.policy.is_file():
    parser.error(f"M20 policy not found: {args.policy}")
args.enable_cameras = True
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

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
DEFAULT_POLICY_POSE = torch.tensor(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=torch.float32,
)
LEG_ACTION_SCALE = torch.tensor([0.125, 0.25, 0.25] * 4, dtype=torch.float32)

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
            velocity_limit=79.3, stiffness=0.0, damping=0.6,
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
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera", update_period=0.02,
        height=args.image_height, width=args.image_width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                                         clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.38, 0.0, 0.12), convention="world"),
    )
    rear_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_camera", update_period=0.02,
        height=args.image_height, width=args.image_width, data_types=["rgb"],
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


def native_observation(robot: Articulation, joint_ids: list[int], last_action: torch.Tensor) -> torch.Tensor:
    gravity = torch.tensor([[0.0, 0.0, -1.0]], device=robot.device)
    projected_gravity = quat_apply_inverse(robot.data.root_quat_w, gravity)
    command = torch.tensor([[args.command_x, args.command_y, args.command_yaw]], device=robot.device)
    joint_pos = robot.data.joint_pos[:, joint_ids].clone()
    joint_pos[:, 12:] = 0.0
    joint_pos -= DEFAULT_POLICY_POSE.to(robot.device)
    joint_vel = robot.data.joint_vel[:, joint_ids] * 0.05
    return torch.cat([robot.data.root_ang_vel_b * 0.25, projected_gravity, command, joint_pos, joint_vel, last_action], dim=-1)


def rgb(camera) -> np.ndarray:
    return camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)


def scan(lidar, max_distance: float = 20.0) -> np.ndarray:
    hits = lidar.data.ray_hits_w[0]
    origin = lidar.data.pos_w[0].unsqueeze(0)
    values = torch.linalg.vector_norm(hits - origin, dim=-1)
    return torch.nan_to_num(values, nan=max_distance, posinf=max_distance, neginf=0.0).clamp(0.0, max_distance).cpu().numpy().astype(np.float32)


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 57] or session.get_outputs()[0].shape != [1, 16]:
        raise RuntimeError("Native M20 policy is not the expected 57->16 model")
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
        "control_hz": 50.0, "joint_names": POLICY_JOINT_NAMES,
        "observation": {"front_rgb": [args.image_height, args.image_width, 3], "rear_rgb": [args.image_height, args.image_width, 3],
                         "lidar": [ray_count], "proprio": [57], "state": [45]},
        "action": {"shape": [16], "leg": "default_pose + output[:12] * [0.125,0.25,0.25]", "wheel": "output[12:] * 5.0"},
        "success_rule": "terminated_steps == 0 and min_root_height >= 0.45 m",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    default_pose = torch.tensor([0.0, -0.6, 1.0, 0.0, -0.6, 1.0, 0.0, 0.6, -1.0, 0.0, 0.6, -1.0], device=robot.device).unsqueeze(0)
    leg_scale = LEG_ACTION_SCALE.to(robot.device).unsqueeze(0)
    zero_wheels = torch.zeros((1, 4), device=robot.device)

    for episode in range(args.episodes):
        reset_scene(scene)
        for _ in range(args.warmup_steps):
            robot.set_joint_position_target(default_pose, joint_ids=leg_ids)
            robot.set_joint_velocity_target(zero_wheels, joint_ids=wheel_ids)
            scene.write_data_to_sim()
            for _ in range(4):
                sim.step(render=False)
                scene.update(physics_dt)
        path = args.output_dir / f"episode_{episode:04d}.h5"
        video_path = args.video_dir / f"episode_{episode:04d}.mp4"
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (480, 288))
        if not video.isOpened():
            raise RuntimeError(f"Unable to open video writer: {video_path}")
        last_action = torch.zeros((1, 16), device=robot.device)
        min_height = max_height = float(robot.data.root_pos_w[0, 2].item())
        terminated_steps = 0
        start_x = float(robot.data.root_pos_w[0, 0].item())
        with h5py.File(path, "w") as h5:
            obs = h5.create_group("observation")
            front_ds = obs.create_dataset("front_rgb", (args.steps, args.image_height, args.image_width, 3), dtype="u1", compression="lzf")
            rear_ds = obs.create_dataset("rear_rgb", (args.steps, args.image_height, args.image_width, 3), dtype="u1", compression="lzf")
            lidar_ds = obs.create_dataset("lidar", (args.steps, ray_count), dtype="f4", compression="lzf")
            proprio_ds = obs.create_dataset("proprio", (args.steps, 57), dtype="f4", compression="lzf")
            state_ds = obs.create_dataset("state", (args.steps, 45), dtype="f4", compression="lzf")
            action_ds = h5.create_dataset("action", (args.steps, 16), dtype="f4", compression="lzf")
            done_ds = h5.create_dataset("terminated", (args.steps,), dtype="u1")
            h5.attrs["task_text"] = args.task_text
            h5.attrs["command"] = np.asarray([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
            h5.attrs["expert"] = metadata["expert"]
            for step in range(args.steps):
                observation = native_observation(robot, joint_ids, last_action)
                action_np = session.run(["actions"], {"obs": observation.cpu().numpy()})[0]
                action = torch.from_numpy(action_np).to(robot.device)
                robot.set_joint_position_target(default_pose + action[:, :12] * leg_scale, joint_ids=leg_ids)
                robot.set_joint_velocity_target(action[:, 12:] * 5.0, joint_ids=wheel_ids)
                last_action = action
                for _ in range(4):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(physics_dt)
                camera_target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.1]], device=robot.device)
                camera_eye = camera_target + torch.tensor([[-1.4, 1.4, 0.85]], device=robot.device)
                third.set_world_poses_from_view(camera_eye, camera_target)
                frame = rgb(third)
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                state = torch.cat((robot.data.root_pos_w[0], robot.data.root_quat_w[0], robot.data.root_lin_vel_w[0], robot.data.root_ang_vel_w[0], robot.data.joint_pos[0], robot.data.joint_vel[0])).cpu().numpy().astype(np.float32)
                front_ds[step], rear_ds[step], lidar_ds[step] = rgb(front), rgb(rear), scan(lidar)
                proprio_ds[step], state_ds[step], action_ds[step] = observation[0].cpu().numpy(), state, action[0].cpu().numpy()
                height = float(robot.data.root_pos_w[0, 2].item())
                min_height, max_height = min(min_height, height), max(max_height, height)
                gravity_z = float(quat_apply_inverse(robot.data.root_quat_w, torch.tensor([[0.0, 0.0, -1.0]], device=robot.device))[0, 2].item())
                terminated = int(height < 0.45 or gravity_z > -0.5)
                done_ds[step] = terminated
                terminated_steps += terminated
            h5.attrs["x_displacement"] = float(robot.data.root_pos_w[0, 0].item()) - start_x
            h5.attrs["min_root_height"] = min_height
            h5.attrs["max_root_height"] = max_height
            h5.attrs["terminated_steps"] = terminated_steps
            h5.attrs["success"] = bool(terminated_steps == 0 and min_height >= 0.45)
        video.release()
        print(
            f"[M20PRO-NATIVE-EXPERT] episode={episode} x_displacement={float(robot.data.root_pos_w[0, 0].item()) - start_x:.4f} m "
            f"min_root_height={min_height:.4f} m terminated_steps={terminated_steps} "
            f"success={terminated_steps == 0 and min_height >= 0.45} data={path} video={video_path}",
            flush=True,
        )
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
