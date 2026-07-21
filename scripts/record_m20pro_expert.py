"""Record a physics-verified M20 Pro jump expert for imitation learning.

The expert is the open-loop PD sequence verified by ``probe_m20pro_jump.py``.
This is deliberately a data-collection baseline, not a claim that PPO learned
the skill.  Each episode stores synchronized camera, LiDAR, state and action
arrays in an HDF5 file, plus a video montage for visual inspection.
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

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "M20PRO_OUTPUT_ROOT",
        "/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA",
    )
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--steps", type=int, default=100, help="Control steps per episode (matches the verified probe timing).")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "datasets/m20pro_jump_expert_v0")
parser.add_argument("--video-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "videos/m20pro_jump_expert_v0")
parser.add_argument("--task-text", default="跳过前方一米高的障碍物")
parser.add_argument(
    "--pattern",
    choices=["squat_minus_thrust_plus", "squat_plus_thrust_minus"],
    default="squat_minus_thrust_plus",
)
parser.add_argument("--image-width", type=int, default=160)
parser.add_argument("--image-height", type=int, default=96)
parser.add_argument("--decimation", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from assets.m20pro import M20PRO_JUMP_CFG  # noqa: E402


@configclass
class M20ProExpertSceneCfg(InteractiveSceneCfg):
    """One M20 scene with the sensors used by the future VLA policy."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
    )
    robot = M20PRO_JUMP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
        update_period=0.02,
        height=96,
        width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.38, 0.0, 0.12), convention="world"),
    )
    rear_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_camera",
        update_period=0.02,
        height=96,
        width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.38, 0.0, 0.12), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    )
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.16)),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=5.0,
        ),
        max_distance=20.0,
        mesh_prim_paths=["/World/ground"],
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def reset_scene(scene: InteractiveScene) -> None:
    """Restore the robot to its configured pose and clear sensor histories."""
    robot = scene["robot"]
    robot.reset()
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()


def camera_frame(camera, height: int, width: int) -> np.ndarray:
    image = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def lidar_scan(lidar, max_distance: float) -> np.ndarray:
    hits = lidar.data.ray_hits_w[0]
    origin = lidar.data.pos_w[0].unsqueeze(0)
    distances = torch.linalg.vector_norm(hits - origin, dim=-1)
    distances = torch.nan_to_num(distances, nan=max_distance, posinf=max_distance, neginf=0.0)
    return torch.clamp(distances, 0.0, max_distance).detach().cpu().numpy().astype(np.float32)


def expert_action(step: int, steps: int, device: str) -> tuple[torch.Tensor, int]:
    """Return the verified squat -> thrust -> settle action and phase id."""
    action = torch.zeros((1, 12), device=device)
    third = max(1, steps // 3)
    squat, thrust = (-1.0, 1.0) if args.pattern == "squat_minus_thrust_plus" else (1.0, -1.0)
    if step < third:
        action[:, 4:12] = squat
        phase = 0
    elif step < 2 * third:
        action[:, 4:12] = thrust
        phase = 1
    else:
        phase = 2
    return action, phase


def write_metadata(output_dir: Path, scene_cfg: M20ProExpertSceneCfg, joint_names: list[str], ray_count: int) -> None:
    metadata = {
        "format": "m20pro_expert_hdf5_v0",
        "expert": f"open_loop_pd_{args.pattern}",
        "expert_source": "probe_m20pro_jump.py verified USD dynamics; not PPO",
        "task": args.task_text,
        "control_hz": 1.0 / (0.005 * args.decimation),
        "action": {"shape": [12], "meaning": "normalized leg joint position target in [-1, 1]"},
        "observation": {
            "front_rgb": [args.image_height, args.image_width, 3],
            "rear_rgb": [args.image_height, args.image_width, 3],
            "lidar": [ray_count],
            "state": [45],
            "state_order": [
                "root_pos_w(3)",
                "root_quat_w(4)",
                "root_lin_vel_w(3)",
                "root_ang_vel_w(3)",
                "joint_pos(16)",
                "joint_vel(16)",
            ],
        },
        "joint_names": joint_names,
        "sensor_config": {"front_camera": "160x96 RGB", "rear_camera": "160x96 RGB", "lidar": "72 rays, 360 deg"},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    sim_cfg = SimulationCfg(dt=0.005, render_interval=args.decimation, device=args.device or "cuda:0")
    sim = SimulationContext(sim_cfg)
    scene_cfg = M20ProExpertSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    leg_ids, leg_names = robot.find_joints(".*_(hipx|hipy|knee)_joint")
    wheel_ids, _ = robot.find_joints(".*_wheel_joint")
    joint_names = list(robot.joint_names)
    front_camera = scene["front_camera"]
    rear_camera = scene["rear_camera"]
    lidar = scene["lidar"]
    ray_count = int(lidar.data.ray_hits_w.shape[1])
    write_metadata(args.output_dir, scene_cfg, joint_names, ray_count)
    default_wheel_targets = robot.data.default_joint_pos[:, wheel_ids].clone()
    physics_dt = sim.get_physics_dt()

    for episode in range(args.episodes):
        reset_scene(scene)
        h5_path = args.output_dir / f"episode_{episode:04d}.h5"
        video_path = args.video_dir / f"episode_{episode:04d}.mp4"
        video = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / (physics_dt * args.decimation),
            (args.image_width * 2, args.image_height),
        )
        if not video.isOpened():
            raise RuntimeError(f"Unable to open video writer: {video_path}")
        max_height = float(robot.data.root_pos_w[0, 2].item())
        min_height = max_height
        with h5py.File(h5_path, "w") as h5:
            obs = h5.create_group("observation")
            front_ds = obs.create_dataset("front_rgb", (args.steps, args.image_height, args.image_width, 3), dtype="u1", compression="lzf")
            rear_ds = obs.create_dataset("rear_rgb", (args.steps, args.image_height, args.image_width, 3), dtype="u1", compression="lzf")
            lidar_ds = obs.create_dataset("lidar", (args.steps, ray_count), dtype="f4", compression="lzf")
            state_ds = obs.create_dataset("state", (args.steps, 45), dtype="f4", compression="lzf")
            action_ds = h5.create_dataset("action", (args.steps, 12), dtype="f4", compression="lzf")
            phase_ds = h5.create_dataset("phase", (args.steps,), dtype="i1")
            time_ds = h5.create_dataset("timestamp", (args.steps,), dtype="f8")
            h5.attrs["task"] = args.task_text
            h5.attrs["expert"] = f"open_loop_pd_{args.pattern}"
            for step in range(args.steps):
                action, phase = expert_action(step, args.steps, robot.device)
                leg_targets = torch.clamp(action * 2.2, -2.2, 2.2)
                robot.set_joint_position_target(leg_targets, joint_ids=leg_ids)
                robot.set_joint_position_target(default_wheel_targets, joint_ids=wheel_ids)
                for _ in range(args.decimation):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(physics_dt)

                front = camera_frame(front_camera, args.image_height, args.image_width)
                rear = camera_frame(rear_camera, args.image_height, args.image_width)
                scan = lidar_scan(lidar, 20.0)
                state = torch.cat(
                    (
                        robot.data.root_pos_w[0],
                        robot.data.root_quat_w[0],
                        robot.data.root_lin_vel_w[0],
                        robot.data.root_ang_vel_w[0],
                        robot.data.joint_pos[0],
                        robot.data.joint_vel[0],
                    )
                ).detach().cpu().numpy().astype(np.float32)
                front_ds[step] = front
                rear_ds[step] = rear
                lidar_ds[step] = scan
                state_ds[step] = state
                action_ds[step] = action[0].detach().cpu().numpy()
                phase_ds[step] = phase
                time_ds[step] = (step + 1) * physics_dt * args.decimation
                montage = np.concatenate((front, rear), axis=1)
                video.write(cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
                root_height = float(robot.data.root_pos_w[0, 2].item())
                max_height = max(max_height, root_height)
                min_height = min(min_height, root_height)
            h5.attrs["max_root_height"] = max_height
            h5.attrs["min_root_height"] = min_height
            h5.attrs["success"] = bool(max_height >= 0.80 and min_height >= 0.35)
        video.release()
        success = max_height >= 0.80 and min_height >= 0.35
        print(
            f"[M20PRO-EXPERT] episode={episode} max_root_height={max_height:.4f} m "
            f"min_root_height={min_height:.4f} m success={success} "
            f"data={h5_path} video={video_path}",
            flush=True,
        )

    scene.reset()
    print(f"[M20PRO-EXPERT] completed episodes={args.episodes} output={args.output_dir}", flush=True)
    sim.clear_instance()


try:
    main()
finally:
    app.close()
