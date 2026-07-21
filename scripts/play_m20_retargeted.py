"""Replay retargeted public Go2 actions on the M20 with a video."""

from __future__ import annotations

import argparse
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
parser.add_argument("--actions-h5", type=Path, required=True)
parser.add_argument("--video-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "videos/m20_retargeted_v0")
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--decimation", type=int, default=4)
parser.add_argument("--image-width", type=int, default=160)
parser.add_argument("--image-height", type=int, default=96)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from assets.m20pro import M20PRO_JUMP_CFG  # noqa: E402


@configclass
class M20ProReplaySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    robot = M20PRO_JUMP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
        update_period=0.02,
        height=96,
        width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.38, 0.0, 0.12), convention="world"),
    )
    rear_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/rear_camera",
        update_period=0.02,
        height=96,
        width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(pos=(-0.38, 0.0, 0.12), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    )
    third_person_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/third_person_camera",
        update_period=0.02,
        height=192,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
        offset=CameraCfg.OffsetCfg(convention="world"),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def main() -> None:
    with h5py.File(args.actions_h5, "r") as source:
        actions = np.asarray(source["action"], dtype=np.float32)
    steps = min(len(actions), args.steps or len(actions))
    args.video_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=args.decimation, device=args.device or "cuda:0"))
    scene = InteractiveScene(M20ProReplaySceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot = scene["robot"]
    leg_ids, _ = robot.find_joints(".*_(hipx|hipy|knee)_joint")
    wheel_ids, _ = robot.find_joints(".*_wheel_joint")
    third_person_camera = scene["third_person_camera"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()
    wheel_targets = robot.data.default_joint_pos[:, wheel_ids].clone()
    video_path = args.video_dir / f"{args.actions_h5.stem}-step-0.mp4"
    video = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0 / (0.005 * args.decimation),
        (args.image_width * 2, args.image_height * 2),
    )
    min_height = max_height = float(robot.data.root_pos_w[0, 2].item())
    start_x = float(robot.data.root_pos_w[0, 0].item())
    for step in range(steps):
        action = torch.from_numpy(actions[step]).to(robot.device).view(1, 12)
        leg_target = torch.clamp(action * 0.8, -0.8, 0.8)
        robot.set_joint_position_target(leg_target, joint_ids=leg_ids)
        robot.set_joint_position_target(wheel_targets, joint_ids=wheel_ids)
        camera_target = robot.data.root_pos_w.clone()
        camera_eye = camera_target + torch.tensor([[-2.2, 2.2, 1.3]], device=robot.device)
        third_person_camera.set_world_poses_from_view(camera_eye, camera_target)
        for _ in range(args.decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        frame = third_person_camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        height = float(robot.data.root_pos_w[0, 2].item())
        min_height = min(min_height, height)
        max_height = max(max_height, height)
    video.release()
    end_x = float(robot.data.root_pos_w[0, 0].item())
    print(f"[M20PRO-RETARGET-PLAY] source={args.actions_h5}", flush=True)
    print(f"[M20PRO-RETARGET-PLAY] steps={steps} x_displacement={end_x - start_x:.4f} m", flush=True)
    print(f"[M20PRO-RETARGET-PLAY] min_root_height={min_height:.4f} m max_root_height={max_height:.4f} m", flush=True)
    print(f"[M20PRO-RETARGET-PLAY] video={video_path}", flush=True)
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
