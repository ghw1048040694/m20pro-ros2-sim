"""Replay the released Robot Parkour Learning Go1 visual policy in Isaac Lab.

This is a native Go1 validation pass.  It intentionally does not retarget to
M20: the public checkpoint must first be shown to produce a physically valid
legged motion with its original 48-d proprioception and 48x64 depth input.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from isaaclab.app import AppLauncher
from video_utils import finalize_h264_video

DATA_ROOT = Path("/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA")
DEFAULT_CHECKPOINT = DATA_ROOT / "public_experts/parkour_go1/skill/model_674000.pt"
DEFAULT_WALK_CHECKPOINT = DATA_ROOT / "public_experts/parkour_go1/walk/model_107500.pt"
DEFAULT_SOURCE = DATA_ROOT / "public_experts/parkour_go1/rsl_rl"
DEFAULT_VIDEO_DIR = Path(os.environ.get("M20PRO_OUTPUT_ROOT", str(DATA_ROOT))) / "videos/public_parkour_go1"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--policy-mode", choices=("skill", "walk"), default="skill")
parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--obstacle-height", type=float, default=0.45)
parser.add_argument("--obstacle-x", type=float, default=1.25)
parser.add_argument("--command-x", type=float, default=1.0)
parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
parser.add_argument("--video", action="store_true", help="Required: write the third-person MP4.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.video:
    parser.error("--video is required so every expert replay leaves an inspectable MP4")
args.enable_cameras = True
if args.checkpoint is None:
    args.checkpoint = DEFAULT_CHECKPOINT if args.policy_mode == "skill" else DEFAULT_WALK_CHECKPOINT
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(args.source_root))

# Isaac Lab's RL extensions may import their own ``rsl_rl`` before this
# script reaches the policy construction.  The released parkour checkpoint
# depends on its fork, which includes VisualDeterministicRecurrent.  Remove
# the already-imported namespace so the source path above wins consistently.
for _module_name in list(sys.modules):
    if _module_name == "rsl_rl" or _module_name.startswith("rsl_rl."):
        del sys.modules[_module_name]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import importlib.util as _importlib_util  # noqa: E402
_parkour_spec = _importlib_util.find_spec("rsl_rl")
if _parkour_spec is None or str(args.source_root) not in str(_parkour_spec.origin):
    raise RuntimeError(
        "public parkour rsl_rl was shadowed: "
        f"origin={getattr(_parkour_spec, 'origin', None)}; sys.path[0:5]={sys.path[:5]}"
    )
from rsl_rl.modules.visual_actor_critic import VisualDeterministicRecurrent  # noqa: E402
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg, Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # noqa: E402


JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
DEFAULT_JOINT_POS = torch.tensor(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    dtype=torch.float32,
)
ACTION_LOW = torch.tensor(
    [-1.894, -2.526, -2.042, -1.494, -2.526, -2.042, -1.894, -2.926, -2.042, -1.494, -2.926, -2.042],
    dtype=torch.float32,
)
ACTION_HIGH = torch.tensor(
    [1.494, 3.932, 0.926, 1.894, 3.932, 0.926, 1.494, 3.532, 0.926, 1.894, 3.532, 0.926],
    dtype=torch.float32,
)

PUBLIC_GO1_CFG = UNITREE_GO1_CFG.copy()
PUBLIC_GO1_CFG.init_state = PUBLIC_GO1_CFG.init_state.replace(pos=(0.0, 0.0, 0.43))
PUBLIC_GO1_CFG.actuators = {
    "hips": DCMotorCfg(
        joint_names_expr=[".*_hip_joint"],
        effort_limit=20.0,
        saturation_effort=20.0,
        velocity_limit=30.0,
        stiffness=40.0,
        damping=0.5,
    ),
    "thighs": DCMotorCfg(
        joint_names_expr=[".*_thigh_joint"],
        effort_limit=20.0,
        saturation_effort=20.0,
        velocity_limit=30.0,
        stiffness=40.0,
        damping=0.5,
    ),
    "calves": DCMotorCfg(
        joint_names_expr=[".*_calf_joint"],
        effort_limit=25.0,
        saturation_effort=25.0,
        velocity_limit=30.0,
        stiffness=40.0,
        damping=0.5,
    ),
}


@configclass
class ParkourSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot = PUBLIC_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.55, 1.5, args.obstacle_height),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.9),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(args.obstacle_x, 0.0, args.obstacle_height / 2.0)),
    )
    forward_depth = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ForwardDepth",
        update_period=0.02,
        height=60,
        width=106,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.12, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.272, 0.0075, 0.092), convention="world"),
    )
    third_person = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ThirdPerson",
        update_period=0.02,
        height=192,
        width=320,
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
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


def load_policy(device: torch.device):
    if args.policy_mode == "walk":
        policy = ActorCriticRecurrent(
            num_actor_obs=48,
            num_critic_obs=65,
            num_actions=12,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            rnn_type="gru",
            rnn_hidden_size=256,
        ).to(device)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state = payload.get("model_state_dict", payload)
        missing, unexpected = policy.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={list(missing)}, unexpected={list(unexpected)}")
        policy.eval()
        return policy
    obs_segments = OrderedDict(proprioception=(48,), forward_depth=(1, 48, 64))
    policy = VisualDeterministicRecurrent(
        num_actor_obs=48 + 48 * 64,
        num_critic_obs=81,
        num_actions=12,
        obs_segments=obs_segments,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_size=256,
        visual_kwargs={
            "channels": [16, 32, 32],
            "hidden_sizes": [128],
            "kernel_sizes": [5, 4, 3],
            "strides": [2, 2, 1],
            "nonlinearity": "LeakyReLU",
            "use_maxpool": True,
        },
        visual_latent_size=128,
    ).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = payload.get("model_state_dict", payload)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={list(missing)}, unexpected={list(unexpected)}")
    policy.eval()
    return policy


def proprioception(robot: Articulation, joint_ids: list[int], last_actions: torch.Tensor) -> torch.Tensor:
    device = robot.device
    gravity = torch.zeros((1, 3), device=device)
    gravity[:, 2] = -1.0
    ang_vel = robot.data.root_ang_vel_b[:, :3] * 0.25
    projected_gravity = quat_rotate_inverse(robot.data.root_quat_w, gravity.expand(robot.num_instances, -1))
    commands = torch.tensor([args.command_x * 2.0, 0.0, 0.0], device=device).expand(robot.num_instances, -1)
    dof_pos = robot.data.joint_pos[:, joint_ids] - DEFAULT_JOINT_POS.to(device)
    dof_vel = robot.data.joint_vel[:, joint_ids] * 0.05
    return torch.cat(
        [torch.zeros_like(ang_vel), ang_vel, projected_gravity, commands, dof_pos, dof_vel, last_actions], dim=-1
    )


def depth_observation(camera) -> torch.Tensor:
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth = torch.nan_to_num(depth, nan=2.0, posinf=2.0, neginf=0.0).clamp(0.0, 2.0)
    # Match the released RealSense preprocessing: crop the raw 60x106 frame,
    # normalize the configured 0-2 m range, then resize to 48x64.
    depth = depth[:, :-1, 15:-12].unsqueeze(1) / 2.0
    return F.interpolate(depth, size=(48, 64), mode="bilinear", align_corners=False)


def main() -> None:
    device = torch.device(args.device or "cuda:0")
    args.video_dir.mkdir(parents=True, exist_ok=True)
    sim = SimulationContext(SimulationCfg(dt=0.005, render_interval=4, device=str(device)))
    scene = InteractiveScene(ParkourSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True))
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]
    joint_ids, joint_names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if joint_names != JOINT_NAMES:
        raise RuntimeError(f"Go1 joint order mismatch: {joint_names}")
    depth_camera = scene["forward_depth"]
    third_person = scene["third_person"]
    policy = load_policy(robot.device)
    last_actions = torch.zeros((1, 12), device=robot.device)
    video_path = args.video_dir / f"go1-parkour-h{args.obstacle_height:.2f}-step-0.mp4"
    video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (320, 192))
    min_height = float(robot.data.root_pos_w[0, 2].item())
    max_height = min_height
    start_x = float(robot.data.root_pos_w[0, 0].item())
    max_action = 0.0
    min_depth = 2.0
    initial_depth_min = None
    initial_depth_max = None
    try:
        for step in range(args.steps):
            depth_eye = robot.data.root_pos_w + torch.tensor([[0.272, 0.0075, 0.092]], device=robot.device)
            depth_target = robot.data.root_pos_w + torch.tensor([[1.45, 0.0, -0.15]], device=robot.device)
            depth_camera.set_world_poses_from_view(depth_eye, depth_target)
            sim.render()
            scene.update(sim.get_physics_dt())
            proprio = proprioception(robot, joint_ids, last_actions)
            depth = depth_observation(depth_camera)
            if initial_depth_min is None:
                initial_depth_min = float(depth.min().item())
                initial_depth_max = float(depth.max().item())
                depth_preview = (depth[0, 0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                cv2.imwrite(str(args.video_dir / "go1-parkour-depth-step-0.png"), depth_preview)
            observations = proprio if args.policy_mode == "walk" else torch.cat([proprio, depth.flatten(1)], dim=-1)
            with torch.inference_mode():
                actions = policy.act_inference(observations)
            actions = torch.maximum(actions, ACTION_LOW.to(robot.device))
            actions = torch.minimum(actions, ACTION_HIGH.to(robot.device))
            robot.set_joint_position_target(
                DEFAULT_JOINT_POS.to(robot.device).expand(1, -1) + 0.5 * actions,
                joint_ids=joint_ids,
            )
            last_actions = actions.detach()
            camera_target = robot.data.root_pos_w + torch.tensor([[0.0, 0.0, 0.2]], device=robot.device)
            camera_eye = camera_target + torch.tensor([[-2.2, 2.2, 1.25]], device=robot.device)
            third_person.set_world_poses_from_view(camera_eye, camera_target)
            for _ in range(4):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())
            frame = third_person.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            height = float(robot.data.root_pos_w[0, 2].item())
            min_height = min(min_height, height)
            max_height = max(max_height, height)
            max_action = max(max_action, float(actions.abs().max().item()))
            min_depth = min(min_depth, float(depth.min().item()))
    finally:
        finalize_h264_video(video, video_path)
    end_x = float(robot.data.root_pos_w[0, 0].item())
    print(f"[M20PRO-PARKOUR-PLAY] policy_mode={args.policy_mode} checkpoint={args.checkpoint}", flush=True)
    print(f"[M20PRO-PARKOUR-PLAY] obstacle_height={args.obstacle_height:.3f} m", flush=True)
    print(f"[M20PRO-PARKOUR-PLAY] steps={args.steps} x_displacement={end_x - start_x:.4f} m", flush=True)
    print(f"[M20PRO-PARKOUR-PLAY] min_root_height={min_height:.4f} m max_root_height={max_height:.4f} m", flush=True)
    print(f"[M20PRO-PARKOUR-PLAY] min_depth_normalized={min_depth:.4f} max_abs_action={max_action:.4f}", flush=True)
    print(f"[M20PRO-PARKOUR-PLAY] initial_depth_range=[{initial_depth_min:.4f}, {initial_depth_max:.4f}] normalized", flush=True)
    print(f"[M20PRO-PARKOUR-PLAY] video={video_path}", flush=True)
    scene.reset()
    sim.clear_instance()


try:
    main()
finally:
    app.close()
