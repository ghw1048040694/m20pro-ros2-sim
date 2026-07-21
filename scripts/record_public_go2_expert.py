"""Record NVIDIA's published Isaac Lab Go2 expert policy and a tracking video."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "datasets/public_go2_rough_v0")
parser.add_argument("--video-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "videos/public_go2_rough_v0")
parser.add_argument("--command-x", type=float, default=0.8, help="Constant forward velocity command in m/s.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

ISAACLAB_ROOT = Path(__file__).resolve().parents[1] / ".deps/IsaacLab"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ISAACLAB_ROOT / "scripts/reinforcement_learning/rsl_rl"))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import ViewerCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    UnitreeGo2RoughPPORunnerCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (  # noqa: E402
    UnitreeGo2RoughEnvCfg_PLAY,
)


TASK = "Isaac-Velocity-Rough-Unitree-Go2-v0"
SOURCE_URL = "https://github.com/isaac-sim/IsaacLab"


def state_vector(robot) -> np.ndarray:
    return (
        torch.cat(
            (
                robot.data.root_pos_w[0],
                robot.data.root_quat_w[0],
                robot.data.root_lin_vel_w[0],
                robot.data.root_ang_vel_w[0],
                robot.data.joint_pos[0],
                robot.data.joint_vel[0],
            )
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def main() -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", TASK)
    if checkpoint is None:
        raise RuntimeError(f"Published checkpoint is unavailable for {TASK}")

    env_cfg = UnitreeGo2RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env_cfg.sim.device = args.device or "cuda:0"
    env_cfg.log_dir = str(Path(checkpoint).parent)
    env_cfg.viewer = ViewerCfg(
        eye=(2.2, 2.2, 1.2),
        lookat=(0.0, 0.0, 0.25),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.ranges.lin_vel_x = (args.command_x, args.command_x)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.debug_vis = False

    agent_cfg = UnitreeGo2RoughPPORunnerCfg()
    agent_cfg.device = env_cfg.sim.device
    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(args.video_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix="public-go2-rough-expert",
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    observations = env.get_observations()
    obs_dim = int(observations["policy"].shape[-1])
    robot = env.unwrapped.scene["robot"]
    data_path = args.output_dir / "episode_0000.h5"
    dt = float(env.unwrapped.step_dt)
    done_count = 0
    with h5py.File(data_path, "w") as h5:
        obs_ds = h5.create_dataset("observation/policy", (args.steps, obs_dim), dtype="f4", compression="lzf")
        state_ds = h5.create_dataset("observation/state", (args.steps, 37), dtype="f4", compression="lzf")
        action_ds = h5.create_dataset("action", (args.steps, 12), dtype="f4", compression="lzf")
        target_ds = h5.create_dataset("joint_position_target", (args.steps, 12), dtype="f4", compression="lzf")
        command_ds = h5.create_dataset("command", (args.steps, 3), dtype="f4", compression="lzf")
        time_ds = h5.create_dataset("timestamp", (args.steps,), dtype="f8")
        done_ds = h5.create_dataset("done", (args.steps,), dtype="u1")
        h5.attrs["task"] = "在崎岖地形上向前行走"
        h5.attrs["source"] = SOURCE_URL
        h5.attrs["checkpoint"] = str(checkpoint)
        h5.attrs["robot"] = "Unitree Go2"
        for step in range(args.steps):
            with torch.inference_mode():
                actions = policy(observations)
            command = env.unwrapped.command_manager.get_command("base_velocity")[0, :3]
            obs_ds[step] = observations["policy"][0].detach().cpu().numpy()
            state_ds[step] = state_vector(robot)
            action_ds[step] = actions[0].detach().cpu().numpy()
            command_ds[step] = command.detach().cpu().numpy()
            time_ds[step] = step * dt
            observations, _, dones, _ = env.step(actions)
            target_ds[step] = robot.data.joint_pos_target[0, :12].detach().cpu().numpy()
            done = int(dones[0].item())
            done_ds[step] = done
            done_count += done
            policy_nn.reset(dones)

    metadata = {
        "format": "m20pro_public_expert_hdf5_v0",
        "source": SOURCE_URL,
        "license": "Isaac Lab BSD-3-Clause; checkpoint distributed by NVIDIA Isaac Lab",
        "task": TASK,
        "robot": "Unitree Go2",
        "control_hz": 1.0 / dt,
        "observation_dim": obs_dim,
        "state_dim": 37,
        "action_dim": 12,
        "joint_names": list(robot.joint_names),
        "command": [args.command_x, 0.0, 0.0],
        "episode_steps": args.steps,
        "done_count": done_count,
        "retarget_status": "source trajectory only; M20 joint retargeting not yet applied",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"[M20PRO-PUBLIC-EXPERT] checkpoint={checkpoint}", flush=True)
    print(f"[M20PRO-PUBLIC-EXPERT] data={data_path} video_dir={args.video_dir}", flush=True)
    print(f"[M20PRO-PUBLIC-EXPERT] obs_dim={obs_dim} action_dim=12 done_count={done_count}", flush=True)
    env.close()


try:
    main()
finally:
    app.close()
