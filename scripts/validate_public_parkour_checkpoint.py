#!/usr/bin/env python3
"""Validate the public Robot Parkour Learning Go1 visual policy.

This is deliberately simulator-independent.  It checks the exact observation
packing used by the released checkpoint before we write an Isaac Lab adapter.
The checkpoint consumes 48 proprioceptive values and a 48x64 forward depth
image; the image is embedded by the policy before the recurrent actor runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch


DEFAULT_ROOT = Path(
    "/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA"
)
DEFAULT_SOURCE = DEFAULT_ROOT / "public_experts/parkour_go1/rsl_rl"
DEFAULT_CHECKPOINT = DEFAULT_ROOT / "public_experts/parkour_go1/skill/model_674000.pt"


def build_policy(source_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(source_root))
    from rsl_rl.modules.visual_actor_critic import VisualDeterministicRecurrent

    # The 235 value in the archived config is stale.  The released model's
    # visual encoder and recurrent input weights prove this packing:
    # 48 proprioception + (1, 48, 64) forward depth.
    obs_segments = OrderedDict(
        proprioception=(48,),
        forward_depth=(1, 48, 64),
    )
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
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    state = payload.get("model_state_dict", payload)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={list(missing)}, "
            f"unexpected={list(unexpected)}"
        )
    policy.eval()
    return policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 2:
        parser.error("--batch-size must be >= 1 and --steps must be >= 2")

    device = torch.device(args.device)
    with torch.no_grad():
        policy = build_policy(args.source_root, args.checkpoint, device)
        proprio = torch.zeros(args.batch_size, 48, device=device)
        # 2 m is the far-depth cap used in the public training configuration.
        plane = torch.full((args.batch_size, 1, 48, 64), 2.0, device=device)
        obstacle = plane.clone()
        obstacle[:, :, 16:40, 24:48] = 0.45

        observations = torch.cat([proprio, plane.flatten(1)], dim=-1)
        obstacle_observations = torch.cat([proprio, obstacle.flatten(1)], dim=-1)
        actions = []
        for _ in range(args.steps):
            actions.append(policy.act_inference(observations).detach().cpu())
        recurrent_actions = torch.stack(actions)
        policy.reset(torch.ones(args.batch_size, dtype=torch.bool, device=device))
        reset_action = policy.act_inference(observations).detach().cpu()
        obstacle_action = policy.act_inference(obstacle_observations).detach().cpu()

    metrics = {
        "checkpoint": str(args.checkpoint),
        "observation_shape": list(observations.shape),
        "action_shape": list(recurrent_actions.shape),
        "action_min": float(recurrent_actions.min()),
        "action_max": float(recurrent_actions.max()),
        "action_mean_abs": float(recurrent_actions.abs().mean()),
        "mean_step_delta": float(
            recurrent_actions[1:].sub(recurrent_actions[:-1]).abs().mean()
        ),
        "plane_vs_obstacle_mean_abs_delta": float(
            obstacle_action.sub(recurrent_actions[-1]).abs().mean()
        ),
        "reset_action_mean_abs_delta": float(
            reset_action.sub(recurrent_actions[-1]).abs().mean()
        ),
        "hidden_state_present_after_reset": policy.memory_a.hidden_states is not None,
    }
    print("[M20PRO-PARKOUR-CHECK] checkpoint loaded")
    for key, value in metrics.items():
        print(f"[M20PRO-PARKOUR-CHECK] {key}={value}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
