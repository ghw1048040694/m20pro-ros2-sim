#!/usr/bin/env python3

"""Audit a trained M20 SmolVLA checkpoint on held trajectory observations.

This is an action-prediction audit only. It does not use target coordinates and
must not be reported as closed-loop navigation success.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors


DEFAULT_ROOT = Path(
    os.environ.get(
        "M20PRO_VLA_DATA_ROOT",
        "/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_ROOT / "datasets/m20_visible_objectnav_lerobot_v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT / "logs/m20_smolvla_objectnav_offline_v1.json",
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def sample_indices(dataset: LeRobotDataset, max_samples: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset.num_frames, dtype=np.int64)
    if max_samples > 0 and max_samples < len(indices):
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
    return [int(index) for index in indices]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset = LeRobotDataset(
        "m20pro_visible_objectnav_v2", root=args.dataset_root, download_videos=False
    )
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset.meta.stats
    )

    predictions = []
    targets = []
    tasks = []
    for sample_index in sample_indices(dataset, args.max_samples, args.seed):
        raw = dataset[sample_index]
        task = raw["task"]
        batch = preprocessor(raw)
        # Each sampled frame is an independent query; do not reuse the
        # previous 50-step action queue between unrelated observations.
        policy.reset()
        with torch.no_grad():
            predicted = postprocessor(policy.select_action(batch))
        predictions.append(predicted.detach().cpu().numpy().reshape(-1))
        targets.append(raw["action"].detach().cpu().numpy().reshape(-1))
        tasks.append(task)

    predicted = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    mae = np.abs(predicted - target)
    result = {
        "schema": "m20pro_smolvla_objectnav_offline_v1",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "samples": int(len(predicted)),
        "action_dim": int(predicted.shape[1]),
        "mean_action_mae": float(mae.mean()),
        "per_action_mae": mae.mean(axis=0).tolist(),
        "forward_mae": float(mae[:, 0].mean()),
        "yaw_mae": float(mae[:, 2].mean()),
        "stop_accuracy": float(
            np.mean((predicted[:, 3] > 0.5) == (target[:, 3] > 0.5))
        ),
        "search_accuracy": float(
            np.mean((predicted[:, 4] > 0.5) == (target[:, 4] > 0.5))
        ),
        "parkour_accuracy": float(
            np.mean((predicted[:, 5] > 0.5) == (target[:, 5] > 0.5))
        ),
        "predicted_range": {
            "min": predicted.min(axis=0).tolist(),
            "max": predicted.max(axis=0).tolist(),
        },
        "task_examples": tasks[:8],
        "warning": "Offline action audit only; no target pose and no closed-loop success claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
