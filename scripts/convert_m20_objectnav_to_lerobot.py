#!/usr/bin/env python3

"""Convert audited M20 visible ObjectNav v2 episodes to LeRobot.

The SmolVLA base checkpoint accepts a 32-dimensional state.  v2 uses eight
body-frame proprioceptive values and compresses the 72-beam indoor LiDAR into
24 sector minimums; absolute world position is deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_ROOT = Path(
    os.environ.get(
        "M20PRO_VLA_DATA_ROOT",
        "/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA",
    )
)
PROPRIO_DIM = 8
LIDAR_SECTORS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_ROOT / "datasets/m20_visible_objectnav_v2/train",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ROOT / "datasets/m20_visible_objectnav_lerobot_v2",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_ROOT / "logs/m20_smolvla_data_audit_v2.json",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def lidar_sector_minimums(lidar: np.ndarray) -> np.ndarray:
    """Return 24 finite minimum ranges from a 72-beam scan."""
    lidar = np.asarray(lidar, dtype=np.float32)
    if lidar.ndim != 1 or lidar.size % LIDAR_SECTORS != 0:
        raise ValueError(f"Expected a flat 72-beam LiDAR scan, got {lidar.shape}")
    beams_per_sector = lidar.size // LIDAR_SECTORS
    clean = np.nan_to_num(lidar, nan=20.0, posinf=20.0, neginf=0.0)
    return clean.reshape(LIDAR_SECTORS, beams_per_sector).min(axis=1)


def make_state(state: np.ndarray, lidar: np.ndarray) -> np.ndarray:
    proprio = np.asarray(state, dtype=np.float32)
    if proprio.shape != (PROPRIO_DIM,):
        raise ValueError(f"Expected invariant proprio shape {(PROPRIO_DIM,)}, got {proprio.shape}")
    compressed_lidar = lidar_sector_minimums(lidar)
    result = np.concatenate((proprio, compressed_lidar)).astype(np.float32, copy=False)
    if result.shape != (32,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid converted state: shape={result.shape}")
    return result


def eligible_paths(input_root: Path, audit_path: Path) -> list[Path]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    eligible = {
        item["path"]
        for item in audit["episodes"]
        if item.get("smolvla_candidate") and item.get("smolvla_eligible")
    }
    # Audit paths are relative to the audit input root. Support either the
    # dataset root or its train/ subdirectory as the converter input.
    dataset_root = input_root.parent if input_root.name == "train" else input_root
    paths = [dataset_root / relative for relative in eligible]
    paths = sorted(
        path
        for path in paths
        if path.exists() and (input_root.name != "train" or path.parent == input_root)
    )
    if not paths:
        raise RuntimeError("No audited SmolVLA-eligible episodes found")
    return paths


def convert(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output exists; pass --force to replace: {args.output_root}")
        shutil.rmtree(args.output_root)

    paths = eligible_paths(args.input_root, args.audit)
    if args.max_episodes > 0:
        paths = paths[: args.max_episodes]

    with h5py.File(paths[0], "r") as first:
        front_shape = tuple(first["observation/front_rgb"].shape[1:])
        rear_shape = tuple(first["observation/rear_rgb"].shape[1:])
    if front_shape != rear_shape or len(front_shape) != 3 or front_shape[-1] != 3:
        raise ValueError(f"Unexpected camera shapes: {front_shape}, {rear_shape}")

    action_names = ["forward_mps", "lateral_mps", "yaw_rps", "stop", "search", "parkour"]
    state_names = [f"proprio_{index:02d}" for index in range(PROPRIO_DIM)]
    state_names.extend(f"lidar_sector_{index:02d}" for index in range(LIDAR_SECTORS))
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (32,),
            "names": state_names,
        },
        "observation.images.camera1": {
            "dtype": "image",
            "shape": front_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": rear_shape,
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": action_names,
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="m20pro_visible_objectnav_v2",
        fps=50,
        root=args.output_root,
        robot_type="m20pro_sim",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    manifest = []
    try:
        for episode_index, path in enumerate(paths):
            with h5py.File(path, "r") as h5:
                attrs = {key: _json_value(value) for key, value in h5.attrs.items()}
                front = h5["observation/front_rgb"]
                rear = h5["observation/rear_rgb"]
                lidar = h5["observation/lidar"]
                action = np.asarray(h5["high_level_action"], dtype=np.float32)
                smolvla_proprio = h5["observation/smolvla_proprio"]
                if action.shape[1:] != (6,):
                    raise ValueError(f"{path}: high_level_action shape={action.shape}")
                task = str(attrs["task_text"])
                for frame_index in range(action.shape[0]):
                    frame = {
                        "observation.state": make_state(
                            smolvla_proprio[frame_index], lidar[frame_index]
                        ),
                        "observation.images.camera1": np.asarray(front[frame_index], dtype=np.uint8),
                        "observation.images.camera2": np.asarray(rear[frame_index], dtype=np.uint8),
                        "action": action[frame_index],
                        "task": task,
                    }
                    dataset.add_frame(frame)
                dataset.save_episode(parallel_encoding=False)
                manifest.append(
                    {
                        "episode_index": episode_index,
                        "source": str(path),
                        "scenario_episode_id": attrs.get("scenario_episode_id"),
                        "scene_id": attrs.get("scene_id"),
                        "object_category": attrs.get("object_category"),
                        "instruction_template_id": attrs.get("instruction_template_id"),
                        "task_text": task,
                        "frames": int(action.shape[0]),
                    }
                )
            print(f"[M20PRO-CONVERT] {episode_index + 1}/{len(paths)} {path.name}", flush=True)
    finally:
        dataset.finalize()

    conversion = {
        "schema": "m20pro_visible_objectnav_lerobot_v2",
        "source_audit": str(args.audit),
        "episodes": len(manifest),
        "frames": sum(item["frames"] for item in manifest),
        "fps": 50,
        "state": {
            "shape": [32],
            "proprio_indices": list(range(PROPRIO_DIM)),
            "lidar_input": "72 beams compressed to 24 sector minimums",
        },
        "action": {"shape": [6], "names": action_names},
        "camera_keys": ["observation.images.camera1", "observation.images.camera2"],
        "episodes": manifest,
    }
    (args.output_root / "m20pro_conversion_manifest.json").write_text(
        json.dumps(conversion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return conversion


def main() -> None:
    args = parse_args()
    result = convert(args)
    print(json.dumps({key: result[key] for key in ("schema", "episodes", "frames", "state", "action")}, indent=2))


if __name__ == "__main__":
    main()
