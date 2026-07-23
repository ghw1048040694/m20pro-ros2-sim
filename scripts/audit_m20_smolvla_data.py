#!/usr/bin/env python3

"""Audit M20 HDF5 demonstrations before converting them to LeRobot v3."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "M20PRO_VLA_DATA_ROOT",
        "/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA",
    )
)
REQUIRED_DATASETS = (
    "action",
    "observation/front_rgb",
    "observation/rear_rgb",
    "observation/lidar",
    "observation/state",
    "terminated",
)
HOLDOUT_MARKERS = ("eval", "holdout", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit source demonstrations for M20 SmolVLA fine-tuning."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_DATA_ROOT / "datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "logs/m20_smolvla_data_audit_v1.json",
    )
    parser.add_argument("--lidar-max-range", type=float, default=20.0)
    parser.add_argument("--lidar-return-margin", type=float, default=0.1)
    return parser.parse_args()


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def task_text(attrs: h5py.AttributeManager) -> str:
    return str(json_value(attrs.get("task_text", attrs.get("task", ""))))


def is_holdout(path: Path) -> bool:
    lowered = str(path).lower()
    return any(marker in lowered for marker in HOLDOUT_MARKERS)


def inspect_episode(path: Path, root: Path, args: argparse.Namespace) -> dict:
    relative_path = str(path.relative_to(root))
    result = {"path": relative_path, "errors": [], "warnings": []}
    with h5py.File(path, "r") as h5:
        attrs = {key: json_value(value) for key, value in h5.attrs.items()}
        smolvla_candidate = bool(attrs.get("smolvla_candidate", False))
        missing = [key for key in REQUIRED_DATASETS if key not in h5]
        result["errors"].extend(f"missing_dataset:{key}" for key in missing)
        if missing:
            result.update(
                {
                    "task": task_text(h5.attrs),
                    "success": bool(attrs.get("success", False)),
                    "train_eligible": False,
                }
            )
            return result

        lengths = {key: int(h5[key].shape[0]) for key in REQUIRED_DATASETS}
        frame_count = lengths["action"]
        if len(set(lengths.values())) != 1:
            result["errors"].append("inconsistent_dataset_lengths")
        if frame_count <= 0:
            result["errors"].append("empty_episode")

        front = h5["observation/front_rgb"]
        rear = h5["observation/rear_rgb"]
        lidar = np.asarray(h5["observation/lidar"], dtype=np.float32)
        state = np.asarray(h5["observation/state"], dtype=np.float32)
        action = np.asarray(h5["action"], dtype=np.float32)
        terminated = np.asarray(h5["terminated"])
        for name, array in (("lidar", lidar), ("state", state), ("action", action)):
            if not np.isfinite(array).all():
                result["errors"].append(f"non_finite:{name}")
        if front.dtype != np.uint8 or rear.dtype != np.uint8:
            result["errors"].append("rgb_not_uint8")
        if front.shape[1:] != rear.shape[1:] or len(front.shape) != 4 or front.shape[-1] != 3:
            result["errors"].append("invalid_rgb_shape")

        lidar = np.nan_to_num(
            lidar,
            nan=args.lidar_max_range,
            posinf=args.lidar_max_range,
            neginf=0.0,
        )
        return_fraction = float(
            np.mean(lidar < args.lidar_max_range - args.lidar_return_margin)
        )
        task = task_text(h5.attrs)
        success = bool(attrs.get("success", False))
        command_ok = bool(attrs.get("command_ok", True))
        stable_search = (
            "寻找" in task
            and int(attrs.get("terminated_steps", int(np.count_nonzero(terminated)))) == 0
            and float(attrs.get("min_root_height", 0.0)) >= 0.45
        )
        holdout = is_holdout(Path(relative_path))
        split = str(attrs.get("split", "legacy"))
        train_eligible = (
            (success and command_ok or stable_search)
            and not holdout
            and (not smolvla_candidate or split == "train")
        )
        explicit_timestamps = "timestamp" in h5 and "frame_index" in h5
        sensor_alignment = str(attrs.get("sensor_alignment", "undocumented"))
        if explicit_timestamps:
            timestamps = np.asarray(h5["timestamp"], dtype=np.float64)
            frame_indices = np.asarray(h5["frame_index"], dtype=np.int64)
            valid_timestamps = (
                timestamps.shape == (frame_count,)
                and np.isfinite(timestamps).all()
                and (frame_count < 2 or bool(np.all(np.diff(timestamps) > 0.0)))
                and np.array_equal(frame_indices, np.arange(frame_count, dtype=np.int64))
            )
            if not valid_timestamps:
                result["errors"].append("invalid_timestamps_or_frame_indices")
                explicit_timestamps = False
        else:
            result["warnings"].append("missing_explicit_timestamps")
        if sensor_alignment != "pre_action":
            result["warnings"].append("sensor_action_alignment_not_verified")

        command_source = "missing"
        if "expert_command" in h5:
            command = np.asarray(h5["expert_command"], dtype=np.float32)
            command_source = "expert_command"
            if command.shape != (frame_count, 3) or not np.isfinite(command).all():
                result["errors"].append("invalid_expert_command")
        elif "command" in attrs and len(attrs["command"]) == 3:
            command_source = "episode_attribute"
        else:
            result["errors"].append("missing_high_level_action_label")

        high_level_action_valid = False
        if "high_level_action" in h5:
            high_level_action = np.asarray(h5["high_level_action"], dtype=np.float32)
            high_level_action_valid = bool(
                high_level_action.shape == (frame_count, 6)
                and np.isfinite(high_level_action).all()
            )
            if not high_level_action_valid:
                result["errors"].append("invalid_high_level_action")
        elif smolvla_candidate:
            result["errors"].append("missing_high_level_action")

        required_candidate_attrs = (
            "scene_id",
            "split",
            "object_category",
            "object_source",
            "object_usd_path",
            "instruction_template_id",
            "manifest_sha256",
            "scenario_episode_id",
            "task_type",
            "expert_uses_privileged_target_pose",
        )
        missing_candidate_attrs = [
            name for name in required_candidate_attrs if not attrs.get(name)
        ]
        if smolvla_candidate:
            result["errors"].extend(
                f"missing_candidate_attribute:{name}" for name in missing_candidate_attrs
            )
            if str(attrs.get("lidar_mesh_scope", "")) == "ground_only":
                result["errors"].append("candidate_lidar_excludes_scene_geometry")
            if bool(attrs.get("inference_uses_privileged_target_pose", True)):
                result["errors"].append("candidate_uses_privileged_target_pose_at_inference")

        target_visibility_valid = True
        if smolvla_candidate and str(attrs.get("task_type")) == "visible_object_navigation":
            target_visibility_valid = (
                "observation/front_target_pixel_fraction" in h5
                and "observation/rear_target_pixel_fraction" in h5
                and bool(attrs.get("target_visible_at_start", False))
            )
            if target_visibility_valid:
                front_fraction = np.asarray(
                    h5["observation/front_target_pixel_fraction"], dtype=np.float32
                )
                rear_fraction = np.asarray(
                    h5["observation/rear_target_pixel_fraction"], dtype=np.float32
                )
                target_visibility_valid = bool(
                    front_fraction.shape == (frame_count,)
                    and rear_fraction.shape == (frame_count,)
                    and np.isfinite(front_fraction).all()
                    and np.isfinite(rear_fraction).all()
                    and (front_fraction[0] > 0.0 or rear_fraction[0] > 0.0)
                )
            if not target_visibility_valid:
                result["errors"].append("visible_target_not_verified_in_camera")

        temporal_alignment_valid = (
            explicit_timestamps and sensor_alignment == "pre_action"
        )
        smolvla_eligible = bool(
            smolvla_candidate
            and train_eligible
            and temporal_alignment_valid
            and high_level_action_valid
            and not missing_candidate_attrs
            and str(attrs.get("lidar_mesh_scope", "")) != "ground_only"
            and target_visibility_valid
            and not result["errors"]
        )

        result.update(
            {
                "task": task,
                "frames": frame_count,
                "success": success,
                "command_ok": command_ok,
                "stable_search_partial_demo": stable_search,
                "holdout": holdout,
                "train_eligible": train_eligible and not result["errors"],
                "smolvla_candidate": smolvla_candidate,
                "smolvla_eligible": smolvla_eligible,
                "dagger": bool(attrs.get("dagger", False)),
                "target_color": str(attrs.get("target_color", "none")),
                "object_category": attrs.get("object_category"),
                "object_source": attrs.get("object_source"),
                "instruction_template_id": attrs.get("instruction_template_id"),
                "split": split,
                "task_type": attrs.get("task_type"),
                "target_visibility_valid": target_visibility_valid,
                "target_reached": bool(attrs.get("target_reached", False)),
                "scene_id": attrs.get("scene_id"),
                "obstacle_height_m": attrs.get("obstacle_height_m"),
                "command_source": command_source,
                "high_level_action_valid": high_level_action_valid,
                "action_dim": int(action.shape[1]) if action.ndim == 2 else None,
                "state_dim": int(state.shape[1]) if state.ndim == 2 else None,
                "rgb_shape": list(front.shape[1:]),
                "lidar_shape": list(lidar.shape[1:]),
                "lidar_min_m": float(lidar.min()),
                "lidar_mean_m": float(lidar.mean()),
                "lidar_std_m": float(lidar.std()),
                "lidar_return_fraction": return_fraction,
                "lidar_mesh_scope": str(attrs.get("lidar_mesh_scope", "undocumented")),
                "explicit_timestamps": explicit_timestamps,
                "sensor_alignment": sensor_alignment,
                "temporal_alignment_valid": temporal_alignment_valid,
            }
        )
    return result


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.rglob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 episodes found under {args.input_root}")
    episodes = [inspect_episode(path, args.input_root, args) for path in paths]
    eligible = [episode for episode in episodes if episode.get("train_eligible", False)]
    candidates = [episode for episode in episodes if episode.get("smolvla_candidate", False)]
    smolvla_eligible = [
        episode for episode in episodes if episode.get("smolvla_eligible", False)
    ]
    train_candidates = [
        episode for episode in candidates if episode.get("split") == "train"
    ]
    successful = [episode for episode in episodes if episode.get("success", False)]
    holdouts = [episode for episode in episodes if episode.get("holdout", False)]
    tasks = Counter(episode.get("task", "") for episode in smolvla_eligible)
    categories = sorted(
        {
            str(episode["object_category"])
            for episode in smolvla_eligible
            if episode.get("object_category") not in (None, "")
        }
    )
    template_ids = sorted(
        {
            str(episode["instruction_template_id"])
            for episode in smolvla_eligible
            if episode.get("instruction_template_id") not in (None, "")
        }
    )
    scene_ids = sorted(
        {
            str(episode["scene_id"])
            for episode in smolvla_eligible
            if episode.get("scene_id") is not None
        }
    )
    jump_episodes = [
        episode
        for episode in smolvla_eligible
        if "跳" in episode.get("task", "") or episode.get("obstacle_height_m") is not None
    ]
    hidden_search_successes = [
        episode
        for episode in smolvla_eligible
        if episode.get("success")
        and episode.get("target_reached")
        and episode.get("stable_search_partial_demo")
        and episode.get("target_color") not in (None, "none", "")
    ]
    obstacle_lidar_episodes = [
        episode
        for episode in smolvla_eligible
        if episode.get("obstacle_height_m") is not None
        and episode.get("lidar_mesh_scope") != "ground_only"
    ]
    timestamp_aligned = [
        episode
        for episode in smolvla_eligible
        if episode.get("explicit_timestamps") and episode.get("sensor_alignment") == "pre_action"
    ]

    visible_objectnav_gates = {
        "minimum_train_scenes_8": len(scene_ids) >= 8,
        "minimum_object_categories_12": len(categories) >= 12,
        "minimum_instruction_templates_24": len(template_ids) >= 24,
        "all_candidates_valid_and_timestamp_aligned": (
            len(train_candidates) > 0
            and len(smolvla_eligible) == len(train_candidates)
        ),
        "scene_geometry_visible_to_lidar": (
            len(smolvla_eligible) > 0
            and all(
                episode.get("lidar_mesh_scope") != "ground_only"
                for episode in smolvla_eligible
            )
        ),
        "six_dimensional_action_labels_present": (
            len(smolvla_eligible) > 0
            and all(episode.get("high_level_action_valid") for episode in smolvla_eligible)
        ),
    }
    downstream_gates = {
        "hidden_object_search_success_present": len(hidden_search_successes) > 0,
        "one_meter_obstacle_success_present": any(
            float(episode.get("obstacle_height_m") or 0.0) >= 1.0
            and episode.get("success", False)
            for episode in jump_episodes
        ),
        "obstacle_lidar_data_present": len(obstacle_lidar_episodes) > 0,
    }
    gates = {**visible_objectnav_gates, **downstream_gates}
    summary = {
        "schema": "m20pro_smolvla_data_audit_v1",
        "input_root": str(args.input_root),
        "inventory": {
            "episodes": len(episodes),
            "frames": sum(int(episode.get("frames", 0)) for episode in episodes),
            "successful_episodes": len(successful),
            "holdout_episodes": len(holdouts),
            "train_eligible_episodes": len(eligible),
            "train_eligible_frames": sum(int(episode["frames"]) for episode in eligible),
            "smolvla_candidate_episodes": len(candidates),
            "smolvla_eligible_episodes": len(smolvla_eligible),
            "smolvla_eligible_frames": sum(
                int(episode["frames"]) for episode in smolvla_eligible
            ),
            "dagger_train_episodes": sum(bool(episode.get("dagger")) for episode in eligible),
            "stable_search_partial_episodes": sum(
                bool(episode.get("stable_search_partial_demo")) for episode in eligible
            ),
            "annotated_scenes": len(scene_ids),
            "target_categories": categories,
            "instruction_template_ids": template_ids,
            "instruction_counts": dict(sorted(tasks.items())),
            "jump_episodes": len(jump_episodes),
            "hidden_search_successes": len(hidden_search_successes),
            "obstacle_lidar_episodes": len(obstacle_lidar_episodes),
            "timestamp_aligned_episodes": len(timestamp_aligned),
        },
        "gates": gates,
        "visible_objectnav_gates": visible_objectnav_gates,
        "downstream_gates": downstream_gates,
        "ready_for_visible_objectnav_finetune": all(visible_objectnav_gates.values()),
        "ready_for_smolvla_finetune": all(gates.values()),
        "blocking_reasons": [name for name, passed in gates.items() if not passed],
        "notes": [
            "Stable search-only rotations are partial demonstrations, not completed ObjectNav episodes.",
            "A range return does not prove obstacle visibility; obstacle annotations and scene geometry are required.",
            "Legacy files without sensor_alignment were written by a recorder that stored RGB/LiDAR before action and state after action.",
            "Only files explicitly marked smolvla_candidate participate in new fine-tuning readiness gates.",
        ],
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    visible = {
        key: summary[key]
        for key in (
            "schema",
            "inventory",
            "gates",
            "visible_objectnav_gates",
            "downstream_gates",
            "ready_for_visible_objectnav_finetune",
            "ready_for_smolvla_finetune",
            "blocking_reasons",
        )
    }
    print(json.dumps(visible, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
