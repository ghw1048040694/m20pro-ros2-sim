#!/usr/bin/env python3

"""Extract stable, target-visible prefixes from SmolVLA DAgger rollouts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visibility-epsilon", type=float, default=1.0e-4)
    parser.add_argument("--minimum-frames", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def first_index(mask: np.ndarray, default: int) -> int:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else default


def copy_prefix(
    source: h5py.Group,
    destination: h5py.Group,
    source_frames: int,
    prefix_frames: int,
) -> None:
    for name, item in source.items():
        if isinstance(item, h5py.Group):
            copy_prefix(
                item,
                destination.create_group(name),
                source_frames,
                prefix_frames,
            )
            continue
        data = item[:prefix_frames] if item.ndim > 0 and item.shape[0] == source_frames else item[()]
        create_options = {}
        if np.ndim(data) > 0 and item.compression is not None:
            create_options["compression"] = item.compression
            if item.compression_opts is not None:
                create_options["compression_opts"] = item.compression_opts
        copied = destination.create_dataset(name, data=data, dtype=item.dtype, **create_options)
        for key, value in item.attrs.items():
            copied.attrs[key] = value


def extract(path: Path, output_root: Path, args: argparse.Namespace) -> dict:
    with h5py.File(path, "r") as source:
        if not bool(source.attrs.get("smolvla_dagger_labels", False)):
            raise ValueError(f"Not a SmolVLA DAgger rollout: {path}")
        front = np.asarray(source["observation/front_target_pixel_fraction"], dtype=np.float32)
        rear = np.asarray(source["observation/rear_target_pixel_fraction"], dtype=np.float32)
        state = np.asarray(source["observation/state"], dtype=np.float32)
        terminated = np.asarray(source["terminated"], dtype=np.uint8)
        source_frames = int(len(front))
        if not (len(rear) == len(state) == len(terminated) == source_frames):
            raise ValueError(f"Inconsistent source lengths: {path}")

        visibility = np.maximum(front, rear)
        peak_index = int(np.argmax(visibility))
        invisible_after_peak = first_index(
            visibility[peak_index:] <= args.visibility_epsilon,
            source_frames - peak_index,
        )
        visibility_cut = min(source_frames, peak_index + invisible_after_peak)
        root_height = state[:, 2]
        unsafe_cut = first_index(
            (root_height < 0.40) | (terminated > 0),
            source_frames,
        )
        prefix_frames = min(visibility_cut, unsafe_cut)
        if prefix_frames < args.minimum_frames:
            raise ValueError(
                f"Usable prefix too short ({prefix_frames} < {args.minimum_frames}): {path}"
            )
        stop_frames = int(
            np.count_nonzero(
                np.asarray(source["high_level_action"][:prefix_frames, 3]) > 0.5
            )
        )

        output_name = f"episode_dagger_{path.parent.name}_{path.stem.removeprefix('episode_')}.h5"
        output_path = output_root / output_name
        with h5py.File(output_path, "w") as destination:
            copy_prefix(source, destination, source_frames, prefix_frames)
            for key, value in source.attrs.items():
                destination.attrs[key] = value
            destination.attrs["smolvla_dagger_partial"] = True
            destination.attrs["smolvla_dagger_source"] = str(path.resolve())
            destination.attrs["smolvla_dagger_source_frames"] = source_frames
            destination.attrs["smolvla_dagger_prefix_frames"] = prefix_frames
            destination.attrs["smolvla_dagger_visibility_cut"] = visibility_cut
            destination.attrs["smolvla_dagger_unsafe_cut"] = unsafe_cut
            destination.attrs["success"] = False
            destination.attrs["command_ok"] = False
            destination.attrs["target_reached"] = False
            destination.attrs["target_reached_step"] = -1
            destination.attrs["terminated_steps"] = 0
            destination.attrs["min_root_height"] = float(root_height[:prefix_frames].min())
            destination.attrs["max_root_height"] = float(root_height[:prefix_frames].max())
            destination.attrs["max_target_pixel_fraction"] = float(
                visibility[:prefix_frames].max()
            )

    return {
        "source": str(path.resolve()),
        "output": str(output_path.resolve()),
        "source_frames": source_frames,
        "prefix_frames": prefix_frames,
        "peak_index": peak_index,
        "peak_visibility": float(visibility[peak_index]),
        "visibility_cut": visibility_cut,
        "unsafe_cut": unsafe_cut,
        "minimum_root_height": float(root_height[:prefix_frames].min()),
        "stop_frames": stop_frames,
    }


def main() -> None:
    args = parse_args()
    if args.visibility_epsilon < 0.0 or args.minimum_frames <= 0:
        raise ValueError("visibility epsilon must be non-negative and minimum frames positive")
    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output exists; pass --force: {args.output_root}")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    episodes = [extract(path, args.output_root, args) for path in args.inputs]
    manifest = {
        "schema": "m20pro_smolvla_dagger_prefixes_v1",
        "visibility_epsilon": args.visibility_epsilon,
        "minimum_frames": args.minimum_frames,
        "episodes": episodes,
        "total_frames": sum(item["prefix_frames"] for item in episodes),
        "warning": "Partial DAgger steering data; not successful ObjectNav episodes.",
    }
    manifest_path = args.output_root / "m20pro_smolvla_dagger_prefixes.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
