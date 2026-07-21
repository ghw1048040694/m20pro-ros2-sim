"""Retarget a public Go2 joint-target trajectory into M20's 12 leg joints.

This converter only performs name/order/range mapping.  It intentionally does
not claim that the signs or morphology are calibrated; the resulting HDF5 is
the input for a visual M20 validation pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

M20_NAMES = [
    "fl_hipx_joint",
    "fr_hipx_joint",
    "hl_hipx_joint",
    "hr_hipx_joint",
    "fl_hipy_joint",
    "fr_hipy_joint",
    "hl_hipy_joint",
    "hr_hipy_joint",
    "fl_knee_joint",
    "fr_knee_joint",
    "hl_knee_joint",
    "hr_knee_joint",
]
GO2_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]
GO2_DEFAULT = np.array([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)
M20_ACTION_SCALE = 0.8

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--amplitude", type=float, default=0.25, help="Scale Go2 motion around its standing pose.")
parser.add_argument(
    "--signs",
    type=float,
    nargs=12,
    default=[1.0] * 12,
    metavar="S",
    help="Per-joint sign correction. Default keeps the source sign and is unvalidated.",
)
args = parser.parse_args()


def main() -> None:
    if not 0.0 < args.amplitude <= 2.0:
        raise ValueError("--amplitude must be in (0, 2]")
    signs = np.asarray(args.signs, dtype=np.float32)
    if signs.shape != (12,) or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("--signs must contain exactly twelve values, each +1 or -1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as src:
        source_target = np.asarray(src["joint_position_target"], dtype=np.float32)
        source_action = np.asarray(src["action"], dtype=np.float32)
        command = np.asarray(src["command"], dtype=np.float32)
        timestamp = np.asarray(src["timestamp"], dtype=np.float64)
        done = np.asarray(src["done"], dtype=np.uint8)
    # Go2 and M20 share the four-leg x-major ordering. Map RL -> HL and keep
    # the source target relative to Go2's standing pose before scaling.
    target_delta = source_target - GO2_DEFAULT[None, :]
    target_m20 = args.amplitude * signs[None, :] * target_delta
    target_m20 = np.clip(target_m20, -M20_ACTION_SCALE, M20_ACTION_SCALE)
    normalized_action = target_m20 / M20_ACTION_SCALE

    with h5py.File(args.output, "w") as dst:
        dst.create_dataset("observation/joint_position_target", data=target_m20.astype(np.float32), compression="lzf")
        dst.create_dataset("action", data=normalized_action.astype(np.float32), compression="lzf")
        dst.create_dataset("source/action", data=source_action, compression="lzf")
        dst.create_dataset("source/joint_position_target", data=source_target, compression="lzf")
        dst.create_dataset("command", data=command, compression="lzf")
        dst.create_dataset("timestamp", data=timestamp, compression="lzf")
        dst.create_dataset("done", data=done, compression="lzf")
        dst.attrs["task"] = "M20 leg-target retargeting from public Go2 expert"
        dst.attrs["source_dataset"] = str(args.input)
        dst.attrs["source_joint_names"] = json.dumps(GO2_NAMES)
        dst.attrs["target_joint_names"] = json.dumps(M20_NAMES)
        dst.attrs["signs"] = json.dumps(signs.tolist())
        dst.attrs["amplitude"] = args.amplitude
        dst.attrs["validated"] = False
        dst.attrs["calibration_note"] = "Joint signs, offsets and morphology scaling require M20 visual/physics validation."
    print(f"[M20PRO-RETARGET] source_frames={len(source_target)} output={args.output}")
    print(f"[M20PRO-RETARGET] target_range=({target_m20.min():.4f}, {target_m20.max():.4f}) normalized_action_dim=12")
    print("[M20PRO-RETARGET] validated=False; do not use as a final M20 policy yet")


if __name__ == "__main__":
    main()
