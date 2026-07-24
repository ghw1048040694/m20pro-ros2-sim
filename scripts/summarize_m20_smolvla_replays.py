#!/usr/bin/env python3

"""Summarize and gate learner-only M20 SmolVLA closed-loop replays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import fmean

import cv2
import h5py


DEFAULT_ROOT = Path(
    os.environ.get(
        "M20PRO_VLA_DATA_ROOT",
        "/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA",
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_ROOT / "logs/smolvla_objectnav_replay_v4_ensemble4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ROOT
        / "logs/m20_smolvla_objectnav_replay_v4_ensemble4_summary.json",
    )
    parser.add_argument("--expected-episodes", type=int, default=4)
    return parser.parse_args()


def decode_video(path: Path, expected_frames: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"path": str(path), "codec": None, "frames": 0, "passed": False}
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
    frames = 0
    while True:
        decoded, _ = capture.read()
        if not decoded:
            break
        frames += 1
    capture.release()
    codec_ok = codec.lower() in {"avc1", "h264", "x264"}
    return {
        "path": str(path),
        "codec": codec,
        "frames": frames,
        "passed": bool(codec_ok and frames == expected_frames),
    }


def audit_episode(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    data_path = Path(metrics["data"])
    video_path = Path(metrics["video"])
    with h5py.File(data_path, "r") as h5:
        required_diagnostics = {
            "smolvla_stop_score",
            "smolvla_stop_votes",
            "smolvla_execution_command",
        }
        diagnostics_present = required_diagnostics.issubset(h5.keys())
        dagger_labels = bool(h5.attrs.get("smolvla_dagger_labels", True))
        stop_score_max = (
            float(h5["smolvla_stop_score"][:].max())
            if "smolvla_stop_score" in h5
            else None
        )
        stop_votes_max = (
            int(h5["smolvla_stop_votes"][:].max())
            if "smolvla_stop_votes" in h5
            else None
        )

    scenario = metrics.get("scenario") or {}
    reached_step = metrics.get("target_reached_step")
    stop_step = metrics.get("smolvla_stop_latched_step")
    interventions = sum(
        int(metrics.get(key, 0))
        for key in (
            "smolvla_visible_intervention_steps",
            "smolvla_stability_intervention_steps",
            "smolvla_target_intervention_steps",
        )
    )
    stop_after_reach = bool(
        reached_step is not None and stop_step is not None and stop_step >= reached_step
    )
    learner_only = bool(
        metrics.get("smolvla_checkpoint")
        and not dagger_labels
        and interventions == 0
        and scenario.get("inference_uses_privileged_target_pose") is False
    )
    video = decode_video(video_path, int(metrics["frames"]))
    passed = bool(
        metrics.get("success")
        and metrics.get("stable")
        and metrics.get("startup_posture_ok") is True
        and metrics.get("posture_ok") is True
        and int(metrics.get("terminated_steps", 0)) == 0
        and metrics.get("target_reached")
        and metrics.get("smolvla_stop_latched")
        and stop_after_reach
        and learner_only
        and diagnostics_present
        and video["passed"]
    )
    return {
        "episode_id": metrics["episode_id"],
        "success": bool(metrics.get("success")),
        "stable": bool(metrics.get("stable")),
        "startup_posture_ok": metrics.get("startup_posture_ok") is True,
        "posture_ok": metrics.get("posture_ok") is True,
        "terminated_steps": int(metrics.get("terminated_steps", 0)),
        "target_reached_step": reached_step,
        "smolvla_stop_latched_step": stop_step,
        "stop_after_reach": stop_after_reach,
        "min_target_distance_m": metrics.get("min_target_distance_m"),
        "final_target_distance_m": metrics.get("final_target_distance_m"),
        "min_root_height_m": metrics.get("min_root_height_m"),
        "root_height_std_m": metrics.get("root_height_std_m"),
        "max_abs_roll_deg": metrics.get("max_abs_roll_deg"),
        "max_abs_pitch_deg": metrics.get("max_abs_pitch_deg"),
        "startup_max_body_angular_speed_rps": metrics.get(
            "startup_max_body_angular_speed_rps"
        ),
        "final_leg_symmetry_error_rad": metrics.get(
            "final_leg_symmetry_error_rad"
        ),
        "max_post_stop_target_hold_steps": metrics.get(
            "max_post_stop_target_hold_steps", 0
        ),
        "stop_score_max": stop_score_max,
        "stop_votes_max": stop_votes_max,
        "learner_only": learner_only,
        "expert_intervention_steps": interventions,
        "diagnostics_present": diagnostics_present,
        "video": video,
        "passed": passed,
        "metrics": str(path),
    }


def main() -> None:
    args = parse_args()
    if args.expected_episodes <= 0:
        raise ValueError("--expected-episodes must be positive")
    paths = sorted(args.input_root.glob("*/episode_*.json"))
    episodes = [audit_episode(path) for path in paths]
    if not episodes:
        raise RuntimeError(f"No episode metrics found below {args.input_root}")

    successes = sum(item["success"] for item in episodes)
    falls = sum(item["terminated_steps"] > 0 for item in episodes)
    false_stops = sum(
        item["smolvla_stop_latched_step"] is not None and not item["stop_after_reach"]
        for item in episodes
    )
    passed = sum(item["passed"] for item in episodes)
    result = {
        "schema": "m20pro_smolvla_closed_loop_summary_v1",
        "input_root": str(args.input_root),
        "expected_episodes": args.expected_episodes,
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "falls": falls,
        "fall_rate": falls / len(episodes),
        "false_stops": false_stops,
        "posture_passed": sum(item["posture_ok"] for item in episodes),
        "startup_posture_passed": sum(
            item["startup_posture_ok"] for item in episodes
        ),
        "learner_only_episodes": sum(item["learner_only"] for item in episodes),
        "video_passed": sum(item["video"]["passed"] for item in episodes),
        "min_root_height_m": min(item["min_root_height_m"] for item in episodes),
        "mean_final_target_distance_m": fmean(
            item["final_target_distance_m"] for item in episodes
        ),
        "passed_episodes": passed,
        "visible_objectnav_closed_loop_passed": bool(
            len(episodes) == args.expected_episodes and passed == len(episodes)
        ),
        "warning": (
            "Visible-target learner-only gate only; hidden search, place navigation, "
            "and one-meter parkour are not covered."
        ),
        "episode_results": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["visible_objectnav_closed_loop_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
