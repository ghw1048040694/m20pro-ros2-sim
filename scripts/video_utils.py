"""Video finalization helpers shared by M20 simulation scripts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2


def transcode_h264_in_place(path: str | Path) -> Path:
    """Atomically transcode an existing video to H.264/yuv420p."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video file is empty or missing: {path}")

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("imageio-ffmpeg is required to finalize H.264 videos") from exc

    temporary = path.with_name(f".{path.stem}.h264.tmp.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        capture = cv2.VideoCapture(str(temporary))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded, _ = capture.read()
        capture.release()
        if not decoded or frame_count <= 0:
            raise RuntimeError(f"H.264 verification failed: {temporary}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def finalize_h264_video(video: cv2.VideoWriter, path: str | Path) -> Path:
    """Close an OpenCV writer and atomically replace its MP4 with H.264."""
    video.release()
    return transcode_h264_in_place(path)
