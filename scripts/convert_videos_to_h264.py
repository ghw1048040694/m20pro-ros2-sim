#!/usr/bin/env python3
"""Convert existing M20 simulation MP4 files to broadly playable H.264."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg

from video_utils import transcode_h264_in_place


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("root", type=Path)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()


def video_codec(path: Path) -> str:
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        return str(next(reader).get("codec", "unknown"))
    finally:
        reader.close()


def main() -> None:
    paths = sorted(args.root.rglob("*.mp4"))
    converted = skipped = failed = 0
    bytes_before = bytes_after = 0
    for path in paths:
        try:
            codec = video_codec(path)
            if codec == "h264":
                skipped += 1
                continue
            old_size = path.stat().st_size
            print(f"[M20PRO-VIDEO] convert codec={codec} path={path}", flush=True)
            if not args.dry_run:
                transcode_h264_in_place(path)
                new_size = path.stat().st_size
                if video_codec(path) != "h264":
                    raise RuntimeError("output codec is not h264")
                bytes_before += old_size
                bytes_after += new_size
            converted += 1
        except Exception as exc:
            failed += 1
            print(f"[M20PRO-VIDEO] failed path={path} error={exc}", flush=True)
    print(
        f"[M20PRO-VIDEO] files={len(paths)} converted={converted} skipped={skipped} "
        f"failed={failed} bytes_before={bytes_before} bytes_after={bytes_after}",
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
