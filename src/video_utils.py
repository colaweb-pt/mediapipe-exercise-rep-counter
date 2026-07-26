from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2

from .config import VIDEO_EXTENSIONS


def discover_input_video(project_root: Path, input_arg: Optional[str]) -> Path:
    if input_arg:
        path = Path(input_arg).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Input file must be one of: {', '.join(VIDEO_EXTENSIONS)}")
        return path

    candidates: list[Path] = []
    search_dirs = [project_root, project_root / "input"]
    for directory in search_dirs:
        if not directory.exists():
            continue
        for ext in VIDEO_EXTENSIONS:
            candidates.extend(sorted(directory.glob(f"*{ext}")))
            candidates.extend(sorted(directory.glob(f"*{ext.upper()}")))

    unique = sorted({candidate.resolve() for candidate in candidates})
    if not unique:
        raise FileNotFoundError(
            "No video found. Place your MP4 file in the project root or input/ folder, "
            "or run with --input path/to/video.mp4"
        )
    if len(unique) > 1:
        found = "\n".join(f"- {path}" for path in unique)
        raise ValueError(f"Multiple video files found. Please specify --input\n{found}")
    return unique[0]


def get_video_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open input video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 0:
        fps = 24.0
    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "frames": frames,
        "duration_seconds": frames / fps if fps else 0.0,
    }


def create_looped_video(input_path: Path, output_path: Path, loops: int) -> dict:
    if loops < 1:
        raise ValueError("--loops must be >= 1")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions for: {input_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from input video: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot create output video: {output_path}")

    for _ in range(loops):
        for frame in frames:
            writer.write(frame)
    writer.release()

    total_frames = len(frames) * loops
    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "source_frames": len(frames),
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
    }
