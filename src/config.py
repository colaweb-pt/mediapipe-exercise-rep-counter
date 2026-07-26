from __future__ import annotations

from dataclasses import dataclass


EXERCISE_ID = "seated_single_arm_dumbbell_curl"
EXERCISE_NAME = "Seated Single-Arm Dumbbell Curl"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v")


@dataclass(frozen=True)
class SideLandmarks:
    side: str
    shoulder: int
    elbow: int
    wrist: int


SIDES = {
    "left": SideLandmarks("left", shoulder=11, elbow=13, wrist=15),
    "right": SideLandmarks("right", shoulder=12, elbow=14, wrist=16),
}


POSE_LANDMARK_COUNT = 33


def tracked_arm_label(active_side: str) -> str:
    return f"subject-{active_side}"
