from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .config import POSE_LANDMARK_COUNT, SIDES
from .geometry import calculate_angle_deg


@dataclass
class PoseFrame:
    frame_index: int
    timestamp_seconds: float
    pose_detected: bool
    landmarks: Optional[list[dict]] = None
    pixel_landmarks: Optional[list[dict]] = None
    world_landmarks: Optional[list[dict]] = None
    side_angles: dict[str, Optional[float]] = field(default_factory=dict)
    side_visibility: dict[str, dict[str, float]] = field(default_factory=dict)
    selected_raw_angle: Optional[float] = None
    smoothed_angle: Optional[float] = None
    state: str = "UNKNOWN"
    rep_count: int = 0
    active_side: str = "right"
    confidence_status: str = "NO_POSE"
    pose_status: str = "NO_POSE"
    counter_status: str = "INIT"


class PoseExtractor:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_pose = mp.solutions.pose

    def process_video(self, video_path: Path) -> tuple[list[PoseFrame], dict]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open input video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: list[PoseFrame] = []
        pose_detected_count = 0

        try:
            with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as pose:
                frame_index = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    timestamp = frame_index / fps
                    record = PoseFrame(
                        frame_index=frame_index,
                        timestamp_seconds=timestamp,
                        pose_detected=bool(results.pose_landmarks),
                    )

                    if results.pose_landmarks:
                        pose_detected_count += 1
                        record.landmarks = _normalized_landmarks(results.pose_landmarks.landmark)
                        record.pixel_landmarks = _pixel_landmarks(
                            results.pose_landmarks.landmark,
                            width,
                            height,
                        )
                        if results.pose_world_landmarks:
                            record.world_landmarks = _world_landmarks(
                                results.pose_world_landmarks.landmark
                            )
                        _add_side_metrics(record)
                    else:
                        record.confidence_status = "NO_POSE"

                    frames.append(record)
                    frame_index += 1
        finally:
            cap.release()

        if not frames:
            raise RuntimeError("MediaPipe cannot process frames: input video has no readable frames.")

        pose_ratio = pose_detected_count / len(frames)
        metadata = {
            "fps": float(fps),
            "width": width,
            "height": height,
            "total_frames": len(frames),
            "duration_seconds": len(frames) / fps,
            "pose_detected_frames": pose_detected_count,
            "pose_detected_ratio": pose_ratio,
        }
        if pose_detected_count == 0:
            raise RuntimeError("Pose detection quality is low. Try another video or adjust camera angle.")
        return frames, metadata


def _normalized_landmarks(landmarks) -> list[dict]:
    return [
        {
            "index": index,
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
            "visibility": float(landmark.visibility),
        }
        for index, landmark in enumerate(landmarks[:POSE_LANDMARK_COUNT])
    ]


def _pixel_landmarks(landmarks, width: int, height: int) -> list[dict]:
    pixels = []
    for index, landmark in enumerate(landmarks[:POSE_LANDMARK_COUNT]):
        pixels.append(
            {
                "index": index,
                "x": int(round(landmark.x * width)),
                "y": int(round(landmark.y * height)),
                "visibility": float(landmark.visibility),
            }
        )
    return pixels


def _world_landmarks(landmarks) -> list[dict]:
    return [
        {
            "index": index,
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
            "visibility": float(landmark.visibility),
        }
        for index, landmark in enumerate(landmarks[:POSE_LANDMARK_COUNT])
    ]


def _add_side_metrics(record: PoseFrame) -> None:
    if not record.landmarks:
        return
    for side, indices in SIDES.items():
        shoulder = record.landmarks[indices.shoulder]
        elbow = record.landmarks[indices.elbow]
        wrist = record.landmarks[indices.wrist]
        angle = calculate_angle_deg(
            (shoulder["x"], shoulder["y"]),
            (elbow["x"], elbow["y"]),
            (wrist["x"], wrist["y"]),
        )
        record.side_angles[side] = angle
        record.side_visibility[side] = {
            "shoulder": shoulder["visibility"],
            "elbow": elbow["visibility"],
            "wrist": wrist["visibility"],
            "mean": float(
                np.mean([shoulder["visibility"], elbow["visibility"], wrist["visibility"]])
            ),
        }


def select_active_arm(
    frames: list[PoseFrame],
    visibility_threshold: float,
) -> dict:
    metrics: dict[str, dict] = {}
    min_valid_frames = max(5, int(len(frames) * 0.05))

    for side in SIDES:
        angles: list[float] = []
        visibilities: list[float] = []
        wrist_positions: list[tuple[float, float]] = []
        for frame in frames:
            angle = frame.side_angles.get(side)
            visibility = frame.side_visibility.get(side, {}).get("mean")
            if angle is None or visibility is None:
                continue
            if visibility >= visibility_threshold:
                angles.append(float(angle))
                visibilities.append(float(visibility))
                if frame.landmarks:
                    wrist = frame.landmarks[SIDES[side].wrist]
                    wrist_positions.append((float(wrist["x"]), float(wrist["y"])))

        angle_range = max(angles) - min(angles) if angles else 0.0
        mean_visibility = float(np.mean(visibilities)) if visibilities else 0.0
        valid_frames = len(angles)
        wrist_path = _path_length(wrist_positions)
        wrist_y_range = (
            max(point[1] for point in wrist_positions) - min(point[1] for point in wrist_positions)
            if wrist_positions
            else 0.0
        )
        base_score = mean_visibility * angle_range if valid_frames >= min_valid_frames else 0.0
        # In concentration curls, the working wrist travels while the support arm can look
        # angularly noisy. The motion factor prevents a planted support arm from winning.
        motion_factor = 1.0 + min(2.0, wrist_path)
        score = base_score * motion_factor
        metrics[side] = {
            "valid_frames": valid_frames,
            "mean_visibility": mean_visibility,
            "angle_range": float(angle_range),
            "base_score": float(base_score),
            "wrist_path": float(wrist_path),
            "wrist_y_range": float(wrist_y_range),
            "motion_factor": float(motion_factor),
            "score": float(score),
        }

    left_ok = metrics["left"]["valid_frames"] >= min_valid_frames
    right_ok = metrics["right"]["valid_frames"] >= min_valid_frames
    confidence = "high"
    warnings: list[str] = []

    if left_ok and right_ok:
        active_side = "left" if metrics["left"]["score"] > metrics["right"]["score"] else "right"
    elif left_ok:
        active_side = "left"
    elif right_ok:
        active_side = "right"
    else:
        active_side = "right"
        confidence = "low"
        warnings.append("Both arms have too few valid high-visibility frames; defaulted to right arm.")

    selected = SIDES[active_side]
    return {
        "active_side": active_side,
        "selected_landmarks": {
            "shoulder": selected.shoulder,
            "elbow": selected.elbow,
            "wrist": selected.wrist,
        },
        "candidates": metrics,
        "mean_visibility": metrics[active_side]["mean_visibility"],
        "angle_range": metrics[active_side]["angle_range"],
        "confidence": confidence,
        "warnings": warnings,
    }


def _path_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for previous, current in zip(points, points[1:]):
        total += float(
            np.hypot(current[0] - previous[0], current[1] - previous[1])
        )
    return total
