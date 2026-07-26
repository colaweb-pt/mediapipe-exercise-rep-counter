from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import EXERCISE_ID, EXERCISE_NAME, SIDES, tracked_arm_label
from .pose_extractor import PoseFrame
from .rep_counter import RepEvent, Thresholds


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_landmarks_sample(
    path: Path,
    frames: list[PoseFrame],
    metadata: dict,
    source_video: Path,
    looped_video: Path,
    active_info: dict,
    visibility_threshold: float,
    rep_events: list[RepEvent],
) -> None:
    event_frames = {
        frame
        for event in rep_events
        for frame in range(max(0, event.counted_frame - 8), event.counted_frame + 9)
    }
    if len(frames) <= 500:
        selected_frames = frames
    else:
        selected_indices = set(range(500)) | event_frames
        selected_frames = [frame for frame in frames if frame.frame_index in selected_indices]

    payload = {
        "metadata": {
            "exercise_id": EXERCISE_ID,
            "exercise_name": EXERCISE_NAME,
            "source_video_filename": source_video.name,
            "looped_video_filename": looped_video.name,
            "fps": metadata["fps"],
            "width": metadata["width"],
            "height": metadata["height"],
            "total_frames": metadata["total_frames"],
            "active_side": active_info["active_side"],
            "selected_landmarks": active_info["selected_landmarks"],
            "visibility_threshold": visibility_threshold,
        },
        "frame_samples": [frame_to_export_dict(frame, active_info["active_side"]) for frame in selected_frames],
    }
    write_json(path, payload)


def frame_to_export_dict(frame: PoseFrame, active_side: str) -> dict:
    side = SIDES[active_side]
    active_landmarks = {}
    if frame.landmarks:
        active_landmarks = {
            "shoulder": frame.landmarks[side.shoulder],
            "elbow": frame.landmarks[side.elbow],
            "wrist": frame.landmarks[side.wrist],
        }
    return {
        "frame_index": frame.frame_index,
        "timestamp_seconds": frame.timestamp_seconds,
        "pose_detected": frame.pose_detected,
        "active_side": active_side,
        "active_landmarks": active_landmarks,
        "all_landmarks": frame.landmarks,
        "world_landmarks": frame.world_landmarks,
        "raw_angle": frame.selected_raw_angle,
        "smoothed_angle": frame.smoothed_angle,
        "phase_state": frame.state,
        "rep_count": frame.rep_count,
        "confidence_status": frame.confidence_status,
        "pose_status": frame.pose_status,
        "counter_status": frame.counter_status,
    }


def export_rep_events(
    path: Path,
    active_side: str,
    thresholds: Thresholds,
    total_counted_reps: int,
    rep_events: list[RepEvent],
) -> None:
    payload = {
        "exercise_name": EXERCISE_NAME,
        "active_side": active_side,
        "threshold_values_used": asdict(thresholds),
        "total_counted_reps": total_counted_reps,
        "rep_events": [event.to_dict() for event in rep_events],
    }
    write_json(path, payload)


def export_exercise_definition(
    path: Path,
    thresholds: Thresholds,
    min_rep_duration_sec: float = 0.35,
    min_angle_range_for_valid_rep: float = 35.0,
) -> None:
    payload: dict[str, Any] = {
        "exercise_id": EXERCISE_ID,
        "exercise_name": EXERCISE_NAME,
        "category": "strength",
        "difficulty": "simple",
        "movement_type": "single_joint_two_phase",
        "primary_joint": "elbow",
        "main_angle": "shoulder_elbow_wrist",
        "landmarks": {
            "left": {
                "shoulder": SIDES["left"].shoulder,
                "elbow": SIDES["left"].elbow,
                "wrist": SIDES["left"].wrist,
            },
            "right": {
                "shoulder": SIDES["right"].shoulder,
                "elbow": SIDES["right"].elbow,
                "wrist": SIDES["right"].wrist,
            },
        },
        "angle_definition": {
            "points": "shoulder -> elbow -> wrist",
            "coordinate_space": "normalized_2d_pose_landmarks",
            "description": "Elbow flexion angle calculated at the elbow joint using BlazePose shoulder, elbow, and wrist landmarks.",
        },
        "thresholds_used": asdict(thresholds),
        "rep_logic": {
            "requires_initial_down_position": True,
            "init_logic": "Counter starts in INIT and waits for an extended/down arm position before any rep can be counted.",
            "min_rep_duration_sec": min_rep_duration_sec,
            "min_angle_range_for_valid_rep": min_angle_range_for_valid_rep,
            "phases": {
                "INIT": "Waiting for the first extended/down arm position.",
                "UP": "Elbow flexed; angle is low.",
                "DOWN_READY": "Arm has re-extended enough to allow the next curl count.",
            },
            "count_condition": "Count one valid rep when the initialized state machine moves from DOWN_READY to the flexed UP threshold.",
            "reset_condition": "Reset readiness when the selected elbow angle returns above the reset threshold.",
        },
        "visualization": {
            "angle_arc": "The v3 overlay draws an elbow arc between shoulder-elbow and wrist-elbow vectors.",
            "timeline_graph": "The v3 overlay includes a live elbow-angle timeline with threshold lines and rep markers.",
            "summary_screen": "The v3 overlay appends a client-facing demo summary screen.",
        },
        "form_feedback_rules": [
            "partial_rep",
            "low_visibility",
            "no_pose_detected",
            "insufficient_extension",
            "insufficient_flexion",
        ],
        "edge_cases": [
            "partial reps",
            "camera angle variation",
            "dumbbell/wrist occlusion",
            "short/looped demo video",
            "seated posture",
            "side-view limitations",
        ],
        "met_value": {
            "value": 3.5,
            "source": "2011 Compendium of Physical Activities - resistance training, multiple exercises",
            "note": "Placeholder/demo MET value; should be confirmed per client schema.",
        },
    }
    write_json(path, payload)


def export_demo_summary(
    path: Path,
    source_video: Path,
    looped_video: Path,
    overlay_video: Path,
    metadata: dict,
    active_info: dict,
    thresholds: Thresholds,
    counted_reps: int,
    generated_files: list[Path],
    warnings: list[str],
    confidence_stats: dict | None = None,
    output_video_v2: Path | None = None,
    overlay_version: str = "v3",
    tracked_arm_label_value: str | None = None,
) -> None:
    confidence_stats = confidence_stats or {"counts": {}, "percentages": {}}
    tracked_arm_label_value = tracked_arm_label_value or tracked_arm_label(active_info["active_side"])
    payload = {
        "overlay_version": overlay_version,
        "input_video": str(source_video),
        "looped_video": str(looped_video),
        "output_video": str(overlay_video),
        "output_video_v2": str(output_video_v2) if output_video_v2 else None,
        "output_video_v3": str(output_video_v2) if overlay_version == "v3" and output_video_v2 else None,
        "fps": metadata["fps"],
        "frames": metadata["total_frames"],
        "duration": metadata["duration_seconds"],
        "active_side": active_info["active_side"],
        "tracked_arm_label": tracked_arm_label_value,
        "counted_reps": counted_reps,
        "thresholds": asdict(thresholds),
        "mean_visibility": active_info["mean_visibility"],
        "angle_range": active_info["angle_range"],
        "confidence_counts": confidence_stats.get("counts", {}),
        "confidence_percentages": confidence_stats.get("percentages", {}),
        "ok_frames_pct": confidence_stats.get("percentages", {}).get("OK", 0.0),
        "low_visibility_frames_pct": confidence_stats.get("percentages", {}).get("LOW_VISIBILITY", 0.0),
        "no_pose_frames_pct": confidence_stats.get("percentages", {}).get("NO_POSE", 0.0),
        "counter_counts": confidence_stats.get("counter_counts", {}),
        "counter_percentages": confidence_stats.get("counter_percentages", {}),
        "init_logic": "waits for DOWN_READY before counting",
        "pose_status_explanation": "Pose status reflects MediaPipe landmark quality.",
        "counter_status_explanation": "Counter status reflects rep-counting state machine initialization and movement phase.",
        "not_initialized_is_error": False,
        "looping_note": "short source clip looped for demo purposes",
        "generated_files": [str(path) for path in generated_files],
        "warnings": warnings,
    }
    write_json(path, payload)


def export_edge_cases(path: Path) -> None:
    content = """# Edge Case Notes - Seated Single-Arm Dumbbell Curl Demo

## Input video limitations

The source video is a short public exercise clip. For this portfolio demo it was looped to demonstrate 3+ rep counting, landmark extraction, overlay rendering, and JSON export.

## Counter initialization

The counter waits until it observes an extended/down arm position before counting reps. This avoids false first reps when a video starts mid-movement.

## Subject-side labeling

Left/right follows MediaPipe anatomical landmark naming. In the overlay this is shown as subject-left or subject-right to avoid confusion with screen-left and screen-right.

## Landmark visibility

The demo checks shoulder, elbow, and wrist visibility before calculating the elbow angle.

## Camera angle variation

Thresholds may need adjustment for different camera angles, body types, rep speeds, and phone positions.

## Partial reps

Partial reps can be flagged when the angle range is insufficient or when the arm does not reach expected flexion/extension boundaries.

## Wrist/dumbbell occlusion

The wrist may be partially occluded by the dumbbell or body position. In production, this should be tested on more users and camera angles.

## Demo vs production

This is a portfolio demo showing the pipeline: 33 landmarks -> joint angle -> state machine -> rep counting -> overlay video -> JSON export. Production exercise definitions should be validated against multiple videos and acceptance thresholds.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
