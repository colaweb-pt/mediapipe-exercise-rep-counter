from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import cv2
import mediapipe as mp
import numpy as np

from .config import EXERCISE_NAME, SIDES, tracked_arm_label
from .pose_extractor import PoseFrame
from .rep_counter import RepEvent, Thresholds


OK_COLOR = (60, 210, 90)
WARN_COLOR = (0, 190, 255)
BAD_COLOR = (40, 40, 230)
ACTIVE_COLOR = (255, 220, 40)
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (20, 20, 20)
CHART_LINE_COLOR = (70, 230, 255)
UP_THRESHOLD_COLOR = (255, 210, 60)
DOWN_THRESHOLD_COLOR = (90, 230, 110)
RESET_THRESHOLD_COLOR = (180, 180, 255)


def render_overlay_video(
    input_video_path: Path,
    output_video_path: Path,
    frames: list[PoseFrame],
    active_side: str,
    fps: float,
    width: int,
    height: int,
) -> None:
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open looped video for rendering: {input_video_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_available = shutil.which("ffmpeg") is not None
    raw_output_path = (
        output_video_path.with_name(f"{output_video_path.stem}.opencv_raw.mp4")
        if ffmpeg_available
        else output_video_path
    )
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV cannot create overlay video: {raw_output_path}")

    connections = list(mp.solutions.pose.POSE_CONNECTIONS)
    active = SIDES[active_side]
    active_pairs = {(active.shoulder, active.elbow), (active.elbow, active.wrist)}
    active_pairs |= {(b, a) for a, b in active_pairs}

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or index >= len(frames):
            break
        record = frames[index]
        draw_pose(frame, record, connections, active_pairs)
        draw_status_panel(frame, record, active_side)
        writer.write(frame)
        index += 1

    cap.release()
    writer.release()

    if ffmpeg_available:
        _reencode_for_playback(raw_output_path, output_video_path, fps)
        raw_output_path.unlink(missing_ok=True)


def render_overlay_video_v2(
    input_video_path: Path,
    output_video_path: Path,
    frames: list[PoseFrame],
    active_side: str,
    fps: float,
    width: int,
    height: int,
    thresholds: Thresholds,
    rep_events: list[RepEvent],
    summary_screen_sec: float = 2.0,
    title_screen_sec: float = 1.0,
    draw_angle_arc_enabled: bool = True,
    draw_angle_chart_enabled: bool = True,
    contact_sheet_path: Path | None = None,
) -> None:
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open looped video for rendering: {input_video_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_available = shutil.which("ffmpeg") is not None
    raw_output_path = (
        output_video_path.with_name(f"{output_video_path.stem}.opencv_raw.mp4")
        if ffmpeg_available
        else output_video_path
    )
    writer = cv2.VideoWriter(
        str(raw_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV cannot create overlay video: {raw_output_path}")

    connections = list(mp.solutions.pose.POSE_CONNECTIONS)
    active = SIDES[active_side]
    active_pairs = {(active.shoulder, active.elbow), (active.elbow, active.wrist)}
    active_pairs |= {(b, a) for a, b in active_pairs}
    all_angles = [frame.smoothed_angle for frame in frames]

    for _ in range(max(0, int(round(title_screen_sec * fps)))):
        writer.write(_create_title_screen(width, height))

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or index >= len(frames):
            break
        record = frames[index]
        draw_pose(frame, record, connections, active_pairs, draw_angle_label=False)
        if draw_angle_arc_enabled:
            draw_active_angle_arc(frame, record, active_side)
        draw_status_panel_v2(frame, record, active_side)
        if draw_angle_chart_enabled:
            draw_angle_timeline(
                frame,
                all_angles[: index + 1],
                [event for event in rep_events if event.counted_frame <= index],
                thresholds,
                index,
            )
        writer.write(frame)
        index += 1

    summary = _create_summary_screen(width, height, active_side, len([event for event in rep_events if event.valid]))
    for _ in range(max(0, int(round(summary_screen_sec * fps)))):
        writer.write(summary)

    cap.release()
    writer.release()

    if ffmpeg_available:
        _reencode_for_playback(raw_output_path, output_video_path, fps)
        raw_output_path.unlink(missing_ok=True)

    if contact_sheet_path:
        create_contact_sheet_v3(
            output_video_path,
            contact_sheet_path,
            frames,
            fps,
            title_screen_sec,
            summary_screen_sec,
        )


def draw_pose(frame, record: PoseFrame, connections, active_pairs, draw_angle_label: bool = True) -> None:
    if not record.pixel_landmarks:
        return
    points = {
        item["index"]: (int(item["x"]), int(item["y"]), float(item["visibility"]))
        for item in record.pixel_landmarks
    }

    for start, end in connections:
        if start not in points or end not in points:
            continue
        x1, y1, v1 = points[start]
        x2, y2, v2 = points[end]
        if min(v1, v2) < 0.25:
            continue
        is_active = (start, end) in active_pairs
        color = ACTIVE_COLOR if is_active else (180, 180, 180)
        thickness = 7 if is_active else 2
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    for index, (x, y, visibility) in points.items():
        if visibility < 0.25:
            continue
        radius = 8 if any(index in pair for pair in active_pairs) else 3
        color = ACTIVE_COLOR if any(index in pair for pair in active_pairs) else (210, 210, 210)
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)

    active_indices = {idx for pair in active_pairs for idx in pair}
    elbow_candidates = [idx for idx in active_indices if idx in (13, 14)]
    if draw_angle_label and elbow_candidates and record.smoothed_angle is not None:
        elbow = elbow_candidates[0]
        if elbow in points:
            x, y, _ = points[elbow]
            cv2.putText(
                frame,
                f"{record.smoothed_angle:.0f} deg",
                (x + 16, y - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                ACTIVE_COLOR,
                2,
                cv2.LINE_AA,
            )


def draw_active_angle_arc(frame, record: PoseFrame, active_side: str) -> None:
    if not record.pixel_landmarks or record.smoothed_angle is None:
        return
    indices = SIDES[active_side]
    points = {item["index"]: item for item in record.pixel_landmarks}
    required = [indices.shoulder, indices.elbow, indices.wrist]
    if any(index not in points for index in required):
        return
    if min(points[index]["visibility"] for index in required) < 0.35:
        return

    shoulder = (int(points[indices.shoulder]["x"]), int(points[indices.shoulder]["y"]))
    elbow = (int(points[indices.elbow]["x"]), int(points[indices.elbow]["y"]))
    wrist = (int(points[indices.wrist]["x"]), int(points[indices.wrist]["y"]))
    draw_angle_arc(frame, shoulder, elbow, wrist, record.smoothed_angle, ACTIVE_COLOR)


def draw_angle_arc(
    frame,
    shoulder_px: tuple[int, int],
    elbow_px: tuple[int, int],
    wrist_px: tuple[int, int],
    angle_deg: float | None,
    color: tuple[int, int, int],
    radius: int = 58,
) -> None:
    if angle_deg is None:
        return
    elbow = np.asarray(elbow_px, dtype=float)
    shoulder = np.asarray(shoulder_px, dtype=float)
    wrist = np.asarray(wrist_px, dtype=float)
    v1 = shoulder - elbow
    v2 = wrist - elbow
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return

    start = math.atan2(v1[1], v1[0])
    end = math.atan2(v2[1], v2[0])
    delta = (end - start + math.pi) % (2 * math.pi) - math.pi
    arc_angles = np.linspace(start, start + delta, 28)
    points = []
    for theta in arc_angles:
        x = int(round(elbow[0] + radius * math.cos(theta)))
        y = int(round(elbow[1] + radius * math.sin(theta)))
        points.append((x, y))
    if len(points) >= 2:
        cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, 5, cv2.LINE_AA)

    label_point = points[len(points) // 2] if points else elbow_px
    cv2.putText(
        frame,
        f"{angle_deg:.1f} deg",
        (label_point[0] + 10, label_point[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        color,
        3,
        cv2.LINE_AA,
    )


def draw_status_panel(frame, record: PoseFrame, active_side: str) -> None:
    height, width = frame.shape[:2]
    scale = max(0.75, min(1.35, width / 900.0))
    line_height = int(34 * scale)
    margin = int(24 * scale)
    panel_width = min(width - 2 * margin, int(620 * scale))
    panel_height = int(240 * scale)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (margin + panel_width, margin + panel_height),
        PANEL_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    confidence_color = _confidence_color(record.confidence_status)
    angle_text = "N/A" if record.smoothed_angle is None else f"{record.smoothed_angle:.1f} deg"
    rows = [
        f"Exercise: {EXERCISE_NAME}",
        f"Active side: {active_side}",
        f"Elbow angle: {angle_text}",
        f"Phase/state: {record.state}",
        f"Reps: {record.rep_count}",
        f"Confidence: {record.confidence_status}",
    ]

    y = margin + int(34 * scale)
    for row in rows:
        color = confidence_color if row.startswith("Confidence:") else TEXT_COLOR
        cv2.putText(
            frame,
            row,
            (margin + int(18 * scale), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72 * scale,
            color,
            max(2, int(2 * scale)),
            cv2.LINE_AA,
        )
        y += line_height

    footer = "MediaPipe / BlazePose 33-landmark demo"
    cv2.putText(
        frame,
        footer,
        (margin, height - margin),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62 * scale,
        (230, 230, 230),
        max(2, int(2 * scale)),
        cv2.LINE_AA,
    )


def _confidence_color(status: str) -> tuple[int, int, int]:
    if status == "OK":
        return OK_COLOR
    if status in {"LOW_VISIBILITY", "ANGLE_UNAVAILABLE", "NOT_INITIALIZED", "WAITING_FOR_DOWN", "INIT"}:
        return WARN_COLOR
    return BAD_COLOR


def draw_status_panel_v2(frame, record: PoseFrame, active_side: str) -> None:
    height, width = frame.shape[:2]
    scale = max(1.0, min(1.55, width / 780.0))
    margin = int(28 * scale)
    line_height = int(42 * scale)
    panel_width = min(width - 2 * margin, int(620 * scale))
    panel_height = int(360 * scale)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (margin, margin),
        (margin + panel_width, margin + panel_height),
        PANEL_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    angle_text = "N/A" if record.smoothed_angle is None else f"{record.smoothed_angle:.1f} deg"
    pose_status = getattr(record, "pose_status", record.confidence_status)
    counter_status = getattr(record, "counter_status", record.state)
    counter_text = "Waiting for DOWN" if counter_status in {"INIT", "WAITING_FOR_DOWN"} else counter_status
    rows = [
        ("MediaPipe / BlazePose Demo", TEXT_COLOR),
        ("Exercise: Dumbbell Curl", TEXT_COLOR),
        (f"Tracked arm: {tracked_arm_label(active_side)}", TEXT_COLOR),
        (f"Angle: {angle_text}", ACTIVE_COLOR),
        (f"State: {record.state}", TEXT_COLOR),
        (f"Reps: {record.rep_count}", TEXT_COLOR),
        (f"Pose: {pose_status}", _confidence_color(pose_status)),
        (f"Counter: {counter_text}", _counter_color(counter_status)),
    ]

    y = margin + int(42 * scale)
    for index, (row, color) in enumerate(rows):
        font_scale = 0.86 * scale if index else 0.92 * scale
        thickness = max(2, int(2.2 * scale))
        cv2.putText(
            frame,
            row,
            (margin + int(18 * scale), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


def _counter_color(status: str) -> tuple[int, int, int]:
    if status in {"INIT", "WAITING_FOR_DOWN"}:
        return WARN_COLOR
    if status in {"DOWN_READY", "UP"}:
        return TEXT_COLOR
    return ACTIVE_COLOR


def draw_angle_timeline(
    frame,
    angle_history: list[float | None],
    rep_events_so_far: list[RepEvent],
    thresholds: Thresholds,
    current_frame_index: int,
) -> None:
    if not angle_history:
        return
    height, width = frame.shape[:2]
    margin_x = int(width * 0.055)
    chart_h = int(height * 0.20)
    chart_y = height - chart_h - int(height * 0.035)
    chart_w = width - 2 * margin_x
    chart_x = margin_x

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (chart_x, chart_y),
        (chart_x + chart_w, chart_y + chart_h),
        PANEL_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)

    cv2.putText(
        frame,
        "Elbow angle over time",
        (chart_x + 18, chart_y + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )

    plot_top = chart_y + 52
    plot_bottom = chart_y + chart_h - 24
    plot_left = chart_x + 22
    plot_right = chart_x + chart_w - 22
    draw_threshold_line(frame, thresholds.down_threshold, plot_left, plot_right, plot_top, plot_bottom, DOWN_THRESHOLD_COLOR, "down")
    draw_threshold_line(frame, thresholds.up_threshold, plot_left, plot_right, plot_top, plot_bottom, UP_THRESHOLD_COLOR, "up")
    draw_threshold_line(frame, thresholds.reset_threshold, plot_left, plot_right, plot_top, plot_bottom, RESET_THRESHOLD_COLOR, "reset")

    valid_points = []
    total = max(1, len(angle_history) - 1)
    for idx, angle in enumerate(angle_history):
        if angle is None:
            continue
        x = int(plot_left + (plot_right - plot_left) * idx / total)
        y = map_angle_to_chart_y(angle, plot_top, plot_bottom)
        valid_points.append((x, y))
    if len(valid_points) >= 2:
        cv2.polylines(
            frame,
            [np.asarray(valid_points, dtype=np.int32)],
            False,
            CHART_LINE_COLOR,
            4,
            cv2.LINE_AA,
        )

    for event in rep_events_so_far:
        if current_frame_index <= 0:
            continue
        x = int(plot_left + (plot_right - plot_left) * event.counted_frame / current_frame_index)
        cv2.line(frame, (x, plot_top), (x, plot_bottom), ACTIVE_COLOR, 2, cv2.LINE_AA)
        cv2.circle(frame, (x, plot_top + 12), 6, ACTIVE_COLOR, -1, cv2.LINE_AA)


def draw_threshold_line(
    frame,
    angle: float,
    left: int,
    right: int,
    top: int,
    bottom: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    y = map_angle_to_chart_y(angle, top, bottom)
    cv2.line(frame, (left, y), (right, y), color, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (right - 72, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.54, color, 1, cv2.LINE_AA)


def map_angle_to_chart_y(angle: float, top: int, bottom: int, min_angle: float = 30.0, max_angle: float = 170.0) -> int:
    clipped = max(min_angle, min(max_angle, float(angle)))
    ratio = (clipped - min_angle) / (max_angle - min_angle)
    return int(round(bottom - ratio * (bottom - top)))


def create_contact_sheet(video_path: Path, output_path: Path, frame_count: int = 12) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video for contact sheet: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Cannot create contact sheet from empty video: {video_path}")

    indices = np.linspace(0, total_frames - 1, frame_count, dtype=int)
    thumbs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        thumb = cv2.resize(frame, (270, 480))
        cv2.putText(
            thumb,
            f"frame {idx}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    cap.release()
    if not thumbs:
        raise RuntimeError(f"No frames available for contact sheet: {video_path}")

    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) < rows * cols:
        thumbs.append(blank.copy())
    sheet_rows = [np.hstack(thumbs[i : i + cols]) for i in range(0, len(thumbs), cols)]
    sheet = np.vstack(sheet_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def create_contact_sheet_v3(
    video_path: Path,
    output_path: Path,
    analysis_frames: list[PoseFrame],
    fps: float,
    title_screen_sec: float,
    summary_screen_sec: float,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video for contact sheet: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Cannot create contact sheet from empty video: {video_path}")

    title_frames = int(round(title_screen_sec * fps))
    summary_start = max(0, total_frames - int(round(summary_screen_sec * fps)))
    semantic = _select_contact_sheet_frames(analysis_frames, title_frames, summary_start)
    thumbs = []
    for idx, label in semantic:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, min(total_frames - 1, idx))))
        ok, frame = cap.read()
        if not ok:
            continue
        thumbs.append(_make_labeled_thumbnail(frame, label))
    cap.release()
    if not thumbs:
        raise RuntimeError(f"No frames available for contact sheet: {video_path}")

    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) < rows * cols:
        thumbs.append(blank.copy())
    sheet = np.vstack([np.hstack(thumbs[i : i + cols]) for i in range(0, len(thumbs), cols)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def _select_contact_sheet_frames(
    frames: list[PoseFrame],
    title_offset: int,
    summary_start: int,
) -> list[tuple[int, str]]:
    selected: list[tuple[int, str]] = [(0, "Title screen")]

    def first_matching(predicate, label: str) -> None:
        for frame in frames:
            if predicate(frame):
                selected.append((title_offset + frame.frame_index, label.format(rep=frame.rep_count, state=frame.state)))
                return

    first_matching(lambda frame: frame.state == "INIT", "Frame {state} / Waiting")
    first_matching(lambda frame: frame.state == "DOWN_READY", "Frame DOWN_READY")
    first_matching(lambda frame: frame.state == "UP" and frame.rep_count >= 1, "UP / Rep {rep}")
    first_matching(lambda frame: frame.state == "DOWN_READY" and frame.rep_count >= 2, "DOWN_READY / Rep {rep}")
    first_matching(lambda frame: frame.state == "UP" and frame.rep_count >= 3, "UP / Rep {rep}")
    first_matching(lambda frame: frame.rep_count >= 4, "Final analysis / Rep {rep}")
    selected.append((summary_start, "Summary"))

    deduped: list[tuple[int, str]] = []
    seen: set[int] = set()
    for idx, label in selected:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append((idx, label))
    return deduped[:10]


def _make_labeled_thumbnail(frame, label: str):
    thumb = cv2.resize(frame, (270, 480))
    header_h = 38
    canvas = np.zeros((480 + header_h, 270, 3), dtype=np.uint8)
    canvas[:header_h, :] = (18, 20, 24)
    canvas[header_h:, :] = thumb
    cv2.putText(
        canvas,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _create_title_screen(width: int, height: int):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 20, 24)
    _put_centered_text(frame, "MediaPipe / BlazePose", int(height * 0.38), 1.35, TEXT_COLOR, 3)
    _put_centered_text(frame, "Dumbbell Curl Rep Counter Demo", int(height * 0.45), 1.05, ACTIVE_COLOR, 3)
    _put_centered_text(frame, "33 landmarks -> elbow angle -> state machine -> rep count", int(height * 0.53), 0.72, (210, 210, 210), 2)
    return frame


def _create_summary_screen(width: int, height: int, active_side: str, reps: int):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (16, 18, 22)
    y = int(height * 0.20)
    _put_centered_text(frame, "Demo Summary", y, 1.45, ACTIVE_COLOR, 3)
    rows = [
        f"Exercise: {EXERCISE_NAME}",
        "Pose model: MediaPipe / BlazePose",
        "Landmarks: 33 body landmarks",
        "Main angle: shoulder-elbow-wrist",
        f"Detected reps: {reps}",
        "Deliverables: overlay video, landmarks JSON,",
        "rep events JSON, exercise definition JSON",
        "Notes: short public clip looped for 3+ rep demo",
    ]
    y += 110
    for row in rows:
        _put_centered_text(frame, row, y, 0.78, TEXT_COLOR, 2)
        y += 58
    return frame


def _put_centered_text(frame, text: str, y: int, scale: float, color: tuple[int, int, int], thickness: int) -> None:
    height, width = frame.shape[:2]
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(20, (width - size[0]) // 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _reencode_for_playback(raw_path: Path, final_path: Path, fps: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_path),
        "-r",
        f"{fps:.6f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_path),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        if final_path.exists():
            final_path.unlink()
        raw_path.replace(final_path)
        print(f"Warning: ffmpeg re-encode failed, kept OpenCV MP4 output. Details: {exc}")
