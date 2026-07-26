from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "mediapipe_curl_demo_matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(tempfile.gettempdir(), "mediapipe_curl_demo_cache"),
)

from src.config import EXERCISE_NAME, SIDES, tracked_arm_label
from src.exporters import (
    export_demo_summary,
    export_edge_cases,
    export_exercise_definition,
    export_landmarks_sample,
    export_rep_events,
)
from src.geometry import exponential_smoothing
from src.pose_extractor import PoseExtractor, PoseFrame, select_active_arm
from src.rep_counter import CurlRepCounter
from src.renderer import render_overlay_video, render_overlay_video_v2
from src.video_utils import create_looped_video, discover_input_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local MediaPipe / BlazePose dumbbell curl rep-counting demo."
    )
    parser.add_argument("--input", help="Optional path to local MP4/MOV/M4V input video.")
    parser.add_argument("--output-dir", default="output", help="Directory for generated artifacts.")
    parser.add_argument("--loops", type=int, default=4, help="How many times to loop the source clip.")
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.5,
        help="Minimum mean shoulder/elbow/wrist visibility for arm scoring.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Exponential smoothing factor for selected elbow angle.",
    )
    parser.add_argument("--model-complexity", type=int, default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--v2-overlay", action="store_true", default=True)
    parser.add_argument("--no-v2-overlay", action="store_false", dest="v2_overlay")
    parser.add_argument("--summary-screen-sec", type=float, default=2.0)
    parser.add_argument("--title-screen-sec", type=float, default=1.0)
    parser.add_argument("--draw-angle-arc", action="store_true", default=True)
    parser.add_argument("--no-draw-angle-arc", action="store_false", dest="draw_angle_arc")
    parser.add_argument("--draw-angle-chart", action="store_true", default=True)
    parser.add_argument("--no-draw-angle-chart", action="store_false", dest="draw_angle_chart")
    parser.add_argument("--output-video-name", default="bicep_curl_overlay_v3.mp4")
    parser.add_argument("--overlay-version", choices=["v2", "v3"], default="v3")
    parser.add_argument("--min-rep-duration-sec", type=float, default=0.35)
    parser.add_argument("--min-angle-range-for-valid-rep", type=float, default=35.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = _resolve_output_dir(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_video = discover_input_video(project_root, args.input)
        looped_video = output_dir / "demo_input_looped.mp4"
        overlay_video = output_dir / "bicep_curl_overlay.mp4"
        overlay_video_v2 = output_dir / args.output_video_name
        contact_sheet_v2 = output_dir / f"overlay_contact_sheet_{args.overlay_version}.jpg"
        landmarks_json = output_dir / "landmarks_sample.json"
        rep_events_json = output_dir / "rep_events.json"
        exercise_definition_json = output_dir / "exercise_definition.json"
        demo_summary_json = output_dir / "demo_summary.json"
        edge_cases_md = output_dir / "edge_cases.md"

        loop_metadata = create_looped_video(input_video, looped_video, args.loops)
        extractor = PoseExtractor(
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )
        frames, pose_metadata = extractor.process_video(looped_video)
        metadata = {**loop_metadata, **pose_metadata}

        active_info = select_active_arm(frames, args.visibility_threshold)
        warnings = list(active_info.get("warnings", []))
        if metadata["pose_detected_ratio"] < 0.5:
            warnings.append("Pose detection quality is low. Try another video or adjust camera angle.")

        analyze_frames(frames, active_info["active_side"], args.visibility_threshold, args.alpha)
        counter = CurlRepCounter(
            min_rep_duration_sec=args.min_rep_duration_sec,
            min_angle_range_for_valid_rep=args.min_angle_range_for_valid_rep,
        )
        thresholds = counter.fit_thresholds([frame.smoothed_angle for frame in frames])
        apply_rep_counting(counter, frames)
        confidence_stats = calculate_confidence_stats(frames)

        if counter.rep_count < 3:
            warnings.append(
                "Fewer than 3 reps were counted. The source clip may not show enough full curl motion."
            )

        if metadata["pose_detected_ratio"] < 0.25:
            raise RuntimeError("Pose detection quality is low. Try another video or adjust camera angle.")

        generated_files = [
            looped_video,
            overlay_video,
            overlay_video_v2,
            contact_sheet_v2,
            landmarks_json,
            rep_events_json,
            exercise_definition_json,
            demo_summary_json,
            edge_cases_md,
        ]

        render_overlay_video(
            looped_video,
            overlay_video,
            frames,
            active_info["active_side"],
            metadata["fps"],
            metadata["width"],
            metadata["height"],
        )
        if args.v2_overlay:
            render_overlay_video_v2(
                looped_video,
                overlay_video_v2,
                frames,
                active_info["active_side"],
                metadata["fps"],
                metadata["width"],
                metadata["height"],
                thresholds,
                counter.rep_events,
                summary_screen_sec=args.summary_screen_sec,
                title_screen_sec=args.title_screen_sec,
                draw_angle_arc_enabled=args.draw_angle_arc,
                draw_angle_chart_enabled=args.draw_angle_chart,
                contact_sheet_path=contact_sheet_v2,
            )
        export_landmarks_sample(
            landmarks_json,
            frames,
            metadata,
            input_video,
            looped_video,
            active_info,
            args.visibility_threshold,
            counter.rep_events,
        )
        export_rep_events(
            rep_events_json,
            active_info["active_side"],
            thresholds,
            counter.rep_count,
            counter.rep_events,
        )
        export_exercise_definition(
            exercise_definition_json,
            thresholds,
            min_rep_duration_sec=args.min_rep_duration_sec,
            min_angle_range_for_valid_rep=args.min_angle_range_for_valid_rep,
        )
        export_demo_summary(
            demo_summary_json,
            input_video,
            looped_video,
            overlay_video,
            metadata,
            active_info,
            thresholds,
            counter.rep_count,
            generated_files,
            warnings,
            confidence_stats=confidence_stats,
            output_video_v2=overlay_video_v2,
            overlay_version=args.overlay_version,
            tracked_arm_label_value=tracked_arm_label(active_info["active_side"]),
        )
        export_edge_cases(edge_cases_md)

        print_final_summary(
            output_dir,
            overlay_video_v2.name,
            active_info,
            counter.rep_count,
            thresholds=asdict(thresholds),
            confidence_stats=confidence_stats,
            contact_sheet_name=contact_sheet_v2.name,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _resolve_output_dir(project_root: Path, output_arg: str) -> Path:
    output_dir = Path(output_arg).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    return output_dir


def analyze_frames(
    frames: list[PoseFrame],
    active_side: str,
    visibility_threshold: float,
    alpha: float,
) -> None:
    previous: Optional[float] = None
    for frame in frames:
        frame.active_side = active_side
        frame.selected_raw_angle = frame.side_angles.get(active_side)
        visibility = frame.side_visibility.get(active_side, {})
        mean_visibility = visibility.get("mean")
        frame.smoothed_angle = exponential_smoothing(frame.selected_raw_angle, previous, alpha)
        previous = frame.smoothed_angle

        if not frame.pose_detected:
            frame.confidence_status = "NO_POSE"
        elif mean_visibility is None or mean_visibility < visibility_threshold:
            frame.confidence_status = "LOW_VISIBILITY"
        elif frame.selected_raw_angle is None:
            frame.confidence_status = "ANGLE_UNAVAILABLE"
        else:
            frame.confidence_status = "OK"
        frame.pose_status = frame.confidence_status


def apply_rep_counting(counter: CurlRepCounter, frames: list[PoseFrame]) -> None:
    for frame in frames:
        state, count = counter.process_frame(
            frame.smoothed_angle,
            frame.frame_index,
            frame.timestamp_seconds,
        )
        frame.state = state
        frame.rep_count = count
        frame.counter_status = "WAITING_FOR_DOWN" if state == CurlRepCounter.INIT else state


def calculate_confidence_stats(frames: list[PoseFrame]) -> dict:
    total = len(frames) or 1
    statuses = ["OK", "LOW_VISIBILITY", "NO_POSE", "ANGLE_UNAVAILABLE"]
    counts = {status: 0 for status in statuses}
    for frame in frames:
        counts[frame.pose_status] = counts.get(frame.pose_status, 0) + 1
    counter_counts: dict[str, int] = {}
    for frame in frames:
        counter_counts[frame.counter_status] = counter_counts.get(frame.counter_status, 0) + 1
    return {
        "counts": counts,
        "percentages": {
            status: (counts.get(status, 0) / total) * 100.0
            for status in statuses
        },
        "counter_counts": counter_counts,
        "counter_percentages": {
            status: (count / total) * 100.0
            for status, count in counter_counts.items()
        },
    }


def print_final_summary(
    output_dir: Path,
    output_video_name: str,
    active_info: dict,
    counted_reps: int,
    thresholds: dict,
    confidence_stats: dict,
    contact_sheet_name: str,
) -> None:
    def rel(name: str) -> str:
        path = output_dir / name
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)

    print(
        "\n".join(
            [
                "Demo completed.",
                "",
                "Generated files:",
                f"- {rel(output_video_name)}",
                f"- {rel(contact_sheet_name)}",
                f"- {rel('landmarks_sample.json')}",
                f"- {rel('rep_events.json')}",
                f"- {rel('exercise_definition.json')}",
                f"- {rel('demo_summary.json')}",
                f"- {rel('edge_cases.md')}",
                "",
                "Summary:",
                f"- Active side: {active_info['active_side']}",
                f"- Tracked arm label: {tracked_arm_label(active_info['active_side'])}",
                f"- Counted reps: {counted_reps}",
                f"- OK frames: {confidence_stats['percentages'].get('OK', 0.0):.1f}%",
                f"- Mean visibility: {active_info['mean_visibility']:.3f}",
                f"- Angle range: {active_info['angle_range']:.1f}",
                (
                    "- Thresholds: "
                    f"up={thresholds['up_threshold']:.1f}, "
                    f"down={thresholds['down_threshold']:.1f}, "
                    f"reset={thresholds['reset_threshold']:.1f}, "
                    f"method={thresholds['method']}"
                ),
                "- INIT logic: waits for DOWN_READY before counting",
                "- Notes: short source clip looped for demo purposes",
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
