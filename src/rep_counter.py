from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


@dataclass
class Thresholds:
    up_threshold: float
    down_threshold: float
    reset_threshold: float
    method: str
    observed_min_angle: Optional[float]
    observed_max_angle: Optional[float]
    observed_range: Optional[float]


@dataclass
class RepEvent:
    rep_number: int
    start_frame: Optional[int]
    counted_frame: int
    reset_frame: Optional[int]
    timestamp_seconds: float
    min_angle_during_rep: Optional[float]
    max_angle_during_rep: Optional[float]
    valid: bool
    notes: str
    initialized_from_down_position: bool = True
    duration_sec: Optional[float] = None
    start_down_frame: Optional[int] = None
    counted_up_frame: Optional[int] = None
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class CurlRepCounter:
    INIT = "INIT"
    UP = "UP"
    DOWN_READY = "DOWN_READY"

    def __init__(
        self,
        min_rep_duration_sec: float = 0.35,
        min_angle_range_for_valid_rep: float = 35.0,
    ) -> None:
        self.thresholds: Optional[Thresholds] = None
        self.state = self.INIT
        self.initialized = False
        self.rep_count = 0
        self.rep_events: list[RepEvent] = []
        self.current_rep_start_frame: Optional[int] = None
        self.current_rep_start_time: Optional[float] = None
        self.current_rep_angles: list[float] = []
        self.last_event_index: Optional[int] = None
        self.min_rep_duration_sec = float(min_rep_duration_sec)
        self.min_angle_range_for_valid_rep = float(min_angle_range_for_valid_rep)

    def fit_thresholds(self, angles: list[Optional[float]]) -> Thresholds:
        valid = np.asarray([angle for angle in angles if angle is not None], dtype=float)
        if valid.size >= 10:
            min_angle = float(np.percentile(valid, 10))
            max_angle = float(np.percentile(valid, 90))
            observed_range = max_angle - min_angle
        else:
            min_angle = None
            max_angle = None
            observed_range = None

        if observed_range is not None and observed_range >= 35.0:
            thresholds = Thresholds(
                up_threshold=min_angle + 0.30 * observed_range,
                down_threshold=max_angle - 0.20 * observed_range,
                reset_threshold=min_angle + 0.65 * observed_range,
                method="adaptive_percentile",
                observed_min_angle=min_angle,
                observed_max_angle=max_angle,
                observed_range=observed_range,
            )
        else:
            thresholds = Thresholds(
                up_threshold=70.0,
                down_threshold=145.0,
                reset_threshold=120.0,
                method="static_fallback",
                observed_min_angle=min_angle,
                observed_max_angle=max_angle,
                observed_range=observed_range,
            )
        self.thresholds = thresholds
        return thresholds

    def process_frame(
        self,
        angle: Optional[float],
        frame_index: int,
        timestamp_seconds: float,
    ) -> tuple[str, int]:
        if self.thresholds is None:
            raise RuntimeError("Call fit_thresholds before process_frame.")
        if angle is None:
            return self.state, self.rep_count

        if self.state == self.INIT:
            if angle >= self.thresholds.down_threshold:
                self.initialized = True
                self.state = self.DOWN_READY
                self.current_rep_start_frame = frame_index
                self.current_rep_start_time = timestamp_seconds
                self.current_rep_angles = [angle]
            return self.state, self.rep_count

        self.current_rep_angles.append(angle)

        if self.state == self.DOWN_READY and angle <= self.thresholds.up_threshold:
            self._finish_rep_attempt(
                frame_index,
                timestamp_seconds,
                "Initialized from down position; down-to-up transition detected.",
            )
            self.state = self.UP
            return self.state, self.rep_count

        if self.state == self.UP and angle >= self.thresholds.reset_threshold:
            self.state = self.DOWN_READY
            if self.last_event_index is not None:
                self.rep_events[self.last_event_index].reset_frame = frame_index
            self.current_rep_start_frame = frame_index
            self.current_rep_start_time = timestamp_seconds
            self.current_rep_angles = [angle]
            return self.state, self.rep_count

        return self.state, self.rep_count

    def _finish_rep_attempt(self, frame_index: int, timestamp_seconds: float, notes: str) -> None:
        valid_angles = self.current_rep_angles or []
        min_angle = min(valid_angles) if valid_angles else None
        max_angle = max(valid_angles) if valid_angles else None
        duration_sec = (
            timestamp_seconds - self.current_rep_start_time
            if self.current_rep_start_time is not None
            else None
        )
        angle_range = (
            max_angle - min_angle
            if min_angle is not None and max_angle is not None
            else None
        )
        valid = True
        event_notes = [notes]
        if duration_sec is None or duration_sec < self.min_rep_duration_sec:
            valid = False
            event_notes.append(
                f"Invalid: duration below {self.min_rep_duration_sec:.2f}s minimum."
            )
        if angle_range is None or angle_range < self.min_angle_range_for_valid_rep:
            valid = False
            event_notes.append(
                f"Invalid: angle range below {self.min_angle_range_for_valid_rep:.1f} degree minimum."
            )

        if valid:
            self.rep_count += 1
            rep_number = self.rep_count
        else:
            rep_number = self.rep_count + 1

        event = RepEvent(
            rep_number=rep_number,
            start_frame=self.current_rep_start_frame,
            counted_frame=frame_index,
            reset_frame=None,
            timestamp_seconds=timestamp_seconds,
            min_angle_during_rep=min_angle,
            max_angle_during_rep=max_angle,
            valid=valid,
            notes=" ".join(event_notes),
            initialized_from_down_position=self.initialized,
            duration_sec=duration_sec,
            start_down_frame=self.current_rep_start_frame,
            counted_up_frame=frame_index,
            min_angle=min_angle,
            max_angle=max_angle,
        )
        self.rep_events.append(event)
        self.last_event_index = len(self.rep_events) - 1
