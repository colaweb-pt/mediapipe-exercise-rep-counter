from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def _as_vector(point: Iterable[float]) -> Optional[np.ndarray]:
    try:
        values = np.asarray(list(point), dtype=float)
    except (TypeError, ValueError):
        return None
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        return None
    return values


def calculate_angle_deg(
    a: Iterable[float],
    b: Iterable[float],
    c: Iterable[float],
) -> Optional[float]:
    """Return the angle ABC in degrees, or None if the vectors are invalid."""
    pa = _as_vector(a)
    pb = _as_vector(b)
    pc = _as_vector(c)
    if pa is None or pb is None or pc is None:
        return None

    v1 = pa - pb
    v2 = pc - pb
    len1 = float(np.linalg.norm(v1))
    len2 = float(np.linalg.norm(v2))
    if len1 == 0.0 or len2 == 0.0:
        return None

    cosine = float(np.dot(v1, v2) / (len1 * len2))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def exponential_smoothing(
    current: Optional[float],
    previous: Optional[float],
    alpha: float,
) -> Optional[float]:
    """Simple exponential smoothing for elbow-angle signals."""
    if current is None:
        return previous
    if previous is None:
        return current
    alpha = max(0.0, min(1.0, float(alpha)))
    return alpha * current + (1.0 - alpha) * previous
