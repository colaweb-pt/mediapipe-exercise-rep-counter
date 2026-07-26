import math

from src.geometry import calculate_angle_deg, exponential_smoothing


def test_calculate_right_angle() -> None:
    angle = calculate_angle_deg((1, 0), (0, 0), (0, 1))
    assert angle is not None
    assert math.isclose(angle, 90.0, abs_tol=1e-6)


def test_calculate_straight_angle() -> None:
    angle = calculate_angle_deg((-1, 0), (0, 0), (1, 0))
    assert angle is not None
    assert math.isclose(angle, 180.0, abs_tol=1e-6)


def test_calculate_angle_rejects_zero_length_vector() -> None:
    assert calculate_angle_deg((0, 0), (0, 0), (1, 0)) is None


def test_exponential_smoothing() -> None:
    assert exponential_smoothing(100.0, 80.0, 0.25) == 85.0
    assert exponential_smoothing(100.0, None, 0.25) == 100.0
    assert exponential_smoothing(None, 80.0, 0.25) == 80.0
