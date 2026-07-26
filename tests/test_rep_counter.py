from src.rep_counter import CurlRepCounter


def _counter() -> CurlRepCounter:
    counter = CurlRepCounter(
        min_rep_duration_sec=0.35,
        min_angle_range_for_valid_rep=35.0,
    )
    counter.fit_thresholds([50.0] * 10 + [155.0] * 10)
    return counter


def test_counter_waits_for_down_position() -> None:
    counter = _counter()
    state, count = counter.process_frame(60.0, 0, 0.0)
    assert state == CurlRepCounter.INIT
    assert count == 0


def test_valid_down_to_up_rep_is_counted() -> None:
    counter = _counter()
    counter.process_frame(155.0, 0, 0.0)
    state, count = counter.process_frame(50.0, 12, 0.5)
    assert state == CurlRepCounter.UP
    assert count == 1
    assert counter.rep_events[-1].valid is True


def test_rep_below_minimum_duration_is_rejected() -> None:
    counter = _counter()
    counter.process_frame(155.0, 0, 0.0)
    _, count = counter.process_frame(50.0, 2, 0.1)
    assert count == 0
    assert counter.rep_events[-1].valid is False
    assert "duration below" in counter.rep_events[-1].notes


def test_rep_below_minimum_angle_range_is_rejected() -> None:
    counter = CurlRepCounter(
        min_rep_duration_sec=0.1,
        min_angle_range_for_valid_rep=120.0,
    )
    counter.fit_thresholds([50.0] * 10 + [155.0] * 10)
    counter.process_frame(155.0, 0, 0.0)
    _, count = counter.process_frame(50.0, 12, 0.5)
    assert count == 0
    assert counter.rep_events[-1].valid is False
    assert "angle range below" in counter.rep_events[-1].notes
