import math

from pcs.geometry.lines import (
    build_line_segment,
    line_angle_rad,
    line_length,
    segment_overlap_ratio_with_rect,
)


def test_line_length_and_angle_are_consistent() -> None:
    assert math.isclose(line_length(0.0, 0.0, 3.0, 4.0), 5.0)
    assert math.isclose(line_angle_rad(0.0, 0.0, 3.0, 4.0), math.atan2(4.0, 3.0))

    vertical = build_line_segment(2.0, 0.0, 2.0, 5.0, confidence=0.9, detector_name="test")
    assert math.isclose(vertical.length, 5.0)
    assert math.isclose(vertical.angle_rad, math.pi / 2.0)


def test_segment_overlap_ratio_with_rect_handles_partial_overlap() -> None:
    segment = build_line_segment(0.0, 0.0, 10.0, 0.0, confidence=1.0, detector_name="test")
    overlap = segment_overlap_ratio_with_rect(segment, (2.0, -1.0, 6.0, 1.0))
    assert math.isclose(overlap, 0.4, rel_tol=1e-6)

