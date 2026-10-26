"""
Unit tests for geometric metric calculations (EAR, MAR, solvePnP).
"""

import math
from metrics import calculate_ear, calculate_mar, euclidean_dist


def test_euclidean_distance():
    assert euclidean_dist((0, 0), (3, 4)) == 5.0
    assert euclidean_dist((1, 1), (1, 1)) == 0.0


def test_ear_calculation():
    # Synthetic open eye coordinates
    open_eye = [
        (0.0, 10.0),   # P1 (outer)
        (3.0, 14.0),   # P2 (top-left)
        (7.0, 14.0),   # P3 (top-right)
        (10.0, 10.0),  # P4 (inner)
        (7.0, 6.0),    # P5 (bottom-right)
        (3.0, 6.0),    # P6 (bottom-left)
    ]
    ear_open = calculate_ear(open_eye)
    assert ear_open > 0.25, f"Expected open EAR > 0.25, got {ear_open}"

    # Synthetic closed eye coordinates (vertical height near zero)
    closed_eye = [
        (0.0, 10.0),
        (3.0, 10.2),
        (7.0, 10.2),
        (10.0, 10.0),
        (7.0, 9.8),
        (3.0, 9.8),
    ]
    ear_closed = calculate_ear(closed_eye)
    assert ear_closed < 0.15, f"Expected closed EAR < 0.15, got {ear_closed}"


def test_mar_calculation():
    # Normal closed/relaxed mouth
    mar_normal = calculate_mar((5.0, 8.0), (5.0, 6.0), (0.0, 7.0), (10.0, 7.0))
    assert mar_normal < 0.35, f"Expected normal MAR < 0.35, got {mar_normal}"

    # Wide yawning mouth
    mar_yawn = calculate_mar((5.0, 15.0), (5.0, 5.0), (1.0, 10.0), (9.0, 10.0))
    assert mar_yawn > 0.65, f"Expected yawning MAR > 0.65, got {mar_yawn}"


if __name__ == "__main__":
    test_euclidean_distance()
    test_ear_calculation()
    test_mar_calculation()
    print("All metric tests passed successfully!")
