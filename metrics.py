"""
Kinematic Metrics & Computer Vision Geometry Solvers.

Calculates Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
and 3D Head Pose Euler Angles (Pitch, Yaw, Roll) via cv2.solvePnP.
"""

import math
from typing import Optional
import cv2
import numpy as np

from config import CANONICAL_3D_HEAD_MODEL


def euclidean_dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Compute 2D Euclidean distance between two spatial points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(eye_points: list[tuple[float, float]]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) based on Soukupová and Čech (2016).

    Parameters
    ----------
    eye_points : list of 6 (x, y) tuples ordered:
        [P1 (outer), P2 (top-left), P3 (top-right), P4 (inner), P5 (bottom-right), P6 (bottom-left)]

    Returns
    -------
    float: EAR value (typically 0.28-0.35 when open, <0.20 when closed).
    """
    if len(eye_points) < 6:
        return 0.30

    # Vertical eye distances
    a = euclidean_dist(eye_points[1], eye_points[5])
    b = euclidean_dist(eye_points[2], eye_points[4])

    # Horizontal eye distance
    c = euclidean_dist(eye_points[0], eye_points[3])

    if c < 1e-6:
        return 0.30

    ear = (a + b) / (2.0 * c)
    return round(float(ear), 4)


def calculate_mar(
    top: tuple[float, float],
    bottom: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) to detect yawning.

    Parameters
    ----------
    top, bottom : (x, y) coordinates of upper and lower lip centers.
    left, right : (x, y) coordinates of mouth corners.

    Returns
    -------
    float: MAR value (typically 0.15-0.35 closed/speaking, >0.65 yawning).
    """
    vertical = euclidean_dist(top, bottom)
    horizontal = euclidean_dist(left, right)

    if horizontal < 1e-6:
        return 0.20

    mar = vertical / horizontal
    return round(float(mar), 4)


def estimate_head_pose(
    points_2d: list[tuple[float, float]], frame_shape: tuple[int, int]
) -> tuple[float, float, float]:
    """
    Solve Perspective-n-Point (PnP) problem to derive 3D head orientation.

    Parameters
    ----------
    points_2d : list of 6 2D coordinates corresponding to CANONICAL_3D_HEAD_MODEL:
        [nose_tip, chin, left_eye_outer, right_eye_outer, left_mouth, right_mouth]
    frame_shape : (height, width) of the video frame.

    Returns
    -------
    tuple[float, float, float]: (pitch, yaw, roll) in degrees.
    """
    h, w = frame_shape[:2]

    # Camera intrinsic matrix approximation
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Assume minimal lens distortion

    model_points = np.array(CANONICAL_3D_HEAD_MODEL, dtype=np.float64)
    image_points = np.array(points_2d, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix into Euler angles
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0

    # Convert radians to degrees
    pitch_deg = round(float(math.degrees(pitch)), 1)
    yaw_deg = round(float(math.degrees(yaw)), 1)
    roll_deg = round(float(math.degrees(roll)), 1)

    return pitch_deg, yaw_deg, roll_deg


def calculate_vigilance_index(
    avg_ear: float,
    mar: float,
    pitch: float,
    yaw: float,
    closed_frames: int,
    yawn_count: int,
) -> int:
    """
    Compute a composite 0-100% Driver Vigilance & Alertness score.
    """
    score = 100.0

    # Penalize eye closure duration
    if closed_frames > 5:
        score -= min(45.0, closed_frames * 2.5)

    # Penalize yawning
    if mar > 0.60:
        score -= 20.0
    score -= min(25.0, yawn_count * 5.0)

    # Penalize head tilting down (nodding off)
    if pitch < -10.0:
        score -= min(30.0, abs(pitch + 10.0) * 2.0)

    # Penalize distraction (looking left/right away from road)
    if abs(yaw) > 20.0:
        score -= min(25.0, (abs(yaw) - 20.0) * 1.5)

    return max(0, min(100, int(score)))
