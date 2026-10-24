"""
Configuration & Threshold Constants for DrowsyGuard AI.
Defines facial landmark indices, EAR/MAR threshold defaults, and alert states.
"""

from dataclasses import dataclass


# ── MediaPipe 468 Face Mesh Landmark Indices ──────────────────────────
# Left Eye (6 landmark points for EAR calculation)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Right Eye (6 landmark points for EAR calculation)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Mouth / Lips (Inner & outer landmarks for MAR calculation)
MOUTH_TOP_BOTTOM = (13, 14)
MOUTH_LEFT_RIGHT = (78, 308)
MOUTH_INNER_PAIRS = [(82, 312), (87, 317)]

# Key 3D Facial Feature Points for Head Pose Estimation (PnP solver)
HEAD_POSE_POINTS_3D = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

# Generic 3D head model coordinates in millimeters (anthropometric averages)
CANONICAL_3D_HEAD_MODEL = [
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye outer corner
    (225.0, 170.0, -135.0),   # Right eye outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
]


@dataclass
class DetectionThresholds:
    """Configurable kinematic thresholds for drowsiness and fatigue."""
    ear_threshold: float = 0.22         # Eyes closed if EAR < threshold
    ear_consec_frames: int = 18         # ~0.6 sec at 30 FPS indicates micro-sleep
    mar_threshold: float = 0.65         # Yawning if MAR > threshold
    mar_consec_frames: int = 15         # Frames mouth must stay wide open
    head_pitch_down: float = -14.0      # Head nodding off downwards (degrees)
    head_yaw_distracted: float = 24.0   # Looking away left/right (degrees)
    blink_ear_threshold: float = 0.25   # Fast dip below this registers a blink
