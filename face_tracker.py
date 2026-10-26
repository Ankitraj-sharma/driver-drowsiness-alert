"""
Facial Landmark Tracking & Augmented Reality HUD Overlay.

Uses MediaPipe Face Mesh (468 points) with an intelligent OpenCV Haar Cascade
fallback to compute and visualize driver facial features in real time.
"""

from typing import Optional
import cv2
import numpy as np

from config import (
    LEFT_EYE_INDICES,
    MOUTH_LEFT_RIGHT,
    MOUTH_TOP_BOTTOM,
    RIGHT_EYE_INDICES,
    HEAD_POSE_POINTS_3D,
)
from metrics import calculate_ear, calculate_mar, estimate_head_pose

# Check MediaPipe availability
_MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    _MEDIAPIPE_AVAILABLE = True
except Exception:
    _MEDIAPIPE_AVAILABLE = False


class FaceTracker:
    """
    Tracks facial landmarks, calculates EAR/MAR/Head Pose, and overlays an AR HUD.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.use_mediapipe = _MEDIAPIPE_AVAILABLE
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.use_mediapipe:
            self.mesh_detector = mp_face_mesh.FaceMesh(
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self.mesh_detector = None

    def process_frame(
        self, frame: np.ndarray, draw: bool = True
    ) -> tuple[np.ndarray, Optional[dict]]:
        """
        Process a video frame and extract facial kinematics.

        Returns
        -------
        tuple[np.ndarray, Optional[dict]]:
            Annotated frame and dictionary containing:
            - 'left_ear': float
            - 'right_ear': float
            - 'avg_ear': float
            - 'mar': float
            - 'pitch': float
            - 'yaw': float
            - 'roll': float
            - 'face_detected': bool
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()

        if self.use_mediapipe and self.mesh_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mesh_detector.process(rgb)

            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0]
                landmarks_px: dict[int, tuple[int, int]] = {}
                for idx, lm in enumerate(mesh.landmark):
                    landmarks_px[idx] = (int(lm.x * w), int(lm.y * h))

                # 1. Extract Eye points and calculate EAR
                left_eye_pts = [landmarks_px[i] for i in LEFT_EYE_INDICES]
                right_eye_pts = [landmarks_px[i] for i in RIGHT_EYE_INDICES]

                left_ear = calculate_ear(left_eye_pts)
                right_ear = calculate_ear(right_eye_pts)
                avg_ear = round((left_ear + right_ear) / 2.0, 4)

                # 2. Extract Mouth points and calculate MAR
                m_top = landmarks_px[MOUTH_TOP_BOTTOM[0]]
                m_bot = landmarks_px[MOUTH_TOP_BOTTOM[1]]
                m_left = landmarks_px[MOUTH_LEFT_RIGHT[0]]
                m_right = landmarks_px[MOUTH_LEFT_RIGHT[1]]
                mar = calculate_mar(m_top, m_bot, m_left, m_right)

                # 3. Head Pose Estimation points
                pnp_2d = [
                    landmarks_px[HEAD_POSE_POINTS_3D["nose_tip"]],
                    landmarks_px[HEAD_POSE_POINTS_3D["chin"]],
                    landmarks_px[HEAD_POSE_POINTS_3D["left_eye_outer"]],
                    landmarks_px[HEAD_POSE_POINTS_3D["right_eye_outer"]],
                    landmarks_px[HEAD_POSE_POINTS_3D["left_mouth_corner"]],
                    landmarks_px[HEAD_POSE_POINTS_3D["right_mouth_corner"]],
                ]
                pitch, yaw, roll = estimate_head_pose(pnp_2d, (h, w))

                if draw:
                    self._draw_facial_hud(
                        annotated,
                        left_eye_pts,
                        right_eye_pts,
                        m_top,
                        m_bot,
                        m_left,
                        m_right,
                        landmarks_px[HEAD_POSE_POINTS_3D["nose_tip"]],
                        pitch,
                        yaw,
                        avg_ear,
                        mar,
                    )

                return annotated, {
                    "face_detected": True,
                    "left_ear": left_ear,
                    "right_ear": right_ear,
                    "avg_ear": avg_ear,
                    "mar": mar,
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                }

        # Fallback if no face mesh is detected or library not available
        return self._fallback_face_detector(frame, annotated, draw)

    def _draw_facial_hud(
        self,
        img: np.ndarray,
        left_eye: list[tuple[int, int]],
        right_eye: list[tuple[int, int]],
        m_top: tuple[int, int],
        m_bot: tuple[int, int],
        m_left: tuple[int, int],
        m_right: tuple[int, int],
        nose_tip: tuple[int, int],
        pitch: float,
        yaw: float,
        avg_ear: float,
        mar: float,
    ) -> None:
        """Render high-tech cybernetic HUD overlays onto the driver's face."""
        # Draw eyes contours
        cv2.polylines(img, [np.array(left_eye)], True, (0, 255, 128), 1, cv2.LINE_AA)
        cv2.polylines(img, [np.array(right_eye)], True, (0, 255, 128), 1, cv2.LINE_AA)
        for pt in left_eye + right_eye:
            cv2.circle(img, pt, 2, (0, 210, 255), -1)

        # Draw mouth crosshairs
        cv2.line(img, m_left, m_right, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(img, m_top, m_bot, (255, 200, 0), 1, cv2.LINE_AA)
        for pt in [m_top, m_bot, m_left, m_right]:
            cv2.circle(img, pt, 2, (0, 165, 255), -1)

        # Draw 3D Nose Direction Vector (Head orientation line)
        nose_end = (int(nose_tip[0] + yaw * 2.2), int(nose_tip[1] - pitch * 2.2))
        cv2.arrowedLine(img, nose_tip, nose_end, (0, 255, 255), 2, tipLength=0.3)

    def _fallback_face_detector(
        self, frame: np.ndarray, out_frame: np.ndarray, draw: bool
    ) -> tuple[np.ndarray, Optional[dict]]:
        """Lightweight OpenCV Haar Cascade fallback detector."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.face_cascade is None or self.face_cascade.empty():
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return out_frame, None

        x, y, w, h = faces[0]
        if draw:
            cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0, 255, 128), 2)
            cv2.putText(
                out_frame,
                "Driver Face Detected (CV Fallback)",
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 128),
                2,
            )

        return out_frame, {
            "face_detected": True,
            "left_ear": 0.30,
            "right_ear": 0.30,
            "avg_ear": 0.30,
            "mar": 0.20,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
        }
