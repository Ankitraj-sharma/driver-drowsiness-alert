"""
WebRTC Real-Time Video Processor for Driver Drowsiness Monitoring.

Handles video streaming from browser webcam, updates kinematic states,
and renders an automotive augmented reality heads-up display.
"""

import threading
import time
from typing import Optional

import av
import cv2
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase

from alert_engine import AlertEngine, DriverStatus
from config import DetectionThresholds
from face_tracker import FaceTracker
from metrics import calculate_vigilance_index

# Cloud-ready STUN server configuration for NAT traversal
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
        ]
    }
)


class LiveDrowsinessProcessor(VideoProcessorBase):
    """
    Video processor that runs face mesh tracking, calculates EAR/MAR/Pose,
    and updates the driver alert state machine frame-by-frame.
    """

    def __init__(self) -> None:
        self.tracker = FaceTracker(max_num_faces=1)
        self.alert_engine = AlertEngine()

        # Thread synchronization
        self._lock = threading.Lock()
        self.fps: float = 0.0
        self.current_ear: float = 0.30
        self.current_mar: float = 0.20
        self.current_pitch: float = 0.0
        self.current_yaw: float = 0.0
        self.vigilance_score: int = 100
        self.status: DriverStatus = DriverStatus.NORMAL
        self.should_alarm: bool = False

        # Performance monitoring
        self._frame_count: int = 0
        self._last_fps_time: float = time.time()

    def update_thresholds(self, thresholds: DetectionThresholds) -> None:
        """Update detection thresholds dynamically from UI."""
        self.alert_engine.thresholds = thresholds

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process incoming webcam frame and overlay cockpit HUD."""
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]

        # Calculate FPS
        now = time.time()
        self._frame_count += 1
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._last_fps_time = now

        # Run face tracking and landmark extraction
        annotated, metrics_data = self.tracker.process_frame(img, draw=True)

        if metrics_data and metrics_data.get("face_detected"):
            ear = metrics_data["avg_ear"]
            mar = metrics_data["mar"]
            pitch = metrics_data["pitch"]
            yaw = metrics_data["yaw"]

            # Update state machine
            status, trigger_alarm = self.alert_engine.update(
                ear, mar, pitch, yaw, face_detected=True
            )

            # Compute composite vigilance index
            v_score = calculate_vigilance_index(
                ear,
                mar,
                pitch,
                yaw,
                self.alert_engine.closed_eye_frames,
                self.alert_engine.total_yawns,
            )

            # Render Cockpit HUD
            self._draw_cockpit_hud(annotated, status, ear, mar, pitch, yaw, v_score, w, h)

            with self._lock:
                self.current_ear = ear
                self.current_mar = mar
                self.current_pitch = pitch
                self.current_yaw = yaw
                self.vigilance_score = v_score
                self.status = status
                self.should_alarm = trigger_alarm
        else:
            self.alert_engine.update(0.3, 0.2, 0, 0, face_detected=False)
            self._draw_cockpit_hud(
                annotated, DriverStatus.NO_FACE, 0.0, 0.0, 0.0, 0.0, 50, w, h
            )
            with self._lock:
                self.status = DriverStatus.NO_FACE
                self.should_alarm = False

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _draw_cockpit_hud(
        self,
        img,
        status: DriverStatus,
        ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        v_score: int,
        w: int,
        h: int,
    ) -> None:
        """Render automotive telemetry dashboard onto video frame."""
        # Top banner with status color
        color_map = {
            DriverStatus.NORMAL: (0, 200, 100),       # Green
            DriverStatus.YAWNING: (0, 215, 255),      # Amber
            DriverStatus.DROWSY: (0, 0, 255),         # Bright Red
            DriverStatus.DISTRACTED: (0, 128, 255),   # Orange-Red
            DriverStatus.NO_FACE: (128, 128, 128),    # Gray
        }
        theme_color = color_map.get(status, (0, 200, 100))

        # Translucent top telemetry ribbon
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (11, 15, 25), -1)
        cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

        # Status accent bar
        cv2.line(img, (0, 80), (w, 80), theme_color, 3)

        # Status text
        cv2.putText(
            img,
            f"STATUS: {status.value}",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            theme_color,
            2,
            cv2.LINE_AA,
        )

        # Telemetry metrics on top right
        info_str = f"EAR: {ear:.2f} | MAR: {mar:.2f} | VIGILANCE: {v_score}%"
        cv2.putText(
            img,
            info_str,
            (w - 480, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (220, 230, 240),
            1,
            cv2.LINE_AA,
        )

        # Bottom subtle telemetry bar
        cv2.rectangle(img, (20, h - 35), (w - 20, h - 10), (15, 20, 35), -1)
        pose_str = f"Head Pose: Pitch {pitch:+.1f}° | Yaw {yaw:+.1f}° | Camera Feed: {self.fps} FPS"
        cv2.putText(
            img,
            pose_str,
            (30, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (140, 160, 180),
            1,
            cv2.LINE_AA,
        )

    def get_telemetry(self) -> dict:
        """Safely fetch pipeline states for Streamlit UI."""
        with self._lock:
            summary = self.alert_engine.get_summary()
            return {
                "status": self.status,
                "should_alarm": self.should_alarm,
                "ear": self.current_ear,
                "mar": self.current_mar,
                "pitch": self.current_pitch,
                "yaw": self.current_yaw,
                "vigilance_score": self.vigilance_score,
                "fps": self.fps,
                "summary": summary,
                "incidents": list(self.alert_engine.incident_log),
            }
