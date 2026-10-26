"""
Fatigue State Machine & Multi-Level Alerting Engine.

Tracks consecutive eye closure, yawn duration, and distraction frames,
triggering in-browser synthesized alarms (AudioContext) and vocal safety warnings.
"""

from enum import Enum
import time
from typing import Optional
import streamlit.components.v1 as components

from config import DetectionThresholds


class DriverStatus(str, Enum):
    NORMAL = "ALERT & ATTENTIVE"
    YAWNING = "YAWNING DETECTED"
    DROWSY = "DROWSINESS / MICRO-SLEEP"
    DISTRACTED = "DISTRACTION DETECTED"
    NO_FACE = "NO DRIVER IN VIEW"


class AlertEngine:
    """
    Finite state machine for driver fatigue evaluation and audio alarms.
    """

    def __init__(self, thresholds: Optional[DetectionThresholds] = None) -> None:
        self.thresholds = thresholds or DetectionThresholds()

        # State tracking counters
        self.closed_eye_frames: int = 0
        self.yawn_frames: int = 0
        self.distraction_frames: int = 0
        self.total_blinks: int = 0
        self.total_yawns: int = 0
        self.total_distractions: int = 0

        # Timing & history
        self.start_time: float = time.time()
        self.incident_log: list[dict] = []
        self._last_state: DriverStatus = DriverStatus.NORMAL
        self._last_alert_time: float = 0.0

    def update(
        self,
        avg_ear: float,
        mar: float,
        pitch: float,
        yaw: float,
        face_detected: bool,
    ) -> tuple[DriverStatus, bool]:
        """
        Evaluate current frame metrics and return (DriverStatus, trigger_alarm_boolean).
        """
        if not face_detected:
            return DriverStatus.NO_FACE, False

        trigger_alarm = False
        current_status = DriverStatus.NORMAL

        # 1. Eye state check (EAR)
        if avg_ear < self.thresholds.ear_threshold:
            self.closed_eye_frames += 1
            if self.closed_eye_frames >= self.thresholds.ear_consec_frames:
                current_status = DriverStatus.DROWSY
                trigger_alarm = True
        else:
            # Check if this was a fast blink
            if 2 <= self.closed_eye_frames < self.thresholds.ear_consec_frames:
                self.total_blinks += 1
            self.closed_eye_frames = 0

        # 2. Mouth state check (MAR / Yawn)
        if mar > self.thresholds.mar_threshold:
            self.yawn_frames += 1
            if self.yawn_frames >= self.thresholds.mar_consec_frames:
                current_status = DriverStatus.YAWNING
                if self.yawn_frames == self.thresholds.mar_consec_frames:
                    self.total_yawns += 1
                    self._record_incident("YAWN", f"Yawn detected (MAR: {mar})")
        else:
            self.yawn_frames = 0

        # 3. Head Pose check (Distraction / Nodding)
        if pitch < self.thresholds.head_pitch_down or abs(yaw) > self.thresholds.head_yaw_distracted:
            self.distraction_frames += 1
            if self.distraction_frames >= 20:
                current_status = DriverStatus.DISTRACTED
                trigger_alarm = True
                if self.distraction_frames == 20:
                    self.total_distractions += 1
                    self._record_incident("DISTRACTION", f"Driver looking away (Pitch: {pitch}°, Yaw: {yaw}°)")
        else:
            self.distraction_frames = 0

        # Record drowsiness incidents
        if current_status == DriverStatus.DROWSY and self._last_state != DriverStatus.DROWSY:
            self._record_incident("MICRO_SLEEP", f"Micro-sleep detected! (EAR: {avg_ear})")

        self._last_state = current_status
        return current_status, trigger_alarm

    def _record_incident(self, incident_type: str, details: str) -> None:
        """Store timestamped driver safety incidents."""
        elapsed_sec = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed_sec, 60)
        self.incident_log.append(
            {
                "time": f"{mins:02d}:{secs:02d}",
                "type": incident_type,
                "details": details,
                "timestamp": time.time(),
            }
        )

    def get_summary(self) -> dict:
        """Fetch trip summary statistics."""
        elapsed = time.time() - self.start_time
        mins = max(1.0, elapsed / 60.0)
        return {
            "total_blinks": self.total_blinks,
            "blink_rate_bpm": round(self.total_blinks / mins, 1),
            "total_yawns": self.total_yawns,
            "total_distractions": self.total_distractions,
            "total_incidents": len(self.incident_log),
        }


def trigger_browser_alarm(status: DriverStatus) -> None:
    """
    Synthesize an emergency acoustic siren in the user's browser via Web Audio API.
    Zero external audio files needed; plays directly on laptop/phone speakers.
    """
    if status not in (DriverStatus.DROWSY, DriverStatus.DISTRACTED):
        return

    freq = 880 if status == DriverStatus.DROWSY else 660
    msg = "WAKE UP! EYES ON THE ROAD!" if status == DriverStatus.DROWSY else "ATTENTION! KEEP EYES ON ROAD!"

    js_code = f"""
    <script>
    (function() {{
        try {{
            var ctx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();

            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime({freq}, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime({freq * 1.5}, ctx.currentTime + 0.15);
            osc.frequency.exponentialRampToValueAtTime({freq}, ctx.currentTime + 0.3);

            gain.gain.setValueAtTime(0.5, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start();
            osc.stop(ctx.currentTime + 0.5);

            // Speech alert
            if ('speechSynthesis' in window) {{
                var u = new SpeechSynthesisUtterance("{msg}");
                u.rate = 1.2;
                u.pitch = 1.1;
                window.speechSynthesis.speak(u);
            }}
        }} catch(e) {{
            console.error(e);
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)
