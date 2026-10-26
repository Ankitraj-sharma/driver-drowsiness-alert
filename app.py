"""
DrowsyGuard AI — Real-Time Driver Drowsiness, Yawn & Distraction Monitoring System
===================================================================================
A production-grade automotive computer vision safety application that monitors
driver alertness via Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), 3D Head Pose
estimation (solvePnP), and triggers real-time acoustic sirens and vocal warnings.

Author : Automotive AI & Computer Vision Safety Team
Stack  : Streamlit · MediaPipe · OpenCV · WebRTC · Web Audio API · Plotly
"""

from collections import deque
import time
import pandas as pd
from PIL import Image
import streamlit as st

from alert_engine import AlertEngine, DriverStatus, trigger_browser_alarm
from analytics import ear_mar_time_series, incidents_dataframe, vigilance_gauge
from calibration import DriverCalibrator
from config import DetectionThresholds
from face_tracker import FaceTracker
from live_stream import RTC_CONFIGURATION, LiveDrowsinessProcessor
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ═══════════════════════════════════════════════════════════════════════
#  Page Setup & Cockpit Cyberpunk Dark Theme
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DrowsyGuard AI · Driver Safety & Fatigue Monitor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Cockpit Hero Header ────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0A0F1D 0%, #151F33 50%, #0D1629 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 75, 75, 0.35);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .hero-banner h1 {
        color: #FF4B4B;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #9FB3C8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
    }

    /* ── Telemetry KPI Card ────────────────────────────────────────── */
    .telemetry-card {
        background: linear-gradient(145deg, #131C2E 0%, #0E1626 100%);
        border: 1px solid rgba(255, 75, 75, 0.25);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .telemetry-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 75, 75, 0.6);
    }
    .telemetry-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FF4B4B;
    }
    .telemetry-label {
        font-size: 0.82rem;
        color: #829AB1;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Status Card */
    .status-alert {
        background: rgba(0, 210, 186, 0.1);
        border: 2px solid #00D2BA;
        color: #00D2BA;
    }
    .status-drowsy {
        background: rgba(255, 71, 87, 0.15);
        border: 2px solid #FF4757;
        color: #FF4757;
        animation: pulse 1s infinite alternate;
    }
    .status-yawn {
        background: rgba(251, 197, 49, 0.15);
        border: 2px solid #FBC531;
        color: #FBC531;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  Session State Initialization
# ═══════════════════════════════════════════════════════════════════════
if "ear_history" not in st.session_state:
    st.session_state.ear_history = deque([0.30] * 50, maxlen=60)
if "mar_history" not in st.session_state:
    st.session_state.mar_history = deque([0.20] * 50, maxlen=60)
if "session_incidents" not in st.session_state:
    st.session_state.session_incidents = []
if "standalone_tracker" not in st.session_state:
    st.session_state.standalone_tracker = FaceTracker()
if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = 0.0

# ═══════════════════════════════════════════════════════════════════════
#  Hero Header
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="hero-banner">
        <h1>🚗 DrowsyGuard AI — Driver Drowsiness & Safety System</h1>
        <p>Continuous AI-powered automotive safety monitoring using Eye Aspect Ratio (EAR),
        Mouth Aspect Ratio (MAR), 3D Head Pose kinematics, and instant acoustic alarms.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  Sidebar Cockpit Settings & Calibration
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Cockpit Calibration")
    st.markdown("---")

    preset = st.selectbox(
        "🎛️ Driving Condition Preset",
        ["Standard Daylight", "Night / Low-Light", "Glasses / Spectacles"],
        index=0,
    )

    # Defaults based on preset
    default_ear = 0.20 if preset == "Glasses / Spectacles" else (0.24 if preset == "Night / Low-Light" else 0.22)
    default_mar = 0.65

    ear_thresh = st.slider(
        "👁️ EAR Micro-Sleep Threshold",
        min_value=0.15,
        max_value=0.32,
        value=default_ear,
        step=0.01,
        help="Eyes are considered closed if EAR falls below this value.",
    )

    ear_frames = st.slider(
        "⏱️ Closed Eye Consecutive Frames",
        min_value=8,
        max_value=30,
        value=16,
        help="Number of consecutive frames below threshold to trigger micro-sleep alarm (~0.5s at 30fps).",
    )

    mar_thresh = st.slider(
        "👄 MAR Yawn Threshold",
        min_value=0.50,
        max_value=0.85,
        value=default_mar,
        step=0.02,
        help="Mouth opening ratio considered as an active yawning event.",
    )

    pitch_thresh = st.slider(
        "📐 Head Nodding Pitch Limit (°)",
        min_value=-25.0,
        max_value=-8.0,
        value=-14.0,
        step=1.0,
        help="Negative pitch angle indicating head dropping forward (nodding off).",
    )

    yaw_thresh = st.slider(
        "👀 Head Distraction Yaw Limit (°)",
        min_value=15.0,
        max_value=35.0,
        value=24.0,
        step=1.0,
        help="Degrees turned away left or right from the forward windshield.",
    )

    enable_audio = st.checkbox("🔊 Web Audio Siren Buzzer", value=True)
    enable_voice = st.checkbox("🗣️ Spoken Voice Safety Alerts", value=True)
    enable_night_ir = st.checkbox("🌙 Night Vision (NIR Contrast Mode)", value=False)

    thresholds = DetectionThresholds(
        ear_threshold=ear_thresh,
        ear_consec_frames=ear_frames,
        mar_threshold=mar_thresh,
        head_pitch_down=pitch_thresh,
        head_yaw_distracted=yaw_thresh,
    )

    st.markdown("---")
    if st.button("🗑️ Reset Safety Trip Log", use_container_width=True):
        st.session_state.session_incidents = []
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════
#  Application Tabs
# ═══════════════════════════════════════════════════════════════════════
tab_live, tab_snap, tab_telemetry, tab_calibrate, tab_about = st.tabs(
    [
        "🔴 Live Driver Monitor",
        "📸 Snapshot Inspection",
        "📊 Trip Safety Telemetry",
        "🎛️ Sensitivity Calibration",
        "ℹ️ Research & Mathematical Principles",
    ]
)

# ─────────────────────────────────────────────────────────────────────
#  TAB 1 : Real-Time Live Driver Monitor (Zero Upload Needed!)
# ─────────────────────────────────────────────────────────────────────
with tab_live:
    st.markdown("### 🔴 Real-Time Driver Fatigue & Distraction Monitor")
    st.markdown(
        """
        <div style="background: rgba(255, 75, 75, 0.08); border-left: 4px solid #FF4B4B; padding: 12px 18px; border-radius: 8px; margin-bottom: 20px;">
            <b style="color: #FF4B4B; font-size: 1.05rem;">✨ Continuous Live Video — Zero File Uploads Needed!</b><br>
            Click <b>START</b> below to begin monitoring. Close your eyes for ~1 second or yawn to trigger the live acoustic buzzer and safety alerts!
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_video, col_telemetry = st.columns([1.5, 1.0])

    with col_video:
        ctx = webrtc_streamer(
            key="drowsyguard-live-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=LiveDrowsinessProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            ctx.video_processor.update_thresholds(thresholds)
            ctx.video_processor.night_mode = enable_night_ir

    with col_telemetry:
        st.markdown("#### ⚡ Live Cockpit Telemetry")

        if ctx.state.playing and ctx.video_processor:
            t = ctx.video_processor.get_telemetry()
            status = t["status"]
            ear_val = t["ear"]
            mar_val = t["mar"]
            vigilance = t["vigilance_score"]
            summary = t["summary"]

            # Record rolling history for time series charts
            st.session_state.ear_history.append(ear_val)
            st.session_state.mar_history.append(mar_val)
            if t["incidents"]:
                st.session_state.session_incidents = t["incidents"]

            # Status styling
            css_class = "status-alert"
            if status == DriverStatus.DROWSY:
                css_class = "status-drowsy"
            elif status == DriverStatus.YAWNING or status == DriverStatus.DISTRACTED:
                css_class = "status-yawn"

            st.markdown(
                f"""
                <div class="telemetry-card {css_class}">
                    <div style="font-size: 2.2rem;">{'🟢' if status == DriverStatus.NORMAL else ('🔴' if status == DriverStatus.DROWSY else '🟡')}</div>
                    <div class="telemetry-val" style="font-size: 1.6rem;">{status.value}</div>
                    <div class="telemetry-label">Driver State</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Vigilance Gauge
            st.plotly_chart(vigilance_gauge(vigilance), use_container_width=True, key="live_v_gauge")

            # KPI Grid
            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("👁️ Blinks", summary["total_blinks"])
            with k2:
                st.metric("👄 Yawns", summary["total_yawns"])
            with k3:
                st.metric("⚠️ Distractions", summary["total_distractions"])

            # Trigger Browser Siren Alarm if hazard active
            if t["should_alarm"] and enable_audio:
                now = time.time()
                if now - st.session_state.last_alarm_time > 2.0:
                    trigger_browser_alarm(status)
                    st.session_state.last_alarm_time = now

        else:
            st.markdown(
                """
                <div class="telemetry-card">
                    <div style="font-size: 2.5rem;">📷</div>
                    <div class="telemetry-val" style="color: #829AB1;">Camera Inactive</div>
                    <div class="telemetry-label">Click START on the left to begin driver safety tracking</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(vigilance_gauge(100), use_container_width=True, key="idle_v_gauge")

# ─────────────────────────────────────────────────────────────────────
#  TAB 2 : Snapshot & Single Image Inspection
# ─────────────────────────────────────────────────────────────────────
with tab_snap:
    st.markdown("### 📸 Driver Facial Geometry Snapshot Inspection")
    st.caption("Upload or capture a still driver photo to calculate exact EAR, MAR, and 3D Head Pose angles.")

    col_c, col_u = st.columns(2)
    with col_c:
        cam_pic = st.camera_input("Capture Driver Photo", key="driver_cam_snap")
    with col_u:
        up_pic = st.file_uploader("Or Upload Driver Image", type=["jpg", "png", "jpeg", "webp"])

    chosen_pic = cam_pic or up_pic
    if chosen_pic is not None:
        pil_img = Image.open(chosen_pic).convert("RGB")
        import numpy as np
        import cv2

        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        tracker = st.session_state.standalone_tracker

        with st.spinner("Executing Face Mesh & Kinematic Extraction..."):
            annotated_frame, metrics_res = tracker.process_frame(bgr, draw=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("#### 🎯 468 Landmark Mesh & Orientation Vector")
            st.image(annotated_frame, channels="BGR", use_container_width=True)

        with sc2:
            st.markdown("#### 📋 Kinematic Diagnostic Results")
            if metrics_res and metrics_res.get("face_detected"):
                ear = metrics_res["avg_ear"]
                mar = metrics_res["mar"]
                pitch = metrics_res["pitch"]
                yaw = metrics_res["yaw"]
                roll = metrics_res["roll"]

                st.info(f"**Left EAR:** {metrics_res['left_ear']:.3f} | **Right EAR:** {metrics_res['right_ear']:.3f}")
                st.metric("👁️ Average EAR", f"{ear:.3f}", delta=f"{ear - ear_thresh:+.3f} vs Thresh")
                st.metric("👄 MAR (Mouth Opening)", f"{mar:.3f}", delta=f"{mar - mar_thresh:+.3f} vs Thresh")

                st.markdown("##### 📐 3D Head Orientation")
                p1, p2, p3 = st.columns(3)
                p1.metric("Pitch (Nod)", f"{pitch:+.1f}°")
                p2.metric("Yaw (Turn)", f"{yaw:+.1f}°")
                p3.metric("Roll (Tilt)", f"{roll:+.1f}°")

                if ear < ear_thresh:
                    st.error("⚠️ Status: EYES CLOSED / DROWSINESS HAZARD")
                elif mar > mar_thresh:
                    st.warning("⚠️ Status: YAWNING DETECTED / FATIGUE HAZARD")
                else:
                    st.success("✅ Status: DRIVER ATTENTIVE & VIGILANT")
            else:
                st.warning("No face detected in the photo. Please align your camera directly towards your face.")

# ─────────────────────────────────────────────────────────────────────
#  TAB 3 : Trip Safety Telemetry & Incident Logs
# ─────────────────────────────────────────────────────────────────────
with tab_telemetry:
    st.markdown("### 📊 Trip Safety Telemetry & Incident Audit")
    st.caption("Rolling kinematic curves and chronological audit trail of all fatigue & distraction events.")

    st.plotly_chart(
        ear_mar_time_series(
            list(st.session_state.ear_history),
            list(st.session_state.mar_history),
            ear_thresh,
            mar_thresh,
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### 📋 Trip Incident Log")
    inc_df = incidents_dataframe(st.session_state.session_incidents)
    st.dataframe(inc_df, use_container_width=True, hide_index=True)

    if not inc_df.empty:
        csv_data = inc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Trip Safety Incident Report (CSV)",
            data=csv_data,
            file_name="trip_safety_audit.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────
#  TAB 4 : Sensitivity Calibration
# ─────────────────────────────────────────────────────────────────────
with tab_calibrate:
    st.markdown("### 🎛️ Driver Baseline Sensitivity Calibration")
    st.caption("Perform an empirical 2-step biometric calibration to eliminate false alarms and personalize thresholds.")

    if "calibrator" not in st.session_state:
        st.session_state.calibrator = DriverCalibrator(sample_size=20)

    calib = st.session_state.calibrator
    driver_name = st.text_input("Driver Name / Identifier", value="Driver 1")

    cal1, cal2 = st.columns(2)
    with cal1:
        st.markdown("#### Step 1: Open Eyes (Wakeful Baseline)")
        st.write("Look directly at the screen with relaxed, natural eyes.")
        if st.button("📸 Capture Awake Sample", key="btn_awake"):
            current_ear = list(st.session_state.ear_history)[-1] if st.session_state.ear_history else 0.30
            is_full = calib.add_open_sample(current_ear)
            st.success(f"Sample #{len(calib.open_samples)} added: EAR = {current_ear:.3f}")
            if is_full:
                st.balloons()
        st.progress(min(1.0, len(calib.open_samples) / 20.0), text=f"Samples: {len(calib.open_samples)} / 20")

    with cal2:
        st.markdown("#### Step 2: Closed Eyes (Resting Baseline)")
        st.write("Close your eyes gently for calibration measurement.")
        if st.button("📸 Capture Closed Eye Sample", key="btn_closed"):
            current_ear = list(st.session_state.ear_history)[-1] if st.session_state.ear_history else 0.12
            is_full = calib.add_closed_sample(current_ear)
            st.success(f"Sample #{len(calib.closed_samples)} added: EAR = {current_ear:.3f}")
            if is_full:
                st.balloons()
        st.progress(min(1.0, len(calib.closed_samples) / 20.0), text=f"Samples: {len(calib.closed_samples)} / 20")

    st.markdown("---")
    if len(calib.open_samples) >= 5 and len(calib.closed_samples) >= 5:
        profile = calib.compute_profile(driver_name)
        if profile:
            st.success(f"### ✅ Calibration Profile Generated for {profile.driver_name}")
            cp1, cp2, cp3 = st.columns(3)
            cp1.metric("Awake EAR Baseline", f"{profile.baseline_open_ear:.3f}")
            cp2.metric("Closed EAR Baseline", f"{profile.baseline_closed_ear:.3f}")
            cp3.metric("🎯 Recommended EAR Threshold", f"{profile.recommended_threshold:.3f}")
            st.info(f"💡 Recommended threshold is **{profile.recommended_threshold:.3f}**. You can set this in the left sidebar slider!")
    else:
        st.info("Take at least 5 awake samples and 5 closed samples to calculate your personalized profile.")

    if st.button("🔄 Reset Calibration Data"):
        calib.reset()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────
#  TAB 5 : Research & Mathematical Foundations
# ─────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        ### 🔬 Mathematical Formulations & Computer Vision Foundations

        #### 1. Eye Aspect Ratio (EAR) Formulation
        Proposed by Tereza Soukupová and Jan Čech (Computer Vision Winter Workshop, 2016):

        $$\\text{EAR} = \\frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \\cdot ||p_1 - p_4||}$$

        Where $p_1, \\dots, p_6$ are 2D landmark coordinates of the eye contours. The EAR remains constant
        when eyes are open and rapidly falls towards zero during blinks or eyelid closure.

        ---

        #### 2. Mouth Aspect Ratio (MAR) Formulation

        $$\\text{MAR} = \\frac{||p_{top} - p_{bottom}||}{||p_{left} - p_{right}||}$$

        Detects deep yawns when mouth opening exceeds the nominal speaking ratio by over 200%.

        ---

        #### 3. 3D Head Pose via Perspective-n-Point (solvePnP)
        Derives rotation vector $\\mathbf{r}$ and translation vector $\\mathbf{t}$ by minimizing reprojection error
        between 3D anthropometric face model points and 2D image coordinates:

        $$\\min_{\\mathbf{r}, \\mathbf{t}} \\sum_{i} ||p_i - \\pi(K [R(\\mathbf{r}) | \\mathbf{t}] P_i)||^2$$

        Using Rodrigues transformation, we decompose $R(\\mathbf{r})$ into Euler angles:
        - **Pitch**: Head nodding up/down
        - **Yaw**: Head turning left/right (distraction)
        - **Roll**: Head lateral tilt

        ---

        <div style="text-align: center; color: #829AB1; padding: 1rem;">
            Built with ❤️ for Automotive AI & Road Safety Intelligence · 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
