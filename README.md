# 🚗 DrowsyGuard AI — Real-Time Driver Drowsiness & Distraction Alert System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Computer Vision](https://img.shields.io/badge/CV-MediaPipe%20Face%20Mesh%20468-brightgreen.svg)](https://mediapipe.dev)
[![Kinematics](https://img.shields.io/badge/Math-EAR%20%7C%20MAR%20%7C%20solvePnP-orange.svg)](https://docs.opencv.org)
[![Audio](https://img.shields.io/badge/Alarms-Web%20Audio%20API-teal.svg)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade automotive Computer Vision safety system** that monitors driver alertness via **Eye Aspect Ratio (EAR)**, **Mouth Aspect Ratio (MAR)**, and **3D Head Pose kinematics (solvePnP)**, triggering instant in-browser acoustic sirens and vocal safety warnings.

---

## 🌟 Key Highlights for Competitions & Hackathons

- 🔴 **Zero-Upload Continuous Webcam Stream**: Built with WebRTC — simply click START to monitor live video at 30 FPS.
- 👁️ **Eye Aspect Ratio (EAR) Micro-Sleep Detection**: Tracks eyelid closure dynamics and triggers multi-level alarms when closure exceeds safety limits.
- 👄 **Mouth Aspect Ratio (MAR) Yawning Analytics**: Measures vertical lip separation relative to horizontal width to detect fatigue and yawning.
- 📐 **3D Head Pose Orientation via solvePnP**: Computes 3D Euler angles (Pitch, Yaw, Roll) to detect head nodding off or looking away from the road.
- 🚨 **Real-Time Acoustic Siren (Web Audio API)**: Synthesizes high-frequency buzzer alarms directly through the client browser without external audio files.
- 📊 **Rolling Telemetry & Trip Audit**: Live Plotly dual-metric time-series curves, circular vigilance gauges, and downloadable safety CSV reports.

---

## 🔬 Mathematical Formulations

### 1. Eye Aspect Ratio (EAR)
Formulated by Soukupová and Čech (2016):

$$\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$

Where $p_1, \dots, p_6$ are 2D coordinates of eye contour landmarks. During normal wakefulness, EAR remains between $0.28 - 0.35$. During blinks and micro-sleep, EAR drops sharply below $0.20$.

---

### 2. Mouth Aspect Ratio (MAR)

$$\text{MAR} = \frac{||p_{\text{top}} - p_{\text{bottom}}||}{||p_{\text{left}} - p_{\text{right}}||}$$

Normal speaking registers $\text{MAR} \approx 0.15 - 0.30$. Deep yawning inflates MAR above $0.65$.

---

### 3. Perspective-n-Point 3D Head Orientation
Minimizes reprojection error between anthropometric 3D canonical facial model points and 2D image coordinates:

$$\min_{\mathbf{r}, \mathbf{t}} \sum_{i} ||p_i - \pi(K [R(\mathbf{r}) | \mathbf{t}] P_i)||^2$$

Decomposes into Euler angles:
- **Pitch**: Head nodding downwards (nodding off hazard)
- **Yaw**: Head turning left/right (distracted driving hazard)
- **Roll**: Head tilting laterally

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│            User Browser / Vehicle Camera               │
│      Live MediaStream via WebRTC (P2P Audio/Video)     │
└──────────────────────────┬─────────────────────────────┘
                           │  Continuous 30 FPS Stream
                           ▼
┌────────────────────────────────────────────────────────┐
│        FaceTracker (MediaPipe Face Mesh 468)           │
│    Extracts Eye Contours, Mouth Shape & 3D PnP Points  │
└──────────────────────────┬─────────────────────────────┘
                           │  2D Coordinates
                           ▼
┌────────────────────────────────────────────────────────┐
│         Kinematics Engine (metrics.py)                 │
│    • EAR (Eye Aspect Ratio)                            │
│    • MAR (Mouth Aspect Ratio)                          │
│    • solvePnP 3D Euler Angles (Pitch, Yaw, Roll)       │
│    • Dynamic 0-100% Vigilance Index                    │
└──────────────────────────┬─────────────────────────────┘
                           │  Telemetry State
                           ▼
┌────────────────────────────────────────────────────────┐
│         AlertEngine & Audio Synthesis                  │
│    • Multi-state Finite State Machine                  │
│    • Web Audio API Synthesized Siren Buzzer            │
│    • SpeechSynthesis Vocal Safety Warnings             │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
driver-drowsiness-alert/
├── app.py                  # Main Streamlit application with cockpit HUD
├── face_tracker.py         # 468 facial landmark detector & AR face overlay
├── metrics.py              # Mathematical formulas: EAR, MAR, solvePnP Head Pose
├── alert_engine.py         # State machine, Web Audio siren, and incident logger
├── live_stream.py          # Asynchronous WebRTC processor with continuous HUD
├── analytics.py            # Plotly time-series charts, gauge, and CSV exporter
├── config.py               # Landmark indices and default calibration limits
├── requirements.txt        # Python library dependencies
├── packages.txt            # Debian system dependencies for Cloud deploy
├── .streamlit/
│   └── config.toml         # Automotive dark theme configuration
├── .gitignore
├── LICENSE                 # MIT License
└── README.md               # Documentation & mathematical formulations
```

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.10+
- Webcam

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ankitraj-sharma/driver-drowsiness-alert.git
cd driver-drowsiness-alert

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch application
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🌐 Deploy to Streamlit Cloud

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Select your repository: `Ankitraj-sharma/driver-drowsiness-alert`.
4. Set main file path to: `app.py`.
5. Click **Deploy!**

---

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>Built with ❤️ for Automotive AI & Road Safety Intelligence</b>
</div>
