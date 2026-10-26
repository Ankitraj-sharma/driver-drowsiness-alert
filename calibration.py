"""
Driver Baseline Eye Geometry Calibration Wizard.

Computes customized personal threshold limits based on normal distribution
of individual driver wakefulness vs resting ocular measurements.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CalibrationProfile:
    """Individual driver biometric threshold profile."""
    driver_name: str
    baseline_open_ear: float
    baseline_closed_ear: float
    recommended_threshold: float
    margin_confidence: float


class DriverCalibrator:
    """
    Collects sample ocular ratios and generates an adaptive threshold.
    """

    def __init__(self, sample_size: int = 30) -> None:
        self.sample_size = sample_size
        self.open_samples: list[float] = []
        self.closed_samples: list[float] = []

    def add_open_sample(self, ear: float) -> bool:
        """Add sample when driver is looking straight and alert. Returns True when full."""
        self.open_samples.append(ear)
        return len(self.open_samples) >= self.sample_size

    def add_closed_sample(self, ear: float) -> bool:
        """Add sample when driver has eyes closed. Returns True when full."""
        self.closed_samples.append(ear)
        return len(self.closed_samples) >= self.sample_size

    def compute_profile(self, driver_name: str = "Driver") -> Optional[CalibrationProfile]:
        """Generate personalized thresholds using midpoint & variance analysis."""
        if not self.open_samples or not self.closed_samples:
            return None

        mean_open = float(np.mean(self.open_samples))
        mean_closed = float(np.mean(self.closed_samples))

        # Recommended threshold placed at midpoint between wakeful and resting states
        optimal_thresh = round((mean_open * 0.45) + (mean_closed * 0.55), 3)
        margin = round(mean_open - mean_closed, 3)

        return CalibrationProfile(
            driver_name=driver_name,
            baseline_open_ear=round(mean_open, 3),
            baseline_closed_ear=round(mean_closed, 3),
            recommended_threshold=optimal_thresh,
            margin_confidence=margin,
        )

    def reset(self) -> None:
        """Clear calibration buffers."""
        self.open_samples = []
        self.closed_samples = []
