"""Motion pattern analysis module.

Detects key motion events (angular velocity peaks), computes left-right
symmetry metrics, and provides optional Fourier analysis for periodic
motion detection.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks


def detect_peaks(
    signal: np.ndarray,
    height: float = None,
    distance: int = 5,
    prominence: float = None,
) -> Tuple[np.ndarray, dict]:
    """Detect peaks (local maxima) in a 1-D signal.

    Wraps scipy.signal.find_peaks with sensible defaults for motion analysis.

    Parameters
    ----------
    signal : np.ndarray
        1-D array (e.g., angular velocity).
    height : float, optional
        Minimum peak height. Defaults to mean + 1 std of the signal.
    distance : int
        Minimum number of frames between peaks.
    prominence : float, optional
        Minimum peak prominence. Defaults to 0.5 std of the signal.

    Returns
    -------
    peaks : np.ndarray
        Frame indices of detected peaks.
    properties : dict
        Peak properties from scipy.signal.find_peaks.
    """
    clean = signal[~np.isnan(signal)]
    if len(clean) == 0:
        return np.array([], dtype=int), {}

    if height is None:
        height = np.mean(clean) + np.std(clean)
    if prominence is None:
        prominence = 0.5 * np.std(clean)

    peaks, props = find_peaks(
        signal,
        height=height,
        distance=distance,
        prominence=prominence,
    )
    return peaks, props


def detect_turning_points(
    angle_seq: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Detect turning points where angular velocity peaks.

    A turning point is a frame where the magnitude of angular velocity
    reaches a local maximum, indicating a change in motion direction.

    Parameters
    ----------
    angle_seq : np.ndarray
        Angle time series in degrees, shape (T,).
    fps : float
        Frames per second.

    Returns
    -------
    result : dict
        {'flexion_peaks': np.ndarray, 'extension_peaks': np.ndarray,
         'peak_times_sec': np.ndarray, 'angular_velocity': np.ndarray}
    """
    from .kinematics import angular_velocity, smooth_angles

    smoothed = smooth_angles(angle_seq, window=5)
    vel = angular_velocity(smoothed, fps)

    # Positive velocity peaks (flexion)
    pos_vel = vel.copy()
    pos_vel[vel < 0] = 0
    ext_peaks, _ = detect_peaks(pos_vel)

    # Negative velocity peaks (extension) — detect peaks in -vel
    neg_vel = -vel.copy()
    neg_vel[vel > 0] = 0
    flex_peaks, _ = detect_peaks(neg_vel)

    return {
        "flexion_peaks": flex_peaks,
        "extension_peaks": ext_peaks,
        "peak_times_sec": np.arange(len(vel)) / fps,
        "angular_velocity": vel,
    }


def compute_symmetry(
    left_angles: np.ndarray,
    right_angles: np.ndarray,
) -> Dict[str, float]:
    """Compute left-right symmetry metrics.

    Correlation between left and right angle sequences indicates
    how symmetric the bilateral motion is.

    Parameters
    ----------
    left_angles : np.ndarray
        Left-side angle sequence.
    right_angles : np.ndarray
        Right-side angle sequence.

    Returns
    -------
    metrics : dict
        {'correlation': float, 'rmse': float, 'mean_abs_diff': float}
        All in degrees for rmse/mad.
    """
    valid = ~np.isnan(left_angles) & ~np.isnan(right_angles)
    if valid.sum() < 2:
        return {"correlation": 0.0, "rmse": float("inf"), "mean_abs_diff": float("inf")}

    left = left_angles[valid]
    right = right_angles[valid]

    corr = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 else 0.0
    rmse = float(np.sqrt(np.mean((left - right) ** 2)))
    mad = float(np.mean(np.abs(left - right)))

    return {"correlation": corr, "rmse": rmse, "mean_abs_diff": mad}


def fourier_analysis(
    signal: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, np.ndarray]:
    """Perform FFT-based periodicity analysis on a signal.

    Useful for detecting periodic motions (walking, cycling, etc.).

    Parameters
    ----------
    signal : np.ndarray
        1-D signal (e.g., knee angle over time).
    fps : float
        Frames per second.

    Returns
    -------
    result : dict
        {'frequencies': np.ndarray, 'magnitudes': np.ndarray,
         'dominant_freq_hz': float, 'dominant_period_sec': float}
    """
    clean = signal[~np.isnan(signal)]
    if len(clean) < 2:
        return {
            "frequencies": np.array([]),
            "magnitudes": np.array([]),
            "dominant_freq_hz": 0.0,
            "dominant_period_sec": float("inf"),
        }

    n = len(clean)
    fft = np.fft.rfft(clean - np.mean(clean))
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Dominant frequency (excluding DC component)
    if len(mag) > 1:
        dominant_idx = np.argmax(mag[1:]) + 1
        dom_freq = float(freqs[dominant_idx])
        dom_period = 1.0 / dom_freq if dom_freq > 0 else float("inf")
    else:
        dom_freq = 0.0
        dom_period = float("inf")

    return {
        "frequencies": freqs,
        "magnitudes": mag,
        "dominant_freq_hz": dom_freq,
        "dominant_period_sec": dom_period,
    }
