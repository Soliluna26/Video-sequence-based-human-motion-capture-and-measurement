"""Kinematics measurement module — the computational core of the system.

Computes angles, angular velocities, angular accelerations, trajectory
lengths, and range-of-motion from pose landmark time series.

All angles are in degrees, angular velocities in degrees/second,
and distances in pixels (unless a scale factor is supplied).
"""

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d


def compute_angle(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> float:
    """Compute the angle formed by three points: p1--p2--p3.

    Uses the law of cosines: angle at p2 (vertex).

    Parameters
    ----------
    p1, p2, p3 : (x, y)
        Proximal, vertex, and distal points respectively.

    Returns
    -------
    angle : float
        Angle in degrees [0, 180].
    """
    v1 = np.array(p1) - np.array(p2)  # vector from vertex to proximal
    v2 = np.array(p3) - np.array(p2)  # vector from vertex to distal

    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)

    if norm < 1e-9:
        return 0.0  # degenerate: points coincide

    # Clamp to [-1, 1] to avoid numerical errors
    cos_theta = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_segment_angle(
    start1: Tuple[float, float],
    end1: Tuple[float, float],
    start2: Tuple[float, float],
    end2: Tuple[float, float],
) -> float:
    """Compute the unsigned angle between two directed line segments."""
    v1 = np.array(end1, dtype=np.float64) - np.array(start1, dtype=np.float64)
    v2 = np.array(end2, dtype=np.float64) - np.array(start2, dtype=np.float64)

    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0

    cos_theta = np.clip(float(np.dot(v1, v2)) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_angles_from_landmarks(
    landmarks_seq: np.ndarray,
    angle_definitions: dict,
) -> dict:
    """Compute time series of joint angles from landmark sequences.

    Parameters
    ----------
    landmarks_seq : np.ndarray
        Shape (T, 33, 2) — x, y coordinates for each keypoint over T frames.
    angle_definitions : dict
        {angle_name: (idx_p1, idx_p2, idx_p3)} mappings.

    Returns
    -------
    angles : dict
        {angle_name: np.ndarray shape (T,)} in degrees.
    """
    T = landmarks_seq.shape[0]
    result = {}
    for name, (i1, i2, i3) in angle_definitions.items():
        angles = np.full(T, np.nan)
        for t in range(T):
            p1 = tuple(landmarks_seq[t, i1])
            p2 = tuple(landmarks_seq[t, i2])
            p3 = tuple(landmarks_seq[t, i3])
            if any(np.isnan(pt).any() for pt in [p1, p2, p3]):
                continue
            angles[t] = compute_angle(p1, p2, p3)
        result[name] = angles
    return result


def compute_segment_angles_from_landmarks(
    landmarks_seq: np.ndarray,
    segment_definitions: dict,
) -> dict:
    """Compute time series of angles between two landmark-defined segments.

    segment_definitions maps a name to (start1, end1, start2, end2).
    """
    T = landmarks_seq.shape[0]
    result = {}
    for name, (i1, i2, i3, i4) in segment_definitions.items():
        angles = np.full(T, np.nan)
        for t in range(T):
            pts = landmarks_seq[t, [i1, i2, i3, i4]]
            if np.any(np.isnan(pts)):
                continue
            angles[t] = compute_segment_angle(
                tuple(landmarks_seq[t, i1]),
                tuple(landmarks_seq[t, i2]),
                tuple(landmarks_seq[t, i3]),
                tuple(landmarks_seq[t, i4]),
            )
        result[name] = angles
    return result


def angular_velocity(
    angles: np.ndarray,
    fps: float = 30.0,
) -> np.ndarray:
    """Compute angular velocity via central-difference derivative.

    dθ/dt at frame i:
        (θ[i+1] - θ[i-1]) / (2 * Δt)   for interior points
        (θ[1] - θ[0]) / Δt              at start
        (θ[-1] - θ[-2]) / Δt            at end

    Parameters
    ----------
    angles : np.ndarray
        Angle sequence in degrees, shape (T,). May contain NaNs.
    fps : float
        Frames per second.

    Returns
    -------
    vel : np.ndarray
        Angular velocity in deg/s, same shape as input.
    """
    dt = 1.0 / fps
    vel = np.full_like(angles, np.nan, dtype=np.float64)

    # Forward difference at start
    valid = ~np.isnan(angles)
    if valid[0] and valid[1]:
        vel[0] = (angles[1] - angles[0]) / dt

    # Central difference for interior
    for i in range(1, len(angles) - 1):
        if valid[i - 1] and valid[i + 1]:
            vel[i] = (angles[i + 1] - angles[i - 1]) / (2 * dt)

    # Backward difference at end
    if valid[-1] and valid[-2]:
        vel[-1] = (angles[-1] - angles[-2]) / dt

    return vel


def angular_acceleration(
    angles: np.ndarray,
    fps: float = 30.0,
) -> np.ndarray:
    """Compute angular acceleration (second derivative of angle).

    Applies central-difference twice: accel = d²θ/dt².

    Parameters
    ----------
    angles : np.ndarray
        Angle sequence in degrees.
    fps : float
        Frames per second.

    Returns
    -------
    accel : np.ndarray
        Angular acceleration in deg/s².
    """
    vel = angular_velocity(angles, fps)
    return angular_velocity(vel, fps)


def trajectory_length(
    positions: np.ndarray,
) -> float:
    """Compute total trajectory length as cumulative Euclidean distance.

    Parameters
    ----------
    positions : np.ndarray
        Shape (T, 2), sequence of (x, y) coordinates.

    Returns
    -------
    length : float
        Total path length in pixels.
    """
    valid = ~np.isnan(positions).any(axis=1)
    if valid.sum() < 2:
        return 0.0

    valid_positions = positions[valid]
    diffs = np.diff(valid_positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def range_of_motion(angles: np.ndarray) -> float:
    """Compute Range of Motion (ROM) as max - min of angle values.

    Parameters
    ----------
    angles : np.ndarray
        Angle sequence in degrees.

    Returns
    -------
    rom : float
        Range of motion in degrees.
    """
    valid = angles[~np.isnan(angles)]
    if len(valid) == 0:
        return 0.0
    return float(np.nanmax(valid) - np.nanmin(valid))


def smooth_angles(
    angles: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Apply moving-average smoothing to an angle sequence.

    Parameters
    ----------
    angles : np.ndarray
        Angle sequence in degrees.
    window : int
        Smoothing window size (odd recommended).

    Returns
    -------
    smoothed : np.ndarray
    """
    # Interpolate NaNs before smoothing
    filled = _interp_nans(angles)
    return uniform_filter1d(filled.astype(np.float64), size=window)


def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D array."""
    mask = np.isnan(arr)
    if not mask.any():
        return arr.copy()

    result = arr.copy()
    ok = ~mask
    if not ok.any():
        return np.zeros_like(arr)

    xp = ok.nonzero()[0]
    fp = arr[ok]
    x = mask.nonzero()[0]
    result[mask] = np.interp(x, xp, fp)
    return result
