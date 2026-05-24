"""Trajectory tracking and interpolation module.

Maintains per-keypoint time-series coordinates, handles low-confidence
samples via linear interpolation, and provides moving-average smoothing.
"""

from typing import List, Optional

import numpy as np

from .pose_estimator import PoseResult

CONFIDENCE_THRESHOLD = 0.5


class Tracker:
    """Tracks 33 keypoints across frames with interpolation and smoothing."""

    def __init__(self, num_keypoints: int = 33):
        self.num_keypoints = num_keypoints
        self._positions: List[Optional[np.ndarray]] = []  # each: (33, 3) or None
        self._confidences: List[Optional[np.ndarray]] = []

    def add_frame(self, pose: Optional[PoseResult]) -> None:
        """Register a single frame's pose detection result.

        Parameters
        ----------
        pose : PoseResult or None
            Per-frame detection from PoseEstimator.process_frame().
        """
        if pose is None:
            self._positions.append(None)
            self._confidences.append(None)
            return

        pos = np.array([[p[0], p[1]] for p in pose], dtype=np.float64)  # (33, 2)
        conf = np.array([p[2] for p in pose], dtype=np.float64)         # (33,)
        self._positions.append(pos)
        self._confidences.append(conf)

    def add_frames(self, poses: List[Optional[PoseResult]]) -> None:
        """Register multiple frames."""
        for pose in poses:
            self.add_frame(pose)

    def interpolate(self) -> np.ndarray:
        """Interpolate low-confidence (< 0.5) and missing keypoints.

        For each keypoint independently:
        - Pixels where confidence < 0.5 or frame is missing -> NaN
        - NaN segments are linearly interpolated
        - Leading/trailing NaNs use nearest-neighbour padding

        Returns
        -------
        positions : np.ndarray
            Shape (T, 33, 2), interpolated x, y coordinates.
        """
        T = len(self._positions)

        # Build masked array: NaN where low confidence or missing
        positions = np.full((T, self.num_keypoints, 2), np.nan, dtype=np.float64)
        for t in range(T):
            if self._positions[t] is None:
                continue
            for k in range(self.num_keypoints):
                if self._confidences[t] is not None and self._confidences[t][k] >= CONFIDENCE_THRESHOLD:
                    positions[t, k, 0] = self._positions[t][k, 0]
                    positions[t, k, 1] = self._positions[t][k, 1]

        return _interpolate_positions(positions)

    def smooth(self, positions: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply moving-average smoothing per-keypoint.

        Parameters
        ----------
        positions : np.ndarray
            Shape (T, 33, 2).
        window : int
            Smoothing window size.

        Returns
        -------
        smoothed : np.ndarray
            Shape (T, 33, 2).
        """
        from scipy.ndimage import uniform_filter1d

        T, K, D = positions.shape
        smoothed = np.empty_like(positions)
        for k in range(K):
            for d in range(D):
                ok = ~np.isnan(positions[:, k, d])
                if ok.sum() == 0:
                    smoothed[:, k, d] = positions[:, k, d]
                    continue
                filled = positions[:, k, d].copy()
                if not ok.all():
                    xp = ok.nonzero()[0]
                    fp = positions[ok, k, d]
                    xi = (~ok).nonzero()[0]
                    filled[xi] = np.interp(xi, xp, fp)
                smoothed[:, k, d] = uniform_filter1d(filled, size=window)
        return smoothed

    @property
    def frame_count(self) -> int:
        return len(self._positions)


def _interpolate_positions(positions: np.ndarray) -> np.ndarray:
    """Per-keypoint linear interpolation for NaN positions.

    Parameters
    ----------
    positions : np.ndarray
        Shape (T, K, 2).

    Returns
    -------
    interp : np.ndarray
        Shape (T, K, 2).
    """
    T, K, _ = positions.shape
    result = positions.copy()

    for k in range(K):
        for d in range(2):
            series = result[:, k, d]
            ok = ~np.isnan(series)
            if ok.sum() == 0:
                result[:, k, d] = 0.0
                continue
            xp = ok.nonzero()[0]
            fp = series[ok]
            # Interpolate interior NaNs
            missing = (~ok).nonzero()[0]
            if len(missing) > 0:
                result[missing, k, d] = np.interp(missing, xp, fp)

    return result
