"""Basketball tracker using HSV color filtering and morphological operations.

Targets bright pink/magenta basketballs. Falls back to CSRT tracker
for robustness when HSV detection fails momentarily.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class BallTracker:
    """Track a brightly-colored ball (pink/red) via HSV thresholding.

    Uses two-pass HSV mask (red wraps around 0/180 in OpenCV) plus
    morphological cleanup. Falls back to CSRT when HSV mask is empty.
    """

    def __init__(
        self,
        hue_range: Tuple[int, int] = (155, 180),
        hue_range_red: Tuple[int, int] = (0, 10),
        sat_min: int = 80,
        val_min: int = 80,
        min_contour_area: int = 30,
        smoothing_window: int = 5,
    ):
        """
        Parameters
        ----------
        hue_range : (int, int)
            Primary hue range for pink/magenta [155, 180].
        hue_range_red : (int, int)
            Secondary hue range for red wrap-around [0, 10].
        sat_min : int
            Minimum saturation threshold.
        val_min : int
            Minimum value (brightness) threshold.
        min_contour_area : int
            Minimum contour area to be considered a ball.
        smoothing_window : int
            Moving-average window for position smoothing.
        """
        self.hue_range = hue_range
        self.hue_range_red = hue_range_red
        self.sat_min = sat_min
        self.val_min = val_min
        self.min_contour_area = min_contour_area
        self.smoothing_window = smoothing_window

        self._csrt: Optional[cv2.TrackerCSRT] = None
        self._csrt_bbox: Optional[Tuple[int, int, int, int]] = None
        self._csrt_fail_count: int = 0
        self._position_history = []

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """Detect the ball center in a BGR frame.

        Returns (x, y) in pixel coordinates, or None if not found.
        """
        center = self._detect_hsv(frame_bgr)
        if center is not None:
            self._csrt_fail_count = 0
            self._update_csrt(frame_bgr, center)
        elif self._csrt is not None:
            center = self._detect_csrt(frame_bgr)
            if center is not None:
                self._csrt_fail_count = 0
            else:
                self._csrt_fail_count += 1
                if self._csrt_fail_count > 10:
                    self._csrt = None

        if center is not None:
            self._position_history.append(center)
            if len(self._position_history) > self.smoothing_window:
                self._position_history.pop(0)

        return self._smoothed_position() if self._position_history else center

    def _detect_hsv(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """HSV thresholding detection."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Build mask for pink/magenta hue + red wrap-around
        mask1 = cv2.inRange(
            hsv,
            (self.hue_range[0], self.sat_min, self.val_min),
            (self.hue_range[1], 255, 255),
        )
        mask2 = cv2.inRange(
            hsv,
            (self.hue_range_red[0], self.sat_min, self.val_min),
            (self.hue_range_red[1], 255, 255),
        )
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Select largest valid contour
        best = None
        best_area = self.min_contour_area
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area:
                continue
            # Circularity check (optional filter)
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if area > best_area and circularity > 0.3:
                best_area = area
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    return (float(cx), float(cy))
        return None

    def _update_csrt(self, frame_bgr: np.ndarray, center: Tuple[float, float]):
        """Initialize or update CSRT tracker around detected center."""
        w, h = 40, 40
        x = int(max(0, center[0] - w // 2))
        y = int(max(0, center[1] - h // 2))
        x = min(x, frame_bgr.shape[1] - w)
        y = min(y, frame_bgr.shape[0] - h)
        self._csrt_bbox = (x, y, w, h)
        self._csrt = cv2.TrackerCSRT_create()
        self._csrt.init(frame_bgr, self._csrt_bbox)

    def _detect_csrt(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
        """Fallback detection using CSRT tracker."""
        if self._csrt is None:
            return None
        ok, bbox = self._csrt.update(frame_bgr)
        if ok:
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            return (float(cx), float(cy))
        return None

    def _smoothed_position(self) -> Optional[Tuple[float, float]]:
        """Return moving-average smoothed position."""
        if not self._position_history:
            return None
        xs = [p[0] for p in self._position_history]
        ys = [p[1] for p in self._position_history]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def reset(self):
        """Reset tracker state for a new video."""
        self._csrt = None
        self._csrt_bbox = None
        self._csrt_fail_count = 0
        self._position_history.clear()
