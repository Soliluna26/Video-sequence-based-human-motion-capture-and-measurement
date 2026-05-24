"""Basketball tracker using HSV color filtering and short-term tracking.

Targets ordinary orange basketballs as well as bright red/pink balls.
Falls back to CSRT tracker for robustness when HSV detection fails momentarily.
"""

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class BallTracker:
    """Track a basketball via HSV thresholding plus temporal consistency.

    The detector builds orange/red masks, scores circular candidates, and
    prefers candidates near the previous ball position or the player's hands.
    CSRT bridges brief color-detection failures.
    """

    def __init__(
        self,
        orange_hue_range: Tuple[int, int] = (3, 32),
        pink_hue_range: Tuple[int, int] = (155, 180),
        red_hue_range: Tuple[int, int] = (0, 10),
        sat_min: int = 45,
        val_min: int = 35,
        min_contour_area: int = 30,
        max_contour_area: int = 2600,
        smoothing_window: int = 5,
    ):
        """
        Parameters
        ----------
        orange_hue_range : (int, int)
            Hue range for ordinary orange basketballs.
        pink_hue_range : (int, int)
            Hue range for pink/magenta balls.
        red_hue_range : (int, int)
            Hue range for red wrap-around in OpenCV HSV.
        sat_min : int
            Minimum saturation threshold.
        val_min : int
            Minimum value (brightness) threshold.
        min_contour_area : int
            Minimum contour area to be considered a ball.
        max_contour_area : int
            Maximum contour area to reject large skin/background regions.
        smoothing_window : int
            Moving-average window for position smoothing.
        """
        self.orange_hue_range = orange_hue_range
        self.pink_hue_range = pink_hue_range
        self.red_hue_range = red_hue_range
        self.sat_min = sat_min
        self.val_min = val_min
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.smoothing_window = smoothing_window

        self._csrt: Optional[cv2.TrackerCSRT] = None
        self._csrt_bbox: Optional[Tuple[int, int, int, int]] = None
        self._csrt_fail_count: int = 0
        self._position_history = []
        self._last_center: Optional[Tuple[float, float]] = None
        self._last_radius: float = 18.0
        self._lost_frames: int = 0
        self._stationary_frames: int = 0

    def detect(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Detect the ball center in a BGR frame.

        Returns (x, y) in pixel coordinates, or None if not found.
        """
        hand_points = self._hand_points(landmarks)
        previous_center = self._last_center
        center, radius = self._detect_hsv(frame_bgr, landmarks)
        if center is not None and self._is_static_background(center, previous_center, hand_points):
            center = None
            self._csrt = None
            self._last_center = None
            self._position_history.clear()

        if center is not None:
            if previous_center is not None and hand_points:
                jump = np.linalg.norm(np.array(center) - np.array(previous_center))
                hand_dist = min(np.linalg.norm(np.array(center) - np.array(p)) for p in hand_points)
                if jump > 85.0 and hand_dist < 70.0:
                    self._position_history.clear()
            self._last_center = center
            self._last_radius = radius
            self._lost_frames = 0
            self._csrt_fail_count = 0
            self._update_csrt(frame_bgr, center, radius)
        elif self._csrt is not None:
            center = self._detect_csrt(frame_bgr)
            if center is not None:
                self._last_center = center
                self._lost_frames = 0
                self._csrt_fail_count = 0
            else:
                self._lost_frames += 1
                self._csrt_fail_count += 1
                if self._csrt_fail_count > 10:
                    self._csrt = None
                    self._last_center = None
                    self._position_history.clear()
        else:
            self._lost_frames += 1
            self._stationary_frames = 0
            if self._lost_frames > 8:
                self._last_center = None
                self._position_history.clear()

        if center is not None:
            self._position_history.append(center)
            if len(self._position_history) > self.smoothing_window:
                self._position_history.pop(0)

        if center is None:
            return None
        return self._smoothed_position() if self._position_history else center

    def _is_static_background(
        self,
        center: Tuple[float, float],
        previous_center: Optional[Tuple[float, float]],
        hand_points: List[Tuple[float, float]],
    ) -> bool:
        """Reject static colored background patches after the ball has left hands."""
        if previous_center is None:
            self._stationary_frames = 0
            return False

        motion = np.linalg.norm(np.array(center) - np.array(previous_center))
        if motion < 2.0:
            self._stationary_frames += 1
        else:
            self._stationary_frames = 0

        near_hand = False
        if hand_points:
            hand_dist = min(np.linalg.norm(np.array(center) - np.array(p)) for p in hand_points)
            near_hand = hand_dist < 85.0

        return self._stationary_frames >= 8 and not near_hand

    def _detect_hsv(
        self,
        frame_bgr: np.ndarray,
        landmarks: Optional[Sequence[Tuple[float, float, float]]] = None,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """HSV thresholding detection."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Build mask for ordinary orange basketballs plus red/pink variants.
        masks = [
            cv2.inRange(
                hsv,
                (self.orange_hue_range[0], self.sat_min, self.val_min),
                (self.orange_hue_range[1], 255, 255),
            ),
            cv2.inRange(
                hsv,
                (self.pink_hue_range[0], max(self.sat_min, 70), max(self.val_min, 60)),
                (self.pink_hue_range[1], 255, 255),
            ),
            cv2.inRange(
                hsv,
                (self.red_hue_range[0], max(self.sat_min, 70), max(self.val_min, 60)),
                (self.red_hue_range[1], 255, 255),
            ),
        ]
        mask = masks[0]
        for extra in masks[1:]:
            mask = cv2.bitwise_or(mask, extra)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, self._last_radius

        hand_points = self._hand_points(landmarks)
        candidates: List[Tuple[Tuple[float, float], float, float, float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h else 0.0
            if circularity < 0.18 or not 0.45 <= aspect <= 2.2:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            radius = float(max(w, h) / 2.0)
            candidates.append(((cx, cy), radius, area, circularity))

        candidates.extend(self._hough_candidates(frame_bgr, mask, hand_points))

        best_score = float("-inf")
        best_center: Optional[Tuple[float, float]] = None
        best_radius = self._last_radius
        for center, radius, area, circularity in candidates:
            if self._last_center is not None:
                dist = np.linalg.norm(np.array(center) - np.array(self._last_center))
                hand_dist = float("inf")
                if hand_points:
                    hand_dist = min(np.linalg.norm(np.array(center) - np.array(p)) for p in hand_points)
                if dist > 85.0 and hand_dist >= 70.0:
                    continue
            score = self._candidate_score(center, radius, area, circularity, hand_points)
            if self._last_center is not None and hand_points:
                hand_dist = min(np.linalg.norm(np.array(center) - np.array(p)) for p in hand_points)
                if hand_dist < 70.0:
                    score += 160.0
            if score > best_score:
                best_score = score
                best_center = center
                best_radius = radius

        if best_center is None:
            return None, self._last_radius
        min_score = 35.0 if self._last_center is None else 0.0
        if best_score < min_score:
            return None, self._last_radius
        return best_center, best_radius

    def _hough_candidates(
        self,
        frame_bgr: np.ndarray,
        color_mask: np.ndarray,
        hand_points: List[Tuple[float, float]],
    ) -> List[Tuple[Tuple[float, float], float, float, float]]:
        """Find circular color candidates near hands or the previous ball."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=18,
            param1=80,
            param2=12,
            minRadius=6,
            maxRadius=32,
        )
        if circles is None:
            return []

        h, w = frame_bgr.shape[:2]
        result: List[Tuple[Tuple[float, float], float, float, float]] = []
        anchors = hand_points
        if self._last_center is not None:
            anchors = [self._last_center]

        for x, y, r in circles[0]:
            cx, cy, radius = float(x), float(y), float(r)
            if not 0 <= cx < w or not 0 <= cy < h:
                continue

            if anchors:
                anchor_dist = min(np.linalg.norm(np.array((cx, cy)) - np.array(p)) for p in anchors)
                max_dist = 130.0 if self._last_center is not None else 95.0
                if anchor_dist > max_dist:
                    continue

            x1, x2 = int(max(0, cx - radius)), int(min(w, cx + radius))
            y1, y2 = int(max(0, cy - radius)), int(min(h, cy + radius))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = color_mask[y1:y2, x1:x2]
            color_ratio = cv2.countNonZero(roi) / float(roi.size)
            if color_ratio < 0.22:
                continue

            area = float(np.pi * radius * radius * color_ratio)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue
            result.append(((cx, cy), radius, area, 0.8))

        return result

    def _candidate_score(
        self,
        center: Tuple[float, float],
        radius: float,
        area: float,
        circularity: float,
        hand_points: List[Tuple[float, float]],
    ) -> float:
        """Score color candidates by shape, size, temporal and hand proximity."""
        score = circularity * 80.0

        expected_area = np.pi * self._last_radius * self._last_radius
        if expected_area > 1.0:
            area_ratio = min(area, expected_area) / max(area, expected_area)
            score += area_ratio * 35.0

        if self._last_center is not None:
            dist = np.linalg.norm(np.array(center) - np.array(self._last_center))
            score += max(0.0, 140.0 - dist) * 1.2
        elif hand_points:
            hand_dist = min(np.linalg.norm(np.array(center) - np.array(p)) for p in hand_points)
            score += max(0.0, 170.0 - hand_dist) * 1.4

        # Ordinary basketballs in this video are small-to-medium blobs. This
        # rejects large orange regions from clothing/skin/background.
        if 5.0 <= radius <= 45.0:
            score += 25.0
        else:
            score -= 40.0

        return score

    def _hand_points(
        self,
        landmarks: Optional[Sequence[Tuple[float, float, float]]],
    ) -> List[Tuple[float, float]]:
        """Return reliable wrist/finger points for initial ball selection."""
        if landmarks is None:
            return []
        points: List[Tuple[float, float]] = []
        for idx in (15, 16, 19, 20, 21, 22):
            if idx >= len(landmarks):
                continue
            x, y, conf = landmarks[idx]
            if conf >= 0.25:
                points.append((float(x), float(y)))
        return points

    def _update_csrt(self, frame_bgr: np.ndarray, center: Tuple[float, float], radius: float):
        """Initialize or update CSRT tracker around detected center."""
        size = int(np.clip(radius * 2.6, 24, 72))
        x = int(max(0, center[0] - size // 2))
        y = int(max(0, center[1] - size // 2))
        x = min(x, max(0, frame_bgr.shape[1] - size))
        y = min(y, max(0, frame_bgr.shape[0] - size))
        self._csrt_bbox = (x, y, size, size)
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
            if self._local_color_ratio(frame_bgr, (cx, cy), max(bbox[2], bbox[3]) / 2.0) < 0.12:
                return None
            return (float(cx), float(cy))
        return None

    def _local_color_ratio(
        self,
        frame_bgr: np.ndarray,
        center: Tuple[float, float],
        radius: float,
    ) -> float:
        """Return red/orange pixel ratio around a tracked center."""
        h, w = frame_bgr.shape[:2]
        cx, cy = center
        pad = int(np.clip(radius, 8, 36))
        x1, x2 = int(max(0, cx - pad)), int(min(w, cx + pad))
        y1, y2 = int(max(0, cy - pad)), int(min(h, cy + pad))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (self.orange_hue_range[0], self.sat_min, self.val_min),
            (self.orange_hue_range[1], 255, 255),
        )
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                (self.pink_hue_range[0], max(self.sat_min, 70), max(self.val_min, 60)),
                (self.pink_hue_range[1], 255, 255),
            ),
        )
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(
                hsv,
                (self.red_hue_range[0], max(self.sat_min, 70), max(self.val_min, 60)),
                (self.red_hue_range[1], 255, 255),
            ),
        )
        return cv2.countNonZero(mask) / float(mask.size)

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
        self._last_center = None
        self._last_radius = 18.0
        self._lost_frames = 0
        self._stationary_frames = 0
