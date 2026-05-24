"""QThread worker: reads video frames, runs MediaPipe pose estimation,
basketball tracking, and manual-point optical-flow tracking.
Emits results via PyQt signals.
"""

import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from src.pose_estimator import PoseEstimator, PoseResult
from src.ball_tracker import BallTracker

# Single frame result emitted by the worker:
# (frame_idx, frame_bgr, landmarks, ball_pos, manual_positions)
FrameData = Tuple[
    int,
    np.ndarray,
    Optional[PoseResult],
    Optional[Tuple[float, float]],
    Dict[int, Tuple[float, float]],
]


class VideoWorker(QThread):
    """Background thread for video processing.

    Emits:
        frame_processed(frame_data) — per-frame results
        progress(current, total) — processing progress
        finished_processing() — all frames processed
        error_occurred(message) — error notification
    """

    frame_processed = pyqtSignal(tuple)
    progress = pyqtSignal(int, int)
    finished_processing = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path: str, max_frames: int = None):
        super().__init__()
        self.video_path = video_path
        self.max_frames = max_frames
        self._running = True
        self._paused = False

        # Manual point tracking (optical flow)
        self._manual_points: Dict[int, dict] = {}
        self._next_manual_id: int = 0
        self._pending_points: List[Tuple[float, float]] = []
        self._lock = threading.Lock()

        # Control flags for the caller to initialise manual points before start
        self.initial_manual_points: List[Tuple[float, float]] = []

    def add_manual_point(self, x: float, y: float):
        """Thread-safe: add a new manual point to be tracked from the next frame."""
        with self._lock:
            self._pending_points.append((x, y))

    def run(self):
        """Main worker loop: read -> process -> emit."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open video: {self.video_path}")
            return

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            limit = min(total, self.max_frames) if self.max_frames else total

            estimator = PoseEstimator()
            ball_tracker = BallTracker()

            # Pre-load initial manual points
            for x, y in self.initial_manual_points:
                pid = self._next_manual_id
                self._next_manual_id += 1
                self._manual_points[pid] = {
                    "pos": (x, y),
                    "prev_pt": (x, y),
                    "active": True,
                }

            prev_gray = None

            for idx in range(limit):
                if not self._running:
                    break
                while self._paused and self._running:
                    self.msleep(50)

                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Resize if too large for performance
                h, w = frame_bgr.shape[:2]
                if max(w, h) > 960:
                    scale = 960.0 / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_bgr = cv2.resize(frame_bgr, (new_w, new_h))

                # Pick up pending manual points (thread-safe)
                with self._lock:
                    while self._pending_points:
                        px, py = self._pending_points.pop(0)
                        pid = self._next_manual_id
                        self._next_manual_id += 1
                        self._manual_points[pid] = {
                            "pos": (px, py),
                            "prev_pt": (px, py),
                            "active": True,
                        }

                # Convert to grayscale for optical flow
                curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                # Update manual points via LK optical flow
                manual_positions: Dict[int, Tuple[float, float]] = {}
                for pid, pt in self._manual_points.items():
                    if not pt["active"]:
                        continue
                    if prev_gray is not None:
                        p0 = np.array([[pt["prev_pt"]]], dtype=np.float32)
                        p1, status, _ = cv2.calcOpticalFlowPyrLK(
                            prev_gray,
                            curr_gray,
                            p0,
                            None,
                            winSize=(41, 41),
                            maxLevel=4,
                            criteria=(
                                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                30,
                                0.01,
                            ),
                            minEigThreshold=0.001,
                        )
                        if status[0][0] == 1:
                            nx, ny = float(p1[0][0][0]), float(p1[0][0][1])
                            pt["pos"] = (nx, ny)
                            pt["prev_pt"] = (nx, ny)
                            manual_positions[pid] = (nx, ny)
                        else:
                            pt["active"] = False
                            manual_positions[pid] = pt["pos"]
                    else:
                        manual_positions[pid] = pt["pos"]

                prev_gray = curr_gray

                # Pose estimation
                landmarks = estimator.process_frame(frame_bgr)

                # Ball tracking
                ball_pos = ball_tracker.detect(frame_bgr)

                self.frame_processed.emit(
                    (idx, frame_bgr.copy(), landmarks, ball_pos, dict(manual_positions))
                )
                self.progress.emit(idx + 1, limit)

            estimator.release()
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            cap.release()

        self.finished_processing.emit()

    def stop(self):
        """Request graceful stop."""
        self._running = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
