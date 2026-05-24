"""QThread worker: reads video frames, runs MediaPipe pose estimation
and basketball tracking, emits results via PyQt signals.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from src.pose_estimator import PoseEstimator, PoseResult
from src.ball_tracker import BallTracker

# Single frame result emitted by the worker
FrameData = Tuple[int, np.ndarray, Optional[PoseResult], Optional[Tuple[float, float]]]


class VideoWorker(QThread):
    """Background thread for video processing.

    Emits:
        frame_processed(frame_data) — per-frame results
        progress(current, total) — processing progress
        finished_processing() — all frames processed
        error_occurred(message) — error notification
    """

    frame_processed = pyqtSignal(tuple)  # (frame_idx, frame_bgr, landmarks, ball_pos)
    progress = pyqtSignal(int, int)
    finished_processing = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path: str, max_frames: int = None):
        super().__init__()
        self.video_path = video_path
        self.max_frames = max_frames
        self._running = True
        self._paused = False

    def run(self):
        """Main worker loop: read → process → emit."""
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

                # Pose estimation
                landmarks = estimator.process_frame(frame_bgr)

                # Ball tracking
                ball_pos = ball_tracker.detect(frame_bgr)

                self.frame_processed.emit((idx, frame_bgr.copy(), landmarks, ball_pos))
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
