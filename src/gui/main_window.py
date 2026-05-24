"""Main PyQt5 application window for human motion capture and measurement.

Provides: video loading, live recording with background processing,
black-background replay with cyan skeleton, basketball tracking,
and real-time physics measurements.
"""

import os
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.video_worker import VideoWorker
from src.pose_estimator import KEYPOINT_NAMES, PoseResult

# Skeleton bone connections for drawing (MediaPipe pose connections)
BONE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # arms
    (23, 25), (25, 27), (24, 26), (26, 28),    # legs
]

# Selected keypoints for skeleton drawing
SKELETON_KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Colors (BGR for OpenCV drawing)
CYAN = (255, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Default scale factor (pixels to mm) — user-configurable
DEFAULT_MM_PER_PIXEL = 2.0


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human Motion Capture & Measurement System")
        self.setMinimumSize(960, 680)

        # State
        self.video_path: Optional[str] = None
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.mode: str = "IDLE"  # IDLE | RECORDING | REPLAY
        self.mm_per_pixel: float = DEFAULT_MM_PER_PIXEL

        # Recording data: list of (frame_idx, frame_bgr, landmarks, ball_pos)
        self.recording_data: List[Tuple[int, np.ndarray, Optional[PoseResult], Optional[Tuple[float, float]]]] = []

        # Replay state
        self._replay_idx: int = 0
        self._replay_timer: Optional[QTimer] = None
        self._ball_trajectory: List[Tuple[float, float]] = []
        self._ball_velocity_history: deque = deque(maxlen=10)

        # Worker thread
        self._worker: Optional[VideoWorker] = None

        self._setup_ui()
        self._set_button_states()

    # --------------- UI Setup ---------------

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000; border: 2px solid #333;")
        self.video_label.setText("Drop a video file or use File → Open")
        layout.addWidget(self.video_label, stretch=1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.btn_open = QPushButton("Open Video")
        self.btn_open.clicked.connect(self._on_open)
        btn_layout.addWidget(self.btn_open)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start)
        btn_layout.addWidget(self.btn_start)

        self.btn_end = QPushButton("End")
        self.btn_end.clicked.connect(self._on_end)
        btn_layout.addWidget(self.btn_end)

        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self._on_save)
        btn_layout.addWidget(self.btn_save)

        self.btn_replay = QPushButton("Replay")
        self.btn_replay.clicked.connect(self._on_replay)
        btn_layout.addWidget(self.btn_replay)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        btn_layout.addWidget(self.btn_exit)

        layout.addLayout(btn_layout)

        # Style buttons
        for btn in [self.btn_start, self.btn_end, self.btn_save, self.btn_replay]:
            btn.setMinimumWidth(90)

    def _set_button_states(self):
        """Update button enabled states based on current mode."""
        is_idle = self.mode == "IDLE"
        is_recording = self.mode == "RECORDING"
        is_replay_ready = self.mode == "REPLAY_READY"

        self.btn_open.setEnabled(is_idle or is_replay_ready)
        self.btn_start.setEnabled((is_idle or is_replay_ready) and self.video_path is not None)
        self.btn_end.setEnabled(is_recording)
        self.btn_save.setEnabled(is_replay_ready)
        self.btn_replay.setEnabled(is_replay_ready)
        self.progress_bar.setVisible(is_recording)

    # --------------- Slots ---------------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.gif *.webm);;All Files (*)"
        )
        if not path:
            return
        self.video_path = path

        # Probe video
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open: {path}")
            return
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Show first frame as preview
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self._display_frame(frame, is_preview=True)

        # Reset state
        self.recording_data.clear()
        self._ball_trajectory.clear()
        self._ball_velocity_history.clear()
        self.mode = "IDLE"
        self._set_button_states()
        self.setWindowTitle(f"Motion Capture — {os.path.basename(path)}  [{self.frame_count} fr @ {self.fps:.1f} fps]")

    def _on_start(self):
        if not self.video_path:
            return

        self.mode = "RECORDING"
        self._set_button_states()
        self.recording_data.clear()
        self._ball_trajectory.clear()
        self._ball_velocity_history.clear()

        self._worker = VideoWorker(self.video_path)
        self._worker.frame_processed.connect(self._on_frame_processed)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_processing.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_end(self):
        """User clicked End — stop recording, enter replay-ready mode."""
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

        if self.recording_data:
            self.mode = "REPLAY_READY"
            self._compute_measurements()
        else:
            self.mode = "IDLE"
        self._set_button_states()

    def _on_replay(self):
        """Start black-background replay."""
        if not self.recording_data:
            return
        self.mode = "REPLAY"
        self._set_button_states()
        self._replay_idx = 0
        self._ball_trajectory.clear()
        self._ball_velocity_history.clear()

        # Stop existing timer
        if self._replay_timer:
            self._replay_timer.stop()

        interval_ms = int(1000.0 / self.fps)
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._replay_step)
        self._replay_timer.start(interval_ms)

    def _on_frame_processed(self, data: tuple):
        """Receive processed frame from worker thread."""
        frame_idx, frame_bgr, landmarks, ball_pos = data
        self.recording_data.append((frame_idx, frame_bgr, landmarks, ball_pos))

        # During recording, show original video
        if self.mode == "RECORDING":
            self._display_frame(frame_bgr, is_preview=False)

    def _on_progress(self, current: int, total: int):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def _on_finished(self):
        """Worker finished processing all frames."""
        if self.mode == "RECORDING":
            self.mode = "REPLAY_READY"
            self._compute_measurements()
            self._set_button_states()

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Processing Error", message)
        self.mode = "IDLE"
        self._set_button_states()

    def _on_save(self):
        """Save the replay animation as a video file."""
        if not self.recording_data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result Video", "output/replay_result.mp4",
            "MP4 Video (*.mp4);;All Files (*)"
        )
        if not path:
            return
        self._save_replay_video(path)

    # --------------- Display ---------------

    def _display_frame(self, frame_bgr: np.ndarray, is_preview: bool = False):
        """Convert a BGR frame to QPixmap and show it."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Scale to fit label while keeping aspect ratio
        label_w = self.video_label.width()
        label_h = self.video_label.height()
        if label_w > 10 and label_h > 10:
            scale = min(label_w / w, label_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            rgb = cv2.resize(rgb, (new_w, new_h))

        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.setPixmap(pixmap)

    def _display_replay_frame(self, canvas_bgr: np.ndarray):
        """Display a replay canvas."""
        self._display_frame(canvas_bgr, is_preview=False)

    # --------------- Replay Engine ---------------

    def _replay_step(self):
        """Render one replay frame on black background."""
        if self._replay_idx >= len(self.recording_data):
            self._replay_timer.stop()
            self.mode = "REPLAY_READY"
            self._set_button_states()
            QMessageBox.information(self, "Replay Complete", "Replay finished.")
            return

        frame_idx, frame_bgr, landmarks, ball_pos = self.recording_data[self._replay_idx]
        h, w = frame_bgr.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw cyan skeleton
        if landmarks is not None:
            canvas = self._draw_skeleton(canvas, landmarks, CYAN)

        # Track ball trajectory
        if ball_pos is not None:
            self._ball_trajectory.append(ball_pos)
            # Draw red ball
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            bx = np.clip(bx, 0, w - 1)
            by = np.clip(by, 0, h - 1)
            cv2.circle(canvas, (bx, by), 10, RED, -1)
            cv2.circle(canvas, (bx, by), 13, RED, 2)

        # Draw yellow trajectory
        if len(self._ball_trajectory) > 1:
            pts = np.array([(int(p[0]), int(p[1])) for p in self._ball_trajectory], dtype=np.int32)
            for i in range(1, len(pts)):
                cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), YELLOW, 2)

        # Draw measurement overlay
        vel, dist, elapsed = self._calc_measurements(frame_idx)
        canvas = self._draw_measurements(canvas, vel, dist, elapsed)

        self._display_replay_frame(canvas)
        self._replay_idx += 1

    def _draw_skeleton(self, canvas: np.ndarray, landmarks: PoseResult, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw skeleton stick figure on canvas."""
        h, w = canvas.shape[:2]

        # Draw joint points
        for idx in SKELETON_KEYPOINTS:
            if idx >= len(landmarks):
                continue
            x, y, conf = landmarks[idx]
            if conf < 0.3:
                continue
            px, py = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            cv2.circle(canvas, (px, py), 4, color, -1)

        # Draw bone lines
        for p1, p2 in BONE_CONNECTIONS:
            if p1 >= len(landmarks) or p2 >= len(landmarks):
                continue
            x1, y1, c1 = landmarks[p1]
            x2, y2, c2 = landmarks[p2]
            if c1 < 0.3 or c2 < 0.3:
                continue
            pt1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
            pt2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
            cv2.line(canvas, pt1, pt2, color, 2)

        return canvas

    def _calc_measurements(self, frame_idx: int) -> Tuple[float, float, float]:
        """Calculate instantaneous velocity, cumulative distance, elapsed time.

        Velocity is smoothed using a moving average of recent frame-to-frame speeds.
        """
        elapsed = frame_idx / self.fps  # seconds

        # Find ball positions up to current frame
        ball_positions = []
        for i in range(min(frame_idx + 1, len(self.recording_data))):
            _, _, _, bp = self.recording_data[i]
            if bp is not None:
                ball_positions.append(bp)

        # Cumulative distance (pixels → mm)
        dist_px = 0.0
        for i in range(1, len(ball_positions)):
            dx = ball_positions[i][0] - ball_positions[i - 1][0]
            dy = ball_positions[i][1] - ball_positions[i - 1][1]
            dist_px += np.sqrt(dx * dx + dy * dy)
        dist_mm = dist_px * self.mm_per_pixel

        # Instantaneous velocity (px/frame → mm/s, smoothed)
        vel_mm_s = 0.0
        if len(ball_positions) >= 2:
            # Use last few frames for smoothing
            recent = ball_positions[-min(5, len(ball_positions)):]
            speeds = []
            for i in range(1, len(recent)):
                dx = recent[i][0] - recent[i - 1][0]
                dy = recent[i][1] - recent[i - 1][1]
                step_px = np.sqrt(dx * dx + dy * dy)
                speed_mm_s = step_px * self.mm_per_pixel * self.fps
                speeds.append(speed_mm_s)
            if speeds:
                vel_mm_s = float(np.mean(speeds))

        self._ball_velocity_history.append(vel_mm_s)
        # Extra smoothing across time
        if len(self._ball_velocity_history) > 0:
            vel_mm_s = float(np.mean(self._ball_velocity_history))

        return vel_mm_s, dist_mm, elapsed

    def _draw_measurements(self, canvas: np.ndarray, vel: float, dist: float, elapsed: float) -> np.ndarray:
        """Overlay measurement text on canvas."""
        lines = [
            f"Velocity: {vel:.2f} mm/s",
            f"Distance: {dist:.2f} mm",
            f"Times: {elapsed:.2f} s",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = WHITE
        outline = (0, 0, 0)

        y0 = 30
        for i, text in enumerate(lines):
            y = y0 + i * 30
            # Black outline
            cv2.putText(canvas, text, (12, y - 1), font, font_scale, outline, thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (12, y + 1), font, font_scale, outline, thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return canvas

    def _compute_measurements(self):
        """Pre-compute summary measurements after recording finishes."""
        ball_positions = []
        for _, _, _, bp in self.recording_data:
            if bp is not None:
                ball_positions.append(bp)

        if not ball_positions:
            return

        total_dist_px = 0.0
        for i in range(1, len(ball_positions)):
            dx = ball_positions[i][0] - ball_positions[i - 1][0]
            dy = ball_positions[i][1] - ball_positions[i - 1][1]
            total_dist_px += np.sqrt(dx * dx + dy * dy)

        print(f"\n=== Recording Summary ===")
        print(f"Total frames processed: {len(self.recording_data)}")
        print(f"Frames with ball detected: {len(ball_positions)}")
        print(f"Total ball distance: {total_dist_px * self.mm_per_pixel:.2f} mm")
        print(f"Duration: {len(self.recording_data) / self.fps:.2f} s")

    # --------------- Save ---------------

    def _save_replay_video(self, output_path: str):
        """Render replay frames to an MP4 file."""
        if not self.recording_data:
            return

        first_frame = self.recording_data[0][1]
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        saved_trajectory = []

        for frame_idx, frame_bgr, landmarks, ball_pos in self.recording_data:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            if landmarks is not None:
                canvas = self._draw_skeleton(canvas, landmarks, CYAN)

            if ball_pos is not None:
                saved_trajectory.append(ball_pos)
                bx, by = int(ball_pos[0]), int(ball_pos[1])
                bx, by = np.clip(bx, 0, w - 1), np.clip(by, 0, h - 1)
                cv2.circle(canvas, (bx, by), 10, RED, -1)
                cv2.circle(canvas, (bx, by), 13, RED, 2)

            if len(saved_trajectory) > 1:
                pts = np.array([(int(p[0]), int(p[1])) for p in saved_trajectory], dtype=np.int32)
                for i in range(1, len(pts)):
                    cv2.line(canvas, tuple(pts[i - 1]), tuple(pts[i]), YELLOW, 2)

            vel, dist, elapsed = self._calc_measurements(frame_idx)
            canvas = self._draw_measurements(canvas, vel, dist, elapsed)

            writer.write(canvas)

        writer.release()
        QMessageBox.information(self, "Saved", f"Replay video saved to:\n{output_path}")
