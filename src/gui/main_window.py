"""Main PyQt5 application window for human motion capture and measurement.

Provides: video loading, live recording with background processing,
black-background replay with skeleton, manual point tracking,
and trajectory visualization.
"""

import csv
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.video_worker import VideoWorker
from src.point_manager import PointManager
from src.pose_estimator import KEYPOINT_NAMES, PoseResult
from src.action_recognizer import (
    ActionRecognizer,
    TemplateStore,
    extract_angle_features,
)
import src.action_recognizer as ar_mod

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
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)

# Default scale factor (pixels to mm) — user-configurable
DEFAULT_MM_PER_PIXEL = 2.0


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human Motion Capture & Measurement System")
        self.setMinimumSize(1200, 680)

        # State
        self.video_path: Optional[str] = None
        self.fps: float = 30.0
        self.frame_count: int = 0
        self.mode: str = "IDLE"  # IDLE | RECORDING | REPLAY_READY | REPLAY
        self.op_mode: str = "capture"  # capture | register | recognize
        self.mm_per_pixel: float = DEFAULT_MM_PER_PIXEL

        # Recording data: (frame_idx, frame_bgr, landmarks, ball_pos, manual_positions)
        self.recording_data: List[Tuple[
            int, np.ndarray, Optional[PoseResult],
            Optional[Tuple[float, float]], Dict[int, Tuple[float, float]]
        ]] = []

        # Replay state
        self._replay_idx: int = 0
        self._replay_timer: Optional[QTimer] = None
        self._ball_trajectory: List[Optional[Tuple[float, float]]] = []
        self._ball_velocity_history: deque = deque(maxlen=10)

        # Trajectory accumulators for replay rendering (manual points)
        self._manual_trajectories: Dict[int, List[Tuple[float, float]]] = {}

        # Manual point state
        self._adding_manual_point: bool = False
        self._pending_manual_points: List[Tuple[float, float]] = []
        self.point_manager = PointManager()

        # Display scaling info for mouse-coordinate mapping
        self._display_scale: float = 1.0
        self._display_offset_x: int = 0
        self._display_offset_y: int = 0
        self._frame_orig_w: int = 640
        self._frame_orig_h: int = 480

        # Worker thread
        self._worker: Optional[VideoWorker] = None

        # Action recognition state
        self._action_store: Optional[TemplateStore] = None
        self._action_matches: List = []
        self._templates_path: str = str(
            Path(__file__).resolve().parent.parent.parent / "config" / "action_templates.json"
        )
        self._positions_smooth: Optional[np.ndarray] = None  # cached for registration/recognition
        self._angle_defs: dict = {}

        self._setup_ui()
        self._set_button_states()

    # --------------- UI Setup ---------------

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ---- Mode selector bar ----
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(16)

        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #ccc;")
        mode_layout.addWidget(mode_label)

        self.btn_capture = QPushButton("捕捉")
        self.btn_capture.setCheckable(True)
        self.btn_capture.setChecked(True)
        self.btn_capture.clicked.connect(lambda: self._on_switch_mode("capture"))
        self.btn_capture.setMinimumHeight(52)
        self.btn_capture.setMinimumWidth(160)
        mode_layout.addWidget(self.btn_capture)

        self.btn_register = QPushButton("注册")
        self.btn_register.setCheckable(True)
        self.btn_register.clicked.connect(lambda: self._on_switch_mode("register"))
        self.btn_register.setMinimumHeight(52)
        self.btn_register.setMinimumWidth(160)
        mode_layout.addWidget(self.btn_register)

        self.btn_recognize = QPushButton("识别")
        self.btn_recognize.setCheckable(True)
        self.btn_recognize.clicked.connect(lambda: self._on_switch_mode("recognize"))
        self.btn_recognize.setMinimumHeight(52)
        self.btn_recognize.setMinimumWidth(160)
        mode_layout.addWidget(self.btn_recognize)

        mode_layout.addStretch()
        main_layout.addLayout(mode_layout)

        # ---- Body: left panel + right sidebar ----
        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)

        # ==== Left panel: video + progress + buttons ====
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000000; border: 2px solid #333;")
        self.video_label.setText("Drop a video file or use File -> Open")
        self.video_label.installEventFilter(self)
        self.video_label.setMouseTracking(True)
        left_panel.addWidget(self.video_label, stretch=1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(28)
        left_panel.addWidget(self.progress_bar)

        # ---- Button row (common) ----
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(16)

        self.btn_open = QPushButton("Open Video")
        self.btn_open.clicked.connect(self._on_open)
        self.btn_open.setMinimumHeight(52)
        self.btn_open.setMinimumWidth(160)
        btn_layout.addWidget(self.btn_open)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_start.setMinimumHeight(52)
        self.btn_start.setMinimumWidth(130)
        btn_layout.addWidget(self.btn_start)

        self.btn_end = QPushButton("End")
        self.btn_end.clicked.connect(self._on_end)
        self.btn_end.setMinimumHeight(52)
        self.btn_end.setMinimumWidth(130)
        btn_layout.addWidget(self.btn_end)

        # Mode-specific widgets
        self._capture_btns: List[QWidget] = []
        self._register_btns: List[QWidget] = []
        self._recognize_btns: List[QWidget] = []

        # Capture-only buttons
        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setMinimumHeight(52)
        self.btn_save.setMinimumWidth(160)
        btn_layout.addWidget(self.btn_save)
        self._capture_btns.append(self.btn_save)

        self.btn_replay = QPushButton("Replay")
        self.btn_replay.clicked.connect(self._on_replay)
        self.btn_replay.setMinimumHeight(52)
        self.btn_replay.setMinimumWidth(130)
        btn_layout.addWidget(self.btn_replay)
        self._capture_btns.append(self.btn_replay)

        # Register-only widgets
        self.action_name_input = QLineEdit()
        self.action_name_input.setPlaceholderText("Action name (e.g. shooting)")
        self.action_name_input.setMinimumHeight(52)
        self.action_name_input.setMinimumWidth(250)
        self.action_name_input.setVisible(False)
        btn_layout.addWidget(self.action_name_input)
        self._register_btns.append(self.action_name_input)

        self.btn_do_register = QPushButton("Register")
        self.btn_do_register.clicked.connect(self._on_register_action)
        self.btn_do_register.setMinimumHeight(52)
        self.btn_do_register.setMinimumWidth(160)
        self.btn_do_register.setVisible(False)
        btn_layout.addWidget(self.btn_do_register)
        self._register_btns.append(self.btn_do_register)

        # Recognize-only buttons
        self.btn_do_recognize = QPushButton("Recognize")
        self.btn_do_recognize.clicked.connect(self._on_recognize_actions)
        self.btn_do_recognize.setMinimumHeight(52)
        self.btn_do_recognize.setMinimumWidth(160)
        self.btn_do_recognize.setVisible(False)
        btn_layout.addWidget(self.btn_do_recognize)
        self._recognize_btns.append(self.btn_do_recognize)

        self.btn_exit = QPushButton("Exit")
        self.btn_exit.clicked.connect(self.close)
        self.btn_exit.setMinimumHeight(52)
        self.btn_exit.setMinimumWidth(130)
        btn_layout.addWidget(self.btn_exit)

        left_panel.addLayout(btn_layout)
        body_layout.addLayout(left_panel, stretch=1)

        # ==== Right sidebar ====
        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)

        sidebar_title = QLabel("Manual Points")
        sidebar_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ccc;")
        sidebar.addWidget(sidebar_title)

        self.btn_add_point = QPushButton("Add Manual Point")
        self.btn_add_point.setCheckable(True)
        self.btn_add_point.clicked.connect(self._on_toggle_add_point)
        self.btn_add_point.setMinimumHeight(44)
        self.btn_add_point.setStyleSheet(
            "QPushButton { background-color: #3a3a3a; color: #e0e0e0; "
            "border: 2px solid #555; border-radius: 6px; padding: 10px 18px; "
            "font-size: 18px; font-weight: bold; }"
            "QPushButton:checked { background-color: #005a9e; border-color: #0078d4; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }"
        )
        sidebar.addWidget(self.btn_add_point)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        self.btn_export_csv.setMinimumHeight(44)
        sidebar.addWidget(self.btn_export_csv)

        self.manual_points_list = QListWidget()
        self.manual_points_list.setMinimumWidth(200)
        sidebar.addWidget(self.manual_points_list, stretch=1)

        # Action recognition info area
        self.ar_info_label = QLabel("")
        self.ar_info_label.setWordWrap(True)
        self.ar_info_label.setStyleSheet("font-size: 16px; color: #8af; padding: 4px;")
        sidebar.addWidget(self.ar_info_label)

        # Wrap sidebar in a widget with fixed width
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setMaximumWidth(280)
        body_layout.addWidget(sidebar_widget)

        main_layout.addLayout(body_layout, stretch=1)

    def _set_button_states(self):
        """Update button enabled states based on current mode."""
        is_idle = self.mode == "IDLE"
        is_recording = self.mode == "RECORDING"
        is_replay_ready = self.mode == "REPLAY_READY"

        self.btn_open.setEnabled(is_idle or is_replay_ready)
        self.btn_start.setEnabled((is_idle or is_replay_ready) and self.video_path is not None)
        self.btn_end.setEnabled(is_recording)
        self.btn_add_point.setEnabled(self.video_path is not None and not self.mode == "REPLAY")
        self.btn_export_csv.setEnabled(is_replay_ready)
        self.progress_bar.setVisible(is_recording)

        # Capture mode buttons
        for w in self._capture_btns:
            w.setVisible(self.op_mode == "capture")
        self.btn_save.setEnabled(is_replay_ready and self.op_mode == "capture")
        self.btn_replay.setEnabled(is_replay_ready and self.op_mode == "capture")

        # Register mode buttons
        for w in self._register_btns:
            w.setVisible(self.op_mode == "register")
        self.btn_do_register.setEnabled(
            self.op_mode == "register" and is_replay_ready and
            bool(self.action_name_input.text().strip())
        )
        self.action_name_input.setEnabled(
            self.op_mode == "register" and not is_recording
        )

        # Recognize mode buttons
        for w in self._recognize_btns:
            w.setVisible(self.op_mode == "recognize")
        self.btn_do_recognize.setEnabled(
            self.op_mode == "recognize" and is_replay_ready
        )

        # Mode toggle buttons
        self.btn_capture.setChecked(self.op_mode == "capture")
        self.btn_register.setChecked(self.op_mode == "register")
        self.btn_recognize.setChecked(self.op_mode == "recognize")

    # --------------- Mode Switching ---------------

    def _on_switch_mode(self, op_mode: str):
        """Switch between capture / register / recognize modes."""
        if self.mode == "RECORDING":
            return  # can't switch while recording
        self.op_mode = op_mode
        self._set_button_states()

    # --------------- Action Recognition ---------------

    def _ensure_action_store(self):
        """Lazy-init the template store and load from disk."""
        if self._action_store is None:
            self._action_store = TemplateStore(target_len=60)
            tp = self._templates_path
            if os.path.exists(tp):
                try:
                    self._action_store.load(tp)
                except Exception:
                    pass
        return self._action_store

    def _on_register_action(self):
        """Register the processed recording as a new action template."""
        name = self.action_name_input.text().strip()
        if not name or self._positions_smooth is None:
            QMessageBox.warning(self, "Register", "Please enter an action name and process a video first.")
            return

        if not self._angle_defs:
            self._load_angle_defs()

        feats = extract_angle_features(self._positions_smooth, self._angle_defs, window=5)
        store = self._ensure_action_store()
        store.add(feats, name, self.fps)
        store.save(self._templates_path)
        QMessageBox.information(
            self, "Registered",
            f"Action '{name}' registered!\nTotal templates: {len(store)}"
        )
        self._update_ar_info()

    def _on_recognize_actions(self):
        """Run action recognition on the processed recording."""
        if self._positions_smooth is None:
            QMessageBox.warning(self, "Recognize", "Please process a video first.")
            return

        if not self._angle_defs:
            self._load_angle_defs()

        store = self._ensure_action_store()
        if len(store) == 0:
            QMessageBox.information(self, "Recognize", "No templates loaded. Register an action first.")
            return

        feats = extract_angle_features(self._positions_smooth, self._angle_defs, window=5)
        rec = ActionRecognizer(store, sensitivity=2.0)
        self._action_matches = rec.recognize(feats, self.fps)

        if self._action_matches:
            msg = f"Detected {len(self._action_matches)} action(s):\n"
            for m in self._action_matches:
                msg += (f"  * {m.action_name} | "
                        f"frame {m.start_frame}-{m.end_frame} | "
                        f"time {m.start_sec:.1f}s-{m.end_sec:.1f}s | "
                        f"confidence {m.confidence:.0%}\n")
            QMessageBox.information(self, "Recognition Results", msg)
        else:
            QMessageBox.information(self, "Recognition Results", "No actions detected.")
        self._update_ar_info()

    def _load_angle_defs(self):
        """Load angle definitions from landmarks.yaml."""
        config_path = Path(__file__).resolve().parent.parent.parent / "config" / "landmarks.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self._angle_defs = {
                name: tuple(indices)
                for name, indices in config.get("angle_definitions", {}).items()
            }

    def _update_ar_info(self):
        """Update the action recognition info label in the sidebar."""
        parts = []
        if self._action_store and len(self._action_store) > 0:
            parts.append(f"Templates: {len(self._action_store)}")
            for tmpl in self._action_store.templates:
                parts.append(f"  - {tmpl.name}")
        if self._action_matches:
            parts.append(f"Detected: {len(self._action_matches)}")
            for m in self._action_matches:
                parts.append(f"  [{m.action_name}] f{m.start_frame}-{m.end_frame}")
        self.ar_info_label.setText("\n".join(parts))

    # --------------- Event Filter for Mouse Clicks ---------------

    def eventFilter(self, obj, event):
        if obj is self.video_label and event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton and self._adding_manual_point:
                self._handle_video_click(event.pos().x(), event.pos().y())
                return True
        return super().eventFilter(obj, event)

    def _handle_video_click(self, click_x: int, click_y: int):
        """Convert click coordinates to frame coordinates and add a manual point."""
        frame_x = (click_x - self._display_offset_x) / self._display_scale
        frame_y = (click_y - self._display_offset_y) / self._display_scale
        frame_x = max(0.0, min(frame_x, self._frame_orig_w - 1))
        frame_y = max(0.0, min(frame_y, self._frame_orig_h - 1))

        point_id = self.point_manager.add_point(frame_x, frame_y)

        if self.mode == "RECORDING" and self._worker:
            self._worker.add_manual_point(frame_x, frame_y)
        else:
            self._pending_manual_points.append((frame_x, frame_y))

        self._add_point_to_sidebar(point_id)
        self._set_button_states()

    def _add_point_to_sidebar(self, point_id: int):
        """Add a manual point entry to the sidebar list widget."""
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(6, 2, 6, 2)
        item_layout.setSpacing(8)

        label = QLabel(f"Point #{point_id}")
        label.setStyleSheet("color: #8af;")
        item_layout.addWidget(label)

        item_layout.addStretch()

        delete_btn = QPushButton("X")
        delete_btn.setFixedSize(24, 24)
        delete_btn.setStyleSheet(
            "QPushButton { background-color: #5a2020; color: #ff6666; border: 1px solid #833; "
            "border-radius: 2px; font-weight: bold; }"
            "QPushButton:hover { background-color: #7a3030; }"
        )
        delete_btn.clicked.connect(lambda checked, pid=point_id: self._delete_manual_point(pid))
        item_layout.addWidget(delete_btn)

        list_item = QListWidgetItem()
        list_item.setData(Qt.UserRole, point_id)
        list_item.setSizeHint(item_widget.sizeHint())
        self.manual_points_list.addItem(list_item)
        self.manual_points_list.setItemWidget(list_item, item_widget)

    def _delete_manual_point(self, point_id: int):
        """Remove a manual point by ID."""
        self.point_manager.delete_point(point_id)
        for i in range(self.manual_points_list.count()):
            item = self.manual_points_list.item(i)
            if item.data(Qt.UserRole) == point_id:
                self.manual_points_list.takeItem(i)
                break

    # --------------- Slots ---------------

    def _on_toggle_add_point(self):
        self._adding_manual_point = self.btn_add_point.isChecked()
        if self._adding_manual_point:
            self.video_label.setCursor(Qt.CrossCursor)
        else:
            self.video_label.setCursor(Qt.ArrowCursor)

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

        self._manual_trajectories.clear()
        self._pending_manual_points.clear()
        self.point_manager.clear()
        self.manual_points_list.clear()
        self._adding_manual_point = False
        self.btn_add_point.setChecked(False)
        self.video_label.setCursor(Qt.ArrowCursor)
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

        self._manual_trajectories.clear()
        self.point_manager.clear()
        self.manual_points_list.clear()

        # Re-add pending manual points to the fresh PointManager
        for px, py in self._pending_manual_points:
            pid = self.point_manager.add_point(px, py)
            self._add_point_to_sidebar(pid)

        self._worker = VideoWorker(self.video_path)
        self._worker.initial_manual_points = list(self._pending_manual_points)
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
            self._cache_positions_for_ar()
        else:
            self.mode = "IDLE"
        self._adding_manual_point = False
        self.btn_add_point.setChecked(False)
        self.video_label.setCursor(Qt.ArrowCursor)
        self._set_button_states()

    def _on_replay(self):
        """Start black-background replay with trajectories."""
        if not self.recording_data:
            return
        self.mode = "REPLAY"
        self._set_button_states()
        self._replay_idx = 0
        self._ball_trajectory.clear()
        self._ball_velocity_history.clear()

        self._manual_trajectories.clear()

        # Stop existing timer
        if self._replay_timer:
            self._replay_timer.stop()

        interval_ms = int(1000.0 / self.fps)
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._replay_step)
        self._replay_timer.start(interval_ms)

    def _on_frame_processed(self, data: tuple):
        """Receive processed frame from worker thread."""
        frame_idx, frame_bgr, landmarks, ball_pos, manual_positions = data
        self.recording_data.append((frame_idx, frame_bgr, landmarks, ball_pos, manual_positions))

        # Update PointManager history from worker's tracking results
        for pid, pos in manual_positions.items():
            pt = self.point_manager.points.get(pid)
            if pt is not None and pt["active"]:
                pt["pos"] = pos
                pt["history"].append((frame_idx, pos[0], pos[1]))

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
            self._cache_positions_for_ar()
            self._adding_manual_point = False
            self.btn_add_point.setChecked(False)
            self.video_label.setCursor(Qt.ArrowCursor)
            self._set_button_states()

    def _cache_positions_for_ar(self):
        """Build interpolated+smoothed positions array for action recognition."""
        from src.tracker import Tracker
        poses = [d[2] for d in self.recording_data]  # landmarks only
        T = len(poses)
        positions = np.full((T, 33, 2), np.nan, dtype=np.float64)
        for i, pose in enumerate(poses):
            if pose is not None:
                for ki in range(min(33, len(pose))):
                    positions[i, ki, 0] = pose[ki][0]
                    positions[i, ki, 1] = pose[ki][1]
        tracker = Tracker()
        tracker.add_frames(poses)
        positions = tracker.interpolate()
        self._positions_smooth = tracker.smooth(positions, window=5)

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

    def _on_export_csv(self):
        """Export all tracking data (human + manual) to CSV."""
        if not self.recording_data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Tracking CSV", "output/tracking_data.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self._export_tracking_csv(path)

    # --------------- Display ---------------

    def _display_frame(self, frame_bgr: np.ndarray, is_preview: bool = False):
        """Convert a BGR frame to QPixmap and show it."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Scale to fit label while keeping aspect ratio
        label_w = self.video_label.width()
        label_h = self.video_label.height()

        # Store info for mouse-coordinate mapping
        self._frame_orig_w = w
        self._frame_orig_h = h

        if label_w > 10 and label_h > 10:
            scale = min(label_w / w, label_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            rgb = cv2.resize(rgb, (new_w, new_h))
            self._display_scale = scale
            self._display_offset_x = (label_w - new_w) // 2
            self._display_offset_y = (label_h - new_h) // 2
        else:
            self._display_scale = 1.0
            self._display_offset_x = 0
            self._display_offset_y = 0

        qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.setPixmap(pixmap)

    def _display_replay_frame(self, canvas_bgr: np.ndarray):
        """Display a replay canvas."""
        self._display_frame(canvas_bgr, is_preview=False)

    # --------------- Replay Engine ---------------

    def _replay_step(self):
        """Render one replay frame on black background with trajectories."""
        if self._replay_idx >= len(self.recording_data):
            self._replay_timer.stop()
            self.mode = "REPLAY_READY"
            self._set_button_states()
            QMessageBox.information(self, "Replay Complete", "Replay finished.")
            return

        frame_idx, frame_bgr, landmarks, ball_pos, manual_positions = self.recording_data[self._replay_idx]
        h, w = frame_bgr.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw skeleton
        if landmarks is not None:
            canvas = self._draw_skeleton(canvas, landmarks, CYAN)

        # Accumulate and draw detected basketball trajectory (orange).
        # None marks frames where the ball is out of view, so the trail breaks.
        self._ball_trajectory.append(ball_pos)
        # Temporarily hide the basketball trajectory to focus on body motion.
        # self._draw_ball_trajectory(canvas, self._ball_trajectory)

        # Accumulate and draw manual point trajectories (blue)
        for pid, pos in manual_positions.items():
            if pid not in self._manual_trajectories:
                self._manual_trajectories[pid] = []
            self._manual_trajectories[pid].append(pos)

        canvas = self._draw_trajectories(canvas, self._manual_trajectories, BLUE)

        # Draw small circles at each tracked point + larger circle at current position
        for pid, pos in manual_positions.items():
            px, py = int(np.clip(pos[0], 0, w - 1)), int(np.clip(pos[1], 0, h - 1))
            # Current position: larger filled circle + outline
            cv2.circle(canvas, (px, py), 6, BLUE, -1)
            cv2.circle(canvas, (px, py), 9, BLUE, 2)

        # Draw small dots on every tracked sample point along each trajectory
        for pid, pts in self._manual_trajectories.items():
            for px, py in pts:
                cx, cy = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
                cv2.circle(canvas, (cx, cy), 3, BLUE, -1)

        # Draw measurement overlay
        vel, dist, elapsed = self._calc_measurements(frame_idx)
        canvas = self._draw_measurements(canvas, vel, dist, elapsed)

        # Draw action recognition labels
        if self._action_matches:
            canvas = self._draw_action_labels(canvas, frame_idx)

        self._display_replay_frame(canvas)
        self._replay_idx += 1

    def _draw_trajectories(
        self,
        canvas: np.ndarray,
        trajectories: Dict[int, List[Tuple[float, float]]],
        color: Tuple[int, int, int],
    ) -> np.ndarray:
        """Draw all accumulated trajectories on canvas."""
        h, w = canvas.shape[:2]
        for pts in trajectories.values():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                x1, y1 = pts[i - 1]
                x2, y2 = pts[i]
                p1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
                p2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
                cv2.line(canvas, p1, p2, color, 2)
        return canvas

    def _draw_ball_trajectory(self, canvas: np.ndarray, trajectory: List[Optional[Tuple[float, float]]]) -> np.ndarray:
        """Draw the basketball trajectory and current detected position."""
        if not trajectory:
            return canvas

        h, w = canvas.shape[:2]
        for i in range(1, len(trajectory)):
            if trajectory[i - 1] is None or trajectory[i] is None:
                continue
            x1, y1 = trajectory[i - 1]
            x2, y2 = trajectory[i]
            if np.hypot(x2 - x1, y2 - y1) > 95.0:
                continue
            p1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
            p2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
            cv2.line(canvas, p1, p2, ORANGE, 2)

        for point in trajectory[:: max(1, len(trajectory) // 80)]:
            if point is None:
                continue
            x, y = point
            px, py = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            cv2.circle(canvas, (px, py), 2, ORANGE, -1)

        latest = trajectory[-1]
        if latest is None:
            return canvas
        x, y = latest
        px, py = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        cv2.circle(canvas, (px, py), 8, ORANGE, -1)
        cv2.circle(canvas, (px, py), 12, WHITE, 2)
        return canvas

    def _draw_skeleton(self, canvas: np.ndarray, landmarks: PoseResult, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw skeleton stick figure on canvas."""
        h, w = canvas.shape[:2]

        for idx in SKELETON_KEYPOINTS:
            if idx >= len(landmarks):
                continue
            x, y, conf = landmarks[idx]
            if conf < 0.3:
                continue
            px, py = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            cv2.circle(canvas, (px, py), 4, color, -1)

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
        """Calculate instantaneous velocity, cumulative distance, elapsed time."""
        elapsed = frame_idx / self.fps

        ball_positions = []
        for i in range(min(frame_idx + 1, len(self.recording_data))):
            _, _, _, bp, _ = self.recording_data[i]
            if bp is not None:
                ball_positions.append(bp)

        dist_px = 0.0
        for i in range(1, len(ball_positions)):
            dx = ball_positions[i][0] - ball_positions[i - 1][0]
            dy = ball_positions[i][1] - ball_positions[i - 1][1]
            dist_px += np.sqrt(dx * dx + dy * dy)
        dist_mm = dist_px * self.mm_per_pixel

        vel_mm_s = 0.0
        if len(ball_positions) >= 2:
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
            cv2.putText(canvas, text, (12, y - 1), font, font_scale, outline, thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (12, y + 1), font, font_scale, outline, thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return canvas

    def _draw_action_labels(self, canvas: np.ndarray, frame_idx: int) -> np.ndarray:
        """Overlay action name banner on canvas when frame_idx falls within a match."""
        if not self._action_matches:
            return canvas
        h, w = canvas.shape[:2]
        for m in self._action_matches:
            if m.start_frame <= frame_idx <= m.end_frame:
                overlay = canvas.copy()
                banner_h = 60
                cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 100, 0), -1)
                cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
                text = f"{m.action_name}  ({m.confidence:.0%})"
                font_scale = 1.2
                thickness = 3
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                tx = (w - tw) // 2
                ty = (banner_h + th) // 2
                cv2.putText(canvas, text, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), thickness, cv2.LINE_AA)
        return canvas

    def _compute_measurements(self):
        """Pre-compute summary measurements after recording finishes."""
        ball_positions = []
        for _, _, _, bp, _ in self.recording_data:
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
        print(f"Manual points recorded: {len(self._manual_trajectories)}")

    # --------------- Save ---------------

    def _save_replay_video(self, output_path: str):
        """Render replay frames to an MP4 file with trajectories."""
        if not self.recording_data:
            return

        first_frame = self.recording_data[0][1]
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        saved_manual: Dict[int, List[Tuple[float, float]]] = {}
        saved_ball: List[Optional[Tuple[float, float]]] = []

        for frame_idx, frame_bgr, landmarks, ball_pos, manual_positions in self.recording_data:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            if landmarks is not None:
                canvas = self._draw_skeleton(canvas, landmarks, CYAN)

            saved_ball.append(ball_pos)
            # Temporarily hide the basketball trajectory to focus on body motion.
            # self._draw_ball_trajectory(canvas, saved_ball)

            # Accumulate and draw manual trajectories
            for pid, pos in manual_positions.items():
                if pid not in saved_manual:
                    saved_manual[pid] = []
                saved_manual[pid].append(pos)

            canvas = self._draw_trajectories(canvas, saved_manual, BLUE)

            for pid, pos in manual_positions.items():
                px, py = int(np.clip(pos[0], 0, w - 1)), int(np.clip(pos[1], 0, h - 1))
                cv2.circle(canvas, (px, py), 6, BLUE, -1)
                cv2.circle(canvas, (px, py), 9, BLUE, 2)

            # Small dots on every tracked sample point
            for pid, pts in saved_manual.items():
                for px, py in pts:
                    cx, cy = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
                    cv2.circle(canvas, (cx, cy), 3, BLUE, -1)

            vel, dist, elapsed = self._calc_measurements(frame_idx)
            canvas = self._draw_measurements(canvas, vel, dist, elapsed)

            # Draw action labels in saved video
            if self._action_matches:
                canvas = self._draw_action_labels(canvas, frame_idx)

            writer.write(canvas)

        writer.release()
        QMessageBox.information(self, "Saved", f"Replay video saved to:\n{output_path}")

    # --------------- Export ---------------

    def _export_tracking_csv(self, output_path: str):
        """Export all tracking data (human keypoints + manual points) to CSV.

        Format (long form): frame_idx, time_sec, point_type, point_id, name, x, y
        """
        if not self.recording_data:
            return

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_idx", "time_sec", "point_type", "point_id", "name", "x", "y"])

            for frame_idx, frame_bgr, landmarks, ball_pos, manual_positions in self.recording_data:
                time_sec = frame_idx / self.fps

                # Human keypoints
                if landmarks is not None:
                    for kp_idx in range(len(landmarks)):
                        x, y, conf = landmarks[kp_idx]
                        if conf >= 0.3:
                            name = KEYPOINT_NAMES[kp_idx] if kp_idx < len(KEYPOINT_NAMES) else f"kp_{kp_idx}"
                            writer.writerow([frame_idx, f"{time_sec:.4f}", "human", kp_idx, name, f"{x:.2f}", f"{y:.2f}"])

                # Manual points
                for pid, pos in manual_positions.items():
                    writer.writerow([frame_idx, f"{time_sec:.4f}", "manual", pid, f"manual_{pid}", f"{pos[0]:.2f}", f"{pos[1]:.2f}"])

        QMessageBox.information(self, "Exported", f"Tracking data saved to:\n{output_path}")
