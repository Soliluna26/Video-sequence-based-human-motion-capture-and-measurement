"""Main PyQt5 application window for human motion capture and measurement.

Provides: video loading, live recording with background processing,
black-background replay with skeleton, manual point tracking,
and trajectory visualization.
"""

import csv
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.gui.video_worker import VideoWorker
from src.point_manager import PointManager
from src.action_recognizer import (
    ActionRecognizer,
    TemplateStore,
    extract_angle_features,
    recognize_rule_based_actions,
)
from src.kinematics import angular_velocity, compute_segment_angle, compute_segment_angles_from_landmarks

KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]
PoseResult = Any

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
THIGH_SEGMENT_ANGLE_THRESHOLD_DEG = 30.0
MEASURE2_ANGLE_NAMES = [
    ("thigh_jump_angle", "Thigh Jump Angle"),
    ("calf_jump_angle", "Calf Jump Angle"),
    ("foot_angle", "Foot Angle"),
]


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
        self.op_mode: str = "capture"  # capture | register | recognize | measure | measure2
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
        self._thigh_angle_deg: Optional[np.ndarray] = None
        self._thigh_velocity_deg_s: Optional[np.ndarray] = None
        self._thigh_measure_active: Optional[np.ndarray] = None
        self._measure2_angles: Dict[str, np.ndarray] = {}
        self._measure2_velocities: Dict[str, np.ndarray] = {}

        self._setup_ui()
        self._ensure_action_store()
        self._refresh_template_selector()
        self._update_ar_info()
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

        self.btn_measure = QPushButton("测量")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(lambda: self._on_switch_mode("measure"))
        self.btn_measure.setMinimumHeight(52)
        self.btn_measure.setMinimumWidth(160)
        mode_layout.addWidget(self.btn_measure)

        self.btn_measure2 = QPushButton("测2")
        self.btn_measure2.setCheckable(True)
        self.btn_measure2.clicked.connect(lambda: self._on_switch_mode("measure2"))
        self.btn_measure2.setMinimumHeight(52)
        self.btn_measure2.setMinimumWidth(160)
        mode_layout.addWidget(self.btn_measure2)

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
        self._measure_btns: List[QWidget] = []

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

        self.btn_export_angle_velocity_csv = QPushButton("Export Angle Velocity CSV")
        self.btn_export_angle_velocity_csv.clicked.connect(self._on_export_angle_velocity_csv)
        self.btn_export_angle_velocity_csv.setMinimumHeight(44)
        sidebar.addWidget(self.btn_export_angle_velocity_csv)
        self._measure_btns.append(self.btn_export_angle_velocity_csv)

        self.chk_export_thigh_metrics = QCheckBox("Measurement CSV")
        self.chk_export_thigh_metrics.setChecked(True)
        self.chk_export_thigh_metrics.setStyleSheet("font-size: 15px; color: #ddd;")
        sidebar.addWidget(self.chk_export_thigh_metrics)
        self._measure_btns.append(self.chk_export_thigh_metrics)

        self.manual_points_list = QListWidget()
        self.manual_points_list.setMinimumWidth(200)
        sidebar.addWidget(self.manual_points_list, stretch=1)

        # Action recognition info area
        action_title = QLabel("Action Templates")
        action_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #ccc;")
        sidebar.addWidget(action_title)

        self.template_selector = QComboBox()
        self.template_selector.setMinimumHeight(36)
        sidebar.addWidget(self.template_selector)

        self.btn_delete_template = QPushButton("Delete Action")
        self.btn_delete_template.clicked.connect(self._on_delete_action_template)
        self.btn_delete_template.setMinimumHeight(40)
        self.btn_delete_template.setStyleSheet(
            "QPushButton { background-color: #5a2020; color: #ffb0b0; "
            "border: 1px solid #833; border-radius: 6px; padding: 8px 12px; }"
            "QPushButton:hover { background-color: #743030; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }"
        )
        sidebar.addWidget(self.btn_delete_template)

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
            w.setVisible(self.op_mode in ("capture", "measure", "measure2"))
        self.btn_save.setEnabled(is_replay_ready and self.op_mode in ("capture", "measure", "measure2"))
        self.btn_replay.setEnabled(is_replay_ready and self.op_mode in ("capture", "measure", "measure2"))

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

        # Measure mode widgets
        for w in self._measure_btns:
            w.setVisible(self.op_mode in ("measure", "measure2"))
        self.btn_export_angle_velocity_csv.setEnabled(
            self.op_mode in ("measure", "measure2") and is_replay_ready
        )
        self.chk_export_thigh_metrics.setEnabled(
            self.op_mode in ("measure", "measure2") and is_replay_ready
        )

        has_templates = self.template_selector.count() > 0
        self.template_selector.setEnabled(has_templates and not is_recording)
        self.btn_delete_template.setEnabled(has_templates and not is_recording)

        # Mode toggle buttons
        self.btn_capture.setChecked(self.op_mode == "capture")
        self.btn_register.setChecked(self.op_mode == "register")
        self.btn_recognize.setChecked(self.op_mode == "recognize")
        self.btn_measure.setChecked(self.op_mode == "measure")
        self.btn_measure2.setChecked(self.op_mode == "measure2")

    # --------------- Mode Switching ---------------

    def _on_switch_mode(self, op_mode: str):
        """Switch between capture / register / recognize modes."""
        if self.mode == "RECORDING":
            return  # can't switch while recording
        self.op_mode = op_mode
        if self.op_mode == "measure" and self.mode == "REPLAY_READY" and self.recording_data:
            self._cache_thigh_measurements()
        if self.op_mode == "measure2" and self.mode == "REPLAY_READY" and self.recording_data:
            self._cache_measure2_measurements()
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

    def _refresh_template_selector(self):
        """Refresh the registered-action selector in the sidebar."""
        if not hasattr(self, "template_selector"):
            return

        current_id = self.template_selector.currentData()
        self.template_selector.blockSignals(True)
        self.template_selector.clear()

        store = self._action_store
        if store:
            for tmpl in store.templates:
                self.template_selector.addItem(
                    f"{tmpl.name} ({tmpl.template_id})",
                    tmpl.template_id,
                )

        if current_id:
            idx = self.template_selector.findData(current_id)
            if idx >= 0:
                self.template_selector.setCurrentIndex(idx)

        self.template_selector.blockSignals(False)
        self._set_button_states()

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
        self._refresh_template_selector()
        self._update_ar_info()

    def _on_delete_action_template(self):
        """Delete the selected registered action template."""
        store = self._ensure_action_store()
        template_id = self.template_selector.currentData()
        if not template_id:
            QMessageBox.information(self, "Delete Action", "No registered action selected.")
            return

        tmpl = store.get(template_id)
        if tmpl is None:
            QMessageBox.warning(self, "Delete Action", "The selected action no longer exists.")
            self._refresh_template_selector()
            self._update_ar_info()
            return

        reply = QMessageBox.question(
            self,
            "Delete Action",
            f"Delete registered action '{tmpl.name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        if store.remove(template_id):
            store.save(self._templates_path)
            self._action_matches = [
                m for m in self._action_matches
                if getattr(m, "action_name", None) != tmpl.name
            ]
            self._refresh_template_selector()
            self._update_ar_info()
            QMessageBox.information(self, "Delete Action", f"Action '{tmpl.name}' deleted.")

    def _on_recognize_actions(self):
        """Run action recognition on the processed recording."""
        if self._positions_smooth is None:
            QMessageBox.warning(self, "Recognize", "Please process a video first.")
            return

        if not self._angle_defs:
            self._load_angle_defs()

        store = self._ensure_action_store()
        self._action_matches = recognize_rule_based_actions(self._positions_smooth, self.fps)

        if len(store) > 0:
            feats = extract_angle_features(self._positions_smooth, self._angle_defs, window=5)
            rec = ActionRecognizer(store, sensitivity=2.0)
            self._action_matches.extend(rec.recognize(feats, self.fps))
            self._action_matches.sort(key=lambda m: (m.start_frame, m.action_name))

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
        if sys.version_info >= (3, 14):
            QMessageBox.critical(
                self,
                "MediaPipe Environment Error",
                "Current virtual environment uses Python "
                f"{sys.version_info.major}.{sys.version_info.minor}. "
                "MediaPipe Pose cannot run reliably on Python 3.14 in this project.\n\n"
                "Please recreate .venv with Python 3.11 or 3.12 and reinstall requirements."
            )
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
            if self.op_mode == "measure":
                self._cache_thigh_measurements()
            elif self.op_mode == "measure2":
                self._cache_measure2_measurements()
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
            if self.op_mode == "measure":
                self._cache_thigh_measurements()
            elif self.op_mode == "measure2":
                self._cache_measure2_measurements()
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

    def _on_export_angle_velocity_csv(self):
        """Export only thigh segment angle and angular velocity to CSV."""
        if not self.recording_data:
            return
        if self.op_mode not in ("measure", "measure2"):
            QMessageBox.information(self, "Measure", "Please switch to Measure mode first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Angle Velocity CSV", "output/angle_velocity.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self._export_angle_velocity_csv(path)

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

        # Draw action recognition labels
        if self._action_matches:
            canvas = self._draw_action_labels(canvas, frame_idx)

        if self.op_mode == "measure":
            canvas = self._draw_thigh_measurement(canvas, frame_idx)
        elif self.op_mode == "measure2":
            canvas = self._draw_measure2_measurement(canvas, frame_idx)

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

    def _draw_thigh_measurement(self, canvas: np.ndarray, frame_idx: int) -> np.ndarray:
        """Overlay thigh segment angle and angular velocity for the current frame."""
        if self.op_mode != "measure":
            return canvas
        if (
            self._thigh_angle_deg is None or
            self._thigh_velocity_deg_s is None or
            self._thigh_measure_active is None or
            frame_idx >= len(self._thigh_angle_deg)
        ):
            return canvas

        angle = self._thigh_angle_deg[frame_idx]
        if np.isnan(angle):
            return canvas

        active = bool(self._thigh_measure_active[frame_idx])
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        panel_h = 78
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        if active:
            vel = self._thigh_velocity_deg_s[frame_idx]
            vel_text = "n/a" if np.isnan(vel) else f"{vel:.2f} deg/s"
            text = f"Thigh segments angle: {angle:.2f} deg | Angular velocity: {vel_text}"
            color = (120, 255, 120)
        else:
            text = f"Thigh segments angle: {angle:.2f} deg | Waiting > {THIGH_SEGMENT_ANGLE_THRESHOLD_DEG:.0f} deg"
            color = (180, 180, 180)

        cv2.putText(
            canvas, text, (18, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA,
        )
        return canvas

    def _cache_thigh_measurements(self):
        """Cache angle/velocity between left and right thigh segments."""
        T = len(self.recording_data)
        positions = np.full((T, 33, 2), np.nan, dtype=np.float64)
        for i, (_, _, landmarks, _, _) in enumerate(self.recording_data):
            if landmarks is None:
                continue
            for ki in (23, 24, 25, 26):
                if ki >= len(landmarks):
                    continue
                x, y, conf = landmarks[ki]
                if conf >= 0.3:
                    positions[i, ki, 0] = x
                    positions[i, ki, 1] = y

        angles = compute_segment_angles_from_landmarks(
            positions,
            {"thigh_segments_angle": (23, 25, 24, 26)},
        )["thigh_segments_angle"]
        active = angles > THIGH_SEGMENT_ANGLE_THRESHOLD_DEG
        velocities = angular_velocity(angles, self.fps)
        velocities = np.where(active, velocities, np.nan)

        self._thigh_angle_deg = angles
        self._thigh_velocity_deg_s = velocities
        self._thigh_measure_active = active

    @staticmethod
    def _segment_vertical_angle(start: np.ndarray, end: np.ndarray) -> float:
        """Angle between a segment and the image vertical line, folded to 0-90 deg."""
        angle = compute_segment_angle(tuple(start), tuple(end), (0.0, 0.0), (0.0, 1.0))
        return min(angle, 180.0 - angle)

    @staticmethod
    def _nanmean_pair(left: float, right: float) -> float:
        values = np.array([left, right], dtype=np.float64)
        if np.all(np.isnan(values)):
            return np.nan
        return float(np.nanmean(values))

    def _cache_measure2_measurements(self):
        """Cache jump-related thigh, calf, and foot angle measurements."""
        T = len(self.recording_data)
        angles = {key: np.full(T, np.nan, dtype=np.float64) for key, _ in MEASURE2_ANGLE_NAMES}

        for i, (_, _, landmarks, _, _) in enumerate(self.recording_data):
            if landmarks is None:
                continue

            points: Dict[int, np.ndarray] = {}
            for ki in (23, 24, 25, 26, 27, 28, 31, 32):
                if ki >= len(landmarks):
                    continue
                x, y, conf = landmarks[ki]
                if conf >= 0.3:
                    points[ki] = np.array([x, y], dtype=np.float64)

            left_thigh = (
                self._segment_vertical_angle(points[23], points[25])
                if 23 in points and 25 in points else np.nan
            )
            right_thigh = (
                self._segment_vertical_angle(points[24], points[26])
                if 24 in points and 26 in points else np.nan
            )
            angles["thigh_jump_angle"][i] = self._nanmean_pair(left_thigh, right_thigh)

            left_calf = (
                self._segment_vertical_angle(points[25], points[27])
                if 25 in points and 27 in points else np.nan
            )
            right_calf = (
                self._segment_vertical_angle(points[26], points[28])
                if 26 in points and 28 in points else np.nan
            )
            angles["calf_jump_angle"][i] = self._nanmean_pair(left_calf, right_calf)

            left_foot = (
                compute_segment_angle(tuple(points[27]), tuple(points[31]), tuple(points[25]), tuple(points[27]))
                if 25 in points and 27 in points and 31 in points else np.nan
            )
            right_foot = (
                compute_segment_angle(tuple(points[28]), tuple(points[32]), tuple(points[26]), tuple(points[28]))
                if 26 in points and 28 in points and 32 in points else np.nan
            )
            angles["foot_angle"][i] = self._nanmean_pair(left_foot, right_foot)

        self._measure2_angles = angles
        self._measure2_velocities = {
            key: angular_velocity(seq, self.fps)
            for key, seq in angles.items()
        }

    def _draw_measure2_measurement(self, canvas: np.ndarray, frame_idx: int) -> np.ndarray:
        """Overlay Measure2 angles and angular velocities for the current frame."""
        if self.op_mode != "measure2":
            return canvas
        if not self._measure2_angles:
            return canvas

        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        panel_h = 126
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0, canvas)

        y = h - 92
        for key, label in MEASURE2_ANGLE_NAMES:
            values = self._measure2_angles.get(key)
            velocities = self._measure2_velocities.get(key)
            if values is None or frame_idx >= len(values) or np.isnan(values[frame_idx]):
                text = f"{label}: n/a"
            else:
                vel = np.nan
                if velocities is not None and frame_idx < len(velocities):
                    vel = velocities[frame_idx]
                vel_text = "n/a" if np.isnan(vel) else f"{vel:.2f} deg/s"
                text = f"{label}: {values[frame_idx]:.2f} deg | {vel_text}"
            cv2.putText(
                canvas, text, (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (120, 255, 120), 2, cv2.LINE_AA,
            )
            y += 34
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

            # Draw action labels in saved video
            if self._action_matches:
                canvas = self._draw_action_labels(canvas, frame_idx)

            if self.op_mode == "measure":
                canvas = self._draw_thigh_measurement(canvas, frame_idx)
            elif self.op_mode == "measure2":
                canvas = self._draw_measure2_measurement(canvas, frame_idx)

            writer.write(canvas)

        writer.release()
        QMessageBox.information(self, "Saved", f"Replay video saved to:\n{output_path}")

    # --------------- Export ---------------

    def _export_angle_velocity_csv(self, output_path: str):
        """Export only angle and angular velocity measurements."""
        if not self.recording_data:
            return
        if self.op_mode not in ("measure", "measure2"):
            return

        if self.op_mode == "measure2":
            if not self._measure2_angles:
                self._cache_measure2_measurements()
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_idx",
                    "time_sec",
                    "thigh_jump_angle_deg",
                    "thigh_jump_angular_velocity_deg_s",
                    "calf_jump_angle_deg",
                    "calf_jump_angular_velocity_deg_s",
                    "foot_angle_deg",
                    "foot_angular_velocity_deg_s",
                ])
                for frame_idx, _, _, _, _ in self.recording_data:
                    time_sec = frame_idx / self.fps if self.fps > 0 else 0.0
                    row = [frame_idx, f"{time_sec:.4f}"]
                    for key, _ in MEASURE2_ANGLE_NAMES:
                        angle_seq = self._measure2_angles.get(key)
                        vel_seq = self._measure2_velocities.get(key)
                        angle = angle_seq[frame_idx] if angle_seq is not None and frame_idx < len(angle_seq) else np.nan
                        velocity = vel_seq[frame_idx] if vel_seq is not None and frame_idx < len(vel_seq) else np.nan
                        row.extend([
                            "" if np.isnan(angle) else f"{angle:.4f}",
                            "" if np.isnan(velocity) else f"{velocity:.4f}",
                        ])
                    writer.writerow(row)
            QMessageBox.information(self, "Exported", f"Angle velocity data saved to:\n{output_path}")
            return

        if self._thigh_angle_deg is None:
            self._cache_thigh_measurements()

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_idx",
                "time_sec",
                "measure_active",
                "angle_deg",
                "angular_velocity_deg_s",
            ])

            for frame_idx, _, _, _, _ in self.recording_data:
                time_sec = frame_idx / self.fps if self.fps > 0 else 0.0
                angle = np.nan
                velocity = np.nan
                active = False
                if self._thigh_angle_deg is not None and frame_idx < len(self._thigh_angle_deg):
                    angle = self._thigh_angle_deg[frame_idx]
                if self._thigh_velocity_deg_s is not None and frame_idx < len(self._thigh_velocity_deg_s):
                    velocity = self._thigh_velocity_deg_s[frame_idx]
                if self._thigh_measure_active is not None and frame_idx < len(self._thigh_measure_active):
                    active = bool(self._thigh_measure_active[frame_idx])

                writer.writerow([
                    frame_idx,
                    f"{time_sec:.4f}",
                    int(active),
                    "" if np.isnan(angle) else f"{angle:.4f}",
                    "" if np.isnan(velocity) else f"{velocity:.4f}",
                ])

        QMessageBox.information(self, "Exported", f"Angle velocity data saved to:\n{output_path}")

    def _export_tracking_csv(self, output_path: str):
        """Export all tracking data (human keypoints + manual points) to CSV.

        Format (long form): frame_idx, time_sec, point_type, point_id, name, x, y
        """
        if not self.recording_data:
            return

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            include_thigh = self.op_mode == "measure" and self.chk_export_thigh_metrics.isChecked()
            include_measure2 = self.op_mode == "measure2" and self.chk_export_thigh_metrics.isChecked()
            header = ["frame_idx", "time_sec", "point_type", "point_id", "name", "x", "y"]
            if include_thigh:
                header.extend([
                    "thigh_measure_active",
                    "thigh_segments_angle_deg",
                    "thigh_segments_angular_velocity_deg_s",
                ])
            elif include_measure2:
                header.extend([
                    "measure2_angle_deg",
                    "measure2_angular_velocity_deg_s",
                ])
            writer.writerow(header)

            for frame_idx, frame_bgr, landmarks, ball_pos, manual_positions in self.recording_data:
                time_sec = frame_idx / self.fps
                thigh_cols = []
                empty_thigh_cols = []
                if include_thigh:
                    if self._thigh_angle_deg is None:
                        self._cache_thigh_measurements()
                    angle = np.nan
                    vel = np.nan
                    active = False
                    if self._thigh_angle_deg is not None and frame_idx < len(self._thigh_angle_deg):
                        angle = self._thigh_angle_deg[frame_idx]
                    if self._thigh_velocity_deg_s is not None and frame_idx < len(self._thigh_velocity_deg_s):
                        vel = self._thigh_velocity_deg_s[frame_idx]
                    if self._thigh_measure_active is not None and frame_idx < len(self._thigh_measure_active):
                        active = bool(self._thigh_measure_active[frame_idx])
                    thigh_cols = [
                        int(active),
                        "" if np.isnan(angle) else f"{angle:.4f}",
                        "" if np.isnan(vel) else f"{vel:.4f}",
                    ]
                    empty_thigh_cols = ["", "", ""]
                elif include_measure2:
                    if not self._measure2_angles:
                        self._cache_measure2_measurements()
                    empty_thigh_cols = ["", ""]

                # Human keypoints
                if landmarks is not None:
                    for kp_idx in range(len(landmarks)):
                        x, y, conf = landmarks[kp_idx]
                        if conf >= 0.3:
                            name = KEYPOINT_NAMES[kp_idx] if kp_idx < len(KEYPOINT_NAMES) else f"kp_{kp_idx}"
                            writer.writerow([
                                frame_idx, f"{time_sec:.4f}", "human", kp_idx,
                                name, f"{x:.2f}", f"{y:.2f}", *empty_thigh_cols,
                            ])

                # Manual points
                for pid, pos in manual_positions.items():
                    writer.writerow([
                        frame_idx, f"{time_sec:.4f}", "manual", pid,
                        f"manual_{pid}", f"{pos[0]:.2f}", f"{pos[1]:.2f}", *empty_thigh_cols,
                    ])

                if include_thigh:
                    writer.writerow([
                        frame_idx, f"{time_sec:.4f}", "measurement", "thigh_segments",
                        "left_hip_left_knee_vs_right_hip_right_knee", "", "", *thigh_cols,
                    ])
                elif include_measure2:
                    for key, label in MEASURE2_ANGLE_NAMES:
                        angle_seq = self._measure2_angles.get(key)
                        vel_seq = self._measure2_velocities.get(key)
                        angle = angle_seq[frame_idx] if angle_seq is not None and frame_idx < len(angle_seq) else np.nan
                        velocity = vel_seq[frame_idx] if vel_seq is not None and frame_idx < len(vel_seq) else np.nan
                        writer.writerow([
                            frame_idx, f"{time_sec:.4f}", "measurement", key, label, "", "",
                            "" if np.isnan(angle) else f"{angle:.4f}",
                            "" if np.isnan(velocity) else f"{velocity:.4f}",
                        ])

        QMessageBox.information(self, "Exported", f"Tracking data saved to:\n{output_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
