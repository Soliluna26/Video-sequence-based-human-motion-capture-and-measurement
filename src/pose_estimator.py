"""MediaPipe Pose estimation module.

Provides per-frame 33-keypoint detection with (x, y, confidence) output
and optional batch processing for efficiency.

Uses the MediaPipe Tasks PoseLandmarker API (0.10.x).
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

if sys.version_info >= (3, 14):
    raise RuntimeError(
        "MediaPipe Pose is not compatible with this project's Python 3.14 "
        "environment. Please recreate the virtual environment with Python "
        "3.11 or 3.12, then reinstall requirements."
    )

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as mp_base_options

# 33 keypoint names in MediaPipe index order
KEYPOINT_NAMES = [
    "nose",                # 0
    "left_eye_inner",      # 1
    "left_eye",            # 2
    "left_eye_outer",      # 3
    "right_eye_inner",     # 4
    "right_eye",           # 5
    "right_eye_outer",     # 6
    "left_ear",            # 7
    "right_ear",           # 8
    "mouth_left",          # 9
    "mouth_right",         # 10
    "left_shoulder",       # 11
    "right_shoulder",      # 12
    "left_elbow",          # 13
    "right_elbow",         # 14
    "left_wrist",          # 15
    "right_wrist",         # 16
    "left_pinky",          # 17
    "right_pinky",         # 18
    "left_index",          # 19
    "right_index",         # 20
    "left_thumb",          # 21
    "right_thumb",         # 22
    "left_hip",            # 23
    "right_hip",           # 24
    "left_knee",           # 25
    "right_knee",          # 26
    "left_ankle",          # 27
    "right_ankle",         # 28
    "left_heel",           # 29
    "right_heel",          # 30
    "left_foot_index",     # 31
    "right_foot_index",    # 32
]

# Map keypoint name -> index
KEYPOINT_INDEX = {name: idx for idx, name in enumerate(KEYPOINT_NAMES)}

# Single keypoint: (x, y, confidence)
Keypoint = Tuple[float, float, float]
# Per-frame result: list of 33 keypoints
PoseResult = List[Keypoint]

# Model download URL (MediaPipe Pose Landmarker Lite)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_DEFAULT_MODEL_DIR = Path.home() / ".mediapipe" / "models"


def _ensure_model(model_path: Optional[str] = None) -> str:
    """Ensure the pose landmarker model file exists; download if needed.

    Parameters
    ----------
    model_path : str, optional
        Path to a .task model file. If None, uses default location.

    Returns
    -------
    str : Path to the model file.
    """
    if model_path and os.path.exists(model_path):
        return model_path

    local_path = _DEFAULT_MODEL_DIR / "pose_landmarker_lite.task"
    if local_path.exists():
        return str(local_path)

    # Download model
    _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"       Downloading MediaPipe Pose Landmarker model...")
    urllib.request.urlretrieve(_MODEL_URL, str(local_path))
    print(f"       Model saved to: {local_path}")
    return str(local_path)


class PoseEstimator:
    """MediaPipe Pose Landmarker wrapper providing 33-keypoint detection per frame."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to a .task model file. Downloads automatically if not provided.
        min_detection_confidence : float
            Minimum confidence for person detection [0, 1].
        min_tracking_confidence : float
            Minimum confidence for landmark tracking [0, 1].
        """
        model = _ensure_model(model_path)

        base_options = mp_base_options.BaseOptions(model_asset_path=model)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp_image = mp.Image

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[PoseResult]:
        """Detect pose landmarks on a single BGR frame.

        Returns
        -------
        landmarks : list of (x, y, confidence) or None
            None if no person detected. x and y are in pixel coordinates.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        landmarks = []
        for lm in result.pose_landmarks[0]:
            px = lm.x * w
            py = lm.y * h
            landmarks.append((px, py, lm.visibility or 0.0))
        return landmarks

    def process_batch(self, frames: np.ndarray) -> List[Optional[PoseResult]]:
        """Process multiple frames (N, H, W, 3 BGR array)."""
        return [self.process_frame(frames[i]) for i in range(len(frames))]

    def release(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    @staticmethod
    def get_keypoint_name(index: int) -> str:
        return KEYPOINT_NAMES[index]

    @staticmethod
    def get_keypoint_index(name: str) -> int:
        return KEYPOINT_INDEX[name]
