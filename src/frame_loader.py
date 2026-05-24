"""Video / GIF frame loading module.

Supports mp4, avi, mov, and gif formats. Outputs unified RGB/BGR image arrays.
"""

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from PIL import Image


class FrameLoader:
    """Load frames from video or GIF files with optional frame limiting."""

    def __init__(self, input_path: str, max_frames: Optional[int] = None):
        """
        Parameters
        ----------
        input_path : str
            Path to the video or GIF file.
        max_frames : int, optional
            Maximum number of frames to load. None means load all frames.
        """
        self.input_path = Path(input_path)
        self.max_frames = max_frames

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.suffix = self.input_path.suffix.lower()
        self._is_gif = self.suffix == ".gif"

    @property
    def fps(self) -> float:
        """Frames per second of the source video."""
        if self._is_gif:
            # GIFs don't have a reliable FPS; default to 30
            return 30.0
        cap = cv2.VideoCapture(str(self.input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 30.0

    @property
    def frame_count(self) -> int:
        """Total number of frames available."""
        if self._is_gif:
            with Image.open(self.input_path) as img:
                return getattr(img, "n_frames", 1)
        cap = cv2.VideoCapture(str(self.input_path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count

    def iter_frames(self) -> Iterator[np.ndarray]:
        """Yield frames as BGR numpy arrays (OpenCV format).

        Yields
        ------
        frame : np.ndarray
            Shape (H, W, 3), dtype uint8, BGR order.
        """
        total = self.frame_count
        limit = min(total, self.max_frames) if self.max_frames else total

        if self._is_gif:
            yield from self._iter_gif_frames(limit)
        else:
            yield from self._iter_video_frames(limit)

    def _iter_video_frames(self, limit: int) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(self.input_path))
        count = 0
        while cap.isOpened() and count < limit:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            count += 1
        cap.release()

    def _iter_gif_frames(self, limit: int) -> Iterator[np.ndarray]:
        pil_img = Image.open(self.input_path)
        for i in range(limit):
            pil_img.seek(i)
            # GIF frames may be in palette mode — convert to RGB
            frame_rgb = pil_img.convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
            yield frame_bgr

    def load_frames(self) -> np.ndarray:
        """Load all (or limited) frames into a single numpy array.

        Returns
        -------
        frames : np.ndarray
            Shape (N, H, W, 3), dtype uint8, BGR order.
        """
        return np.stack(list(self.iter_frames()), axis=0)
