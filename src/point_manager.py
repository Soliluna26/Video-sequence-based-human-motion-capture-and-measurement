"""Manual tracking point manager.

Stores user-defined tracking points with optical-flow-based position updates.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class PointManager:
    """Manages manual tracking points with Lucas-Kanade optical flow."""

    def __init__(self):
        self._points: Dict[int, dict] = {}
        self._next_id: int = 0

    @property
    def points(self):
        return self._points

    def add_point(self, x: float, y: float) -> int:
        """Add a new manual tracking point. Returns the point ID."""
        pid = self._next_id
        self._next_id += 1
        self._points[pid] = {
            "pos": (x, y),
            "prev_pt": (x, y),
            "active": True,
            "history": [(0, x, y)],  # (frame_idx, x, y) — frame_idx filled later
        }
        return pid

    def delete_point(self, pid: int):
        """Remove a manual point by ID."""
        self._points.pop(pid, None)

    def clear(self):
        """Remove all manual points."""
        self._points.clear()
        self._next_id = 0

    def set_initial_frame(self, frame_idx: int):
        """Set the starting frame index for all points' history."""
        for pt in self._points.values():
            if pt["history"] and pt["history"][0][0] == 0:
                x, y = pt["history"][0][1], pt["history"][0][2]
                pt["history"][0] = (frame_idx, x, y)

    def update_all(
        self,
        frame_idx: int,
        prev_gray: Optional[np.ndarray],
        curr_gray: np.ndarray,
    ) -> Dict[int, Tuple[float, float]]:
        """Update all active points using LK optical flow.

        Returns dict of {point_id: (x, y)} for all active points.
        """
        positions: Dict[int, Tuple[float, float]] = {}

        for pid, pt in self._points.items():
            if not pt["active"]:
                continue
            if prev_gray is None:
                positions[pid] = pt["pos"]
                continue

            p0 = np.array([[pt["prev_pt"]]], dtype=np.float32)
            p1, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                curr_gray,
                p0,
                None,
                winSize=(41, 41),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                minEigThreshold=0.001,
            )

            if status[0][0] == 1:
                nx, ny = float(p1[0][0][0]), float(p1[0][0][1])
                pt["pos"] = (nx, ny)
                pt["prev_pt"] = (nx, ny)
                pt["history"].append((frame_idx, nx, ny))
                positions[pid] = (nx, ny)
            else:
                pt["active"] = False
                positions[pid] = pt["pos"]

        return positions

    def get_history(self, pid: int) -> List[Tuple[int, float, float]]:
        """Get the full history for a point."""
        pt = self._points.get(pid)
        if pt is None:
            return []
        return pt["history"]

    def get_all_histories(self) -> Dict[int, List[Tuple[int, float, float]]]:
        """Get histories for all points."""
        return {pid: pt["history"] for pid, pt in self._points.items()}

    def get_active_count(self) -> int:
        """Return number of active points."""
        return sum(1 for pt in self._points.values() if pt["active"])
