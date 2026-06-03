"""Action recognition via DTW template matching on joint angle sequences.

Provides a full pipeline for few-shot action learning and detection:
    - Register action templates from example videos
    - Detect learned actions in multi-action videos using sliding-window DTW
    - Persist template libraries to JSON

Core classes:
    ActionTemplate   — single normalized action exemplar
    TemplateStore    — collection of templates with JSON persistence
    ActionRecognizer — sliding-window detector with NMS
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Angle keys used as the 10-D feature vector (matches landmarks.yaml)
# ---------------------------------------------------------------------------
ANGLE_KEYS: List[str] = [
    "left_knee_angle",
    "right_knee_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_ankle_angle",
    "right_ankle_angle",
]

# ---------------------------------------------------------------------------
# DTW core
# ---------------------------------------------------------------------------

def dtw_distance(
    template: np.ndarray,
    segment: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """Compute normalized DTW distance between two multivariate sequences.

    Parameters
    ----------
    template : np.ndarray, shape (T1, D)
        Reference sequence.
    segment : np.ndarray, shape (T2, D)
        Query sequence to compare against the template.
    window : int, optional
        Sakoe-Chiba band width.  Defaults to ``max(|T1|, |T2|)`` (no constraint).

    Returns
    -------
    distance : float
        Normalized DTW distance = total cost / path length.
        Lower values mean more similar sequences.
    """
    T1, T2 = template.shape[0], segment.shape[0]

    if T1 < 2 or T2 < 2:
        # Degenerate: direct Euclidean on means
        return float(np.linalg.norm(template.mean(axis=0) - segment.mean(axis=0)))

    # Local cost matrix: squared Euclidean distance between every frame pair
    # C[i, j] = ||template[i] - segment[j]||²
    cost = np.zeros((T1, T2), dtype=np.float64)
    for d in range(template.shape[1]):
        diff = template[:, d, np.newaxis] - segment[np.newaxis, :, d]
        cost += diff * diff
    cost = np.sqrt(cost)  # Euclidean, not squared

    # Accumulated cost matrix
    D = np.full((T1, T2), np.inf, dtype=np.float64)
    D[0, 0] = cost[0, 0]

    w = window if window is not None else max(T1, T2)

    for i in range(T1):
        j_start = max(0, i - w)
        j_end = min(T2, i + w + 1)
        for j in range(j_start, j_end):
            if i == 0 and j == 0:
                continue
            candidates = [np.inf]
            if i > 0:
                candidates.append(D[i - 1, j])          # insertion
            if j > 0:
                candidates.append(D[i, j - 1])          # deletion
            if i > 0 and j > 0:
                candidates.append(D[i - 1, j - 1])     # match
            D[i, j] = cost[i, j] + min(candidates)

    # Backtrack to get path length (for normalization)
    i, j = T1 - 1, T2 - 1
    path_len = 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            steps = [
                (D[i - 1, j - 1], i - 1, j - 1),
                (D[i - 1, j], i - 1, j),
                (D[i, j - 1], i, j - 1),
            ]
            _, i, j = min(steps, key=lambda x: x[0])
        path_len += 1

    return float(D[T1 - 1, T2 - 1]) / path_len


# ---------------------------------------------------------------------------
# Action template
# ---------------------------------------------------------------------------

@dataclass
class ActionTemplate:
    """A single normalized action template.

    Attributes
    ----------
    name : str
        Human-readable action label (e.g. "shooting", "walking").
    features : np.ndarray
        Normalized angle matrix of shape (target_len, 10).
    source_fps : float
        Original video FPS (for reference, not used in matching).
    template_id : str
        Unique identifier.
    target_len : int
        Length to which templates are temporally resampled.
    """

    name: str
    features: np.ndarray
    source_fps: float
    template_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    target_len: int = 60

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "features": self.features.tolist(),
            "source_fps": self.source_fps,
            "template_id": self.template_id,
            "target_len": self.target_len,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionTemplate":
        """Deserialize from a dict."""
        features = np.array(d["features"], dtype=np.float64)
        return cls(
            name=d["name"],
            features=features,
            source_fps=d["source_fps"],
            template_id=d.get("template_id", uuid.uuid4().hex[:8]),
            target_len=d.get("target_len", 60),
        )


# ---------------------------------------------------------------------------
# Template store
# ---------------------------------------------------------------------------

class TemplateStore:
    """A collection of action templates with JSON persistence.

    Parameters
    ----------
    target_len : int
        Number of frames to which every template is resampled.
    """

    def __init__(self, target_len: int = 60):
        self.target_len = target_len
        self._templates: Dict[str, ActionTemplate] = {}

    # -- properties ----------------------------------------------------------
    @property
    def templates(self) -> List[ActionTemplate]:
        return list(self._templates.values())

    @property
    def names(self) -> List[str]:
        return [t.name for t in self._templates.values()]

    def __len__(self) -> int:
        return len(self._templates)

    def __contains__(self, template_id: str) -> bool:
        return template_id in self._templates

    # -- CRUD ---------------------------------------------------------------
    def add(
        self,
        angles_seq: np.ndarray,
        name: str,
        fps: float = 30.0,
    ) -> ActionTemplate:
        """Normalize an angle sequence and store it as a new template.

        Parameters
        ----------
        angles_seq : np.ndarray, shape (T, D)
            Raw angle time series (D should be the 10 ANGLE_KEYS).
        name : str
            Action label.
        fps : float
            Original video FPS.

        Returns
        -------
        template : ActionTemplate
        """
        features = _normalize_angles(angles_seq, self.target_len)
        template = ActionTemplate(
            name=name,
            features=features,
            source_fps=fps,
            target_len=self.target_len,
        )
        self._templates[template.template_id] = template
        return template

    def remove(self, template_id: str) -> bool:
        """Remove a template by id.  Returns True if it existed."""
        if template_id in self._templates:
            del self._templates[template_id]
            return True
        return False

    def get(self, template_id: str) -> Optional[ActionTemplate]:
        return self._templates.get(template_id)

    def clear(self) -> None:
        self._templates.clear()

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist all templates to a JSON file."""
        data = {
            "target_len": self.target_len,
            "templates": [t.to_dict() for t in self._templates.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load templates from a JSON file (adds to current store)."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.target_len = data.get("target_len", 60)
        for item in data.get("templates", []):
            template = ActionTemplate.from_dict(item)
            template.target_len = self.target_len
            self._templates[template.template_id] = template

    # -- auto-threshold -----------------------------------------------------
    def compute_threshold(self, template_id: str, multiplier: float = 2.0) -> float:
        """Return a detection threshold for *template_id*.

        The baseline is the RMS energy of the template multiplied by a
        scaling factor, with a sensible floor.  This is more robust than
        self-DTW (which is trivially 0) because matching a different
        instance of the same action always has a small but non-zero DTW
        distance due to resampling and natural variation.
        """
        tmpl = self._templates.get(template_id)
        if tmpl is None:
            return float("inf")
        # RMS of the (de-meaned) template — measures action "energy"
        rms = float(np.sqrt(np.mean(tmpl.features ** 2)))
        # Scale: a typical same-action DTW distance is ~0.3 × RMS
        return max(rms * 0.5 * multiplier, 1.0)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_angles(angles: np.ndarray, target_len: int) -> np.ndarray:
    """Normalize an angle sequence for template storage.

    1. Interpolate NaNs (linear)
    2. Resample temporally to *target_len* frames
    3. De-mean each dimension so DTW focuses on shape

    Parameters
    ----------
    angles : np.ndarray, shape (T, D)
    target_len : int

    Returns
    -------
    normalized : np.ndarray, shape (target_len, D)
    """
    T, D = angles.shape

    # ---- fill NaNs per dimension ------------------------------------------
    filled = np.copy(angles)
    for d in range(D):
        col = filled[:, d]
        mask = np.isnan(col)
        if mask.any():
            ok = ~mask
            if ok.sum() == 0:
                col[:] = 0.0
            else:
                xp = ok.nonzero()[0]
                fp = col[ok]
                col[mask] = np.interp(mask.nonzero()[0], xp, fp)

    # ---- temporal resampling ----------------------------------------------
    if T != target_len:
        src_x = np.linspace(0, 1, T)
        dst_x = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, D), dtype=np.float64)
        for d in range(D):
            resampled[:, d] = np.interp(dst_x, src_x, filled[:, d])
    else:
        resampled = filled.copy()

    # ---- de-mean ----------------------------------------------------------
    means = resampled.mean(axis=0, keepdims=True)
    resampled -= means

    return resampled


def _normalize_segment(segment: np.ndarray, target_len: int) -> np.ndarray:
    """Normalize a query segment consistently with template normalization.

    Resamples to *target_len* and de-means.  Unlike ``_normalize_angles``,
    this does NOT do NaN filling (the caller is expected to have done that
    on the full video already).

    Parameters
    ----------
    segment : np.ndarray, shape (W, D)
    target_len : int

    Returns
    -------
    normalized : np.ndarray, shape (target_len, D)
    """
    resampled = _resample_sequence(segment, target_len)
    means = resampled.mean(axis=0, keepdims=True)
    resampled -= means
    return resampled


# ---------------------------------------------------------------------------
# Action recognizer
# ---------------------------------------------------------------------------

@dataclass
class ActionMatch:
    """A detected action instance in a video.

    Attributes
    ----------
    action_name : str
    template_id : str
    start_frame : int
    end_frame : int
    start_sec : float
    end_sec : float
    confidence : float
        0–1 confidence where 1 = perfect match (0 distance).
    """

    action_name: str
    template_id: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    confidence: float


class ActionRecognizer:
    """Sliding-window DTW action detector.

    Parameters
    ----------
    store : TemplateStore
        Template library to match against.
    sensitivity : float
        Multiplier for the auto-threshold.  Lower = stricter matching.
        Typical range 1.5–3.0.
    step : int
        Sliding-window step in frames.
    scale_range : Tuple[float, float]
        Window-length multiplier range relative to template length.
    """

    def __init__(
        self,
        store: TemplateStore,
        sensitivity: float = 2.0,
        step: int = 5,
        scale_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.store = store
        self.sensitivity = sensitivity
        self.step = step
        self.scale_range = scale_range

    # ------------------------------------------------------------------
    def recognize(
        self,
        angles_seq: np.ndarray,
        fps: float = 30.0,
    ) -> List[ActionMatch]:
        """Detect all registered actions in an angle sequence.

        Parameters
        ----------
        angles_seq : np.ndarray, shape (T, D)
            Full-video angle matrix (already interpolated + smoothed).
        fps : float

        Returns
        -------
        matches : List[ActionMatch]
            Sorted by start_frame ascending.
        """
        T = angles_seq.shape[0]
        if T < 10 or len(self.store) == 0:
            return []

        # Pre-compute de-meaned feature matrix for the entire video
        filled = _fill_nans(angles_seq)
        means = filled.mean(axis=0, keepdims=True)
        video_norm = filled - means  # (T, D)

        raw_matches: List[dict] = []

        for tmpl in self.store.templates:
            T_tmpl = tmpl.target_len
            threshold = self.store.compute_threshold(
                tmpl.template_id, self.sensitivity
            )

            template_norm = tmpl.features  # already de-meaned

            # Multi-scale window scan
            scales = np.arange(
                self.scale_range[0], self.scale_range[1] + 0.25, 0.25
            )
            for scale in scales:
                win_len = max(10, int(T_tmpl * scale))
                if win_len > T:
                    continue

                for start in range(0, T - win_len, self.step):
                    end = start + win_len
                    segment = video_norm[start:end, :]  # (win_len, D)

                    # Normalize segment consistently with template
                    segment_norm = _normalize_segment(segment, T_tmpl)

                    dist = dtw_distance(template_norm, segment_norm)

                    if dist < threshold and not np.isnan(dist):
                        raw_matches.append({
                            "template": tmpl,
                            "start": start,
                            "end": end,
                            "distance": dist,
                            "threshold": threshold,
                        })

        # ---- non-maximum suppression (NMS) ---------------------------------
        raw_matches.sort(key=lambda m: m["distance"])
        suppressed = _non_max_suppression(raw_matches, iou_threshold=0.5)

        # ---- build results -------------------------------------------------
        results: List[ActionMatch] = []
        for m in suppressed:
            # Confidence: 1 at distance=0, decaying to 0 at threshold
            conf = max(0.0, 1.0 - m["distance"] / m["threshold"])
            results.append(ActionMatch(
                action_name=m["template"].name,
                template_id=m["template"].template_id,
                start_frame=m["start"],
                end_frame=m["end"],
                start_sec=m["start"] / fps,
                end_sec=m["end"] / fps,
                confidence=round(conf, 4),
            ))

        results.sort(key=lambda r: r.start_frame)
        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values per column via linear interpolation."""
    filled = np.copy(arr)
    for d in range(filled.shape[1]):
        col = filled[:, d]
        mask = np.isnan(col)
        if mask.any():
            ok = ~mask
            if ok.sum() == 0:
                col[:] = 0.0
            else:
                xp = ok.nonzero()[0]
                fp = col[ok]
                col[mask] = np.interp(mask.nonzero()[0], xp, fp)
    return filled


def _resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a sequence along the time axis to *target_len* frames."""
    T, D = seq.shape
    if T == target_len:
        return seq.copy()
    src_x = np.linspace(0, 1, T)
    dst_x = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, D), dtype=np.float64)
    for d in range(D):
        out[:, d] = np.interp(dst_x, src_x, seq[:, d])
    return out


def _non_max_suppression(
    matches: List[dict],
    iou_threshold: float = 0.5,
) -> List[dict]:
    """Suppress overlapping matches for the same action name.

    Greedy algorithm: keep the lowest-distance match, remove any that
    overlap with it beyond *iou_threshold*.  Matches for *different*
    action names are not suppressed against each other.

    Parameters
    ----------
    matches : list of dict
        Each must have ``"template"`` (with ``.name``), ``"start"``, ``"end"``.
    iou_threshold : float

    Returns
    -------
    kept : list of dict
    """
    if not matches:
        return []

    kept: List[dict] = []
    suppressed = set()

    for i, m in enumerate(matches):
        if i in suppressed:
            continue
        kept.append(m)
        name_i = m["template"].name
        a_start, a_end = m["start"], m["end"]
        len_a = a_end - a_start
        if len_a <= 0:
            continue

        for j in range(i + 1, len(matches)):
            if j in suppressed:
                continue
            name_j = matches[j]["template"].name
            if name_i != name_j:
                continue  # different actions coexist
            b_start, b_end = matches[j]["start"], matches[j]["end"]
            # IoU
            inter_start = max(a_start, b_start)
            inter_end = min(a_end, b_end)
            inter = max(0, inter_end - inter_start)
            union = (a_end - a_start) + (b_end - b_start) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_threshold:
                suppressed.add(j)

    return kept


# ---------------------------------------------------------------------------
# Convenience: extract angle features from the existing pipeline
# ---------------------------------------------------------------------------

def extract_angle_features(
    landmarks_seq: np.ndarray,
    angle_definitions: dict,
    window: int = 5,
) -> np.ndarray:
    """Extract a smoothed (T, 10) angle feature matrix from landmark positions.

    This wraps the existing ``kinematics.compute_angles_from_landmarks`` and
    ``kinematics.smooth_angles`` into a single call that returns the matrix
    expected by ``ActionRecognizer``.

    Parameters
    ----------
    landmarks_seq : np.ndarray, shape (T, 33, 2)
    angle_definitions : dict
        ``{name: (p1, p2, p3)}`` as loaded from landmarks.yaml.
    window : int
        Smoothing window size.

    Returns
    -------
    features : np.ndarray, shape (T, 10)
        Smoothed angle matrix in the order defined by ANGLE_KEYS.
        Missing keys in *angle_definitions* produce an all-NaN column.
    """
    from .kinematics import compute_angles_from_landmarks, smooth_angles

    T = landmarks_seq.shape[0]
    angles_raw = compute_angles_from_landmarks(landmarks_seq, angle_definitions)

    features = np.full((T, len(ANGLE_KEYS)), np.nan, dtype=np.float64)
    for col, key in enumerate(ANGLE_KEYS):
        if key in angles_raw:
            smoothed = smooth_angles(angles_raw[key], window=window)
            features[:, col] = smoothed

    return features
