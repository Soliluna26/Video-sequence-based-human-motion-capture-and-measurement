#!/usr/bin/env python3
"""Streamlit web interface for the Human Motion Capture & Measurement System.

Launch locally:
    streamlit run web_app.py

Or deploy to Streamlit Community Cloud for zero-download browser access.
Reuses all existing src/ modules without modification.
"""

import base64
import io
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request as _request
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -- Provide libgthread-2.0.so.0 on Debian Trixie (GLib >= 2.72) --
# opencv-contrib-python (pulled by mediapipe) links libgthread-2.0.so.0
# which was removed upstream.  Streamlit Cloud runs Debian Trixie where
# it does not exist.  We first try to download libglib2.0-0 from the
# bullseye repo via apt, then fall back to Debian archive HTTP mirrors.
# Bullseye's GLib 2.66 libgthread depends on libglib-2.0.so.0 (same SONAME
# as Trixie's GLib 2.82 — backward ABI-compatible).
import subprocess as _sp  # noqa: E402

_GTHREAD_DIR = "/tmp/mocap_lib"
_GTHREAD_FLAG = _GTHREAD_DIR + "/.extracted"
_GTHREAD_SO = "libgthread-2.0.so.0"
_GTHREAD_SO_PATH = _GTHREAD_DIR + "/" + _GTHREAD_SO


def _gthread_log(msg: str):
    print(f"[mocap] {msg}", file=sys.stderr, flush=True)


def _gthread_setup() -> bool:
    """Download and cache libgthread-2.0.so.0; add to LD_LIBRARY_PATH."""
    if sys.platform != "linux":
        return False

    # Already done in this deployment?
    if os.path.exists(_GTHREAD_FLAG) and os.path.exists(_GTHREAD_SO_PATH):
        _gthread_set_ld_path()
        _gthread_log("libgthread-2.0.so.0 already cached, LD_LIBRARY_PATH set")
        return True

    os.makedirs(_GTHREAD_DIR, exist_ok=True)

    # --- Strategy 1: apt-get download from Debian Bullseye ---
    if _gthread_try_apt():
        _gthread_set_ld_path()
        return True

    # --- Strategy 2: download .deb via HTTP from archive.debian.org ---
    if _gthread_try_http():
        _gthread_set_ld_path()
        return True

    _gthread_log("All strategies failed — cv2 import will likely crash")
    return False


def _gthread_try_apt() -> bool:
    """Use apt-get -o to download libglib2.0-0 from bullseye into /tmp."""
    deb_path = _GTHREAD_DIR + "/._apt_libglib.deb"
    src_list = _GTHREAD_DIR + "/bullseye.list"
    apt_lists = _GTHREAD_DIR + "/apt-lists"
    apt_cache = _GTHREAD_DIR + "/apt-cache"

    try:
        # Write a custom sources.list in /tmp (writable)
        with open(src_list, "w") as f:
            f.write("deb http://archive.debian.org/debian bullseye main\n")

        os.makedirs(apt_lists + "/partial", exist_ok=True)
        os.makedirs(apt_cache, exist_ok=True)

        _apt_opts = [
            "-o", f"Dir::Etc::sourcelist={src_list}",
            "-o", "Dir::Etc::sourceparts=-",  # disable default sources
            "-o", f"Dir::State::lists={apt_lists}",
            "-o", f"Dir::Cache={apt_cache}",
            "-o", "APT::Get::AllowUnauthenticated=true",
            "-o", "Acquire::AllowInsecureRepositories=true",
            "-o", "Acquire::AllowDowngradeToInsecureRepositories=true",
        ]

        _sp.run(
            ["apt-get"] + _apt_opts + ["update", "-qq"],
            capture_output=True, timeout=120,
        )

        # Find the available version
        madison = _sp.run(
            ["apt-cache"] + _apt_opts + ["madison", "libglib2.0-0"],
            capture_output=True, text=True, timeout=30,
        )
        _gthread_log(f"Available versions: {madison.stdout.strip()}")

        result = _sp.run(
            ["apt-get"] + _apt_opts + ["download", "-qq", "libglib2.0-0"],
            capture_output=True, timeout=120, cwd=_GTHREAD_DIR,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")[:300]
            _gthread_log(f"apt-get download failed: {err}")
            return False

        # Find the downloaded .deb (version is dynamic)
        for fn in os.listdir(_GTHREAD_DIR):
            if fn.startswith("libglib2.0-0_") and fn.endswith("_amd64.deb"):
                src = os.path.join(_GTHREAD_DIR, fn)
                os.rename(src, deb_path)
                _gthread_log(f"Downloaded via apt: {fn} ({os.path.getsize(deb_path)} bytes)")
                return _gthread_unpack_deb(deb_path)

        _gthread_log("apt-get ran but no .deb found in output dir")
        return False
    except Exception as e:
        _gthread_log(f"apt strategy exception: {e}")
        return False
    finally:
        for p in (deb_path, src_list):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        shutil.rmtree(apt_lists, ignore_errors=True)
        shutil.rmtree(apt_cache, ignore_errors=True)


def _gthread_try_http() -> bool:
    """Fallback: download .deb via HTTP from Debian snapshot/archive."""
    deb_path = _GTHREAD_DIR + "/._http_libglib.deb"
    # snapshot.debian.org has every version; archive.debian.org has the final one
    _URLS = [
        # First try: snapshot with a known bullseye version (2.66.8-1+deb11u1)
        "https://snapshot.debian.org/archive/debian/20230320T024800Z"
        "/pool/main/g/glib2.0/libglib2.0-0_2.66.8-1+deb11u1_amd64.deb",
        # Second try: archive.debian.org (Debian archive)
        "http://archive.debian.org/debian/pool/main/g/glib2.0/"
        "libglib2.0-0_2.66.8-1+deb11u1_amd64.deb",
        "http://archive.debian.org/debian/pool/main/g/glib2.0/"
        "libglib2.0-0_2.66.8-1_amd64.deb",
        # Third try: generic snapshot (try to list directory)
        "https://snapshot.debian.org/archive/debian/20230101T000000Z"
        "/pool/main/g/glib2.0/libglib2.0-0_2.66.8-1_amd64.deb",
    ]
    for url in _URLS:
        _gthread_log(f"Trying {url} ...")
        try:
            _request.urlretrieve(url, deb_path)
            sz = os.path.getsize(deb_path)
            if sz > 500_000:
                return _gthread_unpack_deb(deb_path)
            _gthread_log(f"Too small ({sz} bytes), likely an error page")
        except Exception as e:
            _gthread_log(f"HTTP error: {e}")
    return False


def _gthread_unpack_deb(deb_path: str) -> bool:
    """Unpack a .deb and extract libgthread-2.0.so.0."""
    data_tar = _gthread_extract_data_tar(deb_path)
    if data_tar is None:
        _gthread_log("Failed to extract data.tar from .deb (bad ar format?)")
        return False
    _gthread_log(f"Extracted data tarball: {os.path.getsize(data_tar)} bytes")
    try:
        # Extract .so from data tarball
        extract_dir = tempfile.mkdtemp(dir=_GTHREAD_DIR)
        try:
            with tarfile.open(data_tar, "r:*") as tf:
                tf.extractall(extract_dir, filter="data")
            found = False
            for root, _dirs, files in os.walk(extract_dir):
                for fn in files:
                    if fn == _GTHREAD_SO or fn.startswith(_GTHREAD_SO + "."):
                        fp = os.path.join(root, fn)
                        if os.path.islink(fp) or os.path.getsize(fp) < 1024:
                            continue
                        with open(fp, "rb") as fh:
                            if fh.read(4) != b"\x7fELF":
                                continue
                        shutil.copy2(fp, _GTHREAD_SO_PATH)
                        os.chmod(_GTHREAD_SO_PATH, 0o755)
                        _gthread_log(
                            f"Copied {_GTHREAD_SO} "
                            f"({os.path.getsize(_GTHREAD_SO_PATH)} bytes)"
                        )
                        found = True
                        break
                if found:
                    break
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
    finally:
        for p in (deb_path, data_tar if data_tar and os.path.exists(data_tar) else None):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    if not os.path.exists(_GTHREAD_SO_PATH):
        _gthread_log("libgthread-2.0.so.0 NOT found in .deb data tarball")
        return False

    # Mark success for subsequent runs
    with open(_GTHREAD_FLAG, "w") as f:
        f.write("1\n")
    return True


def _gthread_extract_data_tar(deb_path: str) -> str | None:
    """Parse a .deb (ar archive) and extract the data.tar.* member."""
    with open(deb_path, "rb") as f:
        if f.read(8) != b"!<arch>\n":
            _gthread_log("Not a valid ar archive (missing !<arch> magic)")
            return None
        while True:
            header = f.read(60)
            if len(header) < 60:
                break
            name = header[0:16].rstrip(b" ").decode("ascii", errors="replace")
            name = name.rstrip("/")
            size_str = header[48:58].rstrip(b" ").decode("ascii", errors="replace")
            try:
                size = int(size_str)
            except ValueError:
                return None
            data = f.read(size)
            if size % 2 != 0:
                f.read(1)  # ar padding byte
            if name.startswith("data.tar"):
                out = _GTHREAD_DIR + "/_data.tar"
                with open(out, "wb") as wf:
                    wf.write(data)
                return out
    return None


def _gthread_set_ld_path():
    """Prepend cache dir to LD_LIBRARY_PATH and preload the .so via ctypes."""
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if _GTHREAD_DIR not in cur.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = (
            _GTHREAD_DIR + os.pathsep + cur if cur else _GTHREAD_DIR
        )
    _gthread_log(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '(empty)')}")

    # Preload with ctypes to register the library in the dynamic linker's
    # namespace BEFORE cv2's .so is dlopen'd.  This satisfies the NEEDED
    # dependency on libgthread-2.0.so.0 by SONAME.
    if os.path.exists(_GTHREAD_SO_PATH):
        try:
            import ctypes as _ct
            _ct.CDLL(
                _GTHREAD_SO_PATH,
                mode=os.RTLD_GLOBAL | os.RTLD_NOW,
            )
            _gthread_log("Preloaded libgthread-2.0.so.0 into process")
        except Exception as _e:
            _gthread_log(f"ctypes preload failed (will rely on LD_LIBRARY_PATH): {_e}")


if not _gthread_setup():
    _gthread_log("WARNING: libgthread-2.0.so.0 not available")

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.frame_loader import FrameLoader
from src.pose_estimator import PoseEstimator, KEYPOINT_NAMES, PoseResult
from src.ball_tracker import BallTracker
from src.point_manager import PointManager

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Human Motion Capture",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BONE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
SKELETON_KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
CYAN = (255, 255, 0)        # OpenCV BGR
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DEFAULT_MAX_FRAMES = 300
MAX_DISPLAY_DIM = 720
DEFAULT_MM_PER_PIXEL = 2.0


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_pose_estimator() -> PoseEstimator:
    """Cached PoseEstimator — model is downloaded once and reused."""
    return PoseEstimator()


# ---------------------------------------------------------------------------
# Drawing helpers (adapted from src/gui/main_window.py)
# ---------------------------------------------------------------------------
def draw_skeleton(
    canvas: np.ndarray,
    landmarks: PoseResult,
    color: Tuple[int, int, int] = CYAN,
) -> np.ndarray:
    """Draw skeleton stick figure on a BGR canvas."""
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


def draw_trajectories(
    canvas: np.ndarray,
    trajectories: Dict[int, List[Tuple[float, float]]],
    color: Tuple[int, int, int],
) -> np.ndarray:
    """Draw point trajectories on canvas."""
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


def render_overlay_frame(
    frame_bgr: np.ndarray,
    landmarks: Optional[PoseResult],
    ball_pos: Optional[Tuple[float, float]],
    manual_positions: Dict[int, Tuple[float, float]],
    manual_trajectories: Dict[int, List[Tuple[float, float]]],
    vel_mm_s: float = 0.0,
    dist_mm: float = 0.0,
    elapsed_s: float = 0.0,
    show_measurements: bool = True,
) -> np.ndarray:
    """Render a frame with skeleton, ball, manual points, and trajectories."""
    canvas = frame_bgr.copy()

    if landmarks is not None:
        canvas = draw_skeleton(canvas, landmarks, CYAN)

    canvas = draw_trajectories(canvas, manual_trajectories, BLUE)

    h, w = canvas.shape[:2]
    for pid, pos in manual_positions.items():
        px, py = int(np.clip(pos[0], 0, w - 1)), int(np.clip(pos[1], 0, h - 1))
        cv2.circle(canvas, (px, py), 6, BLUE, -1)
        cv2.circle(canvas, (px, py), 9, BLUE, 2)
        cv2.putText(
            canvas, str(pid), (px + 12, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1, cv2.LINE_AA,
        )

    if ball_pos is not None:
        bx, by = int(np.clip(ball_pos[0], 0, w - 1)), int(np.clip(ball_pos[1], 0, h - 1))
        cv2.circle(canvas, (bx, by), 10, GREEN, -1)
        cv2.circle(canvas, (bx, by), 14, GREEN, 2)

    if show_measurements:
        lines = [
            f"Velocity: {vel_mm_s:.2f} mm/s",
            f"Distance: {dist_mm:.2f} mm",
            f"Time: {elapsed_s:.2f} s",
        ]
        y0 = 30
        for i, text in enumerate(lines):
            y = y0 + i * 28
            cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, WHITE, 2, cv2.LINE_AA)

    return canvas


def render_replay_frame(
    frame_bgr: np.ndarray,
    landmarks: Optional[PoseResult],
    manual_positions: Dict[int, Tuple[float, float]],
    manual_trajectories: Dict[int, List[Tuple[float, float]]],
    vel_mm_s: float = 0.0,
    dist_mm: float = 0.0,
    elapsed_s: float = 0.0,
) -> np.ndarray:
    """Render a replay frame on black background with trajectories."""
    h, w = frame_bgr.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if landmarks is not None:
        canvas = draw_skeleton(canvas, landmarks, CYAN)

    canvas = draw_trajectories(canvas, manual_trajectories, BLUE)

    for pid, pos in manual_positions.items():
        px, py = int(np.clip(pos[0], 0, w - 1)), int(np.clip(pos[1], 0, h - 1))
        cv2.circle(canvas, (px, py), 6, BLUE, -1)
        cv2.circle(canvas, (px, py), 9, BLUE, 2)

    for pid, pts in manual_trajectories.items():
        for px, py in pts:
            cx, cy = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
            cv2.circle(canvas, (cx, cy), 3, BLUE, -1)

    lines = [
        f"Velocity: {vel_mm_s:.2f} mm/s",
        f"Distance: {dist_mm:.2f} mm",
        f"Time: {elapsed_s:.2f} s",
    ]
    y0 = 30
    for i, text in enumerate(lines):
        y = y0 + i * 28
        cv2.putText(canvas, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, WHITE, 2, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Optical-flow tracking
# ---------------------------------------------------------------------------
def run_optical_flow_tracking(
    frames_gray: List[np.ndarray],
    start_frame: int,
    initial_positions: Dict[int, Tuple[float, float]],
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Run LK optical flow tracking from start_frame to end.

    Returns {point_id: [(frame_idx, x, y), ...]}.
    """
    if not initial_positions:
        return {}

    tracks: Dict[int, List[Tuple[int, float, float]]] = {
        pid: [(start_frame, x, y)] for pid, (x, y) in initial_positions.items()
    }
    active: Dict[int, Tuple[float, float]] = dict(initial_positions)

    for fi in range(start_frame + 1, len(frames_gray)):
        if not active:
            break
        prev_gray = frames_gray[fi - 1]
        curr_gray = frames_gray[fi]
        new_active: Dict[int, Tuple[float, float]] = {}
        for pid, (px, py) in active.items():
            p0 = np.array([[[px, py]]], dtype=np.float32)
            p1, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, None,
                winSize=(41, 41), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                minEigThreshold=0.001,
            )
            if status[0][0] == 1:
                nx, ny = float(p1[0][0][0]), float(p1[0][0][1])
                tracks[pid].append((fi, nx, ny))
                new_active[pid] = (nx, ny)
        active = new_active
    return tracks


# ---------------------------------------------------------------------------
# Ball distance / velocity calculation
# ---------------------------------------------------------------------------
def calc_ball_metrics(
    ball_positions: List[Optional[Tuple[float, float]]],
    up_to_frame: int,
    fps: float,
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
) -> Tuple[float, float, float]:
    """Calculate cumulative distance, velocity, and elapsed time."""
    elapsed = up_to_frame / fps if fps > 0 else 0.0
    positions = [p for p in ball_positions[:up_to_frame + 1] if p is not None]

    dist_px = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dist_px += np.sqrt(dx * dx + dy * dy)
    dist_mm = dist_px * mm_per_pixel

    vel_mm_s = 0.0
    if len(positions) >= 2 and fps > 0:
        recent = positions[-min(5, len(positions)):]
        speeds = []
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            speeds.append(np.sqrt(dx * dx + dy * dy) * mm_per_pixel * fps)
        if speeds:
            vel_mm_s = float(np.mean(speeds))

    return vel_mm_s, dist_mm, elapsed


def build_manual_trajectories_up_to(
    tracks: Dict[int, List[Tuple[int, float, float]]],
    frame_idx: int,
) -> Dict[int, List[Tuple[float, float]]]:
    """Extract trajectory positions up to a given frame index."""
    result: Dict[int, List[Tuple[float, float]]] = {}
    for pid, history in tracks.items():
        pts = [(x, y) for fi, x, y in history if fi <= frame_idx]
        if pts:
            result[pid] = pts
    return result


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------
def generate_replay_video(
    frames_bgr: List[np.ndarray],
    poses: List[Optional[PoseResult]],
    ball_positions: List[Optional[Tuple[float, float]]],
    manual_tracks: Dict[int, List[Tuple[int, float, float]]],
    fps: float,
    mm_per_pixel: float = DEFAULT_MM_PER_PIXEL,
) -> bytes:
    """Generate a replay MP4 video with skeleton and trajectories. Returns bytes."""
    if not frames_bgr:
        return b""
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = os.path.join(tempfile.gettempdir(), "mocap_replay.mp4")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    for fi, frame_bgr in enumerate(frames_bgr):
        manual_pos = {}
        for pid, history in manual_tracks.items():
            for f_idx, x, y in history:
                if f_idx == fi:
                    manual_pos[pid] = (x, y)
                    break

        traj = build_manual_trajectories_up_to(manual_tracks, fi)
        vel, dist, elapsed = calc_ball_metrics(ball_positions, fi, fps, mm_per_pixel)

        canvas = render_replay_frame(
            frame_bgr, poses[fi], manual_pos, traj, vel, dist, elapsed,
        )
        writer.write(canvas)

    writer.release()
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialize all session state keys."""
    defaults = {
        "video_path": None,
        "video_name": "",
        "fps": 30.0,
        "frame_count": 0,
        "frames_bgr": [],
        "frames_gray": [],
        "poses": [],
        "ball_positions": [],
        "processing_done": False,
        "current_frame": 0,
        "manual_tracks": {},      # pid -> [(frame_idx, x, y), ...]
        "next_manual_id": 0,
        "pending_manual_points": {},  # pid -> (frame_idx, x, y) — added but not tracked yet
        "processing_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Clickable image component
# ---------------------------------------------------------------------------
def clickable_image(frame_bgr: np.ndarray, key: str) -> Optional[Dict]:
    """Display a clickable BGR frame. Returns {'x': int, 'y': int} on click.

    Uses an HTML component that captures click events and maps them to
    original frame coordinates.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    scale = min(MAX_DISPLAY_DIM / w, MAX_DISPLAY_DIM / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    if scale < 1.0:
        pil_img = pil_img.resize((disp_w, disp_h), Image.LANCZOS)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    html = f"""
    <style>
        #img-{key} {{ cursor: crosshair; display: block; max-width: 100%; }}
        #info-{key} {{ color: #888; font-size: 12px; margin-top: 4px; }}
    </style>
    <div>
        <img id="img-{key}" src="data:image/png;base64,{img_b64}"
             data-w="{w}" data-h="{h}">
        <div id="info-{key}">Click on the image to add a tracking point</div>
    </div>
    <script>
    (function() {{
        const img = document.getElementById('img-{key}');
        const info = document.getElementById('info-{key}');
        if (!img) return;
        img.addEventListener('click', function(e) {{
            const rect = img.getBoundingClientRect();
            const origW = parseInt(img.dataset.w);
            const origH = parseInt(img.dataset.h);
            const scaleX = origW / rect.width;
            const scaleY = origH / rect.height;
            const x = Math.round((e.clientX - rect.left) * scaleX);
            const y = Math.round((e.clientY - rect.top) * scaleY);
            info.textContent = 'Point added at (' + x + ', ' + y +
                ') — processing tracking...';
            window.parent.postMessage({{
                isStreamlitMessage: true,
                type: 'streamlit:setComponentValue',
                data: {{x: x, y: y}}
            }}, '*');
        }});
    }})();
    </script>
    """

    val = st.components.v1.html(html, height=disp_h + 30)
    if val and isinstance(val, dict) and "x" in val:
        return val
    return None


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    init_session_state()
    ss = st.session_state

    # -- Sidebar --
    with st.sidebar:
        st.title("🏃 Motion Capture")
        st.caption("Video-based Human Motion Capture & Measurement")
        st.divider()

        st.subheader("1. Upload Video")
        uploaded = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "gif", "webm"],
            key="video_uploader",
        )
        if uploaded is not None:
            # Check if it's a new file
            if ss.video_name != uploaded.name:
                ss.video_name = uploaded.name
                ss.processing_done = False
                ss.frames_bgr = []
                ss.frames_gray = []
                ss.poses = []
                ss.ball_positions = []
                ss.manual_tracks = {}
                ss.next_manual_id = 0
                ss.pending_manual_points = {}
                ss.current_frame = 0
                # Save to temp file
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
                tmp.write(uploaded.read())
                tmp.close()
                ss.video_path = tmp.name
                # Probe video
                cap = cv2.VideoCapture(ss.video_path)
                if cap.isOpened():
                    ss.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    ss.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

        if ss.video_path:
            st.caption(f"File: {ss.video_name}")
            st.caption(f"Frames: {ss.frame_count} @ {ss.fps:.1f} fps")

        st.divider()
        st.subheader("2. Settings")
        max_frames = st.number_input(
            "Max frames", min_value=10, max_value=2000,
            value=DEFAULT_MAX_FRAMES, step=10,
            help="Limit frames to process (saves memory).",
        )
        mm_per_pixel = st.number_input(
            "Scale (mm/pixel)", min_value=0.1, max_value=100.0,
            value=DEFAULT_MM_PER_PIXEL, step=0.1,
            help="Conversion factor from pixels to millimeters.",
        )

        st.divider()
        st.subheader("3. Process")
        do_process = st.button(
            "▶ Start Processing", type="primary",
            disabled=not ss.video_path,
            use_container_width=True,
        )

        if ss.processing_done:
            st.success(f"Processed {len(ss.poses)} frames")

        st.divider()
        st.subheader("4. Export")
        col_a, col_b = st.columns(2)
        with col_a:
            export_video_btn = st.button(
                "🎬 Replay Video", disabled=not ss.processing_done,
                use_container_width=True,
            )
        with col_b:
            export_csv_btn = st.button(
                "📊 CSV Data", disabled=not ss.processing_done,
                use_container_width=True,
            )

        st.divider()
        st.caption("Powered by MediaPipe + OpenCV")
        st.caption("Streamlit Community Cloud ready")

    # -- Processing --
    if do_process and ss.video_path:
        with st.spinner("Loading frames..."):
            loader = FrameLoader(ss.video_path, max_frames=max_frames)
            ss.fps = loader.fps
            frames = loader.load_frames()
            T = len(frames)

            # Store BGR frames individually for random access
            ss.frames_bgr = [frames[i] for i in range(T)]
            ss.frames_gray = [
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in range(T)
            ]
            del frames

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        # Pose estimation
        status_text.text("Running pose estimation...")
        estimator = get_pose_estimator()
        ss.poses = []
        for i in range(T):
            pose = estimator.process_frame(ss.frames_bgr[i])
            ss.poses.append(pose)
            progress_bar.progress((i + 1) / (T * 2))

        # Ball tracking
        status_text.text("Running ball tracking...")
        ball_tracker = BallTracker()
        ss.ball_positions = []
        for i in range(T):
            bp = ball_tracker.detect(ss.frames_bgr[i])
            ss.ball_positions.append(bp)
            progress_bar.progress(0.5 + (i + 1) / (T * 2))

        ss.frame_count = T
        ss.current_frame = 0
        ss.processing_done = True
        ss.manual_tracks = {}
        ss.next_manual_id = 0
        ss.pending_manual_points = {}

        estimator.release()
        progress_bar.empty()
        status_text.empty()
        st.success(f"Done! Processed {T} frames. Navigate frames below.")
        st.rerun()

    # -- Error display --
    if ss.processing_error:
        st.error(ss.processing_error)
        ss.processing_error = None

    # -- Main content area --
    if not ss.processing_done:
        st.markdown("""
        ## Welcome to Human Motion Capture & Measurement

        This web interface lets you analyze human motion from video **without downloading any files**.

        ### How to use:
        1. **Upload a video** using the sidebar (mp4, avi, mov, gif, webm)
        2. **Adjust settings** — limit frames for faster processing
        3. **Click "Start Processing"** to run pose estimation and tracking
        4. **View results** frame-by-frame with skeleton overlay
        5. **Add manual tracking points** by clicking on the video
        6. **Export** the replay video or CSV data

        ---
        Upload a video to begin.
        """)
        return

    # -- Results view --
    T = ss.frame_count
    if T == 0:
        return

    st.markdown("### Frame Viewer")
    frame_slider = st.slider(
        "Navigate frames", 0, T - 1, ss.current_frame, 1,
        key="frame_nav",
    )
    ss.current_frame = frame_slider

    # Build frame display
    frame_bgr = ss.frames_bgr[frame_slider]
    landmarks = ss.poses[frame_slider] if frame_slider < len(ss.poses) else None
    ball_pos = ss.ball_positions[frame_slider] if frame_slider < len(ss.ball_positions) else None
    vel, dist, elapsed = calc_ball_metrics(ss.ball_positions, frame_slider, ss.fps, mm_per_pixel)

    # Get manual positions at this frame
    manual_pos: Dict[int, Tuple[float, float]] = {}
    for pid, history in ss.manual_tracks.items():
        for f_idx, x, y in history:
            if f_idx == frame_slider:
                manual_pos[pid] = (x, y)
                break

    traj_up_to = build_manual_trajectories_up_to(ss.manual_tracks, frame_slider)

    overlaid = render_overlay_frame(
        frame_bgr, landmarks, ball_pos, manual_pos, traj_up_to,
        vel, dist, elapsed,
    )

    # Display clickable image
    col_img, col_info = st.columns([3, 1])

    with col_img:
        click_result = clickable_image(overlaid, f"f{frame_slider}")
        if click_result and isinstance(click_result, dict) and "x" in click_result:
            px, py = int(click_result["x"]), int(click_result["y"])
            pid = ss.next_manual_id
            ss.next_manual_id += 1
            # Run optical flow tracking from this frame
            initial = {pid: (float(px), float(py))}
            tracks = run_optical_flow_tracking(
                ss.frames_gray, frame_slider, initial,
            )
            ss.manual_tracks[pid] = tracks[pid]
            st.toast(f"Point #{pid} added and tracked!", icon="✅")
            st.rerun()

    with col_info:
        st.metric("Frame", f"{frame_slider} / {T - 1}")
        st.metric("Time", f"{elapsed:.2f} s")
        st.metric("Velocity", f"{vel:.2f} mm/s")
        st.metric("Distance", f"{dist:.2f} mm")

        if landmarks is not None:
            detected_kps = sum(1 for lm in landmarks if lm[2] >= 0.3)
            st.caption(f"Keypoints detected: {detected_kps}/33")

        if ball_pos is not None:
            st.caption(f"Ball: ({ball_pos[0]:.0f}, {ball_pos[1]:.0f})")

        # Manual points list
        if ss.manual_tracks:
            st.divider()
            st.markdown("**Manual Points**")
            for pid in sorted(ss.manual_tracks.keys()):
                history = ss.manual_tracks[pid]
                last = history[-1] if history else None
                tracked_frames = len(history)
                col_pt, col_del = st.columns([3, 1])
                with col_pt:
                    st.caption(f"#{pid}: {tracked_frames} frames tracked")
                with col_del:
                    if st.button("X", key=f"del_{pid}"):
                        del ss.manual_tracks[pid]
                        st.rerun()

    st.divider()
    st.caption(
        "Click on the frame to add manual tracking points. "
        "Points are automatically tracked using Lucas-Kanade optical flow."
    )

    # -- Export handlers --
    if export_video_btn:
        with st.spinner("Generating replay video..."):
            video_data = generate_replay_video(
                ss.frames_bgr, ss.poses, ss.ball_positions,
                ss.manual_tracks, ss.fps, mm_per_pixel,
            )
        if video_data:
            st.download_button(
                "⬇ Download Replay Video",
                video_data,
                file_name="mocap_replay.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
            st.success("Replay video ready! Click above to download.")
        else:
            st.error("Failed to generate video.")

    if export_csv_btn:
        import csv as csv_mod
        buf = io.StringIO()
        writer = csv_mod.writer(buf)
        writer.writerow(["frame_idx", "time_sec", "point_type", "point_id", "name", "x", "y"])
        for fi in range(T):
            t = fi / ss.fps if ss.fps > 0 else 0.0
            pose = ss.poses[fi] if fi < len(ss.poses) else None
            if pose is not None:
                for ki in range(len(pose)):
                    x, y, conf = pose[ki]
                    if conf >= 0.3:
                        name = KEYPOINT_NAMES[ki] if ki < len(KEYPOINT_NAMES) else f"kp_{ki}"
                        writer.writerow([fi, f"{t:.4f}", "human", ki, name, f"{x:.2f}", f"{y:.2f}"])
            for pid, history in ss.manual_tracks.items():
                for f_idx, mx, my in history:
                    if f_idx == fi:
                        writer.writerow([fi, f"{t:.4f}", "manual", pid, f"manual_{pid}", f"{mx:.2f}", f"{my:.2f}"])
                        break

        st.download_button(
            "⬇ Download CSV Data",
            buf.getvalue(),
            file_name="mocap_tracking.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
