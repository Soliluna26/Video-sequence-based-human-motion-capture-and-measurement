#!/usr/bin/env python3
"""Streamlit web interface for the Human Motion Capture & Measurement System.

Launch locally:
    streamlit run web_app.py

Deploy to HuggingFace Spaces (Docker SDK):
    Uses python:3.11-slim-bullseye (Debian 11, GLib 2.66) which provides
    libgthread-2.0.so.0 natively — no runtime hacks needed.
"""

import base64
import io
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    ball_trajectory: List[Tuple[float, float]] | None = None,
    vel_mm_s: float = 0.0,
    dist_mm: float = 0.0,
    elapsed_s: float = 0.0,
    show_measurements: bool = True,
) -> np.ndarray:
    """Render a frame with skeleton, ball, manual points, and trajectories."""
    canvas = frame_bgr.copy()

    if landmarks is not None:
        canvas = draw_skeleton(canvas, landmarks, CYAN)

    # Ball trajectory (green)
    if ball_trajectory and len(ball_trajectory) >= 2:
        for i in range(1, len(ball_trajectory)):
            x1, y1 = ball_trajectory[i - 1]
            x2, y2 = ball_trajectory[i]
            p1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
            p2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
            cv2.line(canvas, p1, p2, GREEN, 2)

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
    ball_positions: List[Optional[Tuple[float, float]]],
    ball_trajectory: List[Tuple[float, float]],
    vel_mm_s: float = 0.0,
    dist_mm: float = 0.0,
    elapsed_s: float = 0.0,
) -> np.ndarray:
    """Render a replay frame on black background with all trajectories."""
    h, w = frame_bgr.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if landmarks is not None:
        canvas = draw_skeleton(canvas, landmarks, CYAN)

    # Ball trajectory (green)
    if len(ball_trajectory) >= 2:
        for i in range(1, len(ball_trajectory)):
            x1, y1 = ball_trajectory[i - 1]
            x2, y2 = ball_trajectory[i]
            p1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
            p2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
            cv2.line(canvas, p1, p2, GREEN, 2)
    # Current ball position
    for bp in reversed(ball_positions):
        if bp is not None:
            bx, by = int(np.clip(bp[0], 0, w - 1)), int(np.clip(bp[1], 0, h - 1))
            cv2.circle(canvas, (bx, by), 10, GREEN, -1)
            cv2.circle(canvas, (bx, by), 14, GREEN, 2)
            break
    # Ball sampling dots
    for px, py in ball_trajectory:
        cx, cy = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
        cv2.circle(canvas, (cx, cy), 3, GREEN, -1)

    # Manual point trajectories (blue)
    canvas = draw_trajectories(canvas, manual_trajectories, BLUE)
    for pid, pos in manual_positions.items():
        px, py = int(np.clip(pos[0], 0, w - 1)), int(np.clip(pos[1], 0, h - 1))
        cv2.circle(canvas, (px, py), 6, BLUE, -1)
        cv2.circle(canvas, (px, py), 9, BLUE, 2)
    for pid, pts in manual_trajectories.items():
        for px, py in pts:
            cx, cy = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
            cv2.circle(canvas, (cx, cy), 3, BLUE, -1)

    # Measurements
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
    """Generate a replay MP4 video with skeleton, ball, and trajectories."""
    if not frames_bgr:
        return b""
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = os.path.join(tempfile.gettempdir(), "mocap_replay.mp4")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    # Accumulate ball trajectory frame by frame
    ball_traj: List[Tuple[float, float]] = []

    for fi, frame_bgr in enumerate(frames_bgr):
        manual_pos = {}
        for pid, history in manual_tracks.items():
            for f_idx, x, y in history:
                if f_idx == fi:
                    manual_pos[pid] = (x, y)
                    break

        # Ball trajectory
        bp = ball_positions[fi] if fi < len(ball_positions) else None
        if bp is not None:
            ball_traj.append(bp)
        ball_pos_snapshot = [ball_positions[fi] if fi < len(ball_positions) else None]

        traj = build_manual_trajectories_up_to(manual_tracks, fi)
        vel, dist, elapsed = calc_ball_metrics(ball_positions, fi, fps, mm_per_pixel)

        canvas = render_replay_frame(
            frame_bgr, poses[fi], manual_pos, traj,
            ball_pos_snapshot, list(ball_traj),
            vel, dist, elapsed,
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
        "manual_tracks": {},
        "next_manual_id": 0,
        "pending_manual_points": {},
        "processing_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Clickable image component
# ---------------------------------------------------------------------------
def clickable_image(frame_bgr: np.ndarray, key: str) -> Optional[Dict]:
    """Display a clickable BGR frame. Returns {'x': int, 'y': int} on click."""
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

    # -- Reset handler --
    if getattr(ss, "_do_reset", False):
        for _k in list(ss.keys()):
            del ss[_k]
        st.rerun()

    # -- Convenience: _s(key, default) --
    def _s(key, default=None):
        if key not in ss:
            ss[key] = default
        return ss[key]

    # -- Determine app mode --
    app_mode = _s("app_mode", "idle")
    has_video = bool(_s("video_path", ""))

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
            if _s("video_name", "") != uploaded.name:
                ss.video_name = uploaded.name
                ss.app_mode = "ready"
                ss.video_path = None
                ss.frames_bgr = []
                ss.frames_gray = []
                ss.poses = []
                ss.ball_positions = []
                ss.manual_tracks = {}
                ss.next_manual_id = 0
                ss.current_frame = 0
                ss.processing_done = False
                ss.play_mode = False
                ss.frame_idx = 0
                ss._preview_frame = None
                ss._replay_data = None
                # Save video to temp
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded.name).suffix,
                )
                tmp.write(uploaded.read())
                tmp.close()
                ss.video_path = tmp.name
                # Grab first frame for preview
                cap = cv2.VideoCapture(ss.video_path)
                if cap.isOpened():
                    ss.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    ss.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    ret, preview = cap.read()
                    if ret:
                        ss._preview_frame = preview
                cap.release()
                st.rerun()

        if has_video:
            st.caption(f"File: {_s('video_name')}")
            st.caption(f"Frames: {_s('frame_count')} @ {_s('fps', 30):.1f} fps")

        st.divider()
        st.subheader("2. Settings")
        max_frames = st.number_input(
            "Max frames", 10, 2000, DEFAULT_MAX_FRAMES, 10,
            help="Limit frames to process.",
        )
        mm_per_pixel = st.number_input(
            "Scale (mm/pixel)", 0.1, 100.0, DEFAULT_MM_PER_PIXEL, 0.1,
            help="Pixels to millimeters conversion.",
        )

        # -- Buttons depend on mode --
        st.divider()
        st.subheader("3. Controls")

        if app_mode == "ready":
            if st.button("▶ Start", type="primary",
                         use_container_width=True, key="btn_start"):
                ss.app_mode = "processing"
                st.rerun()

        if app_mode == "done":
            st.success(f"Processed {len(ss.poses)} frames")
            if st.button("🎬 Replay Video", use_container_width=True, key="btn_replay"):
                ss._do_replay = True
                ss._replay_data = None
                st.rerun()
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📊 CSV Data", use_container_width=True, key="btn_csv"):
                    ss._do_csv = True
                    st.rerun()
            with col_b:
                if st.button("🔄 Reset All", use_container_width=True, key="btn_reset"):
                    ss._do_reset = True
                    st.rerun()

        st.divider()
        st.caption("Powered by MediaPipe + OpenCV")

    # ================================================================
    # Processing mode
    # ================================================================
    if app_mode == "processing" and has_video:
        with st.spinner("Loading frames..."):
            loader = FrameLoader(ss.video_path, max_frames=max_frames)
            ss.fps = loader.fps
            frames = loader.load_frames()
            T = len(frames)
            ss.frames_bgr = [frames[i] for i in range(T)]
            ss.frames_gray = [
                cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in range(T)
            ]
            del frames

        pb = st.progress(0.0)
        sts = st.empty()

        sts.text("Running pose estimation...")
        estimator = get_pose_estimator()
        ss.poses = []
        for i in range(T):
            ss.poses.append(estimator.process_frame(ss.frames_bgr[i]))
            pb.progress((i + 1) / (T * 2))

        sts.text("Running ball tracking...")
        bt = BallTracker()
        ss.ball_positions = []
        for i in range(T):
            ss.ball_positions.append(bt.detect(ss.frames_bgr[i]))
            pb.progress(0.5 + (i + 1) / (T * 2))

        estimator.release()

        # Track pending manual points from preview
        pending = _s("_pending_pts", [])
        for pid, px, py in pending:
            tracks = run_optical_flow_tracking(
                ss.frames_gray, 0, {pid: (px, py)},
            )
            ss.manual_tracks[pid] = tracks[pid]
            if pid >= ss.next_manual_id:
                ss.next_manual_id = pid + 1
        ss._pending_pts = []

        ss.frame_count = T
        ss.current_frame = 0
        ss.frame_idx = 0
        ss.play_mode = False
        ss.app_mode = "done"
        ss._do_replay = False
        ss._replay_data = None

        pb.empty()
        sts.empty()
        st.success(f"Done! Processed {T} frames.")
        st.rerun()

    # ================================================================
    # Ready mode — show preview + manual point adding
    # ================================================================
    if app_mode == "ready":
        st.markdown("### Video Preview")
        st.info("Add manual tracking points now, or click **Start** to process the video.")

        preview_frame = _s("_preview_frame")
        if preview_frame is not None:
            h, w = preview_frame.shape[:2]

            # Show clickable preview
            pt_mode = st.checkbox("🎯 Point Mode", key="pt_ready",
                                  help="Check to enable clicking on the preview")
            ss._pt_ready = pt_mode

            if pt_mode:
                click_result = clickable_image(preview_frame, "preview")
                if click_result and isinstance(click_result, dict) and "x" in click_result:
                    px, py = int(click_result["x"]), int(click_result["y"])
                    pid = ss.next_manual_id
                    ss.next_manual_id += 1
                    # Save as pending — will be tracked during Start processing
                    _pending = _s("_pending_pts", [])
                    _pending.append((pid, float(px), float(py)))
                    ss._pending_pts = _pending
                    st.toast(f"Point #{pid} added on preview", icon="✅")
                    st.rerun()
            else:
                rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb, use_container_width=True)

            # List pending points
            pending = _s("_pending_pts", [])
            if pending:
                st.markdown("**Pending Points (tracked when Start is clicked):**")
                for pid, px, py in pending:
                    st.caption(f"#{pid}: ({px:.0f}, {py:.0f})")
                    if st.button("✕", key=f"del_pend_{pid}"):
                        ss._pending_pts = [(p, x, y) for p, x, y in pending if p != pid]
                        st.rerun()
        return

    # ================================================================
    # DONE mode — results
    # ================================================================
    if app_mode != "done":
        # Idle or unknown — show welcome
        st.markdown("""
        ## Welcome to Human Motion Capture & Measurement

        1. Upload a video
        2. (Optional) Add manual points on the preview
        3. Click **Start** to process
        4. Frame Viewer, Replay Video, CSV Export
        """)
        return

    T = _s("frame_count", 0)
    if T == 0:
        return

    # --- Replay generation ---
    if getattr(ss, "_do_replay", False):
        ss._do_replay = False
        with st.spinner("Generating replay video..."):
            vdata = generate_replay_video(
                ss.frames_bgr, ss.poses, ss.ball_positions,
                ss.manual_tracks, ss.fps, mm_per_pixel,
            )
        if vdata:
            # Write to temp file for st.video (more reliable than bytes)
            _rp = os.path.join(tempfile.gettempdir(), "mocap_replay.mp4")
            with open(_rp, "wb") as f:
                f.write(vdata)
            ss._replay_path = _rp
            ss._replay_data = vdata
        else:
            st.error("Failed to generate video.")

    # --- Show replay video ---
    if getattr(ss, "_replay_path", None):
        st.markdown("### Replay Video")
        st.video(ss._replay_path)
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.download_button(
                "⬇ Download", ss._replay_data,
                file_name="mocap_replay.mp4", mime="video/mp4",
            )
        with col_r2:
            if st.button("✕ Close Replay", key="close_replay"):
                ss._replay_path = None
                ss._replay_data = None
                st.rerun()
        st.divider()

    # --- Frame Viewer ---
    st.markdown("### Frame Viewer")

    # Point mode
    pt_mode = st.checkbox("🎯 Point Mode", key="pt_done",
                          help="Click on frame to add tracking points (🔵 manual, 🟢 ball)")
    ss.point_mode = pt_mode

    # Playback row
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        fidx = _s("frame_idx", 0)
        fidx = st.slider("Frame", 0, T - 1, fidx, 1, key="fslider",
                         label_visibility="collapsed")
        ss.frame_idx = fidx
    with c2:
        fps_val = st.number_input("FPS", 1, 60, _s("play_fps", 10), 1, key="fpsinp")
        ss.play_fps = fps_val
    with c3:
        playing = _s("play_mode", False)
        if st.button("⏸" if playing else "▶", key="playbtn", use_container_width=True):
            ss.play_mode = not playing

    # Play loop
    if _s("play_mode", False):
        if ss.frame_idx < T - 1:
            ss.frame_idx += 1
            time.sleep(1.0 / max(ss.play_fps, 1))
            st.rerun()
        else:
            ss.play_mode = False
            st.rerun()

    fidx = ss.frame_idx

    frame_bgr = ss.frames_bgr[fidx]
    landmarks = ss.poses[fidx] if fidx < len(ss.poses) else None
    ball_pos = ss.ball_positions[fidx] if fidx < len(ss.ball_positions) else None
    vel, dist, elapsed = calc_ball_metrics(ss.ball_positions, fidx, ss.fps, mm_per_pixel)

    manual_pos: Dict[int, Tuple[float, float]] = {}
    for pid, history in ss.manual_tracks.items():
        for fi, x, y in history:
            if fi == fidx:
                manual_pos[pid] = (x, y)
                break

    traj_up_to = build_manual_trajectories_up_to(ss.manual_tracks, fidx)
    ball_traj = [(p[0], p[1]) for p in ss.ball_positions[:fidx + 1] if p is not None]

    overlaid = render_overlay_frame(
        frame_bgr, landmarks, ball_pos, manual_pos, traj_up_to,
        ball_trajectory=ball_traj,
        vel_mm_s=vel, dist_mm=dist, elapsed_s=elapsed,
    )

    col_img, col_info = st.columns([3, 1])
    with col_img:
        if getattr(ss, "point_mode", False):
            click_result = clickable_image(overlaid, f"f{fidx}")
            if click_result and isinstance(click_result, dict) and "x" in click_result:
                px, py = int(click_result["x"]), int(click_result["y"])
                pid = ss.next_manual_id
                ss.next_manual_id += 1
                tracks = run_optical_flow_tracking(
                    ss.frames_gray, fidx, {pid: (float(px), float(py))},
                )
                ss.manual_tracks[pid] = tracks[pid]
                st.toast(f"Point #{pid} tracked!", icon="✅")
                st.rerun()
        else:
            rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
            st.image(rgb, use_container_width=True)

    with col_info:
        st.metric("Frame", f"{fidx} / {T - 1}")
        st.metric("Time", f"{elapsed:.2f} s")
        st.metric("Velocity", f"{vel:.2f} mm/s")
        st.metric("Distance", f"{dist:.2f} mm")
        if landmarks is not None:
            nk = sum(1 for lm in landmarks if lm[2] >= 0.3)
            st.caption(f"Keypoints: {nk}/33")
        if ball_pos is not None:
            st.caption(f"Ball: ({ball_pos[0]:.0f}, {ball_pos[1]:.0f})")
        if ss.manual_tracks:
            st.divider()
            st.markdown("**Manual Points**")
            for pid in sorted(ss.manual_tracks.keys()):
                h = ss.manual_tracks[pid]
                cp, cd = st.columns([3, 1])
                with cp:
                    st.caption(f"#{pid}: {len(h)} frames")
                with cd:
                    if st.button("X", key=f"del_{pid}"):
                        del ss.manual_tracks[pid]
                        st.rerun()

    st.divider()
    if getattr(ss, "point_mode", False):
        st.info("Point Mode ON — click image to add tracking points")
    else:
        st.caption("Check Point Mode to add tracking markers")

    # --- CSV Export ---
    if getattr(ss, "_do_csv", False):
        ss._do_csv = False
        import csv as csv_mod
        buf = io.StringIO()
        w = csv_mod.writer(buf)
        w.writerow(["frame_idx", "time_sec", "point_type", "point_id", "name", "x", "y"])
        for fi in range(T):
            t = fi / ss.fps if ss.fps > 0 else 0.0
            pose = ss.poses[fi] if fi < len(ss.poses) else None
            if pose is not None:
                for ki in range(len(pose)):
                    x, y, conf = pose[ki]
                    if conf >= 0.3:
                        name = KEYPOINT_NAMES[ki] if ki < len(KEYPOINT_NAMES) else f"kp_{ki}"
                        w.writerow([fi, f"{t:.4f}", "human", ki, name, f"{x:.2f}", f"{y:.2f}"])
            for pid, history in ss.manual_tracks.items():
                for f_idx, mx, my in history:
                    if f_idx == fi:
                        w.writerow([fi, f"{t:.4f}", "manual", pid, f"manual_{pid}", f"{mx:.2f}", f"{my:.2f}"])
                        break
        st.download_button(
            "⬇ Download CSV", buf.getvalue(),
            file_name="mocap_tracking.csv", mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
