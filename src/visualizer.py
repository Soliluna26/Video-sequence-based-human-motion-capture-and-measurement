"""Visualization module.

Generates five types of visual output:
- X-Y trajectory plots
- Kinematics curves (angle / velocity / acceleration vs time)
- Joint angle heatmaps
- Animated trajectory overlay
- 3D keypoint scatter (optional)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

matplotlib.use("Agg")  # non-interactive backend

# MediaPipe bone connections for stick-figure rendering
BONE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (23, 25), (25, 27), (24, 26), (26, 28),   # legs
    (15, 17), (15, 19), (17, 19),              # left hand
    (16, 18), (16, 20), (18, 20),              # right hand
    (27, 29), (27, 31), (29, 31),              # left foot
    (28, 30), (28, 32), (30, 32),              # right foot
]

# Selected keypoints for trajectory overlay (major joints)
TRAJECTORY_KEYPOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
TRAJECTORY_COLORS = [
    "#FF0000", "#0000FF", "#FF4444", "#4444FF", "#FF8888", "#8888FF",
    "#00AA00", "#AA00AA", "#00FF44", "#FF44FF", "#00AAAA", "#AAAAFF",
]


def plot_trajectory(
    positions: np.ndarray,
    keypoint_indices: List[int],
    keypoint_names: List[str],
    output_path: str,
    title: str = "Joint Trajectories (X-Y)",
):
    """Plot spatial (X-Y) trajectories of selected keypoints.

    Parameters
    ----------
    positions : np.ndarray
        Shape (T, 33, 2).
    keypoint_indices : list[int]
        Which keypoints to plot.
    keypoint_names : list[str]
        Names for the legend.
    output_path : str
        Path to save the figure (PNG/PDF).
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, name in zip(keypoint_indices, keypoint_names):
        x = positions[:, idx, 0]
        y = positions[:, idx, 1]
        valid = ~np.isnan(x) & ~np.isnan(y)
        if valid.sum() > 0:
            ax.plot(x[valid], y[valid], linewidth=0.8, label=name, alpha=0.8)
            # Mark start and end
            if valid.sum() >= 2:
                start_t = np.argmax(valid)
                end_t = len(valid) - np.argmax(valid[::-1]) - 1
                ax.scatter(x[start_t], y[start_t], s=20, marker="o", zorder=5)
                ax.scatter(x[end_t], y[end_t], s=20, marker="s", zorder=5)

    ax.invert_yaxis()
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_kinematics(
    time_sec: np.ndarray,
    signals: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Kinematics Curves",
    xlabel: str = "Time (s)",
    ylabel: str = "Angle (deg)",
    overlay_signals: Optional[Dict[str, np.ndarray]] = None,
    overlay_ylabel: str = "Angular Velocity (deg/s)",
):
    """Plot kinematics curves, optionally with overlaid secondary signals.

    Parameters
    ----------
    time_sec : np.ndarray
        Time axis in seconds.
    signals : dict
        Primary signals {label: array}.
    output_path : str
    title, xlabel, ylabel : str
    overlay_signals : dict, optional
        Secondary signals plotted on a twin y-axis.
    overlay_ylabel : str
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    for label, sig in signals.items():
        valid = ~np.isnan(sig)
        if valid.sum() > 0:
            ax1.plot(time_sec[valid], sig[valid], linewidth=1.2, label=label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    if overlay_signals:
        ax2 = ax1.twinx()
        for label, sig in overlay_signals.items():
            valid = ~np.isnan(sig)
            if valid.sum() > 0:
                ax2.plot(time_sec[valid], sig[valid], linewidth=0.8,
                         linestyle="--", alpha=0.6, label=label)
        ax2.set_ylabel(overlay_ylabel)
        ax2.legend(loc="upper right", fontsize=8)

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_angle_heatmap(
    time_sec: np.ndarray,
    angle_names: List[str],
    angles_matrix: np.ndarray,
    output_path: str,
    title: str = "Joint Angle Heatmap",
):
    """Plot joint angles as a heatmap (angles × time).

    Parameters
    ----------
    time_sec : np.ndarray
        Time axis.
    angle_names : list[str]
        Joint names (y-axis labels).
    angles_matrix : np.ndarray
        Shape (num_angles, T), each row is one angle's time series.
    output_path : str
    title : str
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        angles_matrix,
        aspect="auto",
        cmap="inferno",
        interpolation="bilinear",
        extent=[time_sec[0], time_sec[-1], len(angle_names) - 0.5, -0.5],
    )
    ax.set_yticks(range(len(angle_names)))
    ax.set_yticklabels(angle_names)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Angle (deg)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def animate_with_trajectory(
    frames_bgr: np.ndarray,
    positions: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    trail_length: int = 20,
    show_skeleton: bool = True,
):
    """Create an animation with trajectory overlay on original video.

    Parameters
    ----------
    frames_bgr : np.ndarray
        Shape (T, H, W, 3), original video frames.
    positions : np.ndarray
        Shape (T, 33, 2), interpolated keypoint positions.
    output_path : str
        Path for output MP4/GIF.
    fps : float
        Output animation FPS.
    trail_length : int
        Number of past frames to show as fading trail.
    show_skeleton : bool
        Whether to draw stick-figure skeleton.
    """
    T = min(len(frames_bgr), len(positions))
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    if suffix == ".gif":
        _animate_gif(frames_bgr, positions, T, output_path, fps, trail_length, show_skeleton)
    else:
        _animate_mp4(frames_bgr, positions, T, output_path, fps, trail_length, show_skeleton)


def _draw_overlay(
    frame: np.ndarray,
    positions_t: np.ndarray,
    trail: List[np.ndarray],
    trail_length: int,
    show_skeleton: bool,
    alpha: float = 1.0,
) -> np.ndarray:
    """Draw trajectory trail and optional skeleton on a single frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Draw fading trajectory trail
    for age, pos in enumerate(reversed(trail[-trail_length:])):
        fade = max(0.15, 1.0 - age / trail_length)
        for kp_idx, color_hex in zip(TRAJECTORY_KEYPOINTS, TRAJECTORY_COLORS):
            if kp_idx >= pos.shape[0]:
                continue
            x, y = pos[kp_idx]
            if np.isnan(x) or np.isnan(y):
                continue
            color = _hex_to_bgr(color_hex, fade)
            px, py = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            cv2.circle(canvas, (px, py), 3, color, -1)  # filled circle

    # Draw skeleton
    if show_skeleton:
        for p1, p2 in BONE_CONNECTIONS:
            if p1 >= positions_t.shape[0] or p2 >= positions_t.shape[0]:
                continue
            x1, y1 = positions_t[p1]
            x2, y2 = positions_t[p2]
            if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                continue
            pt1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
            pt2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

    return canvas


def _animate_mp4(frames_bgr, positions, T, output_path, fps, trail_length, show_skeleton):
    """Write animation as MP4 using OpenCV VideoWriter."""
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    trail: List[np.ndarray] = []
    for t in range(T):
        trail.append(positions[t].copy())
        canvas = _draw_overlay(frames_bgr[t], positions[t], trail, trail_length, show_skeleton)
        writer.write(canvas)
    writer.release()


def _animate_gif(frames_bgr, positions, T, output_path, fps, trail_length, show_skeleton):
    """Write animation as GIF using imageio (via Pillow)."""
    from PIL import Image

    trail: List[np.ndarray] = []
    pil_frames = []
    for t in range(T):
        trail.append(positions[t].copy())
        canvas = _draw_overlay(frames_bgr[t], positions[t], trail, trail_length, show_skeleton)
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(canvas_rgb))

    duration = int(1000 / fps) if fps > 0 else 33
    pil_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )


def plot_landmarks_3d(
    positions: np.ndarray,
    output_path: str,
    frame_idx: int = 0,
):
    """Plot 3D scatter of keypoints (2D positions, z = keypoint index).

    Parameters
    ----------
    positions : np.ndarray
        Shape (T, 33, 2).
    output_path : str
    frame_idx : int
        Which frame to plot.
    """
    if frame_idx >= len(positions):
        frame_idx = len(positions) - 1

    pos = positions[frame_idx]  # (33, 2)
    valid = ~np.isnan(pos).any(axis=1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = pos[valid, 0]
    ys = pos[valid, 1]
    zs = np.arange(33)[valid]

    ax.scatter(xs, ys, zs, c=zs, cmap="viridis", s=30)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Keypoint Index")
    ax.set_title(f"3D Keypoint Scatter — Frame {frame_idx}")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _hex_to_bgr(hex_color: str, alpha: float = 1.0) -> Tuple[int, int, int]:
    """Convert hex color to BGR tuple with optional alpha blending on black."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (int(b * alpha), int(g * alpha), int(r * alpha))
