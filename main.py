#!/usr/bin/env python3
"""Video-based Human Motion Capture and Measurement System.

Two modes:
    GUI mode (default):    python main.py
    CLI mode:              python main.py --input data/sample.mp4 --max_frames 200
"""

import sys
import warnings

warnings.filterwarnings("ignore")


def run_gui():
    """Launch the PyQt5 GUI application."""
    from PyQt5.QtWidgets import QApplication
    from src.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark stylesheet — large fonts & buttons
    app.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; }
        QLabel { color: #cccccc; font-size: 18px; }
        QPushButton {
            background-color: #3a3a3a; color: #e0e0e0;
            border: 2px solid #555; border-radius: 6px;
            padding: 12px 28px; font-size: 22px; font-weight: bold;
        }
        QPushButton:hover { background-color: #4a4a4a; }
        QPushButton:pressed { background-color: #2a2a2a; }
        QPushButton:disabled { background-color: #2a2a2a; color: #666; }
        QPushButton:checked { background-color: #005a9e; border-color: #0078d4; }
        QProgressBar {
            border: 1px solid #555; border-radius: 3px;
            background-color: #2a2a2a; text-align: center; color: #ccc;
            font-size: 16px;
        }
        QProgressBar::chunk { background-color: #0078d4; border-radius: 2px; }
        QLineEdit {
            background-color: #2a2a2a; color: #e0e0e0;
            border: 2px solid #555; border-radius: 6px;
            padding: 10px 16px; font-size: 22px;
        }
        QListWidget {
            background-color: #2a2a2a; border: 1px solid #555;
            color: #ccc; font-size: 16px;
        }
        QListWidget::item { padding: 4px; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


def run_cli():
    """Run the CLI pipeline (original functionality)."""
    import argparse
    from pathlib import Path

    import numpy as np
    import yaml

    from src.frame_loader import FrameLoader
    from src.pose_estimator import PoseEstimator, KEYPOINT_NAMES
    from src.tracker import Tracker
    from src.kinematics import (
        compute_angles_from_landmarks,
        angular_velocity,
        angular_acceleration,
        trajectory_length,
        range_of_motion,
        smooth_angles,
    )
    from src.analyzer import detect_turning_points, compute_symmetry, fourier_analysis
    from src.visualizer import (
        plot_trajectory,
        plot_kinematics,
        plot_angle_heatmap,
        animate_with_trajectory,
        plot_landmarks_3d,
        TRAJECTORY_KEYPOINTS,
    )
    from src.exporter import export_csv, export_json, export_mat

    parser = argparse.ArgumentParser(description="Video-based Human Motion Capture & Measurement")
    parser.add_argument("--input", required=True, help="Path to input video/GIF")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--export_format", default="csv,json")
    parser.add_argument("--no_animation", action="store_true")
    parser.add_argument("--config", default="config/landmarks.yaml")
    # Action recognition
    parser.add_argument("--register", action="store_true",
                        help="Register the input video as an action template")
    parser.add_argument("--action_name", default=None,
                        help="Action name for registration (required with --register)")
    parser.add_argument("--recognize", action="store_true",
                        help="Recognize actions in the input video using saved templates")
    parser.add_argument("--templates", default="config/action_templates.json",
                        help="Path to action templates JSON file")
    parser.add_argument("--sensitivity", type=float, default=2.0,
                        help="DTW threshold multiplier (lower=stricter, default 2.0)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_angle_definitions(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return {name: tuple(indices) for name, indices in config.get("angle_definitions", {}).items()}

    print(f"[1/7] Loading frames from: {args.input}")
    loader = FrameLoader(args.input, max_frames=args.max_frames)
    fps = loader.fps
    frames = loader.load_frames()
    T = len(frames)
    print(f"       Loaded {T} frames at {fps:.1f} FPS")

    print("[2/7] Running MediaPipe Pose estimation...")
    estimator = PoseEstimator()
    poses = estimator.process_batch(frames)
    detected = sum(1 for p in poses if p is not None)
    print(f"       Pose detected in {detected}/{T} frames")
    estimator.release()

    print("[3/7] Tracking and interpolating keypoints...")
    tracker = Tracker()
    tracker.add_frames(poses)
    positions = tracker.interpolate()
    positions_smooth = tracker.smooth(positions, window=5)

    print("[4/7] Computing kinematics...")
    angle_defs = load_angle_definitions(args.config)
    angles = compute_angles_from_landmarks(positions_smooth, angle_defs)
    time_sec = np.arange(T) / fps

    metrics = {"fps": fps, "total_frames": T, "duration_sec": float(T / fps)}
    kinematics_data = {}
    for name, angle_seq in angles.items():
        vel = angular_velocity(angle_seq, fps)
        accel = angular_acceleration(angle_seq, fps)
        smooth = smooth_angles(angle_seq, window=5)
        rom = range_of_motion(angle_seq)
        metrics[f"{name}_rom_deg"] = float(rom)
        kinematics_data[f"{name}_angle_deg"] = smooth
        kinematics_data[f"{name}_velocity_deg_s"] = vel
        kinematics_data[f"{name}_acceleration_deg_s2"] = accel

    traj_lengths = {}
    for kp_idx in TRAJECTORY_KEYPOINTS:
        tl = trajectory_length(positions_smooth[:, kp_idx, :])
        name = KEYPOINT_NAMES[kp_idx]
        traj_lengths[f"{name}_trajectory_px"] = tl
        metrics[f"{name}_trajectory_length_px"] = float(tl)
    metrics["trajectory_lengths"] = traj_lengths
    print(f"       Computed {len(angles)} joint angles and kinematics")

    print("[5/7] Analyzing motion patterns...")
    if "left_knee_angle" in angles:
        knee = angles["left_knee_angle"]
        tp = detect_turning_points(knee, fps)
        metrics["left_knee_turning_points"] = {
            "flexion_peaks": tp["flexion_peaks"].tolist(),
            "extension_peaks": tp["extension_peaks"].tolist(),
        }
        if "right_knee_angle" in angles:
            sym = compute_symmetry(knee, angles["right_knee_angle"])
            metrics["knee_symmetry"] = {k: float(v) for k, v in sym.items()}
        fft = fourier_analysis(knee, fps)
        metrics["left_knee_fourier"] = {
            "dominant_freq_hz": float(fft["dominant_freq_hz"]),
            "dominant_period_sec": float(fft["dominant_period_sec"]),
        }

    print("[6/7] Generating visualizations...")
    plot_trajectory(positions_smooth, TRAJECTORY_KEYPOINTS,
                    [KEYPOINT_NAMES[i] for i in TRAJECTORY_KEYPOINTS],
                    str(output_dir / "trajectory_xy.png"))
    key_angles = {n: angles[n] for n in ["left_knee_angle", "right_knee_angle",
                  "left_elbow_angle", "right_elbow_angle"] if n in angles}
    key_velocities = {f"{n}_vel": kinematics_data[f"{n}_velocity_deg_s"] for n in key_angles}
    plot_kinematics(time_sec, key_angles, str(output_dir / "kinematics_curves.png"),
                    ylabel="Angle (deg)", overlay_signals=key_velocities)
    angle_names = list(angles.keys())
    if angle_names:
        plot_angle_heatmap(time_sec, angle_names, np.array([angles[n] for n in angle_names]),
                           str(output_dir / "angle_heatmap.png"))
    if not args.no_animation:
        print("       Generating animation...")
        animate_with_trajectory(frames, positions_smooth, str(output_dir / "animation.mp4"),
                                fps=fps, trail_length=20, show_skeleton=True)
    plot_landmarks_3d(positions_smooth, str(output_dir / "landmarks_3d.png"), frame_idx=T // 2)

    print("[7/7] Exporting data...")
    export_formats = [f.strip() for f in args.export_format.split(",")]
    if "csv" in export_formats:
        csv_data = {"frame": np.arange(T, dtype=np.float64)}
        for n, s in angles.items():
            csv_data[n] = s
        export_csv(time_sec, csv_data, str(output_dir / "kinematics.csv"))
    if "json" in export_formats:
        export_json(metrics, str(output_dir / "metrics.json"))
    if "mat" in export_formats:
        try:
            mat_data = {"time_sec": time_sec}
            for n, s in kinematics_data.items():
                mat_data[n] = s
            export_mat(mat_data, str(output_dir / "kinematics.mat"))
        except Exception as e:
            print(f"       [WARN] MAT export failed: {e}")

    # ---- Action recognition (optional) ---------------------------------
    if args.register or args.recognize:
        from src.action_recognizer import (
            ActionRecognizer,
            TemplateStore,
            extract_angle_features,
            recognize_rule_based_actions,
            ANGLE_KEYS,
        )

        store = TemplateStore(target_len=60)
        templates_path = Path(args.templates)
        if templates_path.exists():
            store.load(str(templates_path))
            print(f"\n[Action] Loaded {len(store)} template(s) from {templates_path}")

        if args.register:
            if not args.action_name:
                print("[Action] ERROR: --action_name is required with --register")
                return 1
            # Extract the full-video angle matrix
            feat = extract_angle_features(positions_smooth, angle_defs, window=5)
            store.add(feat, args.action_name, fps)
            store.save(str(templates_path))
            print(f"[Action] Registered template '{args.action_name}' "
                  f"({len(store)} total). Saved to {templates_path}")

        if args.recognize:
            matches = recognize_rule_based_actions(positions_smooth, fps)
            if len(store) > 0:
                feat = extract_angle_features(positions_smooth, angle_defs, window=5)
                recognizer = ActionRecognizer(store, sensitivity=args.sensitivity)
                matches.extend(recognizer.recognize(feat, fps))
                matches.sort(key=lambda m: (m.start_frame, m.action_name))
            else:
                print("[Action] No DTW templates loaded; running built-in rule-based pose recognition.")

            if matches:
                print(f"\n[Action] Detected {len(matches)} action(s):")
                for m in matches:
                    print(f"  * {m.action_name:12s} | "
                          f"frame {m.start_frame:4d}-{m.end_frame:4d} | "
                          f"time {m.start_sec:.1f}s-{m.end_sec:.1f}s | "
                          f"confidence {m.confidence:.2%}")
            else:
                print("[Action] No actions detected.")

    print(f"\nDone! Results saved to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and "--input" in sys.argv:
        sys.exit(run_cli())
    else:
        run_gui()
