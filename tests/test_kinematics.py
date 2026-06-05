"""Unit tests for the kinematics module."""

import sys
from pathlib import Path

# Allow running tests from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.kinematics import (
    compute_angle,
    compute_angles_from_landmarks,
    compute_segment_angle,
    compute_segment_angles_from_landmarks,
    angular_velocity,
    angular_acceleration,
    trajectory_length,
    range_of_motion,
    smooth_angles,
)


class TestComputeAngle:
    """Test the three-point angle computation using known geometric cases."""

    def test_right_angle(self):
        """p1=(1,0), p2=(0,0), p3=(0,1) -> 90 degrees."""
        angle = compute_angle((1, 0), (0, 0), (0, 1))
        assert abs(angle - 90.0) < 1e-6

    def test_straight_line(self):
        """p1=(1,0), p2=(0,0), p3=(-1,0) -> 180 degrees."""
        angle = compute_angle((1, 0), (0, 0), (-1, 0))
        assert abs(angle - 180.0) < 1e-6

    def test_zero_angle(self):
        """p1=(1,0), p2=(0,0), p3=(2,0) -> 0 degrees (same direction)."""
        angle = compute_angle((1, 0), (0, 0), (2, 0))
        assert abs(angle - 0.0) < 1e-6

    def test_acute_angle_45(self):
        """p1=(1,0), p2=(0,0), p3=(1,1) -> 45 degrees."""
        angle = compute_angle((1, 0), (0, 0), (1, 1))
        assert abs(angle - 45.0) < 1e-6

    def test_degenerate_same_points(self):
        """All three points coincide -> 0 degrees."""
        angle = compute_angle((0, 0), (0, 0), (0, 0))
        assert angle == 0.0

    def test_nonzero_vertex_offset(self):
        """Angle at (5,5): p1=(6,5), p2=(5,5), p3=(5,6) -> 90 deg."""
        angle = compute_angle((6, 5), (5, 5), (5, 6))
        assert abs(angle - 90.0) < 1e-6


class TestComputeSegmentAngle:
    """Test angles between two directed line segments."""

    def test_perpendicular_segments(self):
        angle = compute_segment_angle((0, 0), (1, 0), (0, 0), (0, 1))
        assert abs(angle - 90.0) < 1e-6

    def test_parallel_segments(self):
        angle = compute_segment_angle((1, 1), (3, 1), (4, 4), (8, 4))
        assert abs(angle - 0.0) < 1e-6


class TestAngularVelocity:
    """Test angular velocity using a known sine wave input."""

    def test_constant_angle(self):
        """Zero velocity for a constant angle sequence."""
        angles = np.array([45.0, 45.0, 45.0, 45.0, 45.0])
        vel = angular_velocity(angles, fps=30.0)
        valid = vel[~np.isnan(vel)]
        assert np.allclose(valid, 0.0, atol=1e-9)

    def test_sine_velocity(self):
        """For theta(t) = A * sin(2*pi*f*t), dθ/dt = A*2*pi*f*cos(2*pi*f*t).

        Test at frame spacing dt = 1/fps, with fps >> f for accuracy.
        """
        fps = 1000.0
        dt = 1.0 / fps
        f = 1.0  # 1 Hz
        A = 30.0  # 30 deg amplitude

        t = np.arange(0, 1.0, dt)
        angles = A * np.sin(2 * np.pi * f * t)
        vel_numeric = angular_velocity(angles, fps)
        vel_analytic = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t)

        # Compare interior points (skip boundaries where forward/backward diff is used)
        interior = slice(5, -5)
        assert np.allclose(vel_numeric[interior], vel_analytic[interior], atol=0.05)


class TestAngularAcceleration:
    """Test second derivative computation."""

    def test_acceleration_of_sine(self):
        """For theta(t) = A*sin(2pi*f*t), d²θ/dt² = -A*(2pi*f)²*sin(2pi*f*t)."""
        fps = 1000.0
        dt = 1.0 / fps
        f = 1.0
        A = 30.0

        t = np.arange(0, 1.0, dt)
        angles = A * np.sin(2 * np.pi * f * t)
        accel_numeric = angular_acceleration(angles, fps)
        accel_analytic = -A * (2 * np.pi * f) ** 2 * np.sin(2 * np.pi * f * t)

        interior = slice(10, -10)
        # Allow larger tolerance since second-order central difference has error O(dt²)
        assert np.allclose(accel_numeric[interior], accel_analytic[interior], atol=2.0)


class TestTrajectoryLength:
    """Test cumulative Euclidean distance."""

    def test_straight_line(self):
        """10 pixels per step, 5 steps -> 50 pixels."""
        pos = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]],
                       dtype=np.float64)
        assert abs(trajectory_length(pos) - 50.0) < 1e-9

    def test_diagonal(self):
        """Each step is (3,4), so step length = 5. 3 steps -> 15."""
        pos = np.array([[0, 0], [3, 4], [6, 8], [9, 12]], dtype=np.float64)
        assert abs(trajectory_length(pos) - 15.0) < 1e-9

    def test_single_point(self):
        """Single point -> 0 length."""
        pos = np.array([[5, 10]], dtype=np.float64)
        assert trajectory_length(pos) == 0.0

    def test_with_nans(self):
        """NaN entries should be skipped."""
        pos = np.array([[0, 0], [np.nan, np.nan], [10, 0], [20, 0]], dtype=np.float64)
        # Valid: (0,0)->(10,0)->(20,0) = 20
        assert abs(trajectory_length(pos) - 20.0) < 1e-9


class TestRangeOfMotion:
    """Test ROM computation."""

    def test_normal_range(self):
        angles = np.array([30.0, 45.0, 60.0, 40.0, 20.0])
        assert abs(range_of_motion(angles) - 40.0) < 1e-9  # 60 - 20

    def test_all_nans(self):
        angles = np.array([np.nan, np.nan, np.nan])
        assert range_of_motion(angles) == 0.0


class TestComputeAnglesFromLandmarks:
    """Test batch angle computation from landmark sequences."""

    def test_right_angle_knee(self):
        """Synthetic landmarks that form a right angle at the knee."""
        # 2 frames, 33 keypoints, 2 coordinates each
        landmarks = np.zeros((2, 33, 2), dtype=np.float64)
        # Frame 0: hip=(10,0), knee=(0,0), ankle=(0,10) -> 90 deg
        landmarks[0, 23] = [10, 0]   # left_hip
        landmarks[0, 25] = [0, 0]    # left_knee
        landmarks[0, 27] = [0, 10]   # left_ankle
        # Frame 1: same
        landmarks[1, 23] = [10, 0]
        landmarks[1, 25] = [0, 0]
        landmarks[1, 27] = [0, 10]

        defs = {"left_knee_angle": (23, 25, 27)}
        result = compute_angles_from_landmarks(landmarks, defs)
        assert "left_knee_angle" in result
        assert np.allclose(result["left_knee_angle"], [90.0, 90.0], atol=1e-6)

    def test_thigh_segment_angle(self):
        """Synthetic thigh segments: left horizontal, right vertical -> 90 deg."""
        landmarks = np.full((2, 33, 2), np.nan, dtype=np.float64)
        landmarks[:, 23] = [0, 0]
        landmarks[:, 25] = [1, 0]
        landmarks[:, 24] = [2, 0]
        landmarks[:, 26] = [2, 1]

        defs = {"thigh_segments_angle": (23, 25, 24, 26)}
        result = compute_segment_angles_from_landmarks(landmarks, defs)

        assert "thigh_segments_angle" in result
        assert np.allclose(result["thigh_segments_angle"], [90.0, 90.0], atol=1e-6)


class TestSmoothAngles:
    """Test moving-average smoothing."""

    def test_constant_signal_unchanged(self):
        angles = np.array([45.0, 45.0, 45.0, 45.0, 45.0])
        smoothed = smooth_angles(angles, window=3)
        assert np.allclose(smoothed, 45.0, atol=1e-6)

    def test_nan_interpolation(self):
        """NaNs should be interpolated before smoothing."""
        angles = np.array([10.0, np.nan, 30.0, np.nan, 50.0], dtype=np.float64)
        smoothed = smooth_angles(angles, window=3)
        # Should not contain NaNs
        assert not np.isnan(smoothed).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
