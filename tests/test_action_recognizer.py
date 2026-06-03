"""Unit tests for the action recognition module."""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.action_recognizer import (
    ActionRecognizer,
    ActionTemplate,
    TemplateStore,
    dtw_distance,
    extract_angle_features,
    _normalize_angles,
    _resample_sequence,
    _fill_nans,
    _non_max_suppression,
    ANGLE_KEYS,
)


# ---------------------------------------------------------------------------
# DTW distance
# ---------------------------------------------------------------------------
class TestDTWDistance:
    """Test multidimensional DTW distance computation."""

    def test_identical_sequences(self):
        """DTW distance of a sequence to itself should be ~0."""
        seq = np.array([[30.0, 45.0, 60.0, 90.0, 120.0, 45.0, 30.0],
                        [30.0, 45.0, 60.0, 90.0, 120.0, 45.0, 30.0]]).T  # (7, 2)
        dist = dtw_distance(seq, seq)
        assert dist < 5.0, f"Expected ~0, got {dist}"

    def test_different_sequences_larger_distance(self):
        """Different sequences should have larger DTW distance."""
        seq_a = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float64)
        seq_b = np.array([[10, 10], [11, 11], [12, 12], [13, 13], [14, 14]],
                         dtype=np.float64)
        dist_ab = dtw_distance(seq_a, seq_b)
        dist_aa = dtw_distance(seq_a, seq_a)
        assert dist_aa < dist_ab, f"Self: {dist_aa}, Cross: {dist_ab}"

    def test_warped_sequence(self):
        """DTW should handle different-length, time-warped sequences."""
        # A sine wave (30 frames) vs a stretched sine wave (45 frames)
        t1 = np.linspace(0, np.pi, 30)
        t2 = np.linspace(0, np.pi, 45)
        seq1 = np.sin(t1).reshape(-1, 1)
        seq2 = np.sin(t2).reshape(-1, 1)
        dist = dtw_distance(seq1, seq2)
        # Should be relatively small (same shape, different speeds)
        assert dist < 1.0, f"Expected < 1.0 for warped sine, got {dist}"

    def test_single_frame_input(self):
        """Degenerate case: single-frame sequences."""
        seq = np.ones((1, 10))
        dist = dtw_distance(seq, seq)
        assert not np.isnan(dist)

    def test_window_constraint(self):
        """DTW with Sakoe-Chiba window constraint should still work."""
        seq = np.random.RandomState(42).randn(50, 3)
        dist_no_window = dtw_distance(seq, seq)
        dist_windowed = dtw_distance(seq, seq, window=10)
        # Both should be ~0 for self-match
        assert dist_no_window < 1.0
        assert dist_windowed < 1.0


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
class TestNormalizeAngles:
    """Test angle sequence normalization for template storage."""

    def test_resample_to_target_len(self):
        """Output should have exactly target_len frames."""
        angles = np.random.RandomState(0).randn(100, 10)
        result = _normalize_angles(angles, target_len=60)
        assert result.shape == (60, 10)

    def test_demean(self):
        """Each dimension should have ~0 mean after normalization."""
        angles = np.ones((80, 10)) * np.arange(10)  # different per column
        result = _normalize_angles(angles, target_len=40)
        means = result.mean(axis=0)
        assert np.allclose(means, 0.0, atol=1e-9)

    def test_nan_interpolation(self):
        """NaN values should be interpolated, not propagate."""
        angles = np.full((50, 10), 45.0)
        angles[20:25, :] = np.nan  # gap in the middle
        result = _normalize_angles(angles, target_len=50)
        assert not np.isnan(result).any()
        # The interpolated region should be close to 45
        assert np.allclose(result[20:25], 0.0, atol=1e-6)  # de-meaned, so ~0

    def test_all_nan_column(self):
        """Column with all NaN should become zeros after fill."""
        angles = np.full((30, 10), np.nan)
        result = _normalize_angles(angles, target_len=30)
        assert not np.isnan(result).any()


class TestFillNans:
    """Test NaN filling helper."""

    def test_no_nans(self):
        arr = np.ones((10, 3))
        result = _fill_nans(arr)
        assert np.array_equal(result, arr)

    def test_internal_nans(self):
        arr = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
        result = _fill_nans(arr)
        assert not np.isnan(result).any()
        assert result[1, 0] == pytest.approx(2.0)
        assert result[1, 1] == pytest.approx(3.0)


class TestResampleSequence:
    """Test temporal resampling."""

    def test_same_length(self):
        seq = np.random.RandomState(1).randn(30, 5)
        result = _resample_sequence(seq, 30)
        assert np.allclose(result, seq)

    def test_double_length(self):
        seq = np.array([[0.0], [10.0], [20.0]])  # (3, 1)
        result = _resample_sequence(seq, 5)
        assert result.shape == (5, 1)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[-1, 0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# ActionTemplate
# ---------------------------------------------------------------------------
class TestActionTemplate:
    """Test ActionTemplate serialization."""

    def test_roundtrip(self):
        features = np.random.RandomState(7).randn(60, 10)
        tmpl = ActionTemplate(name="jumping", features=features, source_fps=30.0)
        d = tmpl.to_dict()
        restored = ActionTemplate.from_dict(d)
        assert restored.name == "jumping"
        assert restored.source_fps == 30.0
        assert np.allclose(restored.features, features)
        assert restored.template_id == tmpl.template_id


# ---------------------------------------------------------------------------
# TemplateStore
# ---------------------------------------------------------------------------
class TestTemplateStore:
    """Test template store CRUD and persistence."""

    def test_add_and_retrieve(self):
        store = TemplateStore(target_len=60)
        angles = np.random.RandomState(2).randn(80, 10)
        tmpl = store.add(angles, "walking", fps=30.0)
        assert tmpl.name == "walking"
        assert tmpl.features.shape == (60, 10)
        assert len(store) == 1
        assert tmpl.template_id in store

    def test_remove(self):
        store = TemplateStore()
        angles = np.random.RandomState(3).randn(100, 10)
        tmpl = store.add(angles, "running")
        assert len(store) == 1
        assert store.remove(tmpl.template_id)
        assert len(store) == 0
        assert not store.remove("nonexistent")

    def test_names(self):
        store = TemplateStore()
        store.add(np.random.randn(80, 10), "jump")
        store.add(np.random.randn(90, 10), "squat")
        assert set(store.names) == {"jump", "squat"}

    def test_json_persistence(self):
        store = TemplateStore(target_len=30)
        angles = np.random.RandomState(4).randn(70, 10)
        store.add(angles, "shooting", fps=25.0)
        store.add(np.random.RandomState(5).randn(60, 10), "dribbling", fps=30.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            store.save(f.name)
            saved_path = f.name

        try:
            loaded = TemplateStore()
            loaded.load(saved_path)
            assert len(loaded) == 2
            assert set(loaded.names) == {"shooting", "dribbling"}
            tmpl = loaded.templates[0]
            assert tmpl.features.shape[0] == 30  # target_len
        finally:
            os.unlink(saved_path)

    def test_clear(self):
        store = TemplateStore()
        store.add(np.random.randn(80, 10), "test")
        store.clear()
        assert len(store) == 0

    def test_compute_threshold(self):
        store = TemplateStore(target_len=30)
        angles = np.zeros((50, 10))
        store.add(angles, "constant")
        thresh = store.compute_threshold(store.templates[0].template_id)
        assert thresh >= 0.5  # minimum floor
        assert not np.isnan(thresh)


# ---------------------------------------------------------------------------
# ActionRecognizer (integration)
# ---------------------------------------------------------------------------
class TestActionRecognizer:
    """Test the full recognition pipeline."""

    def test_no_templates_returns_empty(self):
        store = TemplateStore()
        rec = ActionRecognizer(store)
        angles = np.random.RandomState(6).randn(100, 10)
        matches = rec.recognize(angles)
        assert matches == []

    def test_self_recognition(self):
        """A video that *is* the template action should be detected."""
        store = TemplateStore(target_len=60)
        # Create a synthetic "knee-bending" pattern
        t = np.linspace(0, 2 * np.pi, 80)
        angles = np.zeros((80, 10))
        angles[:, 0] = 90 + 30 * np.sin(t)   # left knee oscillates
        angles[:, 1] = 90 + 30 * np.sin(t)   # right knee
        angles[:, 2] = 45 + 20 * np.sin(t)   # left elbow

        store.add(angles, "squat", fps=30.0)
        rec = ActionRecognizer(store, sensitivity=3.0)
        matches = rec.recognize(angles, fps=30.0)

        # Should detect at least one squat
        assert len(matches) > 0, "Should detect the squat in its own template"
        assert matches[0].action_name == "squat"

    def test_no_false_positive_on_different_action(self):
        """A very different angle sequence should NOT trigger detection."""
        store = TemplateStore(target_len=60)
        # Template: knee oscillation
        t = np.linspace(0, 2 * np.pi, 60)
        template = np.zeros((60, 10))
        template[:, 0] = 90 + 40 * np.sin(t)

        store.add(template, "knee_swing", fps=30.0)

        # Query: completely constant angles (no motion)
        query = np.ones((100, 10)) * 45.0

        rec = ActionRecognizer(store, sensitivity=2.0)
        matches = rec.recognize(query, fps=30.0)
        assert len(matches) == 0, "Should not detect action in constant signal"

    def test_multiple_templates(self):
        """Recognition works with multiple templates."""
        store = TemplateStore(target_len=40)
        t1 = np.linspace(0, np.pi, 60)
        angles1 = np.zeros((60, 10))
        angles1[:, 0] = 90 + 50 * np.sin(t1)  # deep knee bend

        t2 = np.linspace(0, 2 * np.pi, 60)
        angles2 = np.zeros((60, 10))
        angles2[:, 2] = 45 + 60 * np.sin(t2)  # elbow swing

        store.add(angles1, "deep_squat")
        store.add(angles2, "arm_wave")

        # Query matches angles1 exactly
        rec = ActionRecognizer(store, sensitivity=2.5)
        matches = rec.recognize(angles1, fps=30.0)
        detected_names = {m.action_name for m in matches}
        assert "deep_squat" in detected_names


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------
class TestNMS:
    """Test non-maximum suppression."""

    def test_suppresses_overlapping_same_action(self):
        tmpl = ActionTemplate(name="walk", features=np.zeros((10, 10)), source_fps=30.0)
        matches = [
            {"template": tmpl, "start": 10, "end": 50, "distance": 1.0, "threshold": 3.0},
            {"template": tmpl, "start": 15, "end": 55, "distance": 2.0, "threshold": 3.0},
        ]
        kept = _non_max_suppression(matches, iou_threshold=0.3)
        assert len(kept) == 1
        assert kept[0]["distance"] == 1.0  # lowest distance kept

    def test_keeps_different_actions(self):
        tmpl_a = ActionTemplate(name="run", features=np.zeros((10, 10)), source_fps=30.0)
        tmpl_b = ActionTemplate(name="jump", features=np.zeros((10, 10)), source_fps=30.0)
        matches = [
            {"template": tmpl_a, "start": 10, "end": 50, "distance": 1.0, "threshold": 3.0},
            {"template": tmpl_b, "start": 10, "end": 50, "distance": 1.5, "threshold": 3.0},
        ]
        kept = _non_max_suppression(matches, iou_threshold=0.3)
        assert len(kept) == 2  # different actions, both kept

    def test_empty_input(self):
        assert _non_max_suppression([]) == []


# ---------------------------------------------------------------------------
# extract_angle_features (integration with kinematics)
# ---------------------------------------------------------------------------
class TestExtractAngleFeatures:
    """Test the convenience function that bridges kinematics → action features."""

    def test_output_shape(self):
        """Should produce (T, 10) matrix."""
        # Synthetic landmarks: 30 frames, 33 keypoints, 2 coords
        landmarks = np.zeros((30, 33, 2), dtype=np.float64)
        # Set up a simple configuration for each angle
        for col, key in enumerate(ANGLE_KEYS):
            # Each angle needs 3 points; put them at known positions
            landmarks[:, 23] = [1, 0]  # left hip
            landmarks[:, 25] = [0, 0]  # left knee
            landmarks[:, 27] = [0, 1]  # left ankle

        angle_defs = {
            "left_knee_angle": (23, 25, 27),
            "right_knee_angle": (24, 26, 28),
            "left_elbow_angle": (11, 13, 15),
            "right_elbow_angle": (12, 14, 16),
            "left_hip_angle": (11, 23, 25),
            "right_hip_angle": (12, 24, 26),
            "left_shoulder_angle": (13, 11, 23),
            "right_shoulder_angle": (14, 12, 24),
            "left_ankle_angle": (25, 27, 31),
            "right_ankle_angle": (26, 28, 32),
        }
        features = extract_angle_features(landmarks, angle_defs)
        assert features.shape == (30, 10)

    def test_missing_angle_key(self):
        """Missing keys produce NaN columns that get smoothed."""
        landmarks = np.zeros((10, 33, 2))
        landmarks[:, 23] = [1, 0]
        landmarks[:, 25] = [0, 0]
        landmarks[:, 27] = [0, 1]
        angle_defs = {"left_knee_angle": (23, 25, 27)}
        features = extract_angle_features(landmarks, angle_defs)
        assert features.shape == (10, 10)
        # left_knee_angle column (index 0) should be valid
        assert not np.isnan(features[:, 0]).all()
        # Missing keys should still be NaN after smoothing (all NaN → all 0 after interp)
        # Actually, smoothing fills NaN via interp, so missing keys become 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
