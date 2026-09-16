"""
Unit tests for src/detection.py

Classification requires a trained model, so these tests build a small
CNN once and save it to a temporary path. The network is untrained, so
the tests assert the *output contract* (valid label, confidence in
range, no mutation of inputs) rather than predictive accuracy —
accuracy is measured separately by src/evaluation.py against a held-out
set. A separate test covers the no-model case.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import detection as detection_module
from src.detection import (
    Detection,
    DetectionError,
    ModelUnavailableError,
    analyze_image,
    classify_face,
    detect_faces,
    draw_annotations,
)
from src.utils import _draw_synthetic_face


def _reset_model_cache():
    """Clear the module-level model cache so a new path is picked up."""
    detection_module._cnn_model = None
    detection_module._model_load_attempted = False


class TestDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._original_model_path = config.MODEL_PATH
        cls._tmp_dir = tempfile.mkdtemp()
        config.MODEL_PATH = os.path.join(cls._tmp_dir, "test_model.keras")

        from src.model import build_model
        build_model().save(config.MODEL_PATH)
        _reset_model_cache()

    @classmethod
    def tearDownClass(cls):
        config.MODEL_PATH = cls._original_model_path
        _reset_model_cache()
        import shutil
        shutil.rmtree(cls._tmp_dir, ignore_errors=True)

    def setUp(self):
        self.blank = np.full((200, 200, 3), 128, dtype=np.uint8)
        self.masked_face = _draw_synthetic_face(masked=True, size=200)
        self.bare_face = _draw_synthetic_face(masked=False, size=200)

    def test_detect_faces_returns_list(self):
        boxes = detect_faces(self.blank)
        self.assertIsInstance(boxes, list)

    def test_detect_faces_box_format(self):
        boxes = detect_faces(self.bare_face)
        for box in boxes:
            self.assertEqual(len(box), 4)
            self.assertTrue(all(isinstance(v, int) for v in box))

    def test_detect_faces_none_raises(self):
        with self.assertRaises(DetectionError):
            detect_faces(None)

    def test_classify_returns_valid_label(self):
        label, conf = classify_face(self.masked_face)
        self.assertIn(label, config.CLASS_NAMES)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_classify_raises_when_no_model_available(self):
        """
        With no trained model the system must refuse to guess. This is the
        regression test for a real defect: an earlier edge-density fallback
        labelled every sample image 'with_mask', which in a compliance
        system means violations silently going unrecorded.
        """
        original = config.MODEL_PATH
        try:
            config.MODEL_PATH = os.path.join(self._tmp_dir, "does_not_exist.keras")
            _reset_model_cache()
            with self.assertRaises(ModelUnavailableError):
                classify_face(self.masked_face)
        finally:
            config.MODEL_PATH = original
            _reset_model_cache()

    def test_analyze_image_propagates_missing_model(self):
        """A missing model must reach the caller, not be swallowed per face."""
        original = config.MODEL_PATH
        try:
            config.MODEL_PATH = os.path.join(self._tmp_dir, "does_not_exist.keras")
            _reset_model_cache()
            with self.assertRaises(ModelUnavailableError):
                analyze_image(self.bare_face)
        finally:
            config.MODEL_PATH = original
            _reset_model_cache()

    def test_analyze_image_returns_detections(self):
        results = analyze_image(self.bare_face)
        self.assertIsInstance(results, list)
        for det in results:
            self.assertIsInstance(det, Detection)
            self.assertIn(det.label, config.CLASS_NAMES)

    def test_analyze_blank_image_no_crash(self):
        results = analyze_image(self.blank)
        self.assertIsInstance(results, list)

    def test_draw_annotations_preserves_shape(self):
        dets = [Detection(box=(10, 10, 50, 50), label="with_mask", confidence=0.9)]
        annotated = draw_annotations(self.bare_face, dets)
        self.assertEqual(annotated.shape, self.bare_face.shape)

    def test_draw_annotations_does_not_mutate_original(self):
        original = self.bare_face.copy()
        dets = [Detection(box=(10, 10, 50, 50), label="without_mask", confidence=0.8)]
        draw_annotations(self.bare_face, dets)
        np.testing.assert_array_equal(self.bare_face, original)


if __name__ == "__main__":
    unittest.main()
