"""Unit tests for src/preprocessing.py"""

import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.preprocessing import (
    PreprocessingError,
    augment_image,
    crop_region,
    list_images_in_dir,
    load_image,
    resize_and_normalize,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    def test_resize_and_normalize_shape(self):
        out = resize_and_normalize(self.image)
        self.assertEqual(out.shape, (config.IMG_SIZE, config.IMG_SIZE, 3))

    def test_resize_and_normalize_range(self):
        out = resize_and_normalize(self.image)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)
        self.assertEqual(out.dtype, np.float32)

    def test_resize_custom_size(self):
        out = resize_and_normalize(self.image, size=64)
        self.assertEqual(out.shape, (64, 64, 3))

    def test_resize_rejects_empty(self):
        with self.assertRaises(PreprocessingError):
            resize_and_normalize(np.array([]))

    def test_crop_region_valid(self):
        crop = crop_region(self.image, (10, 20, 50, 60))
        self.assertEqual(crop.shape, (60, 50, 3))

    def test_crop_region_clips_to_bounds(self):
        # Box extends beyond the image; it should be clipped, not crash.
        crop = crop_region(self.image, (300, 230, 100, 100))
        self.assertEqual(crop.shape[0], 10)
        self.assertEqual(crop.shape[1], 20)

    def test_crop_region_invalid_raises(self):
        with self.assertRaises(PreprocessingError):
            crop_region(self.image, (500, 500, 10, 10))

    def test_augment_produces_variants(self):
        variants = augment_image(self.image)
        self.assertEqual(len(variants), 4)
        for v in variants:
            self.assertEqual(v.shape, self.image.shape)

    def test_load_image_missing_file(self):
        with self.assertRaises(PreprocessingError):
            load_image("/tmp/definitely_not_a_real_file_12345.jpg")

    def test_load_image_roundtrip(self):
        path = os.path.join(config.SAMPLE_IMAGES_DIR, "_unittest_tmp.png")
        cv2.imwrite(path, self.image)
        try:
            loaded = load_image(path)
            self.assertEqual(loaded.shape, self.image.shape)
        finally:
            os.remove(path)

    def test_list_images_in_dir_rejects_nondir(self):
        with self.assertRaises(PreprocessingError):
            list_images_in_dir("/tmp/not_a_directory_98765")


if __name__ == "__main__":
    unittest.main()
