"""
preprocessing.py
-----------------
Functional Module 2: Data Input & Processing.

Handles loading images/video frames and turning them into the
normalized tensors the CNN expects. Also provides simple
augmentation utilities used by the training pipeline.

Input  : file path (image) OR a raw BGR numpy frame (from video/webcam)
Output : normalized float32 numpy array of shape (IMG_SIZE, IMG_SIZE, 3)
"""

import os
from typing import List, Tuple

import cv2
import numpy as np

import config
from src.logger_setup import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class PreprocessingError(Exception):
    """Raised when an image cannot be loaded or processed."""


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    if not os.path.exists(path):
        raise PreprocessingError(f"File not found: {path}")
    image = cv2.imread(path)
    if image is None:
        raise PreprocessingError(f"OpenCV could not decode image: {path}")
    return image


def resize_and_normalize(image: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    """Resize to a fixed square shape and scale pixel values to [0, 1]."""
    if image is None or image.size == 0:
        raise PreprocessingError("Cannot process an empty image.")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized


def crop_region(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop a (x, y, w, h) region from an image, clipped to image bounds."""
    x, y, w, h = box
    height, width = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    if x2 <= x1 or y2 <= y1:
        raise PreprocessingError(f"Invalid crop region {box} for image of size {image.shape}")
    return image[y1:y2, x1:x2]


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """
    Produce a small set of augmented variants (horizontal flip, brightness
    jitter) used only during model training to improve generalization.
    """
    variants = [image]

    flipped = cv2.flip(image, 1)
    variants.append(flipped)

    bright = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
    variants.append(bright)

    dark = cv2.convertScaleAbs(image, alpha=1.0, beta=-30)
    variants.append(dark)

    return variants


def list_images_in_dir(directory: str) -> List[str]:
    """Return sorted list of supported image file paths within a directory."""
    if not os.path.isdir(directory):
        raise PreprocessingError(f"Not a directory: {directory}")
    files = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    logger.info("Found %d supported images in %s", len(files), directory)
    return files
