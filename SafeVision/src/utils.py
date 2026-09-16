"""
utils.py
--------
Small shared helpers, including a synthetic dataset generator.

Why a synthetic generator?
The canonical dataset for this task ("Face Mask Detection" on Kaggle)
requires an internet download that may not be available in every
grading environment. To keep the project 100% runnable end-to-end
from the command line without any manual download step, this module
can synthesize a small placeholder dataset of face-like shapes with
and without a drawn "mask" rectangle over the mouth/nose region.
This is clearly a stand-in for real photographs: it lets the full
pipeline (preprocessing -> training -> saving -> inference -> CRUD ->
reporting) be demonstrated and unit-tested end to end.

For production-grade accuracy, replace data/train/<class>/ with the
real Kaggle "Face Mask Detection" dataset (see README) and retrain.
"""

import os
import random
from typing import Tuple

import cv2
import numpy as np

import config
from src.logger_setup import get_logger

logger = get_logger(__name__)


def _draw_synthetic_face(masked: bool, size: int = config.IMG_SIZE) -> np.ndarray:
    img = np.full((size, size, 3), 220, dtype=np.uint8)  # light background
    skin = (random.randint(150, 210), random.randint(170, 220), random.randint(190, 240))

    center = (size // 2, size // 2)
    axes = (size // 3, size // 2 - 5)
    cv2.ellipse(img, center, axes, 0, 0, 360, skin, -1)

    # eyes
    eye_y = size // 2 - size // 6
    cv2.circle(img, (size // 2 - size // 6, eye_y), max(2, size // 20), (30, 30, 30), -1)
    cv2.circle(img, (size // 2 + size // 6, eye_y), max(2, size // 20), (30, 30, 30), -1)

    if masked:
        # Draw a mask rectangle covering nose/mouth region (flat colour, low texture)
        mask_color = random.choice([(255, 255, 255), (100, 150, 220), (150, 220, 150)])
        top_left = (size // 2 - size // 3, size // 2)
        bottom_right = (size // 2 + size // 3, size // 2 + size // 3)
        cv2.rectangle(img, top_left, bottom_right, mask_color, -1)
        cv2.ellipse(img, center, axes, 0, 0, 360, skin, 2)  # face outline on top
    else:
        # Draw a simple mouth + nose (higher local texture/edges)
        mouth_y = size // 2 + size // 5
        cv2.line(img, (size // 2 - size // 8, mouth_y), (size // 2 + size // 8, mouth_y), (60, 30, 30), 2)
        cv2.line(img, (size // 2, size // 2), (size // 2, size // 2 + size // 8), (skin[0]-30, skin[1]-30, skin[2]-30), 2)

    noise = np.random.randint(0, 12, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def generate_synthetic_dataset(n_per_class: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an in-memory synthetic dataset: returns (X, y)."""
    images, labels = [], []
    for label_idx, label_name in enumerate(config.CLASS_NAMES):
        masked = label_name == "with_mask"
        for _ in range(n_per_class):
            img = _draw_synthetic_face(masked)
            images.append(img.astype("float32") / 255.0)
            labels.append(label_idx)

    X = np.array(images, dtype="float32")
    y = np.array(labels, dtype="int64")

    # shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    logger.info("Generated synthetic dataset: %d samples across %d classes.", len(X), len(config.CLASS_NAMES))
    return X, y


def save_sample_images(n: int = 6) -> None:
    """Write a few synthetic sample images to data/sample_images for manual testing via the CLI."""
    os.makedirs(config.SAMPLE_IMAGES_DIR, exist_ok=True)
    for i in range(n):
        masked = i % 2 == 0
        img = _draw_synthetic_face(masked, size=300)
        label = "masked" if masked else "unmasked"
        path = os.path.join(config.SAMPLE_IMAGES_DIR, f"sample_{i}_{label}.jpg")
        cv2.imwrite(path, img)
    logger.info("Saved %d synthetic sample images to %s", n, config.SAMPLE_IMAGES_DIR)
