"""
model.py
--------
Defines the CNN architecture used for mask/no-mask classification
and provides train / save / load utilities.

This is intentionally a compact CNN (not a huge pretrained backbone)
so it can be trained quickly on a CPU-only grading machine, which
supports the Performance and Resource Efficiency non-functional
requirements.
"""

from typing import Tuple

import numpy as np

import config
from src.logger_setup import get_logger

logger = get_logger(__name__)


def build_model(input_shape: Tuple[int, int, int] = (config.IMG_SIZE, config.IMG_SIZE, 3)):
    """Build and compile the CNN classifier. Imports TensorFlow lazily so
    that modules which don't need it (e.g. CRUD/reporting) stay lightweight
    and importable even in minimal environments."""
    from tensorflow.keras import layers, models, optimizers

    # BatchNormalization momentum is lowered from the Keras default of 0.99.
    # With a small dataset the number of gradient steps is low, so the default
    # momentum leaves the moving mean/variance far from the true statistics.
    # That makes training accuracy look perfect while validation accuracy
    # collapses (inference uses the moving stats, training uses batch stats).
    bn_momentum = 0.90

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(momentum=bn_momentum),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(momentum=bn_momentum),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(momentum=bn_momentum),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(config.CLASS_NAMES), activation="softmax"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 16,
          validation_split: float = 0.2):
    """Train the classifier and persist it to config.MODEL_PATH."""
    model = build_model()
    logger.info("Starting training: %d samples, %d epochs.", len(X), epochs)

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    model.save(config.MODEL_PATH)
    logger.info("Model trained and saved to %s", config.MODEL_PATH)
    return history


def load_trained_model():
    """Load a previously trained model from disk."""
    import os
    from tensorflow.keras.models import load_model

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_PATH}. "
            f"Run `python train_model.py` first."
        )
    return load_model(config.MODEL_PATH)
