"""
config.py
----------
Central configuration for the SafeVision system.
Keeping all tunables/paths in one place supports the
Maintainability and Scalability non-functional requirements.
"""

import os

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_IMAGES_DIR = os.path.join(DATA_DIR, "sample_images")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "safevision.db")

for _d in (DATA_DIR, SAMPLE_IMAGES_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, DATABASE_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Image / model parameters
# ---------------------------------------------------------------------------
IMG_SIZE = 100                # width/height the CNN expects (square)
IMG_CHANNELS = 3
CLASS_NAMES = ["with_mask", "without_mask"]
MODEL_PATH = os.path.join(MODELS_DIR, "mask_classifier.keras")

# OpenCV ships Haar cascades with the package itself -> no external
# download required, keeping the project runnable offline.
import cv2  # noqa: E402
HAAR_FACE_CASCADE = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
FACE_DETECT_SCALE_FACTOR = 1.1
FACE_DETECT_MIN_NEIGHBORS = 5
FACE_DETECT_MIN_SIZE = (40, 40)
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.60

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
PASSWORD_HASH_ALGO = "pbkdf2_sha256"
SESSION_TIMEOUT_MINUTES = 30

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(LOGS_DIR, "safevision.log")
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 3
