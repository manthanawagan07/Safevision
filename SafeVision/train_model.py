#!/usr/bin/env python3
"""
train_model.py
---------------
Trains the CNN mask classifier and saves it to models/.

Two data sources are supported:

  1. Real dataset (recommended for accuracy):
         data/train/with_mask/*.jpg
         data/train/without_mask/*.jpg
     Use --data-dir data/train

  2. Synthetic dataset (default, zero-download):
     Generated on the fly by src/utils.py so the full training
     pipeline can be demonstrated on any machine without an
     internet connection.

Usage:
    python train_model.py                       # synthetic, 10 epochs
    python train_model.py --epochs 20
    python train_model.py --data-dir data/train # real images
"""

import argparse
import os
import sys
import time

import numpy as np

import config
from src.logger_setup import get_logger
from src.preprocessing import list_images_in_dir, load_image, resize_and_normalize
from src.utils import generate_synthetic_dataset

logger = get_logger("train")


def load_real_dataset(data_dir: str):
    """Load images from data_dir/<class_name>/ into (X, y) arrays."""
    images, labels = [], []

    for label_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Expected class folder missing: {class_dir}\n"
                f"Structure must be: {data_dir}/{{{', '.join(config.CLASS_NAMES)}}}/*.jpg"
            )
        paths = list_images_in_dir(class_dir)
        if not paths:
            raise ValueError(f"No images found in {class_dir}")

        for path in paths:
            try:
                img = load_image(path)
                images.append(resize_and_normalize(img))
                labels.append(label_idx)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping unreadable training image %s: %s", path, exc)

    X = np.array(images, dtype="float32")
    y = np.array(labels, dtype="int64")

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the SafeVision mask classifier.")
    parser.add_argument("--data-dir", help="Directory with class subfolders. Omit to use synthetic data.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=200,
                        help="Synthetic samples per class (ignored with --data-dir).")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction held out for final evaluation (default 0.2).")
    args = parser.parse_args()

    try:
        if args.data_dir:
            print(f"Loading real dataset from {args.data_dir} ...")
            X, y = load_real_dataset(args.data_dir)
        else:
            print("No --data-dir given: generating synthetic dataset.")
            print("(For real-world accuracy see the Dataset section of README.md)")
            X, y = generate_synthetic_dataset(n_per_class=args.samples_per_class)

        print(f"Dataset ready: X={X.shape}, y={y.shape}")
        counts = {name: int((y == i).sum()) for i, name in enumerate(config.CLASS_NAMES)}
        print(f"Class distribution: {counts}")

        # Hold out a test split that the model never sees during training,
        # so the reported metrics are honest rather than memorized.
        split = int(len(X) * (1.0 - args.test_split))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        print(f"Train/test split: {len(X_train)} train, {len(X_test)} held-out test")

        from src.model import train
        from src.evaluation import evaluate_model, format_report

        start = time.perf_counter()
        history = train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)
        elapsed = time.perf_counter() - start

        final_acc = history.history["accuracy"][-1]
        final_val = history.history.get("val_accuracy", [float("nan")])[-1]

        print("\n" + "=" * 55)
        print(f"Training finished in {elapsed:.1f}s")
        print(f"Final training accuracy  : {final_acc:.4f}")
        print(f"Final validation accuracy: {final_val:.4f}")
        print(f"Model saved to           : {config.MODEL_PATH}")

        print("\n" + "-" * 55)
        print("HELD-OUT TEST SET EVALUATION")
        print("-" * 55)
        from src.model import load_trained_model
        model = load_trained_model()
        result = evaluate_model(model, X_test, y_test)
        print(format_report(result))
        print("=" * 55)
        return 0

    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed.")
        print(f"[ERROR] Training failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
