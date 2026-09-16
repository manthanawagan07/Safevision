"""
evaluation.py
--------------
Model validation utilities.

Computes accuracy, per-class precision / recall / F1 and a confusion
matrix on a held-out test set. These are the numbers reported in the
project report, and the `evaluate` CLI command exposes them to the
grader without needing a notebook.

Implemented with plain NumPy so that no extra dependency beyond what
the project already uses is required.
"""

from typing import Dict, List

import numpy as np

import config
from src.logger_setup import get_logger

logger = get_logger(__name__)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Return an (n_classes x n_classes) matrix; rows = true, cols = predicted."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def per_class_metrics(matrix: np.ndarray) -> List[Dict]:
    """Precision, recall and F1 derived from a confusion matrix."""
    results = []
    for idx in range(matrix.shape[0]):
        true_positive = matrix[idx, idx]
        false_positive = matrix[:, idx].sum() - true_positive
        false_negative = matrix[idx, :].sum() - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        results.append({
            "class": config.CLASS_NAMES[idx],
            "support": int(matrix[idx, :].sum()),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        })
    return results


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Run the model over a test set and return a full metrics dictionary."""
    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = float((y_pred == y_test).mean())
    matrix = confusion_matrix(y_test, y_pred, n_classes=len(config.CLASS_NAMES))
    metrics = per_class_metrics(matrix)

    macro_f1 = float(np.mean([m["f1"] for m in metrics]))

    result = {
        "n_test_samples": int(len(y_test)),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": matrix.tolist(),
        "per_class": metrics,
    }
    logger.info("Evaluation complete: accuracy=%.4f macro_f1=%.4f", accuracy, macro_f1)
    return result


def format_report(result: Dict) -> str:
    """Render the metrics dictionary as a readable console table."""
    lines = []
    lines.append(f"Test samples : {result['n_test_samples']}")
    lines.append(f"Accuracy     : {result['accuracy']:.4f}")
    lines.append(f"Macro F1     : {result['macro_f1']:.4f}")
    lines.append("")
    lines.append(f"{'CLASS':<16}{'PREC':>8}{'RECALL':>9}{'F1':>8}{'SUPPORT':>10}")
    lines.append("-" * 51)
    for m in result["per_class"]:
        lines.append(
            f"{m['class']:<16}{m['precision']:>8.4f}{m['recall']:>9.4f}"
            f"{m['f1']:>8.4f}{m['support']:>10}"
        )

    lines.append("")
    lines.append("Confusion matrix (rows = actual, cols = predicted):")
    header = " " * 16 + "".join(f"{name[:12]:>14}" for name in config.CLASS_NAMES)
    lines.append(header)
    for idx, row in enumerate(result["confusion_matrix"]):
        lines.append(f"{config.CLASS_NAMES[idx][:15]:<16}" + "".join(f"{v:>14}" for v in row))

    return "\n".join(lines)
