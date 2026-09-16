"""
analytics.py
------------
Functional Module 5: Reporting & Analytics.

Aggregates the detection_records table into compliance statistics
and writes a timestamped CSV + text summary report to /reports.
"""

import csv
import os
import sqlite3
from datetime import datetime
from typing import Dict

import config
from src.logger_setup import get_logger

logger = get_logger(__name__)


def _get_connection() -> sqlite3.Connection:
    return sqlite3.connect(config.DATABASE_PATH)


def compute_summary() -> Dict:
    with _get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM detection_records").fetchone()[0]
        with_mask = conn.execute(
            "SELECT COUNT(*) FROM detection_records WHERE label = 'with_mask'"
        ).fetchone()[0]
        without_mask = conn.execute(
            "SELECT COUNT(*) FROM detection_records WHERE label = 'without_mask'"
        ).fetchone()[0]
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM detection_records"
        ).fetchone()[0] or 0.0
        by_source = conn.execute(
            """
            SELECT source, label, COUNT(*)
            FROM detection_records
            GROUP BY source, label
            ORDER BY source
            """
        ).fetchall()

    compliance_rate = (with_mask / total * 100) if total else 0.0

    return {
        "total_detections": total,
        "with_mask": with_mask,
        "without_mask": without_mask,
        "compliance_rate_pct": round(compliance_rate, 2),
        "average_confidence": round(avg_conf, 3),
        "by_source": by_source,
    }


def generate_report() -> str:
    """Write a CSV + human-readable summary to /reports and return the path."""
    summary = compute_summary()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = os.path.join(config.REPORTS_DIR, f"report_{timestamp}.csv")
    txt_path = os.path.join(config.REPORTS_DIR, f"report_{timestamp}.txt")

    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT record_id, source, label, confidence, created_at, created_by "
            "FROM detection_records ORDER BY record_id"
        ).fetchall()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "source", "label", "confidence", "created_at", "created_by"])
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SafeVision Compliance Report\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("=" * 45 + "\n")
        f.write(f"Total detections     : {summary['total_detections']}\n")
        f.write(f"With mask            : {summary['with_mask']}\n")
        f.write(f"Without mask         : {summary['without_mask']}\n")
        f.write(f"Compliance rate      : {summary['compliance_rate_pct']}%\n")
        f.write(f"Average confidence   : {summary['average_confidence']}\n")
        f.write("-" * 45 + "\n")
        f.write("By source:\n")
        for source, label, count in summary["by_source"]:
            f.write(f"  {source} | {label}: {count}\n")

    logger.info("Generated report: %s / %s", csv_path, txt_path)
    return txt_path
