"""
records.py
----------
Functional Module 4: CRUD operations for detection records.

Every time an image/frame is analyzed, one row per detected face is
persisted here. This gives the system an auditable history and is the
data source for the analytics/reporting module.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import config
from src.detection import Detection
from src.logger_setup import get_logger

logger = get_logger(__name__)


@dataclass
class Record:
    record_id: int
    source: str
    label: str
    confidence: float
    box: str
    created_at: str
    created_by: str


def _get_connection() -> sqlite3.Connection:
    return sqlite3.connect(config.DATABASE_PATH)


def init_records_table() -> None:
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detection_records (
                record_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                source      TEXT NOT NULL,
                label       TEXT NOT NULL,
                confidence  REAL NOT NULL,
                box         TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                created_by  TEXT NOT NULL
            );
            """
        )
        conn.commit()
    logger.info("detection_records table verified/created.")


# ---------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------
def create_record(source: str, detection: Detection, created_by: str) -> int:
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO detection_records (source, label, confidence, box, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                detection.label,
                detection.confidence,
                str(detection.box),
                datetime.now().isoformat(timespec="seconds"),
                created_by,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def bulk_create_records(source: str, detections: List[Detection], created_by: str) -> List[int]:
    return [create_record(source, det, created_by) for det in detections]


# ---------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------
def get_record(record_id: int) -> Optional[Record]:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT record_id, source, label, confidence, box, created_at, created_by "
            "FROM detection_records WHERE record_id = ?",
            (record_id,),
        ).fetchone()
    return Record(*row) if row else None


def list_records(limit: int = 50, label_filter: Optional[str] = None) -> List[Record]:
    query = "SELECT record_id, source, label, confidence, box, created_at, created_by FROM detection_records"
    params: tuple = ()
    if label_filter:
        query += " WHERE label = ?"
        params = (label_filter,)
    query += " ORDER BY record_id DESC LIMIT ?"
    params = params + (limit,)

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [Record(*row) for row in rows]


# ---------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------
def update_record_label(record_id: int, new_label: str) -> bool:
    """Allow a human reviewer to correct a misclassified record."""
    if new_label not in config.CLASS_NAMES:
        raise ValueError(f"new_label must be one of {config.CLASS_NAMES}")
    with _get_connection() as conn:
        cursor = conn.execute(
            "UPDATE detection_records SET label = ? WHERE record_id = ?",
            (new_label, record_id),
        )
        conn.commit()
        updated = cursor.rowcount > 0
    if updated:
        logger.info("Record %d label corrected to '%s'.", record_id, new_label)
    return updated


# ---------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------
def delete_record(record_id: int) -> bool:
    with _get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM detection_records WHERE record_id = ?", (record_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Record %d deleted.", record_id)
    return deleted
