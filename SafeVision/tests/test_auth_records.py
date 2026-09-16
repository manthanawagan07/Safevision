"""
Unit tests for src/auth.py, src/records.py and src/analytics.py.

Each test class redirects config.DATABASE_PATH to a temporary file so
the real application database is never touched.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import analytics, auth, records
from src.detection import Detection


class _TempDBTestCase(unittest.TestCase):
    """Base class that swaps in a throwaway SQLite file."""

    def setUp(self):
        self._original_db = config.DATABASE_PATH
        fd, self.tmp_db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        config.DATABASE_PATH = self.tmp_db
        auth.init_auth_table()
        records.init_records_table()

    def tearDown(self):
        config.DATABASE_PATH = self._original_db
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)


class TestAuth(_TempDBTestCase):

    def test_register_and_login(self):
        auth.register_user("alice", "secret123", role="operator")
        user = auth.login("alice", "secret123")
        self.assertEqual(user.username, "alice")
        self.assertEqual(user.role, "operator")

    def test_password_is_not_stored_plaintext(self):
        import sqlite3
        auth.register_user("bob", "mypassword")
        with sqlite3.connect(config.DATABASE_PATH) as conn:
            row = conn.execute("SELECT pwd_hash FROM users WHERE username='bob'").fetchone()
        self.assertNotIn("mypassword", row[0])

    def test_same_password_different_hashes(self):
        """Per-user salts must make identical passwords hash differently."""
        import sqlite3
        auth.register_user("u1", "samepass")
        auth.register_user("u2", "samepass")
        with sqlite3.connect(config.DATABASE_PATH) as conn:
            hashes = conn.execute(
                "SELECT pwd_hash FROM users WHERE username IN ('u1','u2')"
            ).fetchall()
        self.assertNotEqual(hashes[0][0], hashes[1][0])

    def test_wrong_password_rejected(self):
        auth.register_user("carol", "correct1")
        with self.assertRaises(auth.AuthError):
            auth.login("carol", "wrongpass")

    def test_unknown_user_rejected(self):
        with self.assertRaises(auth.AuthError):
            auth.login("ghost", "whatever")

    def test_duplicate_username_rejected(self):
        auth.register_user("dave", "pass1234")
        with self.assertRaises(auth.AuthError):
            auth.register_user("dave", "pass5678")

    def test_short_password_rejected(self):
        with self.assertRaises(auth.AuthError):
            auth.register_user("eve", "123")

    def test_invalid_role_rejected(self):
        with self.assertRaises(auth.AuthError):
            auth.register_user("frank", "pass1234", role="superuser")

    def test_require_admin_blocks_operator(self):
        auth.register_user("grace", "pass1234", role="operator")
        user = auth.login("grace", "pass1234")
        with self.assertRaises(auth.AuthError):
            auth.require_admin(user)

    def test_require_admin_allows_admin(self):
        auth.register_user("root", "pass1234", role="admin")
        user = auth.login("root", "pass1234")
        auth.require_admin(user)  # should not raise


class TestRecordsCRUD(_TempDBTestCase):

    def _sample_detection(self, label="with_mask", conf=0.91):
        return Detection(box=(10, 20, 30, 40), label=label, confidence=conf)

    def test_create_and_read(self):
        rid = records.create_record("img1.jpg", self._sample_detection(), "tester")
        record = records.get_record(rid)
        self.assertIsNotNone(record)
        self.assertEqual(record.source, "img1.jpg")
        self.assertEqual(record.label, "with_mask")

    def test_get_missing_record_returns_none(self):
        self.assertIsNone(records.get_record(99999))

    def test_bulk_create(self):
        dets = [self._sample_detection(), self._sample_detection("without_mask", 0.7)]
        ids = records.bulk_create_records("img2.jpg", dets, "tester")
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(records.list_records()), 2)

    def test_list_with_label_filter(self):
        records.create_record("a.jpg", self._sample_detection("with_mask"), "t")
        records.create_record("b.jpg", self._sample_detection("without_mask"), "t")
        only_violations = records.list_records(label_filter="without_mask")
        self.assertEqual(len(only_violations), 1)
        self.assertEqual(only_violations[0].label, "without_mask")

    def test_update_label(self):
        rid = records.create_record("c.jpg", self._sample_detection("with_mask"), "t")
        self.assertTrue(records.update_record_label(rid, "without_mask"))
        self.assertEqual(records.get_record(rid).label, "without_mask")

    def test_update_invalid_label_raises(self):
        rid = records.create_record("d.jpg", self._sample_detection(), "t")
        with self.assertRaises(ValueError):
            records.update_record_label(rid, "banana")

    def test_update_missing_record_returns_false(self):
        self.assertFalse(records.update_record_label(99999, "with_mask"))

    def test_delete_record(self):
        rid = records.create_record("e.jpg", self._sample_detection(), "t")
        self.assertTrue(records.delete_record(rid))
        self.assertIsNone(records.get_record(rid))

    def test_delete_missing_returns_false(self):
        self.assertFalse(records.delete_record(99999))


class TestAnalytics(_TempDBTestCase):

    def test_empty_summary(self):
        summary = analytics.compute_summary()
        self.assertEqual(summary["total_detections"], 0)
        self.assertEqual(summary["compliance_rate_pct"], 0.0)

    def test_compliance_rate_calculation(self):
        for _ in range(3):
            records.create_record("x.jpg", Detection((0, 0, 1, 1), "with_mask", 0.9), "t")
        records.create_record("x.jpg", Detection((0, 0, 1, 1), "without_mask", 0.8), "t")

        summary = analytics.compute_summary()
        self.assertEqual(summary["total_detections"], 4)
        self.assertEqual(summary["with_mask"], 3)
        self.assertEqual(summary["without_mask"], 1)
        self.assertEqual(summary["compliance_rate_pct"], 75.0)

    def test_generate_report_creates_file(self):
        records.create_record("y.jpg", Detection((0, 0, 1, 1), "with_mask", 0.95), "t")
        path = analytics.generate_report()
        self.assertTrue(os.path.exists(path))
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Compliance rate", content)
        os.remove(path)
        csv_path = path.replace(".txt", ".csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
