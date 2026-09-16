"""
auth.py
-------
Functional Module 1: User Management.

Provides registration, login, and role-based access control
(admin / operator) backed by SQLite. Passwords are never stored
in plain text (Security non-functional requirement).
"""

import hashlib
import hmac
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

import config
from src.logger_setup import get_logger

logger = get_logger(__name__)

VALID_ROLES = ("admin", "operator")


@dataclass
class User:
    user_id: int
    username: str
    role: str


class AuthError(Exception):
    """Raised for authentication / authorization failures."""


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_auth_table() -> None:
    """Create the users table if it does not exist yet."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT UNIQUE NOT NULL,
                salt        TEXT NOT NULL,
                pwd_hash    TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('admin', 'operator')),
                created_at  TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
    logger.info("Auth table verified/created.")


def _hash_password(password: str, salt: Optional[bytes] = None):
    """PBKDF2-HMAC-SHA256 password hashing with a random per-user salt."""
    if salt is None:
        salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex(), pwd_hash.hex()


def register_user(username: str, password: str, role: str = "operator") -> None:
    if not username or not password:
        raise AuthError("Username and password cannot be empty.")
    if role not in VALID_ROLES:
        raise AuthError(f"Role must be one of {VALID_ROLES}.")
    if len(password) < 6:
        raise AuthError("Password must be at least 6 characters long.")

    salt_hex, hash_hex = _hash_password(password)

    try:
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, salt, pwd_hash, role) VALUES (?, ?, ?, ?)",
                (username, salt_hex, hash_hex, role),
            )
            conn.commit()
        logger.info("Registered new user '%s' with role '%s'.", username, role)
    except sqlite3.IntegrityError as exc:
        logger.warning("Attempt to register duplicate username '%s'.", username)
        raise AuthError(f"Username '{username}' already exists.") from exc


def login(username: str, password: str) -> User:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT user_id, salt, pwd_hash, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        logger.warning("Login failed: unknown username '%s'.", username)
        raise AuthError("Invalid username or password.")

    user_id, salt_hex, stored_hash_hex, role = row
    salt = bytes.fromhex(salt_hex)
    _, computed_hash_hex = _hash_password(password, salt)

    if not hmac.compare_digest(computed_hash_hex, stored_hash_hex):
        logger.warning("Login failed: bad password for '%s'.", username)
        raise AuthError("Invalid username or password.")

    logger.info("User '%s' logged in successfully.", username)
    return User(user_id=user_id, username=username, role=role)


def require_admin(user: User) -> None:
    if user.role != "admin":
        logger.warning("Unauthorized admin action attempted by '%s'.", user.username)
        raise AuthError("This action requires administrator privileges.")


def ensure_default_admin() -> None:
    """Create a default admin (admin/admin123) on first run, for grading convenience."""
    with _get_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        register_user("admin", "admin123", role="admin")
        logger.info("Default admin account created (username=admin, password=admin123).")
