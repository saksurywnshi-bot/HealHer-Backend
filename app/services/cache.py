"""SQLite Translation Cache Service.

Schema:
- key: string
- language: string ('hi', 'bn', 'ta')
- source_text: original English text
- source_hash: sha256 hash of source_text
- translated_text: translated text
- status: 'machine' | 'reviewed'
- updated_at: ISO-8601 timestamp string

Unique constraint on (key, language).
A 'reviewed' row always wins and is never overwritten by machine translations.
Machine translations are re-translated only when the source_hash changes.
"""

import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from app.config import DATABASE_PATH


def compute_source_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of source text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TranslationCache:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the SQLite translations table and indexes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    key TEXT NOT NULL,
                    language TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('machine', 'reviewed')),
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (key, language)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_translations_lang_status
                ON translations (language, status)
                """
            )
            conn.commit()

    def get_translation(self, key: str, language: str) -> Optional[Dict[str, Any]]:
        """Fetch a single translation record by key and language."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT key, language, source_text, source_hash, translated_text, status, updated_at
                FROM translations
                WHERE key = ? AND language = ?
                """,
                (key, language),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_translations_batch(self, keys: List[str], language: str) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple translations by keys and language in one query."""
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT key, language, source_text, source_hash, translated_text, status, updated_at
                FROM translations
                WHERE language = ? AND key IN ({placeholders})
                """,
                [language] + keys,
            )
            rows = cursor.fetchall()
            return {row["key"]: dict(row) for row in rows}

    def save_machine_translation(
        self, key: str, language: str, source_text: str, translated_text: str
    ) -> None:
        """Save or update a machine translation.

        A 'reviewed' row ALWAYS wins and is never overwritten.
        """
        source_hash = compute_source_hash(source_text)
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO translations (
                    key, language, source_text, source_hash, translated_text, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'machine', ?)
                ON CONFLICT(key, language) DO UPDATE SET
                    source_text = excluded.source_text,
                    source_hash = excluded.source_hash,
                    translated_text = excluded.translated_text,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                WHERE translations.status != 'reviewed'
                """,
                (key, language, source_text, source_hash, translated_text, now),
            )
            conn.commit()

    def update_reviewed_translation(
        self, key: str, language: str, translated_text: str
    ) -> Optional[Dict[str, Any]]:
        """Update an existing translation as reviewed by an admin.

        Sets status='reviewed' and updates updated_at.
        Returns the updated record, or None if not found.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            # Check if record exists
            cursor = conn.execute(
                "SELECT * FROM translations WHERE key = ? AND language = ?",
                (key, language),
            )
            existing = cursor.fetchone()
            if not existing:
                return None

            conn.execute(
                """
                UPDATE translations
                SET translated_text = ?, status = 'reviewed', updated_at = ?
                WHERE key = ? AND language = ?
                """,
                (translated_text, now, key, language),
            )
            conn.commit()

            cursor = conn.execute(
                "SELECT * FROM translations WHERE key = ? AND language = ?",
                (key, language),
            )
            updated = cursor.fetchone()
            return dict(updated) if updated else None

    def list_translations(
        self, status: Optional[str] = None, language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List translations with optional filtering by status and language."""
        query = "SELECT key, language, source_text, source_hash, translated_text, status, updated_at FROM translations"
        conditions = []
        params = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        if language:
            conditions.append("language = ?")
            params.append(language)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def export_all(self) -> List[Dict[str, Any]]:
        """Export all translations for CSV generation."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT key, language, source_text, source_hash, translated_text, status, updated_at
                FROM translations
                ORDER BY language ASC, key ASC
                """
            )
            return [dict(row) for row in cursor.fetchall()]


# Global cache instance
cache_service = TranslationCache()
