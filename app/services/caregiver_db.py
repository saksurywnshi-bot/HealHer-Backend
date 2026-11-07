"""Database operations for caregiver tip history.

Privacy & Safety Rules:
- Stores tip IDs, date shown, and user reference only.
- NEVER stores questionnaire answers, themes, or clinical scores.
- Idempotent: safe to write once per day per user.
- Supports complete user data deletion.
- Admin stats aggregate counts only without user identifiers.
"""

from datetime import date, datetime, timedelta, timezone
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Set

import app.config as config

logger = logging.getLogger("healher_backend.caregiver_db")


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_caregiver_db(db_path: Optional[str] = None) -> None:
    """Create caregiver_tip_history table and indexes."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS caregiver_tip_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_ref TEXT NOT NULL,
                tip_id TEXT NOT NULL,
                shown_on TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_ref, tip_id, shown_on)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_caregiver_tip_user_date
            ON caregiver_tip_history (user_ref, shown_on)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_caregiver_tip_shown
            ON caregiver_tip_history (shown_on)
        """)
        conn.commit()


def record_shown_tips(
    user_ref: str,
    tip_ids: List[str],
    day: date,
    db_path: Optional[str] = None,
) -> None:
    """Record tips shown to user for the given local date (idempotent)."""
    if not tip_ids:
        return
    day_str = day.isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for tid in tip_ids:
            cursor.execute(
                """
                INSERT OR IGNORE INTO caregiver_tip_history (user_ref, tip_id, shown_on, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_ref, tid, day_str, now_iso),
            )
        conn.commit()


def get_user_tip_history(
    user_ref: str, db_path: Optional[str] = None
) -> Dict[str, date]:
    """Retrieve mapping of tip_id -> most recent date it was shown to user."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT tip_id, MAX(shown_on) as latest_shown
            FROM caregiver_tip_history
            WHERE user_ref = ?
            GROUP BY tip_id
            """,
            (user_ref,),
        )
        rows = cursor.fetchall()
        history: Dict[str, date] = {}
        for r in rows:
            try:
                history[r["tip_id"]] = date.fromisoformat(r["latest_shown"])
            except Exception:
                continue
        return history


def delete_user_caregiver_history(
    user_ref: str, db_path: Optional[str] = None
) -> int:
    """Delete all recorded caregiver tip history for a user. Returns count of deleted rows."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM caregiver_tip_history WHERE user_ref = ?",
            (user_ref,),
        )
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count


def get_caregiver_admin_stats(
    db_path: Optional[str] = None, days: int = 30
) -> Dict[str, Any]:
    """Compute aggregate counts for the last N days without user identifiers.

    Returns:
        {
            "tip_counts": {tip_id: total_times_shown_last_30_days},
            "days_with_fewer_than_five": number_of_user_days_with_less_than_5_tips
        }
    """
    cutoff_day = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Counts per tip
        cursor.execute(
            """
            SELECT tip_id, COUNT(*) as count
            FROM caregiver_tip_history
            WHERE shown_on >= ?
            GROUP BY tip_id
            ORDER BY count DESC
            """,
            (cutoff_day,),
        )
        tip_counts = {r["tip_id"]: r["count"] for r in cursor.fetchall()}

        # 2. Days with fewer than 5 tips
        cursor.execute(
            """
            SELECT user_ref, shown_on, COUNT(*) as cnt
            FROM caregiver_tip_history
            WHERE shown_on >= ?
            GROUP BY user_ref, shown_on
            HAVING cnt < 5
            """,
            (cutoff_day,),
        )
        fewer_than_five_days = len(cursor.fetchall())

        return {
            "tip_counts": tip_counts,
            "tip_counts_last_30_days": tip_counts,
            "days_with_fewer_than_five": fewer_than_five_days,
            "days_with_fewer_than_5_tips": fewer_than_five_days,
            "period_days": days,
        }


record_daily_tips = record_shown_tips


def get_recent_tip_ids(
    user_ref: str, days: int = 30, db_path: Optional[str] = None
) -> Set[str]:
    """Return set of tip IDs shown to the user within the last N days."""
    history = get_user_tip_history(user_ref, db_path)
    today = date.today()
    cutoff = today - timedelta(days=days)
    return {tid for tid, d in history.items() if d >= cutoff}
