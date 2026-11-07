"""Database service for storing sentiment analysis results.

Privacy & Safety:
- Never stores raw note text or model output transcripts.
- Stores only clinical metadata, labels, themes, and calculated flags.
- User identity is referenced solely via SHA-256 user_ref.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from app.config import DATABASE_PATH


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_sentiment_db() -> None:
    """Create sentiment_results and suggestion_history tables and performance indices."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_results (
                id TEXT PRIMARY KEY,
                user_ref TEXT NOT NULL,
                checkin_id TEXT,
                stage TEXT NOT NULL,
                tone TEXT NOT NULL,
                intensity TEXT NOT NULL,
                themes_json TEXT NOT NULL,
                risk_language INTEGER NOT NULL,
                support INTEGER NOT NULL,
                mismatch INTEGER NOT NULL,
                sleep_hours REAL,
                mood_band TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment_user_created
            ON sentiment_results (user_ref, created_at)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suggestion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_ref TEXT NOT NULL,
                suggestion_id TEXT NOT NULL,
                shown_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_suggestion_history_user_time
            ON suggestion_history (user_ref, shown_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_suggestion_history_time
            ON suggestion_history (shown_at)
        """)
        conn.commit()


def save_sentiment_result(
    user_ref: str,
    stage: str,
    tone: str,
    intensity: str,
    themes: List[str],
    risk_language: bool,
    support: bool,
    mismatch: bool,
    checkin_id: Optional[str] = None,
    sleep_hours: Optional[float] = None,
    mood_band: Optional[str] = None,
    created_at: Optional[str] = None,
) -> str:
    """Save an analyzed sentiment result to SQLite."""
    record_id = str(uuid.uuid4())
    now_iso = created_at or datetime.now(timezone.utc).isoformat()
    themes_str = json.dumps(themes, ensure_ascii=False)

    init_sentiment_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sentiment_results (
                id, user_ref, checkin_id, stage, tone, intensity,
                themes_json, risk_language, support, mismatch,
                sleep_hours, mood_band, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record_id,
                user_ref,
                checkin_id,
                stage,
                tone,
                intensity,
                themes_str,
                1 if risk_language else 0,
                1 if support else 0,
                1 if mismatch else 0,
                sleep_hours,
                mood_band,
                now_iso,
            ),
        )
        conn.commit()

    return record_id


def get_sentiment_results_for_period(
    user_ref: str,
    days: int,
    reference_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all sentiment records for a user within the past `days`."""
    now = reference_time or datetime.now(timezone.utc)
    since_iso = (now - timedelta(days=days)).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM sentiment_results
            WHERE user_ref = ? AND created_at >= ?
            ORDER BY created_at ASC
        """,
            (user_ref, since_iso),
        )
        rows = cursor.fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["themes"] = json.loads(d["themes_json"])
        d["risk_language"] = bool(d["risk_language"])
        d["support"] = bool(d["support"])
        d["mismatch"] = bool(d["mismatch"])
        results.append(d)
    return results


def get_sentiment_results_for_range(
    user_ref: str,
    start_iso: str,
    end_iso: str,
) -> List[Dict[str, Any]]:
    """Retrieve records between start_iso (inclusive) and end_iso (exclusive)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM sentiment_results
            WHERE user_ref = ? AND created_at >= ? AND created_at < ?
            ORDER BY created_at ASC
        """,
            (user_ref, start_iso, end_iso),
        )
        rows = cursor.fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["themes"] = json.loads(d["themes_json"])
        d["risk_language"] = bool(d["risk_language"])
        d["support"] = bool(d["support"])
        d["mismatch"] = bool(d["mismatch"])
        results.append(d)
    return results


def clear_sentiment_results(user_ref: str) -> int:
    """Delete all sentiment records for a user (useful for demo resets)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sentiment_results WHERE user_ref = ?", (user_ref,))
        deleted = cursor.rowcount
        conn.commit()
    return deleted


def record_shown_suggestions(
    user_ref: str,
    suggestion_ids: List[str],
    shown_at: Optional[str] = None,
) -> None:
    """Store shown suggestion IDs in suggestion_history table.

    STRICT PRIVACY RULE: Stores IDs and timestamps ONLY, never note text, themes, or scores.
    """
    if not user_ref or not suggestion_ids:
        return
    now_iso = shown_at or datetime.now(timezone.utc).isoformat()
    init_sentiment_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for sid in suggestion_ids:
            cursor.execute(
                """
                INSERT INTO suggestion_history (user_ref, suggestion_id, shown_at)
                VALUES (?, ?, ?)
            """,
                (user_ref, sid, now_iso),
            )
        conn.commit()


def get_recently_shown_ids(
    user_ref: str,
    days: int = 14,
    reference_time: Optional[datetime] = None,
) -> List[str]:
    """Retrieve list of suggestion IDs shown to this user in the last `days` days."""
    if not user_ref:
        return []
    init_sentiment_db()
    now = reference_time or datetime.now(timezone.utc)
    since_iso = (now - timedelta(days=days)).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT suggestion_id FROM suggestion_history
            WHERE user_ref = ? AND shown_at >= ?
        """,
            (user_ref, since_iso),
        )
        rows = cursor.fetchall()
    return [r[0] for r in rows]


def get_suggestion_last_shown_times(
    user_ref: str,
    days: int = 14,
    reference_time: Optional[datetime] = None,
) -> Dict[str, str]:
    """Retrieve mapping of suggestion_id -> most recent shown_at within `days`."""
    if not user_ref:
        return {}
    init_sentiment_db()
    now = reference_time or datetime.now(timezone.utc)
    since_iso = (now - timedelta(days=days)).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT suggestion_id, MAX(shown_at) as last_shown FROM suggestion_history
            WHERE user_ref = ? AND shown_at >= ?
            GROUP BY suggestion_id
        """,
            (user_ref, since_iso),
        )
        rows = cursor.fetchall()
    return {r[0]: r[1] for r in rows}


def delete_user_suggestion_history(user_ref: str) -> int:
    """Delete all suggestion history records for a user."""
    if not user_ref:
        return 0
    init_sentiment_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM suggestion_history WHERE user_ref = ?", (user_ref,))
        deleted = cursor.rowcount
        conn.commit()
    return deleted


def get_sentiment_library_stats(
    days_shown: int = 30,
    repeat_window_days: int = 14,
    reference_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Retrieve telemetry counts for suggestion library admin stats.

    STRICT PRIVACY: Contains counts ONLY, absolutely NO user identifiers or text.
    """
    init_sentiment_db()
    now = reference_time or datetime.now(timezone.utc)
    since_30d_iso = (now - timedelta(days=days_shown)).isoformat()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # How often each suggestion id was shown in the last 30 days
        cursor.execute(
            """
            SELECT suggestion_id, COUNT(*) as cnt
            FROM suggestion_history
            WHERE shown_at >= ?
            GROUP BY suggestion_id
            ORDER BY cnt DESC
        """,
            (since_30d_iso,),
        )
        shown_counts = {r[0]: r[1] for r in cursor.fetchall()}
        total_shown = sum(shown_counts.values())

        # Repeat rate: share of shown suggestions (or unique check-in events) where
        # the same suggestion_id was shown to that user within the prior 14 days.
        # Let's compute repeat rate across all suggestion presentations in last 30 days:
        cursor.execute(
            """
            SELECT sh1.id, sh1.user_ref, sh1.suggestion_id, sh1.shown_at
            FROM suggestion_history sh1
            WHERE sh1.shown_at >= ?
        """,
            (since_30d_iso,),
        )
        recent_events = cursor.fetchall()

        repeated_count = 0
        total_events = len(recent_events)
        for ev in recent_events:
            ev_id, u_ref, s_id, s_time = ev[0], ev[1], ev[2], ev[3]
            try:
                ev_dt = datetime.fromisoformat(s_time.replace("Z", "+00:00"))
            except Exception:
                continue
            window_start = (ev_dt - timedelta(days=repeat_window_days)).isoformat()
            cursor.execute(
                """
                SELECT 1 FROM suggestion_history
                WHERE user_ref = ? AND suggestion_id = ?
                  AND shown_at >= ? AND shown_at < ? AND id != ?
                LIMIT 1
            """,
                (u_ref, s_id, window_start, s_time, ev_id),
            )
            if cursor.fetchone():
                repeated_count += 1

        repeat_rate = round(repeated_count / total_events, 4) if total_events > 0 else 0.0

        # Number of results with no suggestions in the last 30 days
        # A result had no suggestions if it had risk_language=1 or support=1 or tone='distressed'
        cursor.execute(
            """
            SELECT COUNT(*) FROM sentiment_results
            WHERE created_at >= ? AND (risk_language = 1 OR support = 1 OR tone = 'distressed')
        """,
            (since_30d_iso,),
        )
        no_sug_row = cursor.fetchone()
        no_suggestions_count = no_sug_row[0] if no_sug_row else 0

    return {
        "shown_counts_30d": shown_counts,
        "total_shown_30d": total_shown,
        "repeat_rate_14d": repeat_rate,
        "no_suggestions_count": no_suggestions_count,
    }


