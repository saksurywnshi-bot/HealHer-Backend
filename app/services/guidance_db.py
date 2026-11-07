"""Database service for Questionnaire Results, Guidance Cache, and Guidance Stats.

Privacy & Security:
- Stores only clinical metadata, scores, and calculated results.
- Never stores user free text, personal names, or raw model prompts.
- User identity is referenced solely via SHA-256 user_ref.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from app.config import DATABASE_PATH
from app.services.auth import hash_user_id


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_guidance_db() -> None:
    """Create questionnaire_results, guidance_cache, and guidance_stats tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Questionnaire Results Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questionnaire_results (
                id TEXT PRIMARY KEY,
                user_ref TEXT NOT NULL,
                instrument TEXT NOT NULL,
                total_score INTEGER NOT NULL,
                band_id TEXT NOT NULL,
                escalation TEXT NOT NULL,
                safety_triggered INTEGER NOT NULL DEFAULT 0,
                stage TEXT NOT NULL DEFAULT 'postnatal',
                answers_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_questionnaire_results_user
            ON questionnaire_results (user_ref, created_at)
        """)

        # 2. Guidance Cache Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guidance_cache (
                cache_key TEXT PRIMARY KEY,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        # 3. Guidance Telemetry & Validation Stats Table (counts only, no content)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guidance_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                failure_type TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_guidance_stats_event
            ON guidance_stats (event_type)
        """)

        conn.commit()


def save_questionnaire_result(
    user_ref: str,
    instrument: str,
    total_score: int,
    band_id: str,
    escalation: str,
    safety_triggered: bool,
    stage: str,
    answers: Dict[str, Any],
    result_dict: Dict[str, Any],
    custom_id: Optional[str] = None,
) -> str:
    """Store scored questionnaire result row."""
    result_id = custom_id or f"qres_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    clean_user_ref = hash_user_id(user_ref)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO questionnaire_results (
                id, user_ref, instrument, total_score, band_id, escalation,
                safety_triggered, stage, answers_json, result_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                clean_user_ref,
                instrument.lower(),
                int(total_score),
                band_id,
                escalation,
                1 if safety_triggered else 0,
                stage.lower(),
                json.dumps(answers, ensure_ascii=False),
                json.dumps(result_dict, ensure_ascii=False),
                now,
            ),
        )
        conn.commit()

    return result_id


def get_questionnaire_result(result_id: str, user_ref: str) -> Optional[Dict[str, Any]]:
    """Fetch stored questionnaire result, verifying that it belongs to user_ref.

    Returns None if result doesn't exist or belongs to another user.
    """
    clean_user_ref = hash_user_id(user_ref)
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check questionnaire_results table
        cursor.execute(
            "SELECT * FROM questionnaire_results WHERE id = ?",
            (result_id,),
        )
        row = cursor.fetchone()
        if row:
            if row["user_ref"] != clean_user_ref:
                # Result exists but belongs to a different user -> return None for 404
                return None
            return {
                "id": row["id"],
                "user_ref": row["user_ref"],
                "instrument": row["instrument"],
                "total_score": row["total_score"],
                "band_id": row["band_id"],
                "escalation": row["escalation"],
                "safety_triggered": bool(row["safety_triggered"]),
                "stage": row["stage"],
                "answers": json.loads(row["answers_json"]),
                "result": json.loads(row["result_json"]) if row["result_json"] else {},
                "created_at": row["created_at"],
            }

        # Check user_screenings table (fallback compatibility)
        cursor.execute(
            "SELECT * FROM user_screenings WHERE id = ?",
            (result_id,),
        )
        scr_row = cursor.fetchone()
        if scr_row:
            if scr_row["user_ref"] != clean_user_ref:
                return None
            answers = json.loads(scr_row["answers_json"]) if scr_row["answers_json"] else {}
            stype = scr_row["screening_type"]
            score = scr_row["total_score"]
            self_harm = bool(scr_row["self_harm_flag"])

            # Map band and escalation for screenings
            if stype == "phq9":
                if score >= 20:
                    band = "severe"
                    esc = "see_clinician_urgent"
                elif score >= 15:
                    band = "moderately_severe"
                    esc = "see_clinician_soon"
                elif score >= 10:
                    band = "moderate"
                    esc = "see_clinician"
                elif score >= 5:
                    band = "mild"
                    esc = "monitor"
                else:
                    band = "minimal"
                    esc = "routine"
            else:  # epds
                if score >= 13:
                    band = "possible_depression"
                    esc = "see_clinician"
                elif score >= 10:
                    band = "possible_mild_depression"
                    esc = "monitor"
                else:
                    band = "well"
                    esc = "routine"

            if self_harm:
                esc = "see_clinician_urgent"

            return {
                "id": scr_row["id"],
                "user_ref": scr_row["user_ref"],
                "instrument": stype,
                "total_score": score,
                "band_id": band,
                "escalation": esc,
                "safety_triggered": self_harm,
                "stage": "postnatal",
                "answers": answers,
                "result": {},
                "created_at": scr_row["created_at"],
            }

    return None


def get_cached_guidance(cache_key: str, max_days: int) -> Optional[Dict[str, Any]]:
    """Retrieve cached guidance response if younger than max_days."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response_json, created_at FROM guidance_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        try:
            created_dt = datetime.fromisoformat(row["created_at"])
            age_days = (datetime.now(timezone.utc) - created_dt).total_seconds() / 86400.0
            if age_days <= max_days:
                return json.loads(row["response_json"])
        except Exception:
            return None

    return None


def set_cached_guidance(cache_key: str, response_data: Dict[str, Any]) -> None:
    """Store guidance response in cache with current timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO guidance_cache (cache_key, response_json, created_at)
            VALUES (?, ?, ?)
            """,
            (cache_key, json.dumps(response_data, ensure_ascii=False), now),
        )
        conn.commit()


def clear_guidance_cache() -> None:
    """Clear guidance cache table (used in tests and maintenance)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM guidance_cache")
        conn.commit()


def record_guidance_stat(event_type: str, failure_type: Optional[str] = None) -> None:
    """Record an event count in guidance stats (strictly no content logged)."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO guidance_stats (event_type, failure_type, created_at)
                VALUES (?, ?, ?)
                """,
                (event_type, failure_type, now),
            )
            conn.commit()
    except Exception:
        pass


def get_guidance_stats() -> Dict[str, Any]:
    """Aggregate guidance statistics counts."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM guidance_stats WHERE event_type = 'ai_selection_used'")
        ai_selections_used = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM guidance_stats WHERE event_type = 'fallback'")
        fallbacks = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM guidance_stats WHERE event_type = 'cache_hit'")
        cache_hits = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT failure_type, COUNT(*) FROM guidance_stats
            WHERE event_type = 'validation_failure' AND failure_type IS NOT NULL
            GROUP BY failure_type
            """
        )
        validation_failures = {row[0]: row[1] for row in cursor.fetchall()}

    return {
        "ai_selections_used": ai_selections_used,
        "fallbacks": fallbacks,
        "cache_hits": cache_hits,
        "validation_failures_by_type": validation_failures,
    }
