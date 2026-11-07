"""Database Storage for BEAM Insights, Consent, User Profiles, and Screenings.

Privacy & Safety:
- Never stores raw prompt transcripts.
- Stores clinical reasons, chosen suggestion IDs, and calculated tiers for clinician audit.
- Uses SHA-256 user_ref for privacy-preserving user identification.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.config import DATABASE_PATH


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_insights_db() -> None:
    """Create all insights-related database tables and indices if not existing."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. User Phase & Sleep Profile
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_phase_profiles (
                user_ref TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                due_date TEXT,
                delivery_date TEXT,
                usual_sleep_hours REAL DEFAULT 7.5,
                history_bipolar_psychosis INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)

        # 2. Insights Informed Consent
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_insights_consent (
                user_ref TEXT PRIMARY KEY,
                consented INTEGER NOT NULL,
                consented_at TEXT NOT NULL
            )
        """)

        # 3. Generated Weekly / Monthly Insights (for clinical audit)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_generated_insights (
                id TEXT PRIMARY KEY,
                user_ref TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                anchor_date TEXT NOT NULL,
                weekly_band TEXT,
                weekly_addon TEXT,
                tier INTEGER NOT NULL,
                tier_name TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                chosen_suggestion_ids_json TEXT NOT NULL,
                phrased_text_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_insights_user_date
            ON user_generated_insights (user_ref, anchor_date)
        """)

        # 4. Screening Questionnaires (EPDS, GAD-7, MDQ)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_screenings (
                id TEXT PRIMARY KEY,
                user_ref TEXT NOT NULL,
                screening_type TEXT NOT NULL,
                total_score INTEGER NOT NULL,
                positive_screen INTEGER NOT NULL,
                self_harm_flag INTEGER DEFAULT 0,
                answers_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_screenings_user_created
            ON user_screenings (user_ref, created_at)
        """)

        conn.commit()


def save_user_profile(
    user_ref: str,
    phase: str,
    due_date: Optional[str] = None,
    delivery_date: Optional[str] = None,
    usual_sleep_hours: float = 7.5,
    history_bipolar_psychosis: bool = False,
) -> Dict[str, Any]:
    """Upsert user phase and sleep baseline profile."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_phase_profiles (
                user_ref, phase, due_date, delivery_date, usual_sleep_hours, history_bipolar_psychosis, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_ref) DO UPDATE SET
                phase=excluded.phase,
                due_date=excluded.due_date,
                delivery_date=excluded.delivery_date,
                usual_sleep_hours=excluded.usual_sleep_hours,
                history_bipolar_psychosis=excluded.history_bipolar_psychosis,
                updated_at=excluded.updated_at
        """, (
            user_ref,
            phase,
            due_date,
            delivery_date,
            float(usual_sleep_hours),
            1 if history_bipolar_psychosis else 0,
            now,
        ))
        conn.commit()

    return {
        "user_ref": user_ref,
        "phase": phase,
        "due_date": due_date,
        "delivery_date": delivery_date,
        "usual_sleep_hours": usual_sleep_hours,
        "history_bipolar_psychosis": history_bipolar_psychosis,
        "updated_at": now,
    }


def get_user_profile(user_ref: str) -> Optional[Dict[str, Any]]:
    """Retrieve user phase and baseline sleep profile."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_phase_profiles WHERE user_ref = ?", (user_ref,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "user_ref": row["user_ref"],
            "phase": row["phase"],
            "due_date": row["due_date"],
            "delivery_date": row["delivery_date"],
            "usual_sleep_hours": row["usual_sleep_hours"],
            "history_bipolar_psychosis": bool(row["history_bipolar_psychosis"]),
            "updated_at": row["updated_at"],
        }


def save_user_consent(user_ref: str, consented: bool = True) -> Dict[str, Any]:
    """Record user informed consent for BEAM insights."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_insights_consent (user_ref, consented, consented_at)
            VALUES (?, ?, ?)
            ON CONFLICT(user_ref) DO UPDATE SET
                consented=excluded.consented,
                consented_at=excluded.consented_at
        """, (user_ref, 1 if consented else 0, now))
        conn.commit()
    return {"user_ref": user_ref, "consented": consented, "consented_at": now}


def get_user_consent(user_ref: str) -> bool:
    """Check if user has granted informed consent."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT consented FROM user_insights_consent WHERE user_ref = ?", (user_ref,))
        row = cursor.fetchone()
        return bool(row["consented"]) if row else False


def save_generated_insight(
    user_ref: str,
    insight_type: str,
    anchor_date: str,
    weekly_band: Optional[str],
    weekly_addon: Optional[str],
    tier: int,
    tier_name: str,
    reasons: List[str],
    chosen_suggestion_ids: List[str],
    phrased_text: Dict[str, Any],
) -> str:
    """Store generated insight record with full clinician audit log."""
    insight_id = f"ins_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_generated_insights (
                id, user_ref, insight_type, anchor_date, weekly_band, weekly_addon,
                tier, tier_name, reasons_json, chosen_suggestion_ids_json, phrased_text_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            insight_id,
            user_ref,
            insight_type,
            anchor_date,
            weekly_band,
            weekly_addon,
            tier,
            tier_name,
            json.dumps(reasons, ensure_ascii=False),
            json.dumps(chosen_suggestion_ids, ensure_ascii=False),
            json.dumps(phrased_text, ensure_ascii=False),
            now,
        ))
        conn.commit()
    return insight_id


def get_recent_insights(user_ref: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Retrieve user's historical insights for review and doctor summary."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM user_generated_insights
            WHERE user_ref = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_ref, limit))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "insight_type": r["insight_type"],
                "anchor_date": r["anchor_date"],
                "weekly_band": r["weekly_band"],
                "weekly_addon": r["weekly_addon"],
                "tier": r["tier"],
                "tier_name": r["tier_name"],
                "reasons": json.loads(r["reasons_json"]),
                "chosen_suggestion_ids": json.loads(r["chosen_suggestion_ids_json"]),
                "phrased_text": json.loads(r["phrased_text_json"]),
                "created_at": r["created_at"],
            })
        return results


def save_screening(
    user_ref: str,
    screening_type: str,
    total_score: int,
    positive_screen: bool,
    self_harm_flag: bool,
    answers: Dict[str, Any],
) -> str:
    """Store screening questionnaire response (EPDS, GAD-7, MDQ)."""
    screening_id = f"scr_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_screenings (
                id, user_ref, screening_type, total_score, positive_screen, self_harm_flag, answers_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            screening_id,
            user_ref,
            screening_type,
            int(total_score),
            1 if positive_screen else 0,
            1 if self_harm_flag else 0,
            json.dumps(answers, ensure_ascii=False),
            now,
        ))
        conn.commit()
    return screening_id
