"""TTS Cache and Storage Service for HealHer.

Stores synthesized audio permanently in SQLite so that:
- Every unique translated text + language + speaker is synthesized at most once.
- Subsequent plays cost 0 Sarvam tokens and return in milliseconds.
- Audio is saved as Base64 WAV data.
"""

import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from app.config import DATABASE_PATH
from app.services.sarvam import synthesize_speech_with_sarvam

logger = logging.getLogger("healher_backend.tts")


def normalize_speaker(speaker: Optional[str]) -> str:
    """Normalize speaker voice to supported bulbul:v3 catalog (e.g. ritu)."""
    if not speaker or speaker.lower() in ("meera", "default"):
        return "ritu"
    return speaker.lower()


def compute_tts_hash(text: str, language: str, speaker: str) -> str:
    """Compute deterministic SHA-256 hash for text, language, and speaker."""
    valid_spk = normalize_speaker(speaker)
    norm = f"{language}:{valid_spk}:{text.strip()}".lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class TTSCacheService:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the SQLite tts_cache table and indexes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tts_cache (
                    text_hash TEXT NOT NULL,
                    language TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    audio_base64 TEXT NOT NULL,
                    audio_format TEXT NOT NULL DEFAULT 'audio/wav',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (text_hash, language, speaker)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tts_cache_lang
                ON tts_cache (language)
                """
            )
            conn.commit()

    def get_cached_audio(
        self, text: str, language: str, speaker: str = "ritu"
    ) -> Optional[Dict[str, Any]]:
        """Fetch cached audio if present in SQLite."""
        valid_spk = normalize_speaker(speaker)
        text_hash = compute_tts_hash(text, language, valid_spk)
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT text_hash, language, speaker, source_text, audio_base64, audio_format, created_at
                FROM tts_cache
                WHERE text_hash = ? AND language = ? AND speaker = ?
                """,
                (text_hash, language, valid_spk),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def save_audio(
        self,
        text: str,
        language: str,
        audio_base64: str,
        speaker: str = "ritu",
        audio_format: str = "audio/wav",
    ) -> None:
        """Save synthesized audio into persistent SQLite cache."""
        valid_spk = normalize_speaker(speaker)
        text_hash = compute_tts_hash(text, language, valid_spk)
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO tts_cache (
                    text_hash, language, speaker, source_text, audio_base64, audio_format, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(text_hash, language, speaker) DO UPDATE SET
                    audio_base64 = excluded.audio_base64,
                    audio_format = excluded.audio_format,
                    created_at = excluded.created_at
                """,
                (text_hash, language, valid_spk, text.strip(), audio_base64, audio_format, now),
            )
            conn.commit()

    async def get_or_synthesize(
        self,
        text: str,
        language: str = "hi",
        speaker: str = "ritu",
        pace: float = 1.0,
    ) -> Tuple[Optional[str], str, bool]:
        """Get cached audio or synthesize via Sarvam AI and store.

        Returns (audio_base64, audio_format, is_cached).
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return (None, "audio/wav", False)

        valid_spk = normalize_speaker(speaker)

        # 1. Check persistent SQLite cache first (0 AI tokens)
        cached = self.get_cached_audio(clean_text, language, valid_spk)
        if cached and cached.get("audio_base64"):
            logger.info("TTS Cache HIT for [%s]: %s chars (0 AI tokens)", language, len(clean_text))
            return (cached["audio_base64"], cached.get("audio_format", "audio/wav"), True)

        # 2. Cache miss -> Synthesize via Sarvam AI
        logger.info("TTS Cache MISS for [%s]. Calling Sarvam AI TTS...", language)
        audio_b64 = await synthesize_speech_with_sarvam(
            text=clean_text,
            language=language,
            speaker=valid_spk,
            pace=pace,
        )

        if audio_b64:
            # 3. Store in SQLite so future requests cost 0 tokens
            self.save_audio(
                text=clean_text,
                language=language,
                audio_base64=audio_b64,
                speaker=valid_spk,
                audio_format="audio/wav",
            )
            logger.info("TTS Audio synthesized and cached permanently for [%s]", language)
            return (audio_b64, "audio/wav", False)

        return (None, "audio/wav", False)


tts_cache_service = TTSCacheService()

