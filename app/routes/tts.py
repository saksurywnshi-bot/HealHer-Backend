"""TTS API Route for HealHer.

Endpoint:
POST /api/tts
Body: { text: str, language: "en" | "hi" | "bn" | "ta", speaker?: str, pace?: float }
Response: { ok: bool, audio_base64: str, format: str, cached: bool, language: str, speaker: str }
"""

import logging
from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.tts_service import tts_cache_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tts"])


class TTSRequest(BaseModel):
    text: str = Field(..., max_length=2000, description="Text to synthesize to speech")
    language: Literal["en", "hi", "bn", "ta"] = Field("en", description="Target language")
    speaker: Optional[str] = Field("ritu", description="Sarvam speaker voice (default: ritu)")
    pace: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="Speech rate/pace")


class TTSResponse(BaseModel):
    ok: bool
    audio_base64: Optional[str] = None
    format: str = "audio/wav"
    cached: bool = False
    language: str
    speaker: str


@router.post(
    "/tts",
    response_model=TTSResponse,
    summary="Text-to-speech using Sarvam AI with persistent SQLite caching",
)
async def generate_speech(payload: TTSRequest):
    """Synthesize speech using Sarvam AI or return from persistent SQLite cache.

    If cached, consumes 0 tokens and responds in milliseconds.
    """
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty.",
        )

    speaker = "ritu" if not payload.speaker or payload.speaker.lower() in ("meera", "default") else payload.speaker.lower()
    pace = payload.pace or 1.0

    audio_b64, audio_fmt, cached = await tts_cache_service.get_or_synthesize(
        text=clean_text,
        language=payload.language,
        speaker=speaker,
        pace=pace,
    )

    if not audio_b64:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS synthesis unavailable. Please use browser speech synthesis fallback.",
        )

    return TTSResponse(
        ok=True,
        audio_base64=audio_b64,
        format=audio_fmt,
        cached=cached,
        language=payload.language,
        speaker=speaker,
    )

