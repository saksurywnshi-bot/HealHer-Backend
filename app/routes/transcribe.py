"""Speech-to-Text Transcription API Endpoint for HealHer.

Endpoint:
- POST /api/transcribe: Transcribes audio in Hindi, Bengali, Tamil, or English.

Privacy & Safety:
- Audio content and transcribed text are NEVER logged.
- All temporary audio files are strictly deleted in finally blocks.
- No third-party network or telemetry calls for audio data.
"""

import os
import tempfile
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.config import MAX_AUDIO_SIZE_BYTES, STT_RATE_LIMIT
from app.limiter import limiter
from app.services.stt import (
    stt_router,
    validate_audio_input,
    convert_to_16k_mono_wav,
)

logger = logging.getLogger("healher_backend.routes.transcribe")

router = APIRouter(prefix="/api", tags=["speech-to-text"])


class TranscribeResponse(BaseModel):
    """Transcription response model."""

    text: str = Field(..., description="Transcribed text in the target language script.")
    language: str = Field(..., description="The language code processed.")
    model: str = Field(..., description="The model adapter utilized for transcription.")


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe spoken audio to text",
    description="Accepts an audio recording (webm, wav, mp3, m4a, ogg) and language code (hi, bn, ta, en), "
    "converts to 16 kHz mono WAV, and transcribes using IndicConformer or faster-whisper.",
)
@limiter.limit(STT_RATE_LIMIT)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio recording file (max 25 MB)"),
    language: str = Form(..., description="Target language code: hi, bn, ta, en"),
) -> TranscribeResponse:
    """Handles audio file upload, validation, conversion, and STT inference."""
    import app.config as app_cfg
    if not getattr(app_cfg, "ENABLE_STT", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice input is not available on this server",
        )

    # Read and validate incoming file size
    raw_temp_path: Optional[str] = None
    converted_wav_path: Optional[str] = None

    try:
        # Determine file extension safely
        filename = file.filename or "recording.webm"
        content_type = file.content_type
        suffix = os.path.splitext(filename)[1].lower() or ".webm"

        # Stream file to temporary storage to avoid keeping large blobs entirely in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as raw_file:
            raw_temp_path = raw_file.name
            total_size = 0
            chunk_size = 64 * 1024  # 64 KB

            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_AUDIO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="Audio file exceeds the maximum allowed size of 25 MB.",
                    )
                raw_file.write(chunk)

        # Validate file size, type, and language
        validate_audio_input(
            filename=filename,
            content_type=content_type,
            size_bytes=total_size,
            language=language,
        )

        # Create path for 16 kHz mono WAV output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            converted_wav_path = wav_file.name

        # Convert to 16 kHz mono WAV in worker thread
        await asyncio.to_thread(convert_to_16k_mono_wav, raw_temp_path, converted_wav_path)

        # Run model inference in worker thread so the server event loop is not blocked
        text, model_name = await asyncio.to_thread(stt_router.transcribe, converted_wav_path, language)

        return TranscribeResponse(
            text=text,
            language=language.strip().lower(),
            model=model_name,
        )

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error("STT transcription failed with error: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech transcription failed. Please try again or type your note.",
        )
    finally:
        # Guarantee strict cleanup of temporary audio files
        if raw_temp_path and os.path.exists(raw_temp_path):
            try:
                os.unlink(raw_temp_path)
            except OSError:
                pass
        if converted_wav_path and os.path.exists(converted_wav_path):
            try:
                os.unlink(converted_wav_path)
            except OSError:
                pass
