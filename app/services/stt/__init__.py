"""Speech-to-Text (STT) Service Package for HealHer.

Provides unified transcription interface across Indic languages and English.
"""

from app.services.stt.base import BaseSTTAdapter
from app.services.stt.indic_conformer_adapter import IndicConformerAdapter
from app.services.stt.whisper_adapter import WhisperAdapter
from app.services.stt.router import STTRouter, stt_router
from app.services.stt.audio_utils import (
    validate_audio_input,
    convert_to_16k_mono_wav,
    MAX_AUDIO_FILE_SIZE_BYTES,
    SUPPORTED_STT_LANGUAGES,
)

__all__ = [
    "BaseSTTAdapter",
    "IndicConformerAdapter",
    "WhisperAdapter",
    "STTRouter",
    "stt_router",
    "validate_audio_input",
    "convert_to_16k_mono_wav",
    "MAX_AUDIO_FILE_SIZE_BYTES",
    "SUPPORTED_STT_LANGUAGES",
]
