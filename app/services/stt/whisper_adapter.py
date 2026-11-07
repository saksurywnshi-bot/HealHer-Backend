"""OpenAI Whisper STT Adapter using faster-whisper.

Privacy & Safety:
- Audio content and transcripts are NEVER logged.
- Only device selection and system lifecycle events are logged.
"""

import logging
import threading
from typing import Optional

from app.config import WHISPER_MODEL_SIZE, STT_DEVICE
from app.services.stt.base import BaseSTTAdapter

logger = logging.getLogger("healher_backend.stt.whisper")


class WhisperAdapter(BaseSTTAdapter):
    """Adapter for English speech-to-text using faster-whisper."""

    def __init__(self, model_size: Optional[str] = None):
        self._model_size = model_size or WHISPER_MODEL_SIZE
        self._model = None
        self._lock = threading.Lock()
        self._device: Optional[str] = None

    @property
    def model_name(self) -> str:
        return "whisper"

    def _determine_device_and_compute(self) -> tuple[str, str]:
        """Detect CUDA availability and select optimal device and compute type."""
        import torch

        pref = STT_DEVICE.lower()
        if pref == "cuda" and torch.cuda.is_available():
            return "cuda", "float16"
        elif pref == "cpu":
            return "cpu", "int8"

        # Auto detection
        if torch.cuda.is_available():
            return "cuda", "float16"
        return "cpu", "int8"

    def _get_model(self):
        """Lazily load and cache the faster-whisper model."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                device, compute_type = self._determine_device_and_compute()
                self._device = device
                logger.info(
                    "Loading faster-whisper model '%s' on device: %s (compute_type: %s)",
                    self._model_size,
                    device,
                    compute_type,
                )
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self._model_size,
                    device=device,
                    compute_type=compute_type,
                )
                logger.info("faster-whisper model '%s' loaded successfully.", self._model_size)

        return self._model

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        """Transcribe audio file to English text using faster-whisper.

        Args:
            audio_path: Path to the 16 kHz mono WAV audio file.
            language: Target language code ('en').

        Returns:
            Transcribed text.
        """
        import numpy as np
        import soundfile as sf

        # Load audio into numpy float32 array to bypass faster-whisper's internal PyAV decoder
        audio_data, sr = sf.read(audio_path, dtype="float32")
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Check audio energy before normalization
        raw_max_abs = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
        raw_rms = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) > 0 else 0.0

        # If audio is silence (e.g. muted mic or macOS permission issue), return empty
        # to prevent Whisper from hallucinating on silence.
        if raw_max_abs < 0.01 or raw_rms < 0.002:
            return ""

        # Normalize audio volume so speech is brought up to optimal listening level
        audio_data = (audio_data / raw_max_abs * 0.95).astype(np.float32)

        model = self._get_model()
        segments, _ = model.transcribe(
            audio_data,
            language=language or "en",
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
        )
        # Concatenate transcribed segments without logging content
        full_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

        # Suppress known Whisper silence hallucinations when audio energy is low
        SILENCE_HALLUCINATIONS = {
            "you", "you.", "you!", "you?", "thank you", "thank you.",
            "thank you for watching", "thank you for watching.",
            "thanks for watching", "thanks for watching.", "bye", "bye."
        }
        if full_text.lower() in SILENCE_HALLUCINATIONS and raw_rms < 0.04:
            return ""

        return full_text



