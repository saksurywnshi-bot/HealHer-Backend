"""STT Language Router.

Routes language codes to their designated STT model adapter.
Adding a new language or changing models requires only a configuration mapping update.

Privacy & Safety:
- Never logs audio content or transcripts.
"""

import logging
from typing import Dict, Optional, Tuple
from app.services.stt.base import BaseSTTAdapter
from app.services.stt.indic_conformer_adapter import IndicConformerAdapter
from app.services.stt.whisper_adapter import WhisperAdapter

logger = logging.getLogger("healher_backend.stt.router")

DEFAULT_LANGUAGE_ROUTING: Dict[str, str] = {
    "hi": "indic-conformer",
    "bn": "indic-conformer",
    "ta": "indic-conformer",
    "en": "whisper",
}


class STTRouter:
    """Routes incoming audio transcription requests to the appropriate STT adapter."""

    def __init__(
        self,
        routing_map: Optional[Dict[str, str]] = None,
        adapters: Optional[Dict[str, BaseSTTAdapter]] = None,
    ):
        self._routing_map = routing_map or dict(DEFAULT_LANGUAGE_ROUTING)
        self._adapters: Dict[str, BaseSTTAdapter] = adapters or {
            "indic-conformer": IndicConformerAdapter(),
            "whisper": WhisperAdapter(),
        }

    def register_adapter(self, name: str, adapter: BaseSTTAdapter) -> None:
        """Register a new or custom model adapter."""
        self._adapters[name] = adapter

    def set_language_model(self, language: str, model_name: str) -> None:
        """Assign a language code to a registered model adapter."""
        if model_name not in self._adapters:
            raise ValueError(f"Unknown STT model adapter: '{model_name}'")
        self._routing_map[language] = model_name

    def get_supported_languages(self) -> list[str]:
        """Return list of all supported language codes."""
        return sorted(list(self._routing_map.keys()))

    def get_adapter_for_language(self, language: str) -> BaseSTTAdapter:
        """Resolve the model adapter for the specified language code.

        Args:
            language: ISO language code (e.g. 'hi', 'bn', 'ta', 'en').

        Returns:
            The corresponding BaseSTTAdapter instance.

        Raises:
            ValueError: If the language code has no configured adapter.
        """
        lang = language.strip().lower()
        model_name = self._routing_map.get(lang)
        if not model_name:
            supported = ", ".join(self.get_supported_languages())
            raise ValueError(
                f"Unsupported language code '{language}'. Supported languages: {supported}"
            )

        adapter = self._adapters.get(model_name)
        if not adapter:
            raise ValueError(
                f"No adapter registered for model '{model_name}' (language '{language}')"
            )

        return adapter

    def transcribe(self, audio_path: str, language: str) -> Tuple[str, str]:
        """Transcribe an audio file using the appropriate model for the language.

        Args:
            audio_path: Path to the 16 kHz mono WAV file.
            language: Language code ('hi', 'bn', 'ta', 'en').

        Returns:
            Tuple of (transcribed_text, model_name).
        """
        adapter = self.get_adapter_for_language(language)
        try:
            text = adapter.transcribe(audio_path, language)
            return text, adapter.model_name
        except Exception as e:
            if adapter.model_name != "whisper" and "whisper" in self._adapters:
                logger.warning(
                    "STT adapter '%s' failed for language '%s' (%s). Gracefully falling back to faster-whisper.",
                    adapter.model_name,
                    language,
                    str(e),
                )
                whisper_adapter = self._adapters["whisper"]
                text = whisper_adapter.transcribe(audio_path, language)
                return text, whisper_adapter.model_name
            raise


# Default singleton instance
stt_router = STTRouter()
