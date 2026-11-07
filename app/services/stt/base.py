"""Base interface for Speech-to-Text (STT) model adapters.

Privacy & Safety:
- No audio content or transcription text may be logged.
"""

from abc import ABC, abstractmethod


class BaseSTTAdapter(ABC):
    """Abstract base class that all STT model adapters must implement."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the human-readable identifier of the model (e.g. 'indic-conformer', 'whisper')."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str, language: str) -> str:
        """Transcribes a 16 kHz mono WAV audio file to text.

        Args:
            audio_path: Absolute path to the 16 kHz mono WAV audio file.
            language: ISO language code (e.g. 'hi', 'bn', 'ta', 'en').

        Returns:
            Transcribed text as a string.
        """
        pass
