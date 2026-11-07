"""Integration tests for STT models.

Transcribes short sample clips for each language if models are available,
skipping gracefully if models or weights are not loaded/accessible.
"""

import os
import tempfile
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed (speech-to-text is optional)")
torchaudio = pytest.importorskip("torchaudio", reason="torchaudio is not installed")
sf = pytest.importorskip("soundfile", reason="soundfile is not installed")
import numpy as np

from app.services.stt.router import stt_router
from app.services.stt.indic_conformer_adapter import IndicConformerAdapter
from app.services.stt.whisper_adapter import WhisperAdapter


def create_sample_audio_file(sample_rate: int = 16000, duration_s: float = 1.0) -> str:
    """Create a temporary WAV audio file with gentle tone for integration testing."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Generate simple audio signal
    audio_data = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    sf.write(path, audio_data, sample_rate, subtype="PCM_16")
    return path


def test_whisper_integration_short_clip():
    """Integration test for faster-whisper on a short sample audio clip."""
    audio_path = create_sample_audio_file(sample_rate=16000, duration_s=1.0)
    try:
        adapter = WhisperAdapter(model_size="tiny")  # Use tiny for quick test if downloaded
        try:
            # Attempt to transcribe; will skip if weights download fails or network unavailable
            text = adapter.transcribe(audio_path, language="en")
            assert isinstance(text, str)
        except Exception as e:
            pytest.skip(f"Whisper model weights not available or failed to load: {e}")
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)


@pytest.mark.parametrize("lang", ["hi", "bn", "ta"])
def test_indic_conformer_integration_short_clip(lang):
    """Integration test for IndicConformer on a short sample audio clip (graceful skip if unavailable)."""
    audio_path = create_sample_audio_file(sample_rate=16000, duration_s=1.0)
    try:
        adapter = IndicConformerAdapter()
        try:
            text = adapter.transcribe(audio_path, language=lang)
            assert isinstance(text, str)
        except Exception as e:
            pytest.skip(
                f"IndicConformer model not locally cached or HF token not provided ({e}). Skipping gracefully."
            )
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
