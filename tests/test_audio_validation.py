"""Unit tests for audio validation and 16 kHz mono conversion."""

import os
import tempfile
import pytest
from fastapi import HTTPException

from app.services.stt.audio_utils import (
    validate_audio_input,
    convert_to_16k_mono_wav,
    MAX_AUDIO_FILE_SIZE_BYTES,
)


def test_validate_audio_valid_inputs():
    """Verify supported formats, languages, and sizes pass validation without exception."""
    for ext in [".webm", ".wav", ".mp3", ".m4a", ".ogg"]:
        for lang in ["hi", "bn", "ta", "en"]:
            validate_audio_input(
                filename=f"test{ext}",
                content_type="audio/webm",
                size_bytes=1024 * 50,  # 50 KB
                language=lang,
            )


def test_validate_audio_invalid_language():
    """Verify that unsupported language codes raise HTTPException(400)."""
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_input(
            filename="test.wav",
            content_type="audio/wav",
            size_bytes=1024,
            language="es",
        )
    assert exc_info.value.status_code == 400
    assert "Unsupported language 'es'" in exc_info.value.detail


def test_validate_audio_size_exceeded():
    """Verify that files over 25 MB raise HTTPException(413)."""
    oversized = MAX_AUDIO_FILE_SIZE_BYTES + 1
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_input(
            filename="large.wav",
            content_type="audio/wav",
            size_bytes=oversized,
            language="hi",
        )
    assert exc_info.value.status_code == 413
    assert "exceeds the maximum allowed size" in exc_info.value.detail


def test_validate_audio_empty_file():
    """Verify that empty files raise HTTPException(400)."""
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_input(
            filename="empty.wav",
            content_type="audio/wav",
            size_bytes=0,
            language="ta",
        )
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()


def test_validate_audio_unsupported_format():
    """Verify that unsupported extensions and non-audio MIME types raise HTTPException(400)."""
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_input(
            filename="document.pdf",
            content_type="application/pdf",
            size_bytes=1024,
            language="en",
        )
    assert exc_info.value.status_code == 400
    assert "Unsupported audio format" in exc_info.value.detail


def test_convert_to_16k_mono_wav():
    """Verify that audio conversion produces a valid 16,000 Hz 1-channel (mono) WAV file."""
    sf = pytest.importorskip("soundfile", reason="soundfile is not installed")
    np = pytest.importorskip("numpy", reason="numpy is not installed")

    # Generate 1 second of stereo 44.1 kHz audio
    sample_rate_in = 44100
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(sample_rate_in * duration_s), endpoint=False)
    # Sine wave
    stereo_data = np.stack([np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 440 * t)], axis=-1).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_file, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
        in_path = in_file.name
        out_path = out_file.name

    try:
        sf.write(in_path, stereo_data, sample_rate_in)

        # Convert
        convert_to_16k_mono_wav(in_path, out_path)

        # Verify output characteristics
        assert os.path.exists(out_path)
        info = sf.info(out_path)
        assert info.samplerate == 16000
        assert info.channels == 1
        assert info.format == "WAV"
        assert info.subtype == "PCM_16"
        assert info.duration > 0.95
    finally:
        for p in [in_path, out_path]:
            if os.path.exists(p):
                os.unlink(p)
