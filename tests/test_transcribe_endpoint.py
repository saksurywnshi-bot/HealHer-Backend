"""Unit and integration tests for POST /api/transcribe endpoint."""

import io
import pytest
from httpx import ASGITransport, AsyncClient

np = pytest.importorskip("numpy", reason="numpy is not installed")
sf = pytest.importorskip("soundfile", reason="soundfile is not installed")

from app.main import app
from app.services.stt.router import stt_router
from app.services.stt.base import BaseSTTAdapter


@pytest.fixture(autouse=True)
def enable_stt_for_tests(monkeypatch):
    """Enable STT for transcribe route tests unless explicitly overridden."""
    import app.config as config
    monkeypatch.setattr(config, "ENABLE_STT", True)


@pytest.mark.asyncio
async def test_transcribe_disabled_returns_503(monkeypatch):
    """Verify POST /api/transcribe returns 503 when ENABLE_STT is false."""
    import app.config as config
    monkeypatch.setattr(config, "ENABLE_STT", False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/transcribe",
            files={"file": ("recording.wav", b"fake-audio-bytes", "audio/wav")},
            data={"language": "en"},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "Voice input is not available on this server"


class MockSTTAdapter(BaseSTTAdapter):
    """Mock adapter for testing the transcription endpoint pipeline."""

    def __init__(self, model_name: str, response_text: str):
        self._model_name = model_name
        self._response_text = response_text

    @property
    def model_name(self) -> str:
        return self._model_name

    def transcribe(self, audio_path: str, language: str) -> str:
        return self._response_text


def generate_test_wav_bytes(duration_s: float = 0.5) -> bytes:
    """Generate in-memory WAV audio bytes."""
    sr = 16000
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_transcribe_endpoint_success_with_mock():
    """Verify POST /api/transcribe returns 200 with text, language, and model."""
    mock_adapter = MockSTTAdapter("test-model", "आज मुझे बहुत थकान लग रही है")
    original_adapters = stt_router._adapters.copy()
    stt_router._adapters["indic-conformer"] = mock_adapter

    try:
        wav_bytes = generate_test_wav_bytes(0.5)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/transcribe",
                files={"file": ("recording.wav", wav_bytes, "audio/wav")},
                data={"language": "hi"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "आज मुझे बहुत थकान लग रही है"
        assert body["language"] == "hi"
        assert body["model"] == "test-model"
    finally:
        stt_router._adapters = original_adapters


@pytest.mark.asyncio
async def test_transcribe_endpoint_invalid_language():
    """Verify POST /api/transcribe rejects invalid language code with 400."""
    wav_bytes = generate_test_wav_bytes(0.2)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/transcribe",
            files={"file": ("recording.wav", wav_bytes, "audio/wav")},
            data={"language": "de"},
        )

    assert response.status_code == 400
    assert "Unsupported language 'de'" in response.json()["detail"]


@pytest.mark.asyncio
async def test_transcribe_endpoint_unsupported_file_format():
    """Verify POST /api/transcribe rejects non-audio file formats with 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/transcribe",
            files={"file": ("notes.txt", b"some plain text notes", "text/plain")},
            data={"language": "en"},
        )

    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


@pytest.mark.asyncio
async def test_transcribe_endpoint_empty_file():
    """Verify POST /api/transcribe rejects empty files with 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/transcribe",
            files={"file": ("empty.wav", b"", "audio/wav")},
            data={"language": "en"},
        )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_transcribe_endpoint_english_whisper():
    """Verify POST /api/transcribe correctly routes English to whisper adapter."""
    mock_whisper = MockSTTAdapter("whisper", "I am feeling very anxious and exhausted today.")
    original_adapters = stt_router._adapters.copy()
    stt_router._adapters["whisper"] = mock_whisper

    try:
        wav_bytes = generate_test_wav_bytes(0.5)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/transcribe",
                files={"file": ("recording.webm", wav_bytes, "audio/webm")},
                data={"language": "en"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "I am feeling very anxious and exhausted today."
        assert body["language"] == "en"
        assert body["model"] == "whisper"
    finally:
        stt_router._adapters = original_adapters
