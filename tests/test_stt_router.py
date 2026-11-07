"""Unit tests for STTRouter and language adapter selection."""

import pytest
from app.services.stt.base import BaseSTTAdapter
from app.services.stt.router import STTRouter, DEFAULT_LANGUAGE_ROUTING


class DummyAdapter(BaseSTTAdapter):
    def __init__(self, name: str, return_text: str = "transcribed"):
        self._name = name
        self._return_text = return_text

    @property
    def model_name(self) -> str:
        return self._name

    def transcribe(self, audio_path: str, language: str) -> str:
        return f"{self._name}:{language}:{self._return_text}"


def test_stt_router_default_routing():
    """Verify default routing map for hi, bn, ta, and en."""
    router = STTRouter()
    assert router.get_supported_languages() == ["bn", "en", "hi", "ta"]

    # Verify adapter resolution
    hi_adapter = router.get_adapter_for_language("hi")
    assert hi_adapter.model_name == "indic-conformer"

    bn_adapter = router.get_adapter_for_language("bn")
    assert bn_adapter.model_name == "indic-conformer"

    ta_adapter = router.get_adapter_for_language("ta")
    assert ta_adapter.model_name == "indic-conformer"

    en_adapter = router.get_adapter_for_language("en")
    assert en_adapter.model_name == "whisper"


def test_stt_router_unsupported_language():
    """Verify that unsupported language codes raise ValueError with helpful message."""
    router = STTRouter()
    with pytest.raises(ValueError) as exc_info:
        router.get_adapter_for_language("fr")
    assert "Unsupported language code 'fr'" in str(exc_info.value)
    assert "hi" in str(exc_info.value)
    assert "en" in str(exc_info.value)


def test_stt_router_custom_adapter_registration():
    """Verify that adding a new model or language needs only a configuration change."""
    custom_adapter = DummyAdapter(name="custom-stt", return_text="hello world")
    router = STTRouter(
        routing_map={"mr": "custom-stt", "hi": "indic-conformer"},
        adapters={"custom-stt": custom_adapter, "indic-conformer": DummyAdapter("indic-conformer")},
    )

    adapter = router.get_adapter_for_language("mr")
    assert adapter.model_name == "custom-stt"

    text, model = router.transcribe("dummy_path.wav", "mr")
    assert text == "custom-stt:mr:hello world"
    assert model == "custom-stt"


def test_stt_router_case_insensitivity():
    """Verify router handles whitespace and casing variations cleanly."""
    router = STTRouter(
        routing_map={"en": "dummy"},
        adapters={"dummy": DummyAdapter("dummy")},
    )
    adapter = router.get_adapter_for_language("  EN  ")
    assert adapter.model_name == "dummy"
