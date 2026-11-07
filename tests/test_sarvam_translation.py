"""Tests for Gemini AI Indic Translation (Hindi, Bengali, Tamil)."""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.llm import (
    get_band_message,
    translate_text_with_gemini,
    translate_batch_with_llm,
    BAND_MESSAGES_TRANSLATED,
)


def test_band_messages_exist_for_all_supported_languages():
    """Verify band messages exist in en, hi, bn, and ta for all bands."""
    bands = [
        "minimal",
        "well",
        "mild",
        "possible_mild_depression",
        "moderate",
        "possible_depression",
        "moderately_severe",
        "severe",
    ]
    for band in bands:
        for lang in ["en", "hi", "bn", "ta"]:
            msg = get_band_message(band, lang)
            assert msg, f"Missing {lang} message for band {band}"
            assert isinstance(msg, str)
            assert len(msg) > 5


@pytest.mark.asyncio
async def test_gemini_translation_with_mock():
    """Mock Gemini translation response to verify request format and caching."""
    with patch(
        "app.services.llm.translate_batch_with_llm",
        new_callable=AsyncMock,
    ) as mock_translate:
        mock_translate.return_value = {
            "sug_0": "आज सुबह बहुत धूप खिली है।"
        }

        res = await translate_text_with_gemini(
            "Today has been a bright and sunny morning.",
            target_language="hi",
        )
        assert res == "आज सुबह बहुत धूप खिली है।"
        assert mock_translate.called

        # Second call with same text should hit in-memory cache without calling mock again
        mock_translate.reset_mock()
        res_cached = await translate_text_with_gemini(
            "Today has been a bright and sunny morning.",
            target_language="hi",
        )
        assert res_cached == "आज सुबह बहुत धूप खिली है।"
        assert not mock_translate.called


@pytest.mark.asyncio
async def test_gemini_translation_fallback_on_error():
    """If Gemini translation throws or fails, gracefully fallback to source text."""
    with patch(
        "app.services.llm.translate_batch_with_llm",
        new_callable=AsyncMock,
        side_effect=Exception("API Error"),
    ):
        res = await translate_text_with_gemini(
            "Unique text that should fallback gracefully",
            target_language="ta",
        )
        assert res == "Unique text that should fallback gracefully"


@pytest.mark.asyncio
async def test_gemini_english_passthrough():
    """English text should not be sent for translation."""
    with patch(
        "app.services.llm.translate_batch_with_llm",
        new_callable=AsyncMock,
    ) as mock_translate:
        res = await translate_text_with_gemini("Rest well.", target_language="en")
        assert res == "Rest well."
        assert not mock_translate.called


@pytest.mark.asyncio
async def test_sarvam_translation_routes_bengali_and_tamil_to_gemini():
    """Verify that translate_with_sarvam delegates Bengali and Tamil to Gemini."""
    from app.services.sarvam import translate_with_sarvam

    with patch(
        "app.services.llm.translate_text_with_gemini",
        new_callable=AsyncMock,
    ) as mock_gemini:
        mock_gemini.return_value = "আপনি কেমন আছেন"
        res_bn = await translate_with_sarvam("How are you feeling?", target_lang="bn")
        assert res_bn == "আপনি কেমন আছেন"
        assert mock_gemini.called

        mock_gemini.reset_mock()
        mock_gemini.return_value = "நீங்கள் எப்படி உணர்கிறீர்கள்"
        res_ta = await translate_with_sarvam("How are you feeling?", target_lang="ta")
        assert res_ta == "நீங்கள் எப்படி உணர்கிறீர்கள்"
        assert mock_gemini.called
