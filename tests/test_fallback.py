"""Tests for fallback to English text when LLM fails."""

import pytest


def test_fallback_on_llm_exception(client):
    async def mock_llm_failing(*args, **kwargs):
        raise TimeoutError("Gemini 10-second timeout exceeded")

    client.app.state.custom_llm_fn = mock_llm_failing

    original_text = "Rest when baby rests."
    payload = {
        "language": "hi",
        "items": [{"key": "tip.rest", "text": original_text}],
    }

    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["translations"]["tip.rest"]["status"] == "fallback"
    assert data["translations"]["tip.rest"]["text"] == original_text


def test_fallback_on_empty_or_invalid_llm_response(client):
    async def mock_llm_empty(*args, **kwargs):
        return {}  # empty dict returned

    client.app.state.custom_llm_fn = mock_llm_empty

    original_text = "Hydration is essential for milk production."
    payload = {
        "language": "ta",
        "items": [{"key": "tip.water", "text": original_text}],
    }

    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["translations"]["tip.water"]["status"] == "fallback"
    assert data["translations"]["tip.water"]["text"] == original_text
