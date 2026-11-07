"""Tests for placeholder validation and conditional retry routing."""

import pytest


def test_placeholder_retry_succeeds_on_second_attempt(client):
    calls = []

    async def mock_llm(items, lang, attempt):
        calls.append((attempt, [it["key"] for it in items]))
        if attempt == 1:
            # Corrupted translation: omitted {{week}} placeholder
            return {
                "greeting": "नमस्ते {{name}}, आपका स्वागत है!",  # missing {{week}}
                "simple": "सरल पाठ",
            }
        else:
            # Fixed translation on attempt 2: preserves both {{name}} and {{week}}
            return {
                "greeting": "नमस्ते {{name}}, सप्ताह {{week}} में आपका स्वागत है!",
            }

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "language": "hi",
        "items": [
            {"key": "greeting", "text": "Hello {{name}}, welcome to week {{week}}!"},
            {"key": "simple", "text": "Simple text"},
        ],
    }

    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify retry occurred: attempt 1 called for both, attempt 2 called only for 'greeting'
    assert len(calls) == 2
    assert calls[0][0] == 1
    assert calls[1][0] == 2
    assert calls[1][1] == ["greeting"]

    # Both succeeded in the end
    assert data["translations"]["greeting"]["status"] == "machine"
    assert "{{name}}" in data["translations"]["greeting"]["text"]
    assert "{{week}}" in data["translations"]["greeting"]["text"]
    assert data["translations"]["simple"]["status"] == "machine"


def test_placeholder_failure_falls_back_after_attempt_two(client):
    calls = []

    async def mock_llm(items, lang, attempt):
        calls.append(attempt)
        # Always omit the %s placeholder
        return {
            "score_display": "आपका स्कोर है",  # omitted %s
        }

    client.app.state.custom_llm_fn = mock_llm

    original_text = "Your score is %s"
    payload = {
        "language": "bn",
        "items": [
            {"key": "score_display", "text": original_text},
        ],
    }

    response = client.post("/api/translate", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Retried once, then fell back
    assert calls == [1, 2]
    assert data["translations"]["score_display"]["status"] == "fallback"
    assert data["translations"]["score_display"]["text"] == original_text
