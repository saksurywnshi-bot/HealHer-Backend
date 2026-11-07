"""Tests for input validation (word/char counts) and crisis rules."""

import pytest


def test_note_with_under_10_words_and_under_50_chars_returns_too_short(client, auth_headers):
    """9-word note with fewer than 50 characters -> status: 'too_short'."""
    llm_called = False

    async def mock_llm(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        return {}

    client.app.state.custom_llm_fn = mock_llm

    # 9 words, 36 characters without spaces
    short_text = "one two three four five six seven eight nine"
    assert len(short_text.split()) == 9
    assert len(short_text.replace(" ", "")) < 50

    payload = {
        "note_text": short_text,
        "language": "en",
        "stage": "prenatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "too_short"
    assert data["tone"] is None
    assert data["reflection"] is None
    assert llm_called is False


def test_note_with_10_words_is_eligible_and_analyzed(client, auth_headers):
    """10-word note -> analyzed by model."""
    llm_called = False

    async def mock_llm(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        return {
            "tone": "neutral",
            "intensity": "low",
            "themes": ["tiredness"],
            "risk_language": False,
            "reflection": "You are feeling quite tired after a busy day today.",
        }

    client.app.state.custom_llm_fn = mock_llm

    ten_word_text = "one two three four five six seven eight nine ten"
    assert len(ten_word_text.split()) == 10

    payload = {
        "note_text": ten_word_text,
        "language": "en",
        "stage": "prenatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["tone"] == "neutral"
    assert data["tone_phrase_key"] == "sentiment.tone.neutral"
    assert llm_called is True


def test_tamil_note_with_few_words_but_over_50_chars_is_analyzed(client, auth_headers):
    """Tamil note with few words but 55 characters -> analyzed."""
    llm_called = False

    async def mock_llm(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        return {
            "tone": "tired_overwhelmed",
            "intensity": "medium",
            "themes": ["body_physical_symptoms"],
            "risk_language": False,
            "reflection": "இன்று உங்கள் உடல் சோர்வாக இருப்பதை உணர்கிறீர்கள்.",
        }

    client.app.state.custom_llm_fn = mock_llm

    # Only 5 words, but 55 characters excluding spaces
    tamil_text = "இன்றுஎன் உடல்மிகவும் சோர்வாகஇருக்கிறது என்னால்முடியவில்லை"
    words = tamil_text.split()
    chars_no_spaces = len(tamil_text.replace(" ", ""))
    assert len(words) < 10
    assert chars_no_spaces >= 50

    payload = {
        "note_text": tamil_text,
        "language": "ta",
        "stage": "postnatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["tone"] == "tired_overwhelmed"
    assert llm_called is True


@pytest.mark.parametrize(
    "crisis_word,lang",
    [
        ("suicide", "en"),
        ("kill myself", "en"),
        ("मरना चाहती हूं", "hi"),
        ("আত্মহত্যা", "bn"),
        ("தற்கொலை", "ta"),
    ],
)
def test_crisis_keyword_in_one_word_note_returns_crisis_without_llm(
    client, auth_headers, crisis_word, lang
):
    """A crisis keyword on any length note immediately triggers crisis with NO LLM call."""
    llm_called = False

    async def mock_llm(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        return {}

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "note_text": crisis_word,
        "language": lang,
        "stage": "postnatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "crisis"
    assert data["support"] is True
    assert data["tone"] is None
    assert data["reflection"] is None
    assert llm_called is False
