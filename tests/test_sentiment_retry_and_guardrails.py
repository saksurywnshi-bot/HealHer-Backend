"""Tests for LangGraph output validation, retries, and clinical guardrails."""

import pytest


def test_invalid_enum_retries_and_returns_unavailable_if_persisting(client, auth_headers):
    """Invalid tone enum retries once and falls back to status: 'unavailable'."""
    call_count = 0

    async def mock_llm(note, lang, stage, retry_instruction):
        nonlocal call_count
        call_count += 1
        return {
            "tone": "invalid_super_happy",  # Invalid enum!
            "intensity": "low",
            "themes": ["sleep"],
            "risk_language": False,
            "reflection": "A gentle reflection sentence.",
        }

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "note_text": "I am writing a sufficiently long note about my daily check in today.",
        "language": "en",
        "stage": "prenatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unavailable"
    assert call_count == 2  # 1 initial attempt + 1 retry


def test_wrong_stage_theme_retries_and_returns_unavailable(client, auth_headers):
    """Postnatal theme (e.g. feeding_difficulty) given in prenatal stage is rejected."""
    call_count = 0

    async def mock_llm(note, lang, stage, retry_instruction):
        nonlocal call_count
        call_count += 1
        return {
            "tone": "anxious",
            "intensity": "medium",
            "themes": ["feeding_difficulty"],  # Prohibited in prenatal stage!
            "risk_language": False,
            "reflection": "You are worried about feeding difficulties today.",
        }

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "note_text": "I am writing a sufficiently long note about my daily check in today.",
        "language": "en",
        "stage": "prenatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unavailable"
    assert call_count == 2


def test_diagnostic_wording_retries_and_returns_unavailable(client, auth_headers):
    """Reflection containing diagnostic wording (e.g. 'you have clinical depression') is rejected."""
    call_count = 0

    async def mock_llm(note, lang, stage, retry_instruction):
        nonlocal call_count
        call_count += 1
        return {
            "tone": "low",
            "intensity": "high",
            "themes": ["tiredness"],
            "risk_language": False,
            "reflection": "You have clinical depression and need medical diagnosis.",  # Prohibited!
        }

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "note_text": "I am writing a sufficiently long note about my daily check in today.",
        "language": "en",
        "stage": "postnatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unavailable"
    assert call_count == 2


def test_retry_succeeds_when_model_corrects_itself(client, auth_headers):
    """If first attempt is invalid, model corrects itself on retry and succeeds."""
    call_count = 0

    async def mock_llm(note, lang, stage, retry_instruction):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt: invalid tone
            return {
                "tone": "invalid_mood",
                "intensity": "low",
                "themes": ["sleep"],
                "risk_language": False,
                "reflection": "First attempt reflection.",
            }
        # Second attempt (retry): valid tone and reflection
        assert retry_instruction is not None
        return {
            "tone": "tired_overwhelmed",
            "intensity": "medium",
            "themes": ["sleep"],
            "risk_language": False,
            "reflection": "You are feeling quite tired and needing more sleep today.",
        }

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "note_text": "I am writing a sufficiently long note about my daily check in today.",
        "language": "en",
        "stage": "postnatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["tone"] == "tired_overwhelmed"
    assert data["tone_phrase_key"] == "sentiment.tone.tired_overwhelmed"
    assert call_count == 2
