"""Tests for escalation logic, clinical mismatch rules, and suggestions library."""

import pytest
from app.services.suggestions_service import pick_suggestions


@pytest.mark.parametrize(
    "tone,intensity,risk_lang,expected_support",
    [
        ("positive", "low", False, False),
        ("neutral", "medium", False, False),
        ("distressed", "low", False, True),  # Distressed tone always escalates
        ("distressed", "high", False, True),
        ("anxious", "high", False, True),  # Anxious high intensity escalates
        ("low", "high", False, True),  # Low high intensity escalates
        ("anxious", "medium", False, False),  # Medium anxiety does not escalate
        ("low", "low", False, False),
        ("positive", "low", True, True),  # Risk language always escalates
    ],
)
def test_escalation_rules(
    client, auth_headers, tone, intensity, risk_lang, expected_support
):
    """Test plain code escalation logic across tones, intensities, and risk flags."""

    async def mock_llm(*args, **kwargs):
        return {
            "tone": tone,
            "intensity": intensity,
            "themes": ["tiredness"],
            "risk_language": risk_lang,
            "reflection": "A gentle reflection on your thoughts today.",
        }

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "note_text": "I am writing a sufficiently long note about how I am feeling today.",
        "language": "en",
        "stage": "postnatal",
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["support"] is expected_support


@pytest.mark.parametrize(
    "tone,mood_band,expected_mismatch",
    [
        ("anxious", "Normal", True),  # Anxious note with Normal band
        ("low", "Normal", True),  # Low note with Normal band
        ("distressed", "Mild Elevated", True),  # Distressed note with Elevated band
        ("positive", "Moderate Depressed", True),  # Positive note with Depressed band
        ("positive", "Normal", False),  # Agreement
        ("low", "Mild Depressed", False),  # Agreement
        ("anxious", None, False),  # No band provided -> no mismatch
    ],
)
def test_mismatch_rules(client, auth_headers, tone, mood_band, expected_mismatch):
    """Test plain code mismatch detection between note tone and clinical mood band."""

    async def mock_llm(*args, **kwargs):
        return {
            "tone": tone,
            "intensity": "medium",
            "themes": ["sleep"],
            "risk_language": False,
            "reflection": "A gentle reflection on your thoughts today.",
        }

    client.app.state.custom_llm_fn = mock_llm

    payload = {
        "note_text": "I am writing a sufficiently long note about how I am feeling today.",
        "language": "en",
        "stage": "postnatal",
        "mood_band": mood_band,
    }
    response = client.post("/api/sentiment/analyze", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["mismatch"] is expected_mismatch


def test_suggestions_draft_items_hidden_unless_dev_include_drafts(monkeypatch):
    """Draft suggestions are hidden when DEV_INCLUDE_DRAFTS=False, visible when True."""
    mock_items = [
        {
            "id": "sug_mock_draft",
            "tones": ["tired_overwhelmed"],
            "themes": ["sleep"],
            "stage": "prenatal",
            "kind": "do",
            "text_en": "Mock draft suggestion text",
            "status": "draft",
        }
    ]
    monkeypatch.setattr(
        "app.services.suggestion_selector.load_served_suggestions",
        lambda dev_include_drafts=None, app_env=None: mock_items if (dev_include_drafts and (app_env or "development") == "development") else []
    )

    # When drafts are excluded (default production behavior)
    picked_no_drafts = pick_suggestions(
        tone="tired_overwhelmed",
        themes=["sleep", "tiredness"],
        stage="prenatal",
        language="en",
        dev_include_drafts=False,
    )
    assert picked_no_drafts == []

    # When drafts are included
    picked_with_drafts = pick_suggestions(
        tone="tired_overwhelmed",
        themes=["sleep", "tiredness"],
        stage="prenatal",
        language="en",
        dev_include_drafts=True,
    )
    assert len(picked_with_drafts) == 1
    assert picked_with_drafts[0]["id"] == "sug_mock_draft"


def test_suggestions_fallback_to_english_when_language_missing(monkeypatch):
    """If requested language text is missing or TODO, falls back to English with text_language='en'."""
    mock_items = [
        {
            "id": "sug_mock_missing",
            "tones": ["anxious"],
            "themes": ["delivery_fear"],
            "stage": "prenatal",
            "kind": "do",
            "text_en": "English fallback suggestion",
            "text_hi": "TODO(Clinician/Translation)",
            "status": "approved",
        }
    ]
    monkeypatch.setattr(
        "app.services.suggestion_selector.load_served_suggestions",
        lambda dev_include_drafts=None, app_env=None: mock_items
    )

    picked = pick_suggestions(
        tone="anxious",
        themes=["delivery_fear"],
        stage="prenatal",
        language="hi",
        dev_include_drafts=True,
    )
    assert len(picked) == 1
    assert picked[0]["text_language"] == "en"
    assert picked[0]["text"] == "English fallback suggestion"
