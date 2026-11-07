"""Unit tests for BEAM Real-time Tier 3 Safety Layer."""

import pytest
from app.services.insights.safety import (
    check_multilingual_safety_language,
    evaluate_entry_safety,
)


def test_safety_severe_elevated_plus_3_triggers_tier_3():
    """Single Severe Elevated (+3) check-in immediately triggers Tier 3."""
    entry = {"log_date": "2026-10-27", "mood_beam": 3, "sleep_hours": 8.0}
    res = evaluate_entry_safety(entry)
    assert res["is_tier3"] is True
    assert "Severe Elevated (+3)" in res["reasons"][0]
    assert res["script"] is not None
    assert "Tele-MANAS" in res["script"]
    assert res["emergency_number"] == "112"


def test_safety_multilingual_crisis_keywords():
    """Verify conservative crisis language detection across en, hi, bn, and ta."""
    # English
    flagged, reasons = check_multilingual_safety_language("I feel like I want to die today", "en")
    assert flagged is True
    assert any("want to die" in r for r in reasons)

    # Hindi
    flagged, reasons = check_multilingual_safety_language("मुझे लगता है कि मैं मरना चाहती हूं", "hi")
    assert flagged is True
    assert any("मरना चाहती हूं" in r for r in reasons)

    # Bengali
    flagged, reasons = check_multilingual_safety_language("আমি আত্মহত্যা করতে চাই", "bn")
    assert flagged is True
    assert any("আত্মহত্যা" in r for r in reasons)

    # Tamil
    flagged, reasons = check_multilingual_safety_language("நான் தற்கொலை செய்து கொள்ள விரும்புகிறேன்", "ta")
    assert flagged is True
    assert any("தற்கொலை" in r for r in reasons)

    # Benign normal note
    flagged, reasons = check_multilingual_safety_language("I had a relaxing walk in the park", "en")
    assert flagged is False
    assert len(reasons) == 0


def test_safety_note_text_triggers_tier_3():
    """Note text containing crisis language triggers Tier 3 even if mood is normal."""
    entry = {"log_date": "2026-10-27", "mood_beam": 0, "sleep_hours": 8.0, "note_text": "I feel like hurting myself"}
    res = evaluate_entry_safety(entry)
    assert res["is_tier3"] is True
    assert any("Crisis phrase" in r for r in res["reasons"])


def test_safety_epds_item_10_positive_triggers_tier_3():
    """EPDS Item 10 > 1 ('Sometimes', 'Yes quite often') triggers Tier 3, while score 1 ('Hardly ever') does not."""
    entry = {"log_date": "2026-10-27", "mood_beam": 0}

    # Case 1: 'Hardly ever' (score 1) does NOT trigger Tier 3 warning
    epds_mild = {1: 0, 2: 1, 10: 1}
    res_mild = evaluate_entry_safety(entry, epds_answers=epds_mild)
    assert res_mild["is_tier3"] is False

    # Case 2: 'Sometimes' (score 2) DOES trigger Tier 3 warning
    epds_urgent = {1: 0, 2: 1, 10: 2}
    res_urgent = evaluate_entry_safety(entry, epds_answers=epds_urgent)
    assert res_urgent["is_tier3"] is True
    assert any("EPDS screening item 10 positive" in r for r in res_urgent["reasons"])


def test_safety_postpartum_consecutive_marked_and_reduced_sleep():
    """Postpartum 2 consecutive marked elevated days with reduced sleep triggers Tier 3."""
    recent_entries = [
        {"log_date": "2026-10-25", "mood_beam": 2, "sleep_hours": 5.0},
    ]
    current_entry = {"log_date": "2026-10-26", "mood_beam": 2, "sleep_hours": 5.2}

    res = evaluate_entry_safety(
        current_entry,
        recent_entries=recent_entries,
        phase="postpartum",
        usual_sleep=8.0,
    )
    assert res["is_tier3"] is True
    assert any("Postpartum warning signs" in r for r in res["reasons"])


def test_safety_layer_fails_safe_on_exception():
    """If unexpected error happens, safety layer must fail safe and return referral script."""
    # Pass an un-parsable entry that might raise error
    class BrokenEntry:
        def get(self, *args):
            raise RuntimeError("Database connection suddenly dropped")

    res = evaluate_entry_safety(BrokenEntry())
    assert res["is_tier3"] is True
    assert res["fail_safe"] is True
    assert res["script"] is not None
    assert "Please contact your doctor" in res["script"]
