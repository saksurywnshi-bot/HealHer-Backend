"""Unit tests for BEAM Phrasing Layer, Output Validator, and Deterministic Fallback."""

import pytest
from app.services.insights.phrasing import (
    validate_model_output,
    build_fallback_template,
    generate_phrased_insights,
)


def test_validator_rejects_diagnosis_words():
    """Output validator strictly rejects words like 'depression', 'bipolar', 'mania', 'psychosis', 'disorder'."""
    bad_texts = [
        "This week shows signs of clinical depression.",
        "Your mood swings look like bipolar disorder.",
        "Watch out for postpartum psychosis.",
        "This pattern indicates a depressive disorder.",
        "Symptoms of mania should be checked.",
    ]
    for text in bad_texts:
        valid, violations = validate_model_output(text, tier=1)
        assert valid is False, f"Expected validation failure for: {text}"
        assert len(violations) > 0


def test_validator_rejects_medication_names():
    """Output validator strictly rejects any medication or pill mentions."""
    bad_texts = [
        "You might benefit from taking sertraline.",
        "Ask your doctor about antidepressants or pills.",
        "Lithium or mood stabilizers could help with sleep.",
        "Consider asking for a sleeping pill or medication.",
    ]
    for text in bad_texts:
        valid, violations = validate_model_output(text, tier=1)
        assert valid is False, f"Expected validation failure for: {text}"
        assert any("medication" in v.lower() for v in violations)


def test_validator_rejects_alarming_terms_below_tier_3():
    """Output validator rejects alarming words like 'danger' or 'crisis' below Tier 3."""
    bad_texts = [
        "You are in danger of getting worse.",
        "This is an emergency situation.",
        "A mental health crisis has been detected.",
    ]
    for text in bad_texts:
        valid, violations = validate_model_output(text, tier=2)
        assert valid is False, f"Expected validation failure for: {text}"


def test_validator_accepts_clean_warm_phrasing():
    """Output validator passes clean, supportive, normalising language."""
    clean_text = (
        "This was a gentle and balanced week. Taking time to notice these steady days "
        "is a wonderful way to care for yourself as you prepare for the baby."
    )
    valid, violations = validate_model_output(clean_text, tier=0)
    assert valid is True
    assert len(violations) == 0


def test_tier_3_text_is_never_produced_by_model():
    """Tier 3 output is strictly the pre-approved fixed clinical script."""
    eval_res = {
        "weekly_band": "tougher_low",
        "tier": 3,
        "tier_name": "tier3_urgent_same_day",
        "reasons": ["Severe Elevated (+3) recorded."],
    }
    suggestions = [{"id": "TG1", "text": "Start small."}]

    # Even with use_llm=True, Tier 3 must use fixed pre-approved template script
    result = generate_phrased_insights(eval_res, suggestions, use_llm=True)
    assert result["source"] == "deterministic_template"
    assert "Tele-MANAS" in result["tier_message"]
    assert "112 in India" in result["tier_message"]


def test_deterministic_template_fallback_structure():
    """Verify fallback template contains all 7 required components."""
    eval_res = {
        "weekly_band": "good_balanced",
        "weekly_strength": "Steady daily tracking.",
        "monthly_summary": {"average_sleep_hours": 7.5},
        "tier": 0,
    }
    suggestions = [
        {"id": "GD1", "text": "This was a balanced week. That's worth noticing and being proud of."},
        {"id": "GD2", "text": "Keep your sleep and meals steady, and rest whenever you can."},
    ]
    fallback = build_fallback_template(eval_res, suggestions)
    assert "balanced week" in fallback["opening"].lower()
    assert fallback["strength"] == "Steady daily tracking."
    assert "7.5 hours" in fallback["sleep_note"]
    assert len(fallback["suggestions"]) == 2
    assert "This app does not diagnose" in fallback["disclaimer"]
