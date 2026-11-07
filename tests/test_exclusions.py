"""Tests for clinical and safety key exclusion rules."""

import pytest


def test_questionnaire_prefix_is_rejected_with_400(client):
    payload = {
        "language": "hi",
        "items": [
            {"key": "questionnaire.phq9_q1", "text": "Little interest or pleasure in doing things?"}
        ],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 400
    errors = response.json()["detail"]["errors"]
    assert any("questionnaire.phq9_q1" in err and "excluded" in err for err in errors)


def test_safety_prefix_is_rejected_with_400(client):
    payload = {
        "language": "bn",
        "items": [
            {"key": "safety.emergency_helpline", "text": "If you are in crisis, please call Tele-MANAS."}
        ],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 400
    errors = response.json()["detail"]["errors"]
    assert any("safety.emergency_helpline" in err and "excluded" in err for err in errors)


def test_mixed_valid_and_excluded_keys_are_rejected(client):
    payload = {
        "language": "ta",
        "items": [
            {"key": "dashboard.title", "text": "Your Dashboard"},
            {"key": "questionnaire.gad7_q2", "text": "Not being able to stop or control worrying"},
        ],
    }
    response = client.post("/api/translate", json=payload)
    assert response.status_code == 400
    errors = response.json()["detail"]["errors"]
    assert any("questionnaire.gad7_q2" in err for err in errors)
