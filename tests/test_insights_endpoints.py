"""Integration tests for BEAM Insights API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
AUTH_HEADERS = {"x-user-ref": "test_user_insight_123"}


def test_check_entry_tier_3_trigger_on_plus_3():
    """POST /api/insights/check-entry triggers Tier 3 on single Severe Elevated entry."""
    response = client.post(
        "/api/insights/check-entry",
        headers=AUTH_HEADERS,
        json={
            "entry": {"log_date": "2026-10-27", "mood_beam": 3, "sleep_hours": 8.0},
            "language": "en",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_tier3"] is True
    assert "Tele-MANAS" in data["script"]
    assert data["emergency_number"] == "112"
    assert data["fail_safe"] is False


def test_check_entry_tier_3_trigger_on_crisis_text():
    """POST /api/insights/check-entry triggers Tier 3 on crisis text regardless of mood."""
    response = client.post(
        "/api/insights/check-entry",
        headers=AUTH_HEADERS,
        json={
            "entry": {"log_date": "2026-10-27", "mood_beam": 0, "sleep_hours": 8.0},
            "note_text": "I feel hopeless and want to end my life",
            "language": "en",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_tier3"] is True
    assert any("end my life" in r for r in data["reasons"])


def test_check_entry_normal_checkin():
    """POST /api/insights/check-entry returns no Tier 3 on balanced day."""
    response = client.post(
        "/api/insights/check-entry",
        headers=AUTH_HEADERS,
        json={
            "entry": {"log_date": "2026-10-27", "mood_beam": 0, "sleep_hours": 8.0},
            "note_text": "Feeling good after a gentle walk in the garden.",
            "language": "en",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_tier3"] is False
    assert len(data["reasons"]) == 0


def test_evaluate_endpoint_tier_2_trigger():
    """POST /api/insights/evaluate evaluates 14 days and triggers Tier 2."""
    entries = [
        {"log_date": f"2026-10-{i+1:02d}", "mood_beam": -1 if i < 8 else 0, "sleep_hours": 8.0}
        for i in range(14)
    ]
    response = client.post(
        "/api/insights/evaluate",
        headers=AUTH_HEADERS,
        json={
            "entries": entries,
            "phase": "pregnancy",
            "today": "2026-10-14",
            "language": "en",
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["tier"] == 2
    assert "offer_epds" in data["evaluation"]["actions"]
    assert len(data["suggestions"]) >= 2
    assert "This app does not diagnose" in data["phrased"]["disclaimer"]


def test_evaluate_endpoint_not_enough_data():
    """POST /api/insights/evaluate returns friendly message when fewer than 4 entries in 7 days."""
    entries = [
        {"log_date": "2026-10-01", "mood_beam": 0, "sleep_hours": 8.0},
        {"log_date": "2026-10-02", "mood_beam": 0, "sleep_hours": 8.0},
    ]
    response = client.post(
        "/api/insights/evaluate",
        headers=AUTH_HEADERS,
        json={
            "entries": entries,
            "phase": "pregnancy",
            "today": "2026-10-07",
            "language": "en",
            "use_llm": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["weekly_band"] == "not_enough_data"
    assert "A quick daily check-in makes these insights more useful" in data["phrased"]["opening"]


def test_user_profile_crud():
    """GET and POST /api/insights/profile."""
    post_res = client.post(
        "/api/insights/profile",
        headers=AUTH_HEADERS,
        json={
            "phase": "postpartum",
            "delivery_date": "2026-10-01",
            "usual_sleep_hours": 7.0,
            "history_bipolar_psychosis": True,
        },
    )
    assert post_res.status_code == 200
    assert post_res.json()["phase"] == "postpartum"
    assert post_res.json()["usual_sleep_hours"] == 7.0

    get_res = client.get("/api/insights/profile", headers=AUTH_HEADERS)
    assert get_res.status_code == 200
    assert get_res.json()["history_bipolar_psychosis"] is True


def test_user_consent_crud():
    """GET and POST /api/insights/consent."""
    post_res = client.post(
        "/api/insights/consent",
        headers=AUTH_HEADERS,
        json={"consented": True},
    )
    assert post_res.status_code == 200
    assert post_res.json()["consented"] is True

    get_res = client.get("/api/insights/consent", headers=AUTH_HEADERS)
    assert get_res.status_code == 200
    assert get_res.json()["consented"] is True


def test_questionnaires_metadata():
    """GET /api/insights/questionnaires returns EPDS, GAD-7, and MDQ definitions."""
    res = client.get("/api/insights/questionnaires")
    assert res.status_code == 200
    data = res.json()
    assert "epds" in data
    assert "gad7" in data
    assert "mdq" in data
    assert "phq9" in data
    assert data["epds"]["cutoff_positive"] == 10
    assert data["gad7"]["cutoff_positive"] == 5
    assert data["mdq"]["cutoff_positive_yes_count"] == 7
    assert data["phq9"]["cutoff_positive"] == 10
    assert "hi" in data["phq9"]["translations"]
    assert "ta" in data["phq9"]["translations"]
    assert "bn" in data["phq9"]["translations"]
    assert "hi" in data["epds"]["translations"]
    assert "ta" in data["epds"]["translations"]
    assert "bn" in data["epds"]["translations"]
    assert len(data["epds"]["translations"]["hi"]["items"]) == 10
    assert len(data["epds"]["translations"]["ta"]["items"]) == 10
    assert len(data["epds"]["translations"]["bn"]["items"]) == 10


def test_questionnaires_submit_phq9_item_9_escalates_to_tier_3():
    """Submitting PHQ-9 with Item 9 > 0 triggers self-harm flag and Tier 3 safety escalation."""
    res = client.post(
        "/api/insights/questionnaires/submit",
        headers=AUTH_HEADERS,
        json={
            "screening_type": "phq9",
            "answers": {
                "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                "6": 0, "7": 0, "8": 0, "9": 2,
            },
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["tier"] == 3
    assert data["self_harm_flag"] is True
    assert "Tele-MANAS" in data["tier_message"]


def test_questionnaires_submit_epds_item_10_escalates_to_tier_3():
    """Submitting EPDS with Item 10 >= 2 ('Sometimes') returns Tier 3, whereas Item 10 == 1 ('Hardly ever') does not."""
    # Case 1: Hardly ever (score 1) does NOT escalate to Tier 3
    res_hardly_ever = client.post(
        "/api/insights/questionnaires/submit",
        headers=AUTH_HEADERS,
        json={
            "screening_type": "epds",
            "answers": {
                "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                "6": 0, "7": 0, "8": 0, "9": 0, "10": 1,
            },
        },
    )
    assert res_hardly_ever.status_code == 200
    data_he = res_hardly_ever.json()
    assert data_he["tier"] == 0
    assert data_he["self_harm_flag"] is False

    # Case 2: Sometimes (score 2) DOES escalate to Tier 3
    res = client.post(
        "/api/insights/questionnaires/submit",
        headers=AUTH_HEADERS,
        json={
            "screening_type": "epds",
            "answers": {
                "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
                "6": 0, "7": 0, "8": 0, "9": 0, "10": 2,
            },
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["tier"] == 3
    assert data["self_harm_flag"] is True
    assert "Tele-MANAS" in data["tier_message"]


def test_doctor_summary_endpoint():
    """GET /api/insights/doctor-summary returns printable clinical report data."""
    res = client.get("/api/insights/doctor-summary", headers=AUTH_HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert "header" in data
    assert "latest_classification" in data
    assert "history" in data
    assert "Not a diagnostic report" in data["header"]["notice"]
